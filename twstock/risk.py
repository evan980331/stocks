"""風險控制模組（集中管理，純函式好測試）。

設計原則：
  - 正式交易（``live``）預設關閉；研究/回測/模擬交易**不得**因設定錯誤而送出真實委託。
  - 所有限額集中於此，不得散落在策略中；未經使用者確認不得放寬。
  - Kill Switch 只「禁止新增正式委託」，不刪除紀錄、不自動大量平倉。

本模組目前只提供「檢查與決策」，實際下單介面（券商/模擬券商）屬後續階段（P4/P5）。
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config import DATA_DIR


class Mode(str, Enum):
    RESEARCH = "research"     # 研究：僅特徵/打分
    BACKTEST = "backtest"     # 歷史回測
    PAPER = "paper"           # 模擬交易（預設可用的最高層級）
    LIVE = "live"             # 正式交易（預設關閉）


def current_mode() -> Mode:
    """目前執行模式。預設 paper；live 必須以 TWSTOCK_MODE=live 顯式指定。"""
    raw = (os.environ.get("TWSTOCK_MODE") or "paper").strip().lower()
    try:
        return Mode(raw)
    except ValueError:
        return Mode.PAPER


def live_enabled() -> bool:
    """正式交易需同時滿足：模式為 live 且已設 TWSTOCK_LIVE_ACK=1（人工確認旗標）。"""
    return current_mode() is Mode.LIVE and os.environ.get("TWSTOCK_LIVE_ACK") == "1"


@dataclass(frozen=True)
class RiskLimits:
    """集中式風險限額（保守預設值；調整需經使用者確認並更新 plan.md）。"""

    max_order_value: float = 100_000.0        # 單筆委託金額上限（TWD）
    max_position_pct: float = 0.10            # 單一股票部位上限（佔總資產）
    max_gross_exposure: float = 0.60          # 整體持倉上限
    max_daily_loss_pct: float = 0.03          # 單日損失上限
    max_drawdown_pct: float = 0.15            # 最大回撤限制
    max_orders_per_day: int = 20              # 單日委託數上限
    min_liquidity_value: float = 10_000_000.0  # 最低 20 日均成交值（TWD）
    max_data_staleness_days: int = 3          # 可接受行情延遲
    max_open_orders: int = 10                 # 最大未完成委託數

    def validate(self) -> None:
        assert 0 < self.max_position_pct <= 1
        assert 0 < self.max_gross_exposure <= 1
        assert self.max_orders_per_day > 0


DEFAULT_LIMITS = RiskLimits()


@dataclass
class RiskDecision:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # 方便 if decision:
        return self.ok


class KillSwitch:
    """以檔案為基礎的緊急停止旗標（本機、可跨行程）。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or (Path(DATA_DIR) / "state" / "KILL_SWITCH"))

    def engaged(self) -> bool:
        return self.path.exists()

    def engage(self, reason: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            f"{dt.datetime.now().isoformat(timespec='seconds')} {reason}\n",
            encoding="utf-8")

    def release(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def reason(self) -> str:
        if not self.engaged():
            return ""
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""


@dataclass
class AccountState:
    """下單前必須能確認的帳戶/資料狀態。無法確認時一律不得下單。"""

    equity: float = 0.0
    position_value_by_stock: dict[str, float] = field(default_factory=dict)
    open_orders: int = 0
    orders_today: int = 0
    day_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    data_staleness_days: int = 0
    liquidity_20d_value: float = 0.0
    realized_position_reconciles: bool = True
    now: dt.datetime | None = None


def preflight(state: AccountState) -> RiskDecision:
    """強制停止條件（與單筆委託無關的全域檢查）。"""
    reasons: list[str] = []
    if state.equity <= 0:
        reasons.append("可用資金無法確認（equity<=0）")
    if not state.realized_position_reconciles:
        reasons.append("本地持倉與券商實際持倉不一致")
    if state.data_staleness_days > DEFAULT_LIMITS.max_data_staleness_days:
        reasons.append(f"行情過期（{state.data_staleness_days} 天）")
    if state.day_pnl_pct <= -DEFAULT_LIMITS.max_daily_loss_pct:
        reasons.append("觸發單日損失上限")
    if state.drawdown_pct <= -DEFAULT_LIMITS.max_drawdown_pct:
        reasons.append("觸發最大回撤限制")
    if state.open_orders >= DEFAULT_LIMITS.max_open_orders:
        reasons.append("未完成委託數達上限")
    if state.orders_today >= DEFAULT_LIMITS.max_orders_per_day:
        reasons.append("單日委託數達上限")
    return RiskDecision(not reasons, reasons)


def check_order(state: AccountState, *, stock_id: str, price: float, qty: int,
                side: str = "BUY") -> RiskDecision:
    """單筆委託檢查。任何一項不通過即不得送出。"""
    reasons = list(preflight(state).reasons)

    if not live_enabled() and current_mode() is Mode.LIVE:
        reasons.append("live 模式缺少人工確認旗標（TWSTOCK_LIVE_ACK）")
    if qty <= 0 or price <= 0:
        reasons.append("委託數量或價格不合法")

    value = price * qty
    if value > DEFAULT_LIMITS.max_order_value:
        reasons.append(f"單筆金額 {value:,.0f} 超過上限 {DEFAULT_LIMITS.max_order_value:,.0f}")

    if side.upper() == "BUY":
        held = state.position_value_by_stock.get(stock_id, 0.0)
        if held + value > state.equity * DEFAULT_LIMITS.max_position_pct:
            reasons.append("超過單一股票部位上限")
        gross = sum(state.position_value_by_stock.values()) + value
        if gross > state.equity * DEFAULT_LIMITS.max_gross_exposure:
            reasons.append("超過整體持倉上限")
        if state.liquidity_20d_value and state.liquidity_20d_value < DEFAULT_LIMITS.min_liquidity_value:
            reasons.append("流動性不足")

    return RiskDecision(not reasons, reasons)


def kill_switch_decision(ks: KillSwitch) -> RiskDecision:
    if ks.engaged():
        return RiskDecision(False, [f"Kill Switch 已啟用：{ks.reason()}"])
    return RiskDecision(True, [])
