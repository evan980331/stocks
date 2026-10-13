"""策略規格（可獨立比較、可公平對照）。

此模組只定義「策略參數」與「比較用變體」，不含執行邏輯；
實際成交模擬由 `twstock.backtest.run_backtest` 負責。

持倉政策：
  - ``fixed``（預設）：進場後持有固定 ``hold_days`` 個交易日，準時出場。
  - ``hysteresis``：設定 ``exit_threshold`` 後改為訊號出場——
    至少持有 ``min_hold_days`` 日，之後只要「最近一個交易日的分數」跌破
    ``exit_threshold`` 就出場；無論分數如何，最長持有 ``hold_days`` 日。

  進出場門檻不對稱（enter >= buy_threshold，exit < exit_threshold < buy_threshold）
  就是「訊號緩衝區」，用來避免分數在門檻附近抖動造成無謂換手。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .config import CONFIG


@dataclass(frozen=True)
class StrategySpec:
    name: str = "baseline"
    top_n: int = 5                      # 同時持有的最大檔數
    hold_days: int = 5                  # 最長（fixed 政策則為固定）持有交易日數
    buy_threshold: float = 0.2          # 進場分數門檻
    exit_threshold: float | None = None  # None => fixed 政策
    min_hold_days: int = 0              # 最短持有交易日數（僅 hysteresis 有效）
    rebalance_days: int = 1             # 每 N 個交易日才評估新進場（再平衡頻率）
    limit_pct: float = CONFIG.limit_pct
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def policy(self) -> str:
        return "fixed" if self.exit_threshold is None else "hysteresis"

    def validate(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n 必須 > 0")
        if self.hold_days <= 0:
            raise ValueError("hold_days 必須 > 0")
        if self.rebalance_days <= 0:
            raise ValueError("rebalance_days 必須 > 0")
        if self.policy == "hysteresis":
            if self.exit_threshold is None:
                raise ValueError("hysteresis 政策必須設定 exit_threshold")
            if self.exit_threshold >= self.buy_threshold:
                raise ValueError(
                    "exit_threshold 必須小於 buy_threshold（否則無緩衝區）")
            if self.min_hold_days < 0 or self.min_hold_days > self.hold_days:
                raise ValueError("min_hold_days 必須在 0 與 hold_days 之間")

    def with_updates(self, **kw) -> "StrategySpec":
        return replace(self, **kw)


def baseline() -> StrategySpec:
    """目前系統預設策略（P2 前基準）。"""
    return StrategySpec(
        name="baseline_rule_h5",
        top_n=CONFIG.top_n,
        hold_days=CONFIG.hold_days,
        buy_threshold=0.2,
        rebalance_days=1,
        description=f"規則策略、持有 {CONFIG.hold_days} 日、每日再平衡、門檻 0.2",
        tags=("baseline", "rule"),
    )


def comparison_variants() -> list[StrategySpec]:
    """公平比較用變體：相同資料、相同成本、相同風險條件下互相比較。

    涵蓋任務要求的：(a) 現有規則策略 (b) 不同持有期間 (c) 降低換手頻率。
    融合策略因 ML 未啟用無法公平比較，於報告中明確標示（見 validation.py）。
    """
    base = baseline()
    variants = [base]
    # (b) 不同持有期間
    for hold in (10, 20):
        variants.append(base.with_updates(
            name=f"rule_h{hold}", hold_days=hold,
            description=f"規則策略、持有 {hold} 日", tags=("hold_period",)))
    # (c) 降低換手：拉長再平衡間隔
    for rb in (5, 10):
        variants.append(base.with_updates(
            name=f"rule_h5_rebal{rb}", rebalance_days=rb,
            description=f"規則策略、持有 5 日、每 {rb} 日再平衡",
            tags=("lower_turnover",)))
    # (c) 降低換手：提高進場門檻（只做高分訊號）
    variants.append(base.with_updates(
        name="rule_h5_thresh35", buy_threshold=0.35,
        description="規則策略、持有 5 日、進場門檻提高到 0.35",
        tags=("lower_turnover",)))
    # (c) 訊號緩衝區 + 最長持有（hysteresis）
    variants.append(base.with_updates(
        name="rule_hysteresis", hold_days=20, exit_threshold=0.0,
        min_hold_days=5, rebalance_days=5,
        description="規則策略、至少持有 5 日、分數 < 0 出場、最長 20 日、每 5 日再平衡",
        tags=("lower_turnover", "buffer")))
    for v in variants:
        v.validate()
    return variants


def variant_by_name(name: str) -> StrategySpec:
    for v in comparison_variants():
        if v.name == name:
            return v
    raise KeyError(f"未知策略 {name}；可用：{[v.name for v in comparison_variants()]}")
