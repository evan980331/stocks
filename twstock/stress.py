"""空頭市況壓力測試（P2.9）：真實壓力期間辨識、防禦能力與訊號檢驗。

本階段**只研究與評估**：不修改因子權重、曝險、持有期間或任何交易規則，
不接入真實下單，也不宣稱策略具備可交易優勢。

三個必須分開的概念（混用任何兩個都會得到錯誤結論）
  1. **期間辨識是事後描述**：壓力期間的「前高、低點、復原日」必然要看到整段
     資料才能界定，這是描述性統計，不是交易決策。
  2. **期間內的交易決策只能用當時已知資訊**：實際回測沿用
     :func:`twstock.backtest.run_backtest`（T 日收盤算分數 → T+1 開盤進場、
     漲跌停不成交），因此期間辨識的事後性質不會滲入下單邏輯。
  3. **報酬與曝險必須分離**：策略在下跌段「少賠」可能只是因為它沒進場
     （低曝險），不是選股好。故同時提供四個對照：

       - **現金基準**：0%（保守，不計無風險利率）
       - **等曝險基準**：以策略的實際平均曝險比例 x，每日再平衡持有 0050；
         報酬 = Π(1 + x·r_t) − 1。用來剝離「市場曝險效果」。
       - **0050 買進持有**：100% 曝險。
       - **隨機選股**：相同規則、相同成本、相同資金限制，只破壞
         「分數 ↔ 股票」的對應（沿用 :func:`twstock.attribution.shuffled_scores`）。

期間門檻事前固定
  回撤分級（10% / 15%）與空頭狀態（20 日與 60 日報酬同時為負）由任務指定；
  高波動門檻 = 20 日年化已實現波動率 ≥ 30%，理由見 ``VOL_THRESHOLD``。
  這些門檻在**任何策略結果產生之前**即固定，並提供敏感度對照，
  不得依事後績效反向挑選。

資料限制如實呈現（詳見 ``docs/BEAR_MARKET_STRESS_TEST.md``）
  - 個股價格自 2024-10-14 起（0050 自 2024-08-01），僅約 2 年。
  - 股票池是「今日仍上市」的 20 檔權值股 → survivorship bias。
  - 不同定義的期間高度重疊，**不得當成獨立樣本**；較嚴重分級是較寬分級的子集合。

指令：``python -m twstock.stress``
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .attribution import shuffled_scores
from .backtest import BacktestResult, _prepare, _schedule_entries, run_backtest
from .config import CONFIG, REPORTS_DIR
from .costs import CostModel, get_cost_model
from .diagnostics import (MIN_DAYS_FOR_IC, attach_forward_returns,
                          build_signal_frame, ic_series, ic_summary,
                          quantile_returns)
from .strategies import StrategySpec, baseline
from .validation import (MIN_TRADES_FOR_CONCLUSION, subset_by_dates,
                         subset_panel)

log = logging.getLogger(__name__)

TRADING_DAYS = 252

# ---- 事前固定的期間定義（在任何策略結果產生之前決定）----------------------
DRAWDOWN_TIERS: tuple[float, ...] = (0.10, 0.15)
TIER_MIN = DRAWDOWN_TIERS[0]
TIER_SEVERE = DRAWDOWN_TIERS[1]

VOL_WINDOW = 20
# 20 日年化已實現波動率門檻，事前固定為 30%：
#   (1) 約為台股長期平均波動（約 18~20%）的 1.5 倍；
#   (2) 本樣本自身 vol20 分布：中位數 20.6%、P75 30.2%、P90 36.9%、最大 62.1%，
#       故 30% 相當於該分布的 P75~P80，能挑出異常波動而非把「正常上漲」算進去。
VOL_THRESHOLD = 0.30
VOL_SENSITIVITY: tuple[float, ...] = (0.25, 0.35)

REGIME_WINDOWS: tuple[int, ...] = (20, 60)   # 兩者同時為負 → 空頭狀態
MIN_MASK_DAYS = 5        # 連續天數低於此不予視為期間（濾除單日雜訊）
MIN_WINDOW_DAYS = 10     # 期間短於此交易日數不做策略比較（樣本不足）

DEFAULT_SIMS = 200       # 隨機選股模擬次數
HORIZON = 5              # 訊號檢驗持有期（= 現行策略 hold_days）
# 少於約 6 個月不做年化：短窗年化會把波動放大成誤導性的數字
MIN_DAYS_FOR_ANNUALIZE = 126
MAX_PERIODS_PER_KIND = 6


# --------------------------------------------------------------------------
# 輸出格式（無法計算一律 N/A，不輸出 nan）
# --------------------------------------------------------------------------
def _pct(x: float, digits: int = 2) -> str:
    if x is None or not math.isfinite(float(x)):
        return "N/A"
    return f"{float(x):+.{digits}%}"


def _num(x: float, digits: int = 2) -> str:
    if x is None or not math.isfinite(float(x)):
        return "N/A"
    return f"{float(x):.{digits}f}"


def _pctx(x: float, digits: int = 1) -> str:
    """無號百分比；無法計算回 N/A（**不得輸出 nan%**）。"""
    if x is None or not math.isfinite(float(x)):
        return "N/A"
    return f"{float(x):.{digits}%}"


def _d(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    try:
        return f"{pd.Timestamp(x):%Y-%m-%d}"
    except (TypeError, ValueError):  # pragma: no cover - 防禦性
        return "—"


# --------------------------------------------------------------------------
# 壓力期間辨識（只用「當時已知」的資訊界定狀態，描述性統計為事後）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StressPeriod:
    """一個壓力期間。``start``／``end`` 為**下跌段**（前高 → 低點）。

    ``recovery_date`` 為回到前高的日期（未復原則 None）；``episode_end``
    為全段終點（= 復原日或資料末日）。遮罩型期間（高波動、空頭狀態）
    沒有前高／低點，相關欄位為 None/NaN。
    """

    kind: str                       # drawdown | volatility | bear_regime
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_dates: int
    severity: float                 # drawdown: 最大回撤（負值）；其他：最大值
    severe: bool = False
    peak_date: pd.Timestamp | None = None
    peak_value: float = float("nan")
    trough_date: pd.Timestamp | None = None
    trough_value: float = float("nan")
    max_drawdown: float = float("nan")
    recovery_date: pd.Timestamp | None = None
    episode_end: pd.Timestamp | None = None
    severity_label: str = ""
    note: str = ""

    def label_long(self) -> str:
        return f"{self.label}｜{_d(self.start)}~{_d(self.end)}（{self.n_dates} 日）"


def _new_high_segments(close: np.ndarray) -> list[tuple[int, int]]:
    """把序列切成「前高 → 下一個新高」的區段 ``[s, e)``。

    ``close[s]`` 必為該區段之前的最高點（新增紀錄高點）。回傳非重疊區段，
    因此由同一段落推出的期間**不會重疊**。
    """
    n = len(close)
    if n < 2:
        return []
    highs: list[int] = [0]
    running = close[0]
    for i in range(1, n):
        if close[i] > running:
            highs.append(i)
            running = close[i]
    out: list[tuple[int, int]] = []
    for k, s in enumerate(highs):
        e = highs[k + 1] if k + 1 < len(highs) else n
        if e - s >= 2:              # 至少一個下跌 bar 才算一段
            out.append((s, e))
    return out


def drawdown_episodes(close: pd.Series, *,
                      tiers: tuple[float, ...] = DRAWDOWN_TIERS,
                      min_tier: float | None = TIER_MIN) -> list[StressPeriod]:
    """由基準收盤序列辨識「自前高回撤達門檻」的期間（非重疊）。

    ``min_tier`` 為 None 時回傳**所有**回撤段落（含輕微），供完整度檢查；
    否則只回傳最大回撤不小於 ``min_tier`` 的期間。
    """
    s = pd.to_numeric(close, errors="coerce").astype(float)
    s = s.dropna()
    if len(s) < 3:
        return []
    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    vals = s.to_numpy(dtype=float)
    peak = np.maximum.accumulate(vals)
    dd = vals / peak - 1.0
    out: list[StressPeriod] = []
    for (a, b) in _new_high_segments(vals):
        seg = dd[a:b]
        if len(seg) == 0:
            continue
        t = a + int(np.argmin(seg))
        mdd = float(seg.min())
        if not math.isfinite(mdd) or mdd >= 0:
            continue
        if min_tier is not None and mdd > -abs(float(min_tier)):
            continue
        recovered = None
        if b < len(vals) and vals[b] >= vals[a]:
            recovered = idx[b]
        ep_end = idx[b] if b < len(vals) else idx[-1]
        # 「較嚴重」取**最高**分級（例如 15%），不是最低分級
        severe = mdd <= -abs(max(tiers)) if tiers else False
        note = "" if recovered is not None else "資料結束前尚未回到前高"
        out.append(StressPeriod(
            kind="drawdown",
            label=(f"回撤 ≥{abs(min_tier):.0%}" if min_tier is not None
                   else "回撤（任意）"),
            start=idx[a], end=idx[t], n_dates=int(t - a + 1),
            severity=mdd, severe=severe,
            peak_date=idx[a], peak_value=float(vals[a]),
            trough_date=idx[t], trough_value=float(vals[t]),
            max_drawdown=mdd, recovery_date=recovered, episode_end=ep_end,
            note=note))
    return out


def mask_episodes(index, mask, *, kind: str, label: str,
                  severity: np.ndarray | None = None,
                  min_days: int = MIN_MASK_DAYS) -> list[StressPeriod]:
    """把布林遮罩的**連續區段**轉為期間（過短的區段濾除）。"""
    idx = pd.DatetimeIndex(index)
    m = np.asarray(mask, dtype=bool)
    if len(idx) != len(m):
        raise ValueError("mask 長度必須與 index 相同")
    out: list[StressPeriod] = []
    i = 0
    while i < len(m):
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(m) and m[j + 1]:
            j += 1
        if j - i + 1 >= min_days:
            sev = float("nan")
            if severity is not None:
                sev = float(np.nanmax(np.asarray(severity, dtype=float)[i:j + 1]))
            out.append(StressPeriod(
                kind=kind, label=label, start=idx[i], end=idx[j],
                n_dates=int(j - i + 1), severity=sev))
        i = j + 1
    return out


def volatility_series(close: pd.Series, *, window: int = VOL_WINDOW) -> pd.Series:
    """年化已實現波動率（滾動標準差 × √252）。"""
    s = pd.to_numeric(close, errors="coerce").astype(float)
    return s.pct_change().rolling(window).std() * math.sqrt(TRADING_DAYS)


def regime_frame(close: pd.Series, *,
                 windows: tuple[int, ...] = REGIME_WINDOWS) -> pd.DataFrame:
    """各觀察窗的累積報酬（小數）。"""
    s = pd.to_numeric(close, errors="coerce").astype(float)
    return pd.DataFrame({f"ret{w}": s.pct_change(w) if w > 0 else s
                         for w in windows})


def benchmark_close(benchmark: pd.DataFrame) -> pd.Series:
    """``(date, close)`` 長表 → 依日期排序的收盤序列（缺值不補）。"""
    if benchmark is None or benchmark.empty:
        return pd.Series(dtype=float)
    if "date" not in benchmark.columns or "close" not in benchmark.columns:
        return pd.Series(dtype=float)
    b = benchmark.copy()
    b["date"] = pd.to_datetime(b["date"]).dt.normalize()
    b["close"] = pd.to_numeric(b["close"], errors="coerce")
    b = (b.dropna(subset=["close"]).sort_values("date")
         .drop_duplicates("date", keep="last"))
    return pd.Series(b["close"].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(b["date"]), name="close")


def identify_periods(benchmark: pd.DataFrame, *,
                     vol_threshold: float = VOL_THRESHOLD
                     ) -> dict[str, list[StressPeriod]]:
    """四種固定規則的壓力期間：回撤 10%／15%、高波動、空頭狀態。

    回傳 ``{kind: [period, ...]}``；``drawdown_severe`` 是 ``drawdown`` 的
    **子集合**，不是獨立樣本。
    """
    close = benchmark_close(benchmark)
    out: dict[str, list[StressPeriod]] = {
        "drawdown": drawdown_episodes(close), "drawdown_severe": [],
        "volatility": [], "bear_regime": [],
    }
    if len(close) < 3:
        return out
    for p in out["drawdown"]:
        if p.max_drawdown <= -TIER_SEVERE:
            out["drawdown_severe"].append(StressPeriod(
                **{**p.__dict__, "label": "回撤 ≥15%（較嚴重）", "severe": True}))
    vol = volatility_series(close)
    out["volatility"] = [
        StressPeriod(**{**p.__dict__,
                        "severity_label": "期間最大 20 日年化波動"})
        for p in mask_episodes(
            vol.index, (vol >= vol_threshold).to_numpy(), kind="volatility",
            label=f"高波動（20 日年化 ≥{vol_threshold:.0%}）",
            severity=vol.to_numpy(dtype=float))]
    reg = regime_frame(close)
    if len(reg.columns) >= 2:
        vals = close.to_numpy(dtype=float)
        dd = vals / np.maximum.accumulate(vals) - 1.0
        bear = (reg.iloc[:, 0] < 0) & (reg.iloc[:, 1] < 0)
        out["bear_regime"] = [
            StressPeriod(**{**p.__dict__, "severity_label": "期間最大回撤"})
            for p in mask_episodes(
                reg.index, bear.to_numpy(), kind="bear_regime",
                label=(f"空頭狀態（{REGIME_WINDOWS[0]} 日與 "
                       f"{REGIME_WINDOWS[1]} 日報酬同時為負）"),
                severity=dd)]
    return out


# --------------------------------------------------------------------------
# 資料可行性稽核
# --------------------------------------------------------------------------
@dataclass
class Coverage:
    """價格與特徵的涵蓋範圍、品質與缺口（全部由實際資料算出）。"""

    pool_size: int = 0
    n_price_rows: int = 0
    n_bench_rows: int = 0
    stock_start: pd.Timestamp | None = None
    stock_end: pd.Timestamp | None = None
    bench_start: pd.Timestamp | None = None
    bench_end: pd.Timestamp | None = None
    n_dates: int = 0
    n_bench_dates: int = 0
    n_bench_only_dates: int = 0
    n_dates_missing_stock: int = 0
    min_stocks_per_day: int = 0
    median_stocks_per_day: float = float("nan")
    n_days_below_pool: int = 0
    panel_start: pd.Timestamp | None = None
    panel_end: pd.Timestamp | None = None
    n_panel_rows: int = 0
    n_score_rows: int = 0
    warmup_dates_without_60d: int = 0
    rows_dropped: int = 0
    actions_applied: int = 0
    actions_skipped: int = 0
    # 價格品質違規計數
    n_nonpositive: int = 0
    n_high_lt_low: int = 0
    n_close_outside: int = 0
    n_zero_volume: int = 0
    news_horizon: str = ""

    def to_text(self) -> str:
        L = ["[1] 資料可行性與涵蓋範圍（實際資料計算，非假設）"]
        L.append(f"  股票池 {self.pool_size} 檔（今日仍上市；見 survivorship 限制）")
        L.append(f"  價格列 {self.n_price_rows:,}；個股區間 "
                 f"{_d(self.stock_start)} ~ {_d(self.stock_end)}"
                 f"（{self.n_dates} 交易日）")
        L.append(f"  基準列 {self.n_bench_rows:,}；區間 {_d(self.bench_start)} ~ "
                 f"{_d(self.bench_end)}（{self.n_bench_dates} 交易日）")
        L.append(f"  特徵列 {self.n_panel_rows:,}；分數列 {self.n_score_rows:,}"
                 f"（{_d(self.panel_start)} ~ {_d(self.panel_end)}）")
        L.append(f"  基準早於個股的可交易日數：{self.n_bench_only_dates}"
                 f"（此段僅有基準，無法用於策略比較 → 歷史缺口）")
        L.append(f"  每日有效股票數：最少 {self.min_stocks_per_day}、"
                 f"中位數 {_num(self.median_stocks_per_day, 1)}、"
                 f"不足 {self.pool_size} 檔的天數 {self.n_days_below_pool}")
        L.append(f"  公司行動：套用 {self.actions_applied}、捨棄 {self.actions_skipped}、"
                 f"移除無效價格列 {self.rows_dropped}")
        L.append(f"  價格品質違規：非正價格 {self.n_nonpositive}、"
                 f"high<low {self.n_high_lt_low}、close 超出 [low,high] "
                 f"{self.n_close_outside}、零成交量 {self.n_zero_volume}")
        L.append(f"  60 日窗尚未成熟的前導交易日：{self.warmup_dates_without_60d}"
                 f"（此期間 60 日動能／均線因子不計入，可用因子較少）")
        if self.news_horizon:
            L.append(f"  新聞因子：{self.news_horizon}")
        return "\n".join(L)


def coverage_report(price_panel: dict[str, pd.DataFrame],
                    panel: pd.DataFrame, benchmark: pd.DataFrame, *,
                    qa: dict | None = None,
                    news_status: str = "") -> Coverage:
    """由實際面板計算涵蓋範圍與品質（不讀任何外部來源）。"""
    cov = Coverage(pool_size=len(CONFIG.pool))
    qa = qa or {}
    rows = []
    for sid, df in (price_panel or {}).items():
        # 基準（0050）不在股票池內，且其歷史比個股長；若混入會把
        # 「個股起始日」與「每日有效股票數」算錯。
        if sid == CONFIG.benchmark or df is None or df.empty:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        d["stock_id"] = sid
        rows.append(d)
    if rows:
        allp = pd.concat(rows, ignore_index=True)
        cov.n_price_rows = int(len(allp))
        for col in ("open", "close", "low", "high"):
            if col in allp.columns:
                allp[col] = pd.to_numeric(allp[col], errors="coerce")
        for col in ("open", "close", "low", "high"):
            if col in allp.columns:
                cov.n_nonpositive += int((allp[col] <= 0).sum())
        if {"high", "low"}.issubset(allp.columns):
            cov.n_high_lt_low = int((allp["high"] < allp["low"]).sum())
        if {"close", "low", "high"}.issubset(allp.columns):
            cov.n_close_outside = int(
                ((allp["close"] < allp["low"]) | (allp["close"] > allp["high"])).sum())
        if "volume" in allp.columns:
            vol = pd.to_numeric(allp["volume"], errors="coerce")
            cov.n_zero_volume = int((vol.fillna(0) <= 0).sum())
        cov.stock_start = pd.Timestamp(allp["date"].min())
        cov.stock_end = pd.Timestamp(allp["date"].max())
        per_day = allp.groupby("date")["stock_id"].nunique()
        cov.n_dates = int(per_day.size)
        cov.min_stocks_per_day = int(per_day.min())
        cov.median_stocks_per_day = float(per_day.median())
        cov.n_days_below_pool = int((per_day < cov.pool_size).sum())
    bclose = benchmark_close(benchmark)
    cov.n_bench_rows = int(len(bclose))
    if len(bclose):
        cov.bench_start = pd.Timestamp(bclose.index.min())
        cov.bench_end = pd.Timestamp(bclose.index.max())
        cov.n_bench_dates = int(len(bclose))
        if cov.stock_start is not None:
            cov.n_bench_only_dates = int(
                (bclose.index < cov.stock_start).sum())
            cov.n_dates_missing_stock = int(
                (bclose.index > cov.stock_end).sum())
    if panel is not None and not panel.empty:
        d = pd.to_datetime(panel["date"]).dt.normalize()
        cov.n_panel_rows = int(len(panel))
        cov.n_score_rows = int(len(panel))
        cov.panel_start = pd.Timestamp(d.min())
        cov.panel_end = pd.Timestamp(d.max())
        # 60 日窗需要前 60 個交易日；不足者該因子不計入
        if "mom_60" in panel.columns:
            first_ok = panel.copy()
            first_ok["_d"] = d
            per_day = first_ok.groupby("_d")["mom_60"].apply(
                lambda s: pd.to_numeric(s, errors="coerce").notna().any())
            cov.warmup_dates_without_60d = int((~per_day).sum())
    cov.rows_dropped = int(qa.get("rows_dropped", 0))
    cov.actions_applied = int(qa.get("actions_applied", 0))
    cov.actions_skipped = int(qa.get("actions_skipped", 0))
    cov.news_horizon = news_status
    return cov


# --------------------------------------------------------------------------
# 期間內策略 vs 基準（同規則、同成本、同資金）
# --------------------------------------------------------------------------
@dataclass
class WindowResult:
    """一個評估視窗內的策略、基準與對照組結果。"""

    label: str
    kind: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    n_dates: int
    years: float
    # 策略
    strat_total: float = 0.0
    strat_cagr: float = float("nan")
    strat_mdd: float = float("nan")
    strat_sharpe: float = float("nan")
    worst_1d: float = float("nan")
    worst_5d: float = float("nan")
    worst_20d: float = float("nan")
    win_rate: float = float("nan")
    n_trades: int = 0
    avg_holding: float = float("nan")
    exposure: float = float("nan")
    cash_share: float = float("nan")
    turnover_ann: float = float("nan")
    cost_total: float = 0.0
    cost_rate: float = float("nan")
    gross_total: float = float("nan")
    cost_drag: float = float("nan")
    # 基準
    bench_total: float = float("nan")
    bench_mdd: float = float("nan")
    cash_total: float = 0.0
    matched_total: float = float("nan")
    excess: float = float("nan")
    excess_matched: float = float("nan")
    # 隨機選股
    rand_median: float = float("nan")
    rand_p05: float = float("nan")
    rand_p95: float = float("nan")
    rand_percentile: float = float("nan")
    rand_p_value: float = float("nan")
    n_rand: int = 0
    # 覆蓋
    n_score_rows: int = 0
    n_evaluable_signals: int = 0
    n_excluded_tail: int = 0
    n_stocks: int = 0
    sufficient: bool = False
    note: str = ""

    def to_row(self) -> str:
        return (f"  {self.label:<30}{_pct(self.strat_total):>10}"
                f"{_pct(self.bench_total):>10}{_pct(self.excess):>10}"
                f"{_pct(self.matched_total):>10}{_pct(self.cash_total):>8}"
                f"{_pct(self.strat_mdd):>10}{self.n_trades:>6}")


def _worst_k(eq: pd.Series, k: int) -> float:
    """最差的 k 日累積報酬（不使用未來資訊：只是在已實現序列上取最小值）。"""
    if eq is None or len(eq) <= k:
        return float("nan")
    r = (eq.shift(-k) / eq - 1.0).dropna()
    return float(r.min()) if len(r) else float("nan")


def _cagr(total: float, years: float) -> float:
    """年化報酬；期間過短或總報酬非正時明確回 NaN（不輸出誤導性數字）。"""
    if not math.isfinite(total) or not math.isfinite(years) or years <= 0:
        return float("nan")
    if years * 252 < MIN_DAYS_FOR_ANNUALIZE:
        return float("nan")
    base = 1.0 + total
    if base <= 0:
        return float("nan")
    return float(base ** (1.0 / years) - 1.0)


def bench_buy_hold(benchmark: pd.DataFrame | pd.Series
                   ) -> tuple[float, float]:
    """基準買進持有：回傳 ``(期間總報酬, 期間最大回撤)``；資料不足回 (NaN, NaN)。"""
    if isinstance(benchmark, pd.Series):
        s = pd.to_numeric(benchmark, errors="coerce").dropna()
        c = s.to_numpy(dtype=float)
    elif benchmark is None or len(benchmark) == 0:
        return float("nan"), float("nan")
    else:
        if "close" not in benchmark.columns:
            return float("nan"), float("nan")
        c = pd.to_numeric(benchmark["close"], errors="coerce").dropna().to_numpy(
            dtype=float)
    if len(c) < 2 or c[0] <= 0:
        return float("nan"), float("nan")
    return (float(c[-1] / c[0] - 1.0),
            float(np.min(c / np.maximum.accumulate(c) - 1.0)))


def equal_exposure_return(bench_equity: pd.Series, exposure: float) -> float:
    """等曝險基準：固定比例 x 每日再平衡持有基準，報酬 = Π(1 + x·r_t) − 1。

    用來把「市場曝險效果」從「選股效果」中剝離。無基準資料時回 NaN。
    """
    if bench_equity is None or len(bench_equity) < 2:
        return float("nan")
    if not math.isfinite(exposure) or exposure <= 0:
        return 0.0
    r = bench_equity.pct_change().dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return float("nan")
    return float(np.prod(1.0 + exposure * r) - 1.0)


def evaluate_window(scores: pd.DataFrame,
                    price_panel: dict[str, pd.DataFrame],
                    benchmark: pd.DataFrame, spec: StrategySpec,
                    cost_model: CostModel, start: pd.Timestamp,
                    end: pd.Timestamp, *, n_sims: int = DEFAULT_SIMS,
                    seed: int = 0,
                    initial_capital: float = 1_000_000.0,
                    label: str = "", kind: str = "",
                    note: str = "", run_gross: bool = True) -> WindowResult:
    """在 ``[start, end]`` 內比較策略與四個對照組。

    價格與分數都**只取視窗內**，因此視窗結束後才發生的任何事都不會進入
    計算（無前視偏差）。代價是最後 ``hold_days`` 個交易日的訊號無法在
    視窗內完成一回合而被排除，該筆數如實記錄於 ``n_excluded_tail``。
    """
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    s = subset_by_dates(scores, lo, hi)
    p = subset_panel(price_panel, lo, hi)
    b = (subset_by_dates(benchmark, lo, hi)
         if benchmark is not None and not benchmark.empty else None)
    res = run_backtest(s, p, strategy=spec, benchmark=b, cost_model=cost_model,
                       initial_capital=initial_capital)
    out = WindowResult(
        label=label or f"{_d(lo)}~{_d(hi)}", kind=kind, start=lo, end=hi,
        n_dates=int(len(res.equity)), years=float(res.years), note=note)

    if not len(s) or not len(p):
        # 個股資料未涵蓋此區間（例：基準有更長歷史）。
        # 回測引擎對空分數會提早回傳，其基準報酬維持預設 0.0；
        # 若直接引用會把「沒有資料」誤報成「0% 報酬」，故這裡自行算基準，
        # 並將策略欄位一律設為 NaN（N/A），不以 0 冒充。
        bt, bm = bench_buy_hold(b)
        out.bench_total, out.bench_mdd = bt, bm
        nan = float("nan")
        out.strat_total = nan
        out.strat_cagr = nan
        out.strat_mdd = nan
        out.strat_sharpe = nan
        out.worst_1d = out.worst_5d = out.worst_20d = nan
        out.win_rate = nan
        out.exposure = nan
        out.cash_share = nan
        out.turnover_ann = nan
        out.cost_total = nan
        out.cost_rate = nan
        out.gross_total = nan
        out.cost_drag = nan
        out.matched_total = nan
        out.excess = nan
        out.excess_matched = nan
        out.n_stocks = int(len(p))
        out.sufficient = False
        out.note = note or ("個股資料未涵蓋此區間 → 策略不予評估"
                            "（不以 0% 冒充，僅列基準走勢）")
        return out

    out.strat_total = float(res.total_return)
    out.strat_cagr = _cagr(res.total_return, res.years)
    out.strat_mdd = float(res.max_drawdown)
    out.strat_sharpe = float(res.sharpe)
    eq = res.equity
    if eq is not None and len(eq):
        out.worst_1d = _worst_k(eq, 1)
        out.worst_5d = _worst_k(eq, 5)
        out.worst_20d = _worst_k(eq, 20)
    out.win_rate = float(res.win_rate) if res.round_trips else float("nan")
    out.n_trades = int(res.round_trips)
    out.avg_holding = float(res.avg_holding_days)
    out.exposure = float(res.exposure)
    out.cash_share = float(max(0.0, 1.0 - res.exposure))
    out.turnover_ann = float(res.portfolio_turnover_annualized)
    out.cost_total = float(res.cost_total)
    out.cost_rate = (float(res.cost_total) / initial_capital
                     if initial_capital else float("nan"))
    out.bench_total = float(res.benchmark_total_return)
    out.bench_mdd = float(res.benchmark_max_drawdown)
    out.matched_total = equal_exposure_return(res.benchmark_equity, out.exposure)
    out.cash_total = 0.0
    if len(res.benchmark_equity) > 1:
        out.excess = out.strat_total - out.bench_total
        out.excess_matched = out.strat_total - out.matched_total

    if run_gross:
        gross = run_backtest(s, p, strategy=spec, benchmark=None,
                             cost_model=get_cost_model("zero"),
                             initial_capital=initial_capital)
        out.gross_total = float(gross.total_return)
        out.cost_drag = out.strat_total - out.gross_total

    if n_sims > 0 and len(s):
        arr = np.asarray([
            run_backtest(shuffled_scores(s, seed=seed + k), p, strategy=spec,
                         benchmark=None, cost_model=cost_model,
                         initial_capital=initial_capital).total_return
            for k in range(n_sims)], dtype=float)
        arr = arr[np.isfinite(arr)]
        out.n_rand = int(len(arr))
        if len(arr):
            out.rand_median = float(np.median(arr))
            out.rand_p05 = float(np.quantile(arr, 0.05))
            out.rand_p95 = float(np.quantile(arr, 0.95))
            out.rand_percentile = float((arr <= out.strat_total).mean() * 100.0)
            out.rand_p_value = float((int((arr >= out.strat_total).sum()) + 1)
                                     / (len(arr) + 1))

    out.n_score_rows = int(len(s))
    out.n_stocks = int(len(p))
    book = _prepare(p)
    if book and len(s):
        sched = _schedule_entries(s, book, spec.hold_days)
        out.n_evaluable_signals = int(sum(len(v) for v in sched.values()))
    out.n_excluded_tail = int(max(0, out.n_score_rows - out.n_evaluable_signals))
    out.sufficient = bool(out.n_trades >= MIN_TRADES_FOR_CONCLUSION
                          and out.n_dates >= MIN_DAYS_FOR_IC
                          and out.n_dates >= MIN_WINDOW_DAYS)
    return out


# --------------------------------------------------------------------------
# 訊號檢驗（空頭 vs 非空頭）
# --------------------------------------------------------------------------
DEFAULT_SIGNALS: tuple[str, ...] = ("score", "rev_yoy", "mom_20")


def date_mask(index, *, lo: pd.Timestamp | None = None,
              hi: pd.Timestamp | None = None) -> pd.Series:
    """以交易日索引建立連續區間遮罩（避免用 [min,max] 誤含期間之間的日子）。"""
    idx = pd.DatetimeIndex(index)
    m = pd.Series(True, index=idx)
    if lo is not None:
        m &= idx >= pd.Timestamp(lo)
    if hi is not None:
        m &= idx <= pd.Timestamp(hi)
    return m


def _subset_mask(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """依日期遮罩取子集（不是取 [min,max] 區間）。"""
    if df is None or df.empty or mask is None or len(mask) == 0:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    keep = d["date"].map(mask).fillna(False).astype(bool)
    return d[keep]


@dataclass
class SignalCheck:
    signal: str
    segment: str
    horizon: int
    n_days: int = 0
    ic_mean: float = float("nan")
    ic_t_nw: float = float("nan")
    ic_p: float = float("nan")
    ic_ci_low: float = float("nan")
    ic_ci_high: float = float("nan")
    q_spread: float = float("nan")
    q_spread_net: float = float("nan")
    monotonicity: float = float("nan")
    ordered: bool = False
    sufficient: bool = False

    def to_row(self) -> str:
        return (f"  {self.signal:<16}{self.segment:<22}{self.n_days:>5}"
                f"{_num(self.ic_mean, 3):>9}{_num(self.ic_t_nw, 2):>8}"
                f"{_num(self.ic_p, 3):>8}"
                f"{_pct(self.q_spread, 3):>10}{_pct(self.q_spread_net, 3):>11}"
                f"{_num(self.monotonicity, 2):>7}"
                f"{'是' if self.ordered else '否':>4}")


def signal_checks(signal_frame: pd.DataFrame,
                  segments: list[tuple[str, pd.Series]],
                  *, signals: tuple[str, ...] = DEFAULT_SIGNALS,
                  horizon: int = HORIZON) -> list[SignalCheck]:
    """各區段的 Rank IC、五分組價差與單調性（觀察單位 = 交易日）。

    ``segments`` 為 ``[(標籤, 日期遮罩), ...]``；**必須用遮罩而非 [起,訖] 區間**，
    否則「壓力日」會把期間之間的正常交易日也算進去。

    時間對齊沿用 ``twstock.diagnostics``：T 日收盤訊號 → T+1 開盤進場，
    因此不會把未來的價格或報酬當成已知資訊。
    """
    out: list[SignalCheck] = []
    for seg_label, mask in segments:
        sub = _subset_mask(signal_frame, mask)
        for sg in signals:
            if signal_frame is None or signal_frame.empty or sg not in sub.columns:
                out.append(SignalCheck(signal=sg, segment=seg_label,
                                       horizon=horizon))
                continue
            ic = ic_series(sub, sg, horizon)
            summ = ic_summary(ic, sg, horizon, segment=seg_label)
            q = quantile_returns(sub, sg, horizon)
            out.append(SignalCheck(
                signal=sg, segment=seg_label, horizon=horizon,
                n_days=int(summ.n_days), ic_mean=summ.ic_mean,
                ic_t_nw=summ.t_nw, ic_p=summ.p_value,
                ic_ci_low=summ.ci_low, ic_ci_high=summ.ci_high,
                q_spread=q.spread, q_spread_net=q.spread_net,
                monotonicity=q.monotonicity, ordered=bool(q.ordered),
                sufficient=bool(summ.sufficient)))
    return out


def news_factor_status(signal_frame: pd.DataFrame,
                       segments: list[tuple[str, pd.Series]],
                       *, column: str = "news_sentiment") -> str:
    """新聞因子在各區段的實際數值狀態（不得宣稱不存在的資訊）。"""
    if signal_frame is None or signal_frame.empty or column not in signal_frame.columns:
        return "無新聞因子欄位"
    parts = []
    for seg_label, mask in segments:
        sub = _subset_mask(signal_frame, mask)
        v = (pd.to_numeric(sub[column], errors="coerce") if len(sub)
             else pd.Series(dtype=float))
        n = int(len(v))
        nz = int((v.fillna(0.0) != 0).sum()) if n else 0
        parts.append(f"{seg_label} 非零 {nz}/{n}")
    return "；".join(parts)


# --------------------------------------------------------------------------
# 報告
# --------------------------------------------------------------------------
@dataclass
class StressReport:
    generated_at: str
    spec: StrategySpec
    cost_model_name: str
    coverage: Coverage
    periods: dict[str, list[StressPeriod]]
    windows: list[WindowResult] = field(default_factory=list)
    signals: list[SignalCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    conclusions: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        L: list[str] = []
        L.append("=" * 96)
        L.append("空頭市況壓力測試報告（P2.9）")
        L.append(f"產生時間：{self.generated_at}")
        L.append("=" * 96)
        L.append(f"策略：{self.spec.name}（{self.spec.description}）"
                 f"｜成本情境：{self.cost_model_name}"
                 f"｜交易規則與因子權重未做任何修改")
        L.append("")
        L.append(self.coverage.to_text())
        L.append("")
        L.append("[2] 壓力期間辨識（門檻事前固定；辨識為事後描述，"
                 "期間內交易仍只用當時已知資訊）")
        for kind in ("drawdown", "drawdown_severe", "volatility", "bear_regime"):
            ps = self.periods.get(kind) or []
            if kind == "drawdown_severe":
                L.append(f"  -- {kind}（**drawdown 的子集合，不是獨立樣本**）："
                         f"{len(ps)} 段")
            else:
                L.append(f"  -- {kind}：{len(ps)} 段")
            for p in ps[:MAX_PERIODS_PER_KIND]:
                if p.kind == "drawdown" or not math.isnan(p.max_drawdown):
                    L.append(
                        f"     {p.label}｜前高 {_d(p.peak_date)}"
                        f"({_num(p.peak_value, 2)}) → 低點 {_d(p.trough_date)}"
                        f"({_num(p.trough_value, 2)})｜最大回撤 {_pct(p.max_drawdown)}"
                        f"｜下跌段 {p.n_dates} 日｜復原 "
                        f"{_d(p.recovery_date) if p.recovery_date is not None else '未復原'}"
                        f"｜全段 {_d(p.episode_end)}"
                        + (f"｜{p.note}" if p.note else ""))
                else:
                    L.append(f"     {p.label}｜{_d(p.start)} ~ {_d(p.end)}"
                             f"（{p.n_dates} 交易日）"
                             f"｜{p.severity_label or '期間最大值'} "
                             f"{_pct(p.severity)}")
            if len(ps) > MAX_PERIODS_PER_KIND:
                L.append(f"     （另有 {len(ps) - MAX_PERIODS_PER_KIND} 段未列）")
        L.append("")
        L.append("[3] 策略 vs 基準（每個視窗為獨立回測，不可相加複合）")
        L.append(f"  {'視窗':<30}{'策略':>10}{'0050':>10}{'超額':>10}"
                 f"{'等曝險':>10}{'現金':>8}{'策略MDD':>10}{'筆數':>6}")
        for w in self.windows:
            L.append(w.to_row())
        L.append("  等曝險 = 以策略實際平均曝險比例每日再平衡持有 0050"
                 "（剝離市場曝險效果）；現金 = 0%。")
        L.append("  拆解：市場曝險效果 = 等曝險 − 現金；選股效果 = 策略 − 等曝險")
        L.append(f"  {'視窗':<30}{'市場曝險效果':>13}{'選股效果':>11}{'等曝險差 0050':>15}")
        for w in self.windows:
            diff = (w.matched_total - w.bench_total
                    if math.isfinite(w.matched_total)
                    and math.isfinite(w.bench_total) else float("nan"))
            L.append(f"  {w.label:<30}{_pct(w.matched_total):>13}"
                     f"{_pct(w.excess_matched):>11}{_pct(diff):>15}")
        L.append("")
        L.append("[4] 曝險、成本與隨機對照拆解")
        L.append(f"  {'視窗':<30}{'曝險':>8}{'現金':>8}{'年化週轉':>10}"
                 f"{'成本':>10}{'成本率':>9}{'扣成本前':>10}{'隨機中位':>10}{'百分位':>8}")
        for w in self.windows:
            L.append(f"  {w.label:<30}{_pctx(w.exposure):>8}"
                     f"{_pctx(w.cash_share):>8}"
                     f"{_num(w.turnover_ann, 2):>10}"
                     f"{_num(w.cost_total, 0):>10}"
                     f"{_num(w.cost_rate, 4):>9}{_pct(w.gross_total):>10}"
                     f"{_pct(w.rand_median):>10}{_num(w.rand_percentile, 1):>8}")
        L.append("")
        L.append("[5] 回撤、波動與交易統計（每個視窗為獨立回測）")
        L.append(f"  {'視窗':<30}{'年化':>9}{'Sharpe':>8}{'策略MDD':>10}"
                 f"{'0050MDD':>9}{'最差1日':>9}{'最差5日':>9}{'最差20日':>10}"
                 f"{'勝率':>7}{'持有':>6}{'排除尾端':>9}{'樣本足夠':>9}")
        for w in self.windows:
            L.append(f"  {w.label:<30}{_pct(w.strat_cagr):>9}"
                     f"{_num(w.strat_sharpe, 2):>8}"
                     f"{_pct(w.strat_mdd):>10}{_pct(w.bench_mdd):>9}"
                     f"{_pct(w.worst_1d):>9}{_pct(w.worst_5d):>9}"
                     f"{_pct(w.worst_20d):>10}"
                     f"{_num(w.win_rate, 3):>7}{_num(w.avg_holding, 1):>6}"
                     f"{w.n_excluded_tail:>9}"
                     f"{'是' if w.sufficient else '否':>9}")
        L.append(f"  視窗內分數列數 / 可評估訊號 / 超過門檻的股票檔數：")
        for w in self.windows:
            L.append(f"    {w.label:<30}{w.n_score_rows:>7}"
                     f"{w.n_evaluable_signals:>10}{w.n_excluded_tail:>10}"
                     f"{w.n_stocks:>7} 檔"
                     + (f"｜{w.note}" if w.note else ""))
        L.append("  年化僅在視窗 ≥ 126 交易日時輸出，否則 N/A；"
                 "『排除尾端』= 因無法在視窗內完成一回合而未納入交易的訊號數（截斷）。")
        L.append("")
        L.append("[6] 空頭期間訊號檢驗（Rank IC、五分組價差；觀察單位 = 交易日）")
        L.append(f"  {'訊號':<16}{'區段':<22}{'天數':>5}{'IC均值':>9}{'NW t':>8}"
                 f"{'p值':>8}{'Q5-Q1':>10}{'扣成本':>11}{'單調':>7}{'有序':>4}")
        for c in self.signals:
            L.append(c.to_row())
        L.append("  註：IC 為全期間多重檢定的一員，未經 FDR 校正者不得單獨宣稱顯著；"
                 "單一期間的樣本數遠低於可推論水準，均屬探索性。")
        L.append("")
        L.append("[7] 已確認結論與無法判定事項（由上面各表的實際數字導出）")
        L.append("  已確認（可直接由表中數字驗算）：")
        for c in self.conclusions:
            L.append("    · " + c)
        L.append("  無法判定（樣本或資料不足，不得取而代之的推論）：")
        for n in self.notes:
            L.append("    - " + n)
        L.append("")
        L.append("[8] 下一步建議（不涉及調整權重、曝險或交易規則）")
        for s in self.next_steps:
            L.append("  - " + s)
        L.append("=" * 96)
        return "\n".join(L)


def _conclusions(report: StressReport) -> list[str]:
    """由實際計算出的數字導出結論（不是事先寫好的文字）。

    任何無法從報表中驗算的句子不得寫在這裡；不確定的一律歸到「無法判定」。
    """
    cov = report.coverage
    out: list[str] = []
    dd = report.periods.get("drawdown", [])
    sev = report.periods.get("drawdown_severe", [])
    legs = [w for w in report.windows
            if w.kind == "drawdown" and math.isfinite(w.strat_total)]
    fulls = [w for w in report.windows
             if w.kind == "drawdown_full" and math.isfinite(w.strat_total)]
    out.append(
        f"可辨識的回撤事件 {len(dd)} 段、其中 {len(sev)} 段回撤 ≥15%；"
        f"在個股資料涵蓋範圍內可實際評估的下跌段 {len(legs)} 段。"
        f"{_d(cov.stock_start)} 之前的 {cov.n_bench_only_dates} 個交易日僅有基準資料，"
        "不可用於策略比較（已標示為歷史缺口，不以較晚資料冒充）。")
    if legs:
        beat = [w for w in legs if math.isfinite(w.excess) and w.excess > 0]
        sel = [w for w in legs if math.isfinite(w.excess_matched)
               and w.excess_matched > 0]
        ex = [w.exposure for w in legs if math.isfinite(w.exposure)]
        out.append(f"下跌段 {len(legs)} 段中，策略優於 0050 買進持有者 {len(beat)} 段、"
                   f"優於**同曝險**基準者 {len(sel)} 段。")
        if ex:
            out.append(f"下跌段策略平均曝險 {np.mean(ex):.1%}"
                       f"（現金 {1 - np.mean(ex):.1%}）：相對抗跌有多少來自低曝險，"
                       "必須同時看「等曝險」與「選股效果」兩欄才能區分。")
        mdds = [w.strat_mdd for w in legs if math.isfinite(w.strat_mdd)]
        if mdds:
            worst = min(legs, key=lambda w: w.strat_mdd)
            out.append(f"下跌段策略自身的最大回撤介於 {min(mdds):.2%} ~ "
                       f"{max(mdds):.2%}；最差為 {worst.label}"
                       f"（{_pct(worst.strat_mdd)}，同期 0050 為 {_pct(worst.bench_mdd)}）。")
    if fulls:
        pos = [w for w in fulls if w.strat_total > 0]
        out.append(f"全段（前高 → 復原）{len(fulls)} 段中，策略為正報酬者 {len(pos)} 段；"
                   "報酬不可相加複合，這只是各段獨立回測的符號統計。")
    ddic = [c for c in report.signals
            if c.segment == "下跌段合計" and c.signal == "score"]
    nst = [c for c in report.signals
           if c.segment == "非壓力日" and c.signal == "score"]
    if ddic and nst:
        a, b = ddic[0], nst[0]
        verdict = ("達顯著（未試 FDR）" if math.isfinite(a.ic_p) and a.ic_p <= 0.05
                   else "未達顯著")
        out.append(f"合成分數在下跌段（{a.n_days} 個交易日）Rank IC 均值 "
                   f"{a.ic_mean:+.3f}（NW t={_num(a.ic_t_nw, 2)}、"
                   f"p={_num(a.ic_p, 3)}，{verdict}）；"
                   f"在非壓力日（{b.n_days} 日）為 {b.ic_mean:+.3f}"
                   f"（NW t={_num(b.ic_t_nw, 2)}、p={_num(b.ic_p, 3)}）。")
        out.append(f"五分組價差 Q5−Q1：下跌段 {_pct(a.q_spread, 3)}"
                   f"（扣來回成本後 {_pct(a.q_spread_net, 3)}）；"
                   f"非壓力日 {_pct(b.q_spread, 3)}（扣成本後 {_pct(b.q_spread_net, 3)}）。")
    rebound = [w for w in report.windows
               if w.kind == "volatility" and math.isfinite(w.excess)]
    if rebound:
        lag = [w for w in rebound if w.excess < 0]
        worst = min(rebound, key=lambda w: w.excess)
        out.append(f"高波動期間 {len(rebound)} 段中，策略落後 0050 者 {len(lag)} 段；"
                   f"落後最多者為 {worst.label}（策略 {_pct(worst.strat_total)}、"
                   f"0050 {_pct(worst.bench_total)}、超額 {_pct(worst.excess)}），"
                   "對應低曝險在急彈段付出的代價。")
    tc = [w for w in report.windows if math.isfinite(w.cost_rate)]
    if tc:
        out.append(f"各視窗交易成本佔初始資金 {min(w.cost_rate for w in tc):.2%} ~ "
                   f"{max(w.cost_rate for w in tc):.2%}；"
                   "扣成本前後的差距見 [4]「扣成本前」欄。")
    out.append("在下跌段未觀察到足以支撐「空頭期間仍具選股能力」的證據；"
               "本階段的樣本只含少數獨立市場事件，不得宣稱策略具防禦能力或可交易優勢。")
    return out


def _next_steps(report: StressReport) -> list[str]:
    """下一步建議（僅限資料與驗證方法，不涉及調參或交易規則）。"""
    steps = [
        "要對「空頭期間的訊號能力」下任何統計結論，必須先擴充真實歷史"
        "（至少涵蓋多個獨立空頭段）；可行方案與限制見 docs/DATA_SOURCES.md §3，"
        "採購或授權需使用者決定。",
        "不得以合成行情、反轉歷史報酬或回填虛構新聞製造空頭樣本；"
        "正式資料庫也不因本階段而做非必要 schema 變更。",
        "已反覆使用的 test 段（B17 那批）不得再度重用；"
        "凍結後新累積的資料在達到事前宣告的最低觀察期間前只報告品質與覆蓋率"
        "（python -m twstock.cli research）。",
        "「低曝險」的來源與代價屬於量測方法問題：建議後續報告一律同時列出"
        "同曝險基準，避免把曝險效果誤讀成選股能力。",
        "將本報告採用的期間定義與時間對齊規則納入測試（tests/test_stress.py），"
        "避免後續改動悄悄破壞 point-in-time 假設。",
        "若未來要比較不同策略在空頭期間的防禦性，需先宣告評估期間與指標，"
        "不得在看到結果後回頭挑選期間（與 P2.8 §6 同一研究紀律）。",
    ]
    if not any(w.sufficient for w in report.windows if w.kind == "drawdown"):
        steps.insert(0, "本階段可評估的下跌段僅少數幾段且部分樣本不足"
                        "（見 [5]「樣本足夠」欄），下一步應先補資料而非調整策略。")
    return steps


def _default_notes(report: StressReport) -> list[str]:
    cov = report.coverage
    notes = [
        "期間辨識是**事後描述**（前高／低點／復原需要看到整段資料才能界定），"
        "但期間內每一筆交易都只用當時已知的分數與價格（T 日收盤 → T+1 開盤），"
        "兩種性質必須分開解讀。",
        f"個股價格僅自 {_d(cov.stock_start)} 起，約 {cov.n_dates} 個交易日；"
        f"基準另有 {cov.n_bench_only_dates} 個交易日的歷史（見 [1]），"
        "該段無法用於策略比較。",
        "股票池為寫死的 20 檔大型權值股、且皆為今日仍上市之個股 → "
        "**survivorship bias 無法消除**；結果不代表台股整體，也不能外推。",
        "回撤 ≥15% 期間是回撤 ≥10% 期間的子集合；高波動與空頭狀態期間亦與其重疊。"
        "它們是同一批市場事件的不同視角，**不得當成獨立樣本累加證據**。",
        "每個視窗皆為獨立起始的回測，報酬不可相加複合；"
        "視窗末端的訊號因無法在視窗內完成一回合而被排除（見 [4] 筆數與覆蓋），"
        "此為截斷而非前視偏差，方向為低估該視窗的交易數。",
        "Sharpe 以日報酬年化，短視窗的樣本數少且報酬非獨立，"
        "數值僅供排序參考，不具統計推論效力。",
        "年化報酬僅在視窗 ≥ 約 6 個月時輸出，否則標示 N/A；"
        "把 2~3 個月的報酬年化會嚴重放大。",
        "基準與策略使用相同的除權息／分割還原；但基準買進持有不計手續費與稅，"
        "策略則計入完整成本，故超額報酬對策略是保守的比較。",
        "隨機選股控制只破壞『分數 ↔ 股票』的對應，其餘規則完全相同，"
        "是公平的同機制對照；但它仍受限於同一股票池與同一期間。",
        "本報告不宣稱策略在空頭市況具備防禦能力或可交易優勢；"
        "樣本內只有少數獨立市場事件，任何結論均屬探索性。",
    ]
    if cov.news_horizon:
        notes.append(f"新聞因子狀態：{cov.news_horizon} → "
                     "發布時間未知的新聞不得納入訊號，故本階段結果"
                     "不受新聞情緒影響。")
    return notes


def run_stress(*, spec: StrategySpec | None = None,
               cost_model: CostModel | None = None,
               n_sims: int = DEFAULT_SIMS,
               horizon: int = HORIZON,
               signals: tuple[str, ...] = DEFAULT_SIGNALS,
               quiet: bool = False) -> StressReport:
    """完整壓力測試：載入本機資料 → 稽核 → 辨識期間 → 比較 → 訊號檢驗。"""
    from .data import loader
    from .validation import load_db_inputs

    qa = loader.new_stats()
    panel, price_panel, benchmark = load_db_inputs(stats=qa)
    if panel is None or panel.empty:
        raise RuntimeError("特徵面板為空，請先執行 ingest 與 run")
    # 基準（0050）不是股票池成員；從價格面板移除，避免「21 檔」這類誤計
    # （交易日曆另由 benchmark 參數補上，因此不影響任何回測結果）
    price_panel = {k: v for k, v in price_panel.items() if k != CONFIG.benchmark}
    sp = spec or baseline()
    cm = cost_model or get_cost_model("typical")

    sf = attach_forward_returns(build_signal_frame(panel), price_panel,
                                (horizon,))
    cal = pd.DatetimeIndex(sorted(
        pd.to_datetime(sf["date"]).dt.normalize().unique()))
    news_status = news_factor_status(sf, [("全部", date_mask(cal))])
    cov = coverage_report(price_panel, panel, benchmark, qa=qa,
                          news_status=news_status)
    periods = identify_periods(benchmark)

    windows: list[WindowResult] = []
    for kind in ("drawdown", "volatility", "bear_regime"):
        for p in periods.get(kind, []):
            # 個股資料是否涵蓋（個股比基準晚開始 → 早期事件無法評估）
            covered = bool(len(cal) and p.start <= cal.max() and p.end >= cal.min())
            note = ("" if covered else
                    "個股資料未涵蓋此區間 → 不予評估（歷史缺口，不以較晚資料冒充）")
            windows.append(evaluate_window(
                sf, price_panel, benchmark, sp, cm, p.start, p.end,
                n_sims=n_sims if covered else 0,
                label=f"{_d(p.start)}~{_d(p.end)}", kind=kind, note=note))
            if p.kind == "drawdown" and p.recovery_date is not None:
                windows.append(evaluate_window(
                    sf, price_panel, benchmark, sp, cm, p.start,
                    p.recovery_date, n_sims=n_sims if covered else 0,
                    label=f"{_d(p.start)}~{_d(p.recovery_date)}(全段)",
                    kind="drawdown_full", note=note))

    # 訊號檢驗區段：以日期遮罩定義，避免把期間之間的正常交易日算入
    segs: list[tuple[str, pd.Series]] = []
    dd_mask = pd.Series(False, index=cal)
    for p in periods.get("drawdown", []):
        dd_mask |= date_mask(cal, lo=p.start, hi=p.end)
    if bool(dd_mask.any()):
        segs.append(("下跌段合計", dd_mask))
    rest = ~dd_mask
    if bool(rest.any()):
        segs.append(("非壓力日", rest))
    for kind in ("bear_regime", "volatility"):
        for i, p in enumerate(periods.get(kind, []), 1):
            m = date_mask(cal, lo=p.start, hi=p.end)
            if bool(m.any()):
                segs.append((f"{kind}#{i}", m))
    checks = signal_checks(sf, segs, signals=signals, horizon=horizon)

    report = StressReport(
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
        spec=sp, cost_model_name=cm.name, coverage=cov, periods=periods,
        windows=windows, signals=checks)
    report.notes = _default_notes(report)
    report.conclusions = _conclusions(report)
    report.next_steps = _next_steps(report)
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="空頭市況壓力測試（P2.9）")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                    help="隨機選股控制模擬次數")
    ap.add_argument("--quick", action="store_true",
                    help="少量模擬（僅供快速檢查，不可用於結論）")
    args = ap.parse_args(argv)
    sims = 20 if args.quick else args.sims
    report = run_stress(n_sims=sims)
    text = report.to_text()
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"stress_{dt.datetime.now():%Y%m%d_%H%M%S}.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n（報告已寫入：{out}）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
