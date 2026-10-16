"""策略失效歸因（P2.4）：把「訊號 → 排序 → 持倉 → 交易結果」逐段拆開。

目的不是改善績效，而是回答一個問題：**若訊號有資訊，為什麼沒有轉化成超額報酬？**

三個控制變因（一次只變一個，其餘完全相同）
  - **真實訊號**：策略原樣。
  - **隨機選股控制**：同日內把分數隨機重排到不同股票（保留分數分布、股票池、
    交易規則、成本與資金限制），重複多次得到一個分布。
    若真實策略落在這個分布裡面，選股就沒有貢獻任何資訊。
  - **完美前瞻上限**：用「T+1 起算的實際持有期報酬」當分數（**使用未來資訊，
    不可交易**）。這是相同機制下選股能做到的上限；用來判斷瓶頸在訊號還是在機制。

拆解項目（對應任務 4.1~4.10）
  1. 訊號與未來報酬的關係 → 由 `twstock.diagnostics`（P2.3）負責。
  2. 選股排序與實際持倉的差異 → ``selection_gap``。
  3. 進場時間及訊號延遲 → ``entry_delay``。
  4. 持有期間與退出規則 → ``holding_stats``。
  5. 換股頻率與訊號反轉 → ``turnover_breakdown``。
  6. 現金比例與市場曝險 → ``exposure_stats``。
  7. 交易成本與滑價 → ``cost_stats``。
  8. 流動性與成交容量 → 回測的 skipped_* 計數。
  9. 個股集中度 → ``concentration_stats``。
  10. 不同市場期間 → 沿用 ``validation.market_periods``。

指令：python -m twstock.attribution
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult, _prepare, _schedule_entries, run_backtest
from .config import REPORTS_DIR
from .costs import CostModel, get_cost_model
from .strategies import StrategySpec, baseline
from .validation import (Segment, build_scores, make_split, market_periods,
                         subset_by_dates, subset_panel)

log = logging.getLogger(__name__)

DEFAULT_SIMS = 200      # 隨機選股控制的模擬次數


def _pct(x: float, digits: int = 2) -> str:
    """百分比輸出；無法計算時明確標示 N/A，不輸出 nan%。"""
    if x is None or not math.isfinite(float(x)):
        return "N/A"
    return f"{float(x):+.{digits}%}"


def _num(x: float, digits: int = 2) -> str:
    if x is None or not math.isfinite(float(x)):
        return "N/A"
    return f"{float(x):.{digits}f}"


# --------------------------------------------------------------------------
# 控制變因輸入
# --------------------------------------------------------------------------
def _rank_pct(values: pd.Series) -> pd.Series:
    """當日橫斷面排名百分位 → [-1, 1]（與 signals._rank_pct 同一定義）。"""
    v = pd.to_numeric(values, errors="coerce")
    n = int(v.notna().sum())
    if n <= 1:
        return pd.Series(0.0, index=values.index)
    r = v.rank(method="average")
    return (r - 1) / (n - 1) * 2 - 1


def shuffled_scores(scores: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """同日內把分數隨機重排給不同股票（保留分布、日期與股票池）。

    這就是「相同交易規則、相同成本、相同資金限制下的隨機選股」。
    """
    if scores is None or scores.empty:
        return scores
    rng = np.random.default_rng(seed)
    # 一律先重設索引：呼叫端常傳入切片後的子表（索引標籤非連續），
    # 若用 groupby().groups 的標籤當位置索引會越界或洗錯列。
    out = scores.reset_index(drop=True).copy()
    new_score = out["score"].to_numpy(dtype=float).copy()
    for _d, g in out.groupby("date", sort=True):
        pos = g.index.to_numpy(dtype=int)
        new_score[pos] = rng.permutation(new_score[pos])
    out["score"] = new_score
    return out


def oracle_scores(signal_frame: pd.DataFrame, *, hold_days: int = 5
                  ) -> pd.DataFrame:
    """完美前瞻分數（**不可交易，僅作上限對照**）。

    以 T+1 起算、持有 ``hold_days`` 日的實際報酬做**當日橫斷面排名百分位**
    映射到 [-1, 1]，因此與真實分數同尺度、同一門檻語意，可公平比較排序能力。
    """
    col = f"fwd_{hold_days}"
    if signal_frame is None or signal_frame.empty or col not in signal_frame.columns:
        return pd.DataFrame(columns=["date", "stock_id", "score"])
    df = signal_frame[["date", "stock_id", col]].copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    if df.empty:
        return pd.DataFrame(columns=["date", "stock_id", "score"])
    df["score"] = df.groupby("date")[col].transform(_rank_pct)
    return df[["date", "stock_id", "score"]]


# --------------------------------------------------------------------------
# 歸因統計
# --------------------------------------------------------------------------
@dataclass
class SelectionGap:
    n_entry_days: int = 0
    n_candidates: int = 0
    n_captured: int = 0
    captured_rate: float = float("nan")

    def label(self) -> str:
        if self.n_candidates == 0:
            return "無候選（分數未達門檻或資料不足）"
        return (f"{self.n_entry_days} 個進場日、候選 {self.n_candidates} 筆、"
                f"實際成交 {self.n_captured} 筆（{self.captured_rate:.1%}）")


def selection_gap(scores: pd.DataFrame, price_panel: dict[str, pd.DataFrame],
                  trades: pd.DataFrame, spec: StrategySpec) -> SelectionGap:
    """策略規則下「應該」買進的候選，與實際成交之間的落差。

    候選完全由 ``backtest._schedule_entries`` 產生（與回測同一來源），
    因此不是另寫一份近似規則。落差來源包含：已持有該股、漲停無法成交、
    資金不足、單檔預算不足、流動性上限。
    """
    res = SelectionGap()
    if scores is None or scores.empty:
        return res
    book = _prepare(price_panel)
    if not book:
        return res
    schedule = _schedule_entries(scores, book, spec.hold_days)
    held: set[tuple[pd.Timestamp, str]] = set()
    if trades is not None and not trades.empty:
        for r in trades.itertuples(index=False):
            held.add((pd.Timestamp(r.entry_date).normalize(), str(r.stock_id)))
    cand_by_day: dict[pd.Timestamp, list[str]] = {}
    for entry_date, cands in schedule.items():
        passed = [c for c in cands if c["score"] >= spec.buy_threshold]
        if not passed:
            continue
        passed.sort(key=lambda c: -c["score"])
        cand_by_day[pd.Timestamp(entry_date)] = [
            c["stock_id"] for c in passed[:spec.top_n]]
    res.n_entry_days = len(cand_by_day)
    res.n_candidates = sum(len(v) for v in cand_by_day.values())
    res.n_captured = sum(1 for d, sids in cand_by_day.items()
                         for sid in sids if (d, sid) in held)
    if res.n_candidates:
        res.captured_rate = res.n_captured / res.n_candidates
    return res


@dataclass
class DelayStats:
    n: int = 0
    median_days: float = float("nan")
    p90_days: float = float("nan")
    max_days: float = float("nan")


def entry_delay(trades: pd.DataFrame) -> DelayStats:
    """訊號日 → 進場日的日曆日延遲（策略設計為 T+1，故中位數應約 1~3 天）。"""
    st = DelayStats()
    if trades is None or trades.empty or "signal_date" not in trades.columns:
        return st
    t = trades.dropna(subset=["signal_date"]).copy()
    if t.empty:
        return st
    lag = (pd.to_datetime(t["entry_date"]) - pd.to_datetime(t["signal_date"])).dt.days
    st.n = int(len(lag))
    if st.n:
        st.median_days = float(lag.median())
        st.p90_days = float(lag.quantile(0.9))
        st.max_days = float(lag.max())
    return st


@dataclass
class HoldingStats:
    n: int = 0
    avg_days: float = float("nan")
    to_expiry: int = 0          # 持有到最長期限才出場
    early_exit: int = 0         # 提早出場（訊號轉弱）
    extended: int = 0           # 因跌停順延而超期

    def label(self, spec: StrategySpec) -> str:
        if self.n == 0:
            return "無交易"
        return (f"{self.n} 筆；平均持有 {self.avg_days:.1f} 交易日；"
                f"到期出場 {self.to_expiry}（{self.to_expiry / self.n:.1%}）、"
                f"提早出場 {self.early_exit}、跌停順延 {self.extended}")


def holding_stats(trades: pd.DataFrame, spec: StrategySpec) -> HoldingStats:
    st = HoldingStats()
    if trades is None or trades.empty:
        return st
    st.n = int(len(trades))
    h = pd.to_numeric(trades["holding_days"], errors="coerce")
    st.avg_days = float(h.mean())
    # 三類互斥且加總等於總筆數：剛好到期 / 提早出場 / 跌停順延超期
    st.to_expiry = int((h == spec.hold_days).sum())
    st.early_exit = int((h < spec.hold_days).sum())
    st.extended = int((h > spec.hold_days).sum())
    assert st.to_expiry + st.early_exit + st.extended == st.n
    return st


@dataclass
class TurnoverBreakdown:
    n: int = 0
    repeat_entries: int = 0        # 同一股票在前次出場後極短時間內再次進場
    repeat_rate: float = float("nan")
    distinct_stocks: int = 0
    per_stock: dict[str, int] = field(default_factory=dict)


def turnover_breakdown(trades: pd.DataFrame, spec: StrategySpec, *,
                       window_days: int | None = None) -> TurnoverBreakdown:
    """區分「必要的換股」與「同一標的被反覆買賣」造成的週轉。"""
    tb = TurnoverBreakdown()
    if trades is None or trades.empty:
        return tb
    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.normalize()
    t = t.sort_values(["stock_id", "entry_date"])
    tb.n = int(len(t))
    tb.distinct_stocks = int(t["stock_id"].nunique())
    tb.per_stock = {str(k): int(v) for k, v in
                    t["stock_id"].value_counts().items()}
    w = window_days if window_days is not None else spec.hold_days
    reps = 0
    for _sid, g in t.groupby("stock_id"):
        prev_exit = None
        for r in g.itertuples(index=False):
            if prev_exit is not None:
                gap = (pd.Timestamp(r.entry_date) - prev_exit).days
                if gap <= max(1, w):
                    reps += 1
            prev_exit = pd.Timestamp(r.exit_date).normalize()
    tb.repeat_entries = reps
    tb.repeat_rate = reps / tb.n if tb.n else float("nan")
    return tb


@dataclass
class ConcentrationStats:
    top1_share: float = float("nan")     # 最大單一標的佔總交易名目比例
    hhi: float = float("nan")            # 交易名目集中度（0~1）

    def label(self) -> str:
        if not math.isfinite(self.hhi):
            return "無交易"
        return (f"最大單一標的佔交易名目 {self.top1_share:.1%}；"
                f"HHI {self.hhi:.4f}（滿倉 5 檔時約 0.20）")


def concentration_stats(trades: pd.DataFrame) -> ConcentrationStats:
    st = ConcentrationStats()
    if trades is None or trades.empty:
        return st
    notional = (pd.to_numeric(trades["qty"], errors="coerce")
                * pd.to_numeric(trades["entry_price"], errors="coerce"))
    tot = float(notional.sum())
    if tot <= 0:
        return st
    share = notional.groupby(trades["stock_id"]).sum() / tot
    st.top1_share = float(share.max())
    st.hhi = float((share ** 2).sum())
    return st


@dataclass
class ExposureStats:
    exposure: float = float("nan")
    avg_positions: float = float("nan")
    cash_share: float = float("nan")


def exposure_stats(result: BacktestResult, spec: StrategySpec) -> ExposureStats:
    st = ExposureStats()
    st.exposure = float(result.exposure)
    st.cash_share = float(max(0.0, 1.0 - result.exposure))
    if result.equity is not None and len(result.equity) and result.years > 0:
        # 平均持倉檔數的粗估：交易筆數 × 平均持有日 ÷ 交易日數
        st.avg_positions = float(
            result.round_trips * result.avg_holding_days
            / max(1.0, result.years * 252)) if result.avg_holding_days else float("nan")
    return st


@dataclass
class ControlRuns:
    label: str
    baseline: BacktestResult
    oracle: BacktestResult | None = None
    randoms: list[BacktestResult] = field(default_factory=list)

    def random_total_returns(self) -> np.ndarray:
        return np.asarray([r.total_return for r in self.randoms], dtype=float)

    def percentile_of_baseline(self) -> float:
        """真實策略在隨機選股分布中的百分位（0 = 最差、100 = 最好）。"""
        arr = self.random_total_returns()
        if len(arr) == 0 or not np.isfinite(self.baseline.total_return):
            return float("nan")
        return float((arr <= self.baseline.total_return).mean() * 100.0)

    def p_value_beats_random(self) -> float:
        """單尾經驗 p 值：真實策略贏過隨機選股的機率有多不容易發生。"""
        arr = self.random_total_returns()
        if len(arr) == 0:
            return float("nan")
        return float((int((arr >= self.baseline.total_return).sum()) + 1)
                     / (len(arr) + 1))


# --------------------------------------------------------------------------
# 報告
# --------------------------------------------------------------------------
@dataclass
class AttributionReport:
    generated_at: str
    segment: Segment
    spec: StrategySpec
    cost_model_name: str
    controls: ControlRuns
    benchmark_total_return: float
    gap: SelectionGap
    delay: DelayStats
    holding: HoldingStats
    turnover: TurnoverBreakdown
    concentration: ConcentrationStats
    exposure: ExposureStats
    periods: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        b = self.controls.baseline
        o = self.controls.oracle
        L: list[str] = []
        L.append("=" * 84)
        L.append("策略失效歸因報告（P2.4）")
        L.append(f"產生時間：{self.generated_at}")
        L.append("=" * 84)
        L.append("")
        L.append("[1] 設定")
        L.append(f"  策略：{self.spec.name}（{self.spec.description}）")
        L.append(f"  區段：{self.segment.label()}")
        L.append(f"  成本：{self.cost_model_name}；"
                 f"隨機控制模擬 {len(self.controls.randoms)} 次")
        L.append("  三個控制變因僅改變『分數』，交易規則／成本／資金限制完全相同。")
        L.append("")
        L.append("[2] 主結果與基準")
        L.append(f"  策略 {b.strategy_name}：總報酬 {b.total_return:+.2%}、"
                 f"年化 {b.cagr:+.2%}、MDD {b.max_drawdown:.2%}、"
                 f"Sharpe {b.sharpe:.2f}、{b.round_trips} 筆")
        if math.isfinite(self.benchmark_total_return):
            L.append(f"  0050 買進持有：總報酬 {self.benchmark_total_return:+.2%}"
                     f"（同期、同一還原方式）")
            L.append(f"  超額：{_pct(b.total_return - self.benchmark_total_return)}")
        else:
            L.append("  0050 買進持有：N/A（基準不可用，不得以 0 取代）")
            L.append("  超額：N/A")
        L.append("")
        L.append("[3] 控制變因比較（判斷瓶頸在訊號還是在機制）")
        arr = self.controls.random_total_returns()
        if len(arr):
            L.append(f"  {'情境':<22}{'總報酬':>10}{'年化':>9}{'MDD':>9}"
                     f"{'Sharpe':>8}{'筆數':>7}")
            rows = [("真實訊號（策略）", b)]
            if o is not None:
                rows.append(("完美前瞻（不可交易）", o))
            L.append(f"  {rows[0][0]:<22}{rows[0][1].total_return:>10.2%}"
                     f"{rows[0][1].cagr:>9.2%}{rows[0][1].max_drawdown:>9.2%}"
                     f"{rows[0][1].sharpe:>8.2f}{rows[0][1].round_trips:>7}")
            if o is not None:
                L.append(f"  {rows[1][0]:<22}{o.total_return:>10.2%}"
                         f"{o.cagr:>9.2%}{o.max_drawdown:>9.2%}"
                         f"{o.sharpe:>8.2f}{o.round_trips:>7}")
            n_med = int(np.median([r.round_trips for r in self.controls.randoms]))
            L.append(f"  {'隨機選股（中位數）':<22}{np.median(arr):>10.2%}"
                     f"{'N/A':>9}{'N/A':>9}{'N/A':>8}{n_med:>7}")
            L.append(f"  {'隨機選股（5%~95%）':<22}"
                     f"{np.quantile(arr, 0.05):>10.2%} ~ {np.quantile(arr, 0.95):.2%}")
            pct = self.controls.percentile_of_baseline()
            p = self.controls.p_value_beats_random()
            L.append(f"  真實策略在隨機分布中的百分位：{_num(pct, 1)}%；"
                     f"單尾經驗 p 值 = {_num(p, 3)}")
            if math.isfinite(p) and p > 0.05:
                L.append("  ⚠️ 真實選股**未顯著優於**相同規則下的隨機選股 → "
                         "在此區段選股沒有貢獻可辨識的資訊。")
            else:
                L.append("  真實選股優於隨機選股（但仍須看是否優於買進持有）。")
        else:
            L.append("  （未執行隨機控制）")
        L.append("")
        L.append("[4] 訊號 → 持倉的轉換落差")
        L.append("  " + self.gap.label())
        L.append("  落差來源包含：已持有該股、漲停無法成交、資金不足、"
                 "單檔預算不足、流動性上限。")
        L.append("")
        L.append("[5] 進場延遲與持有期間")
        d = self.delay
        if d.n:
            L.append(f"  訊號日→進場日：{d.n} 筆；中位數 {d.median_days:.0f} 日曆日、"
                     f"P90 {d.p90_days:.0f} 日、最大 {d.max_days:.0f} 日"
                     f"（T+1 開盤進場，週末會拉長日曆日數）")
        else:
            L.append("  訊號日→進場日：N/A（無交易）")
        L.append("  " + self.holding.label(self.spec))
        L.append("")
        L.append("[6] 週轉率拆解（判斷高換手是必要換股還是反覆買賣）")
        t = self.turnover
        L.append(f"  交易 {t.n} 筆、涉及 {t.distinct_stocks} 檔股票"
                 f"（股票池共 20 檔）")
        rate = (f"{t.repeat_rate:.1%}" if math.isfinite(t.repeat_rate) else "N/A")
        L.append(f"  同一標的在出場後 {self.spec.hold_days} 日內再次進場："
                 f"{t.repeat_entries} 筆（{rate}）")
        if t.per_stock:
            top = sorted(t.per_stock.items(), key=lambda kv: -kv[1])[:5]
            L.append("  交易最頻繁的標的：" +
                     "、".join(f"{k} {v} 筆" for k, v in top))
        L.append(f"  總週轉率（累積雙邊）{b.turnover:.2f}x；"
                 f"年化單邊（資金）{b.portfolio_turnover_annualized:.2f}x")
        L.append("")
        L.append("[7] 曝險與集中度")
        ex_lab = (f"{self.exposure.exposure:.1%}"
                  if math.isfinite(self.exposure.exposure) else "N/A")
        cash_lab = (f"{self.exposure.cash_share:.1%}"
                    if math.isfinite(self.exposure.cash_share) else "N/A")
        L.append(f"  平均曝險 {ex_lab}（現金 {cash_lab}）")
        if math.isfinite(self.exposure.avg_positions):
            L.append(f"  平均持有檔數（粗估）{self.exposure.avg_positions:.1f} 檔"
                     f" / 上限 {self.spec.top_n} 檔")
        L.append("  " + self.concentration.label())
        L.append("")
        L.append("[8] 交易成本與流動性")
        L.append(f"  總成本 {b.cost_total:,.0f}（佔初始資金 "
                 f"{b.cost_total / max(b.initial_capital, 1e-9):.2%}）")
        L.append(f"  漲停無法買進 {b.skipped_limit_up} 次、"
                 f"無有效價格 {b.skipped_no_price} 次、"
                 f"流動性上限 {b.skipped_liquidity} 次")
        L.append("")
        L.append("[9] 不同市場期間（策略 vs 基準；各段為獨立回測）")
        if self.periods:
            for pr in self.periods:
                L.append(f"  {pr.label:<26}{pr.strategy_return:>10.2%}"
                         f"{_pct(pr.benchmark_return):>10}"
                         f"{_pct(pr.excess()):>10}{pr.n_trades:>6}")
            L.append("  註：各列為獨立起始的子期間回測，不可相加複合。")
        L.append("")
        L.append("[10] 結論與限制")
        for n in self.notes:
            L.append("  - " + n)
        L.append("=" * 84)
        return "\n".join(L)


def run_attribution(*, n_sims: int = DEFAULT_SIMS, seed: int = 0,
                    cost_model: CostModel | None = None,
                    spec: StrategySpec | None = None,
                    initial_capital: float = 1_000_000.0) -> AttributionReport:
    """在**最終測試段**執行歸因（不重複使用同一資料挑參數）。"""
    from .diagnostics import attach_forward_returns, build_signal_frame
    from .validation import load_db_inputs

    qa = _new_stats()
    panel, price_panel, benchmark = load_db_inputs(stats=qa)
    if panel.empty:
        raise RuntimeError("特徵面板為空，請先執行 ingest 與 run")

    scores = build_scores(panel)
    dates = sorted(pd.to_datetime(scores["date"]).dt.normalize().unique())
    sp = spec or baseline()
    split = make_split(dates, train_frac=0.5, valid_frac=0.25,
                       embargo_days=sp.hold_days)
    seg = split.test
    cm = cost_model or get_cost_model("typical")

    seg_scores = subset_by_dates(scores, seg.start, seg.end)
    seg_panel = subset_panel(price_panel, seg.start, seg.end)
    seg_bench = (subset_by_dates(benchmark, seg.start, seg.end)
                 if benchmark is not None and not benchmark.empty else None)

    base = run_backtest(seg_scores, seg_panel, strategy=sp, benchmark=seg_bench,
                        cost_model=cm, initial_capital=initial_capital)
    randoms = [
        run_backtest(shuffled_scores(seg_scores, seed=seed + k), seg_panel,
                     strategy=sp, benchmark=None, cost_model=cm,
                     initial_capital=initial_capital)
        for k in range(n_sims)]

    sf = attach_forward_returns(build_signal_frame(panel), price_panel,
                                (sp.hold_days,))
    oracle = None
    orc = oracle_scores(sf, hold_days=sp.hold_days)
    if not orc.empty:
        orc_seg = subset_by_dates(orc, seg.start, seg.end)
        oracle = run_backtest(orc_seg, seg_panel, strategy=sp,
                              benchmark=seg_bench, cost_model=cm,
                              initial_capital=initial_capital)

    controls = ControlRuns(label=seg.label(), baseline=base, oracle=oracle,
                           randoms=randoms)
    bench_ret = float(base.benchmark_total_return)
    notes = [
        "本報告在**最終測試段**執行；未以這些結果回頭挑選參數。",
        "隨機控制保留分數分布、股票池、交易規則、成本與資金限制，只破壞"
        "『分數與股票的對應』，因此是公平的同機制對照。",
        "完美前瞻使用未來資訊，**不可交易**，僅用於判斷瓶頸在訊號或機制。",
        "子期間表為各段獨立回測，不可相加複合回測試段總報酬。",
        "⚠️ 樣本僅約 2 年；股票池 20 檔存在 survivorship bias。"
        "期間內大盤淨上漲，但確有空頭事件（見 P2.9 壓力測試："
        "python -m twstock.cli stress）；本報告的等分切段不足以回答抗跌能力。",
        "⚠️ 新聞情緒因子在歷史面板中幾近全為 0（新聞僅涵蓋少數日曆日），"
        "等於以 0.10 權重佔位而無實際資訊，會稀釋其他因子。",
        "本報告不宣稱策略具備獲利能力，也不提出可交易建議。",
    ]
    if base.round_trips < 20:
        notes.append(f"⚠️ 交易僅 {base.round_trips} 筆，統計上不足以支持任何結論。")
    if not math.isfinite(bench_ret) or seg_bench is None or seg_bench.empty:
        notes.append("⚠️ 基準不可用 → 超額報酬欄位無意義，不得以 0 替代。")

    periods = market_periods(scores, price_panel, split, sp,
                             benchmark if seg_bench is not None else None,
                             cost_model=cm, initial_capital=initial_capital)

    return AttributionReport(
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
        segment=seg, spec=sp, cost_model_name=cm.name, controls=controls,
        benchmark_total_return=bench_ret,
        gap=selection_gap(seg_scores, seg_panel, base.trades, sp),
        delay=entry_delay(base.trades),
        holding=holding_stats(base.trades, sp),
        turnover=turnover_breakdown(base.trades, sp),
        concentration=concentration_stats(base.trades),
        exposure=exposure_stats(base, sp), periods=periods, notes=notes)


def _new_stats() -> dict:
    from .data import loader
    return loader.new_stats()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="策略失效歸因（P2.4）")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                    help="隨機選股控制模擬次數")
    args = ap.parse_args(argv)
    report = run_attribution(n_sims=args.sims)
    text = report.to_text()
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"attribution_{dt.datetime.now():%Y%m%d_%H%M%S}.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n（報告已寫入：{out}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
