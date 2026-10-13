"""嚴格的樣本外驗證流程：時間序列切分、purge/embargo、基準檢查與策略比較。

設計原則
  1. **不隨機打亂**：切分完全依時間順序。
  2. **Purge / embargo**：標籤使用未來 ``hold_days`` 日報酬，因此訓練段尾端
     與測試段開頭之間必須移除重疊的樣本（``purge_boundary``），否則相鄰樣本
     的標籤觀察期間互相重疊會洩漏資訊。
  3. **轉換只用訓練資料擬合**：任何需要 fit 的轉換（如標準化）必須先
     ``FeatureScaler.fit(train)``，再 ``transform(valid/test)``。目前規則策略
     只用固定門檻（無 fit 參數），因此洩漏風險主要在標籤期間，由 purge 處理。
  4. **測試集只用一次**：``HoldoutGuard`` 記錄每個策略在同一執行中評估測試集的
     次數，避免反覆用測試集調參。
  5. **基準缺失必須明確**：``check_benchmark`` 回傳 ``ok=False`` 與原因，
     不把缺值當 0。
  6. **資料不足如實記錄**：樣本/交易數不足時，報告標示「不足以判斷」，
     不產生虛構數字。

指令：
  python -m twstock.validation            # 讀本機 DB，輸出報告
  python -m twstock.validation --quick    # 只用部分策略
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import signals as sig_mod
from .backtest import BacktestResult, run_backtest
from .config import CONFIG, REPORTS_DIR
from .costs import COST_SCENARIOS, CostModel, get_cost_model
from .strategies import StrategySpec, baseline, comparison_variants

log = logging.getLogger(__name__)

MIN_TRADES_FOR_CONCLUSION = 20     # 低於此交易數不宣稱統計顯著
MIN_TEST_DAYS = 60                 # 測試段最少交易日
MIN_FOLD_TEST_DAYS = 20            # 單一 walk-forward fold 的測試窗最少交易日


# --------------------------------------------------------------------------
# 時間序列切分
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_dates: int

    def label(self) -> str:
        return f"{self.start:%Y-%m-%d} ~ {self.end:%Y-%m-%d}（{self.n_dates} 交易日）"


@dataclass(frozen=True)
class TimeSplit:
    train: Segment
    valid: Segment
    test: Segment
    embargo_days: int

    def describe(self) -> str:
        return (
            f"purge/embargo = {self.embargo_days} 交易日（= 標籤觀察期間 hold_days）\n"
            f"  train : {self.train.label()}\n"
            f"  valid : {self.valid.label()}\n"
            f"  test  : {self.test.label()}（最終評估，不得用於調參）")


def _segment(name: str, dates: list[pd.Timestamp]) -> Segment:
    return Segment(name=name, start=dates[0], end=dates[-1], n_dates=len(dates))


def make_split(dates: list[pd.Timestamp], *, train_frac: float = 0.5,
               valid_frac: float = 0.25, embargo_days: int) -> TimeSplit:
    """依時間順序切分 train/valid/test，並在段與段之間移除 embargo。"""
    ds = sorted(pd.Timestamp(d).normalize() for d in dates)
    n = len(ds)
    if n < 3 * (embargo_days + 1):
        raise ValueError(
            f"資料量不足：{n} 個交易日無法切出含 {embargo_days} 日 embargo 的三段")
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)
    train_all = ds[:n_train]
    valid_all = ds[n_train:n_train + n_valid]
    test_all = ds[n_train + n_valid:]
    # purge：訓練段尾端 / 驗證段尾端各移除 embargo 個交易日
    train = train_all[:max(0, len(train_all) - embargo_days)] or train_all
    valid = valid_all[:max(0, len(valid_all) - embargo_days)] or valid_all
    if not (train and valid and test_all):
        raise ValueError("切分後有空段，資料量不足")
    return TimeSplit(_segment("train", list(train)), _segment("valid", list(valid)),
                     _segment("test", list(test_all)), embargo_days)


def purge_boundary(train_dates: list[pd.Timestamp], hold_days: int,
                   boundary: pd.Timestamp) -> list[pd.Timestamp]:
    """移除訓練段中「標籤期間會跨過 boundary」的樣本。"""
    if hold_days <= 0:
        return list(train_dates)
    kept = []
    for i, d in enumerate(train_dates):
        # 該樣本之後還有幾個交易日；若不足 hold_days 代表標籤會落到 boundary 之後
        remaining = len(train_dates) - i - 1
        if d < boundary and remaining >= hold_days:
            kept.append(d)
    return kept


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_valid: int
    n_test: int

    def label(self) -> str:
        return (f"fold{self.index}: train≤{self.train_end:%Y-%m-%d}"
                f"({self.n_train}) valid≤{self.valid_end:%Y-%m-%d}({self.n_valid})"
                f" test {self.test_start:%Y-%m-%d}~{self.test_end:%Y-%m-%d}({self.n_test})")


def walk_forward(dates: list[pd.Timestamp], *, n_folds: int = 3,
                 train_frac: float = 0.5, embargo_days: int) -> list[Fold]:
    """滾動驗證：每個 fold 記錄其訓練/驗證/測試窗口。"""
    ds = sorted(pd.Timestamp(d).normalize() for d in dates)
    n = len(ds)
    if n_folds < 1:
        raise ValueError("n_folds 必須 >= 1")
    folds: list[Fold] = []
    step = max(1, n // (n_folds + 1))
    for k in range(n_folds):
        train_end_i = int(n * train_frac) + k * step
        valid_end_i = train_end_i + step
        test_end_i = min(valid_end_i + step, n)
        if test_end_i <= valid_end_i or train_end_i - embargo_days <= 0:
            break
        train_slice = ds[:max(0, train_end_i - embargo_days)]
        valid_slice = ds[train_end_i:max(train_end_i, valid_end_i - embargo_days)]
        test_slice = ds[valid_end_i:test_end_i]
        # 測試窗過短的 fold 不具意義，直接略過（如資料尾端只剩 1 天）
        if len(test_slice) < MIN_FOLD_TEST_DAYS:
            continue
        if not (train_slice and valid_slice and test_slice):
            break
        folds.append(Fold(
            index=k, train_end=train_slice[-1], valid_end=valid_slice[-1],
            test_start=test_slice[0], test_end=test_slice[-1],
            n_train=len(train_slice), n_valid=len(valid_slice),
            n_test=len(test_slice)))
    return folds


# --------------------------------------------------------------------------
# 轉換（僅用訓練資料擬合）
# --------------------------------------------------------------------------
@dataclass
class FeatureScaler:
    """z-score 標準化；``fit`` 只接受訓練資料，避免測試集資訊洩漏。"""

    cols: list[str] = field(default_factory=list)
    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, df: pd.DataFrame, cols: list[str] | None = None) -> "FeatureScaler":
        self.cols = list(cols or [c for c in df.columns
                                  if pd.api.types.is_numeric_dtype(df[c])])
        for c in self.cols:
            s = pd.to_numeric(df[c], errors="coerce")
            self.mean_[c] = float(s.mean()) if s.notna().any() else 0.0
            sd = float(s.std()) if s.notna().any() else 0.0
            self.std_[c] = sd if sd and sd > 0 else 1.0
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeatureScaler 尚未 fit（必須先以訓練資料擬合）")
        out = df.copy()
        for c in self.cols:
            if c in out.columns:
                out[c] = (pd.to_numeric(out[c], errors="coerce")
                          - self.mean_[c]) / self.std_[c]
        return out


# --------------------------------------------------------------------------
# 基準檢查
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkStatus:
    ok: bool
    reason: str
    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    def label(self) -> str:
        if not self.ok:
            return f"基準不可用：{self.reason}"
        return (f"基準可用：{self.rows} 筆（{self.start:%Y-%m-%d} ~ "
                f"{self.end:%Y-%m-%d}）；價格已套用分割與現金股利還原"
                f"（與策略一致）；未計入手續費，無風險利率不扣除")


def check_benchmark(benchmark: pd.DataFrame | None,
                    start: pd.Timestamp | None = None,
                    end: pd.Timestamp | None = None) -> BenchmarkStatus:
    """驗證基準資料是否存在、時間範圍是否一致。缺失時明確回報，不當 0。"""
    if benchmark is None or benchmark.empty:
        return BenchmarkStatus(False, f"缺少基準（{CONFIG.benchmark}）資料，"
                                      f"無法計算超額報酬")
    b = benchmark.copy()
    if "close" not in b.columns or "date" not in b.columns:
        return BenchmarkStatus(False, "基準資料缺少 date/close 欄位")
    b["date"] = pd.to_datetime(b["date"]).dt.normalize()
    b = b.dropna(subset=["close"])
    if b.empty:
        return BenchmarkStatus(False, "基準資料全為空值")
    lo, hi = b["date"].min(), b["date"].max()
    if start is not None and lo > pd.Timestamp(start):
        return BenchmarkStatus(False, f"基準起始日 {lo:%Y-%m-%d} 晚於策略起始日 "
                                      f"{pd.Timestamp(start):%Y-%m-%d}，時間範圍不一致")
    if end is not None and hi < pd.Timestamp(end):
        return BenchmarkStatus(False, f"基準結束日 {hi:%Y-%m-%d} 早於策略結束日 "
                                      f"{pd.Timestamp(end):%Y-%m-%d}，時間範圍不一致")
    return BenchmarkStatus(True, "", rows=len(b), start=lo, end=hi)


# --------------------------------------------------------------------------
# 測試集使用次數守門
# --------------------------------------------------------------------------
class HoldoutGuard:
    """確保測試集不被反覆用於調參：同一策略在同一執行中只應評估測試集一次。"""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self.violations: list[str] = []

    def record(self, key: str) -> int:
        self._counts[key] = self._counts.get(key, 0) + 1
        if self._counts[key] > 1:
            self.violations.append(f"{key} 被評估 {self._counts[key]} 次")
        return self._counts[key]

    def counts(self) -> dict[str, int]:
        return dict(self._counts)


# --------------------------------------------------------------------------
# 評估
# --------------------------------------------------------------------------
def build_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """由特徵面板產生規則分數（不使用任何 fit 參數）。"""
    if panel is None or panel.empty:
        return pd.DataFrame()
    features_by_stock = {sid: g for sid, g in panel.groupby("stock_id")}
    return sig_mod.score_stocks(features_by_stock)


def subset_by_dates(df: pd.DataFrame, start: pd.Timestamp,
                    end: pd.Timestamp, col: str = "date") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d[col] = pd.to_datetime(d[col]).dt.normalize()
    return d[(d[col] >= pd.Timestamp(start)) & (d[col] <= pd.Timestamp(end))]


def subset_panel(panel: dict[str, pd.DataFrame], start: pd.Timestamp,
                 end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    out = {}
    for sid, df in panel.items():
        sub = subset_by_dates(df, start, end)
        if not sub.empty:
            out[sid] = sub
    return out


@dataclass
class SegmentResult:
    segment: str
    strategy: str
    cost_model: str
    result: BacktestResult
    n_score_rows: int
    note: str = ""

    @property
    def sufficient(self) -> bool:
        return self.result.round_trips >= MIN_TRADES_FOR_CONCLUSION


@dataclass
class MarketPeriodResult:
    label: str
    strategy_return: float
    benchmark_return: float
    n_trades: int

    def excess(self) -> float:
        return self.strategy_return - self.benchmark_return


@dataclass
class ValidationReport:
    generated_at: str
    split: TimeSplit
    folds: list[Fold]
    benchmark: BenchmarkStatus
    rows: list[SegmentResult]
    cost_rows: list[SegmentResult] = field(default_factory=list)
    periods: list[MarketPeriodResult] = field(default_factory=list)
    strategy_benchmark_available: bool = True
    test_guard_violations: list[str] = field(default_factory=list)
    data_notes: list[str] = field(default_factory=list)
    fusion_available: bool = False

    # ---- 報告輸出 ----
    def to_text(self) -> str:
        L: list[str] = []
        L.append("=" * 78)
        L.append("樣本外驗證報告（P2.1）")
        L.append(f"產生時間：{self.generated_at}")
        L.append("=" * 78)
        L.append("")
        L.append("[1] 時間序列切分與 purge/embargo")
        L.append(self.split.describe())
        L.append("")
        if self.folds:
            L.append("[2] 滾動驗證窗口（walk-forward）")
            for f in self.folds:
                L.append("  " + f.label())
            L.append("")
        L.append("[3] 基準檢查")
        L.append("  " + self.benchmark.label())
        if not self.benchmark.ok:
            L.append("  ⚠️ 基準不可用 → 本報告所有「超額報酬」欄位標示為 N/A，"
                     "不以 0 替代。")
        L.append("")
        L.append("[4] 策略比較（同一資料區間、同一成本模型、同一風險條件）")
        L.append(f"  {'策略':<24}{'區段':<8}{'筆數':>5}{'總報酬':>10}{'年化':>9}"
                 f"{'MDD':>9}{'Sharpe':>8}{'勝率':>8}{'邊際':>8}{'超額':>10}")
        for r in self.rows:
            res = r.result
            excess = (f"{res.excess_return():+.2%}"
                      if self.strategy_benchmark_available else "N/A")
            edge = "是" if r.sufficient else "不足"
            L.append(
                f"  {r.strategy:<24}{r.segment:<8}{res.round_trips:>5}"
                f"{res.total_return:>10.2%}{res.cagr:>9.2%}"
                f"{res.max_drawdown:>9.2%}{res.sharpe:>8.2f}"
                f"{res.win_rate:>8.2%}{edge:>8}{excess:>10}")
        L.append("")
        L.append("  註：『筆數 < "
                 f"{MIN_TRADES_FOR_CONCLUSION}』標為邊際不足，不宣稱統計顯著。")
        L.append("")
        L.append("[5] 交易成本敏感度（測試段）")
        if self.cost_rows:
            L.append(f"  {'成本情境':<16}{'總報酬':>10}{'年化':>9}{'Sharpe':>8}"
                     f"{'筆數':>6}{'週轉率(雙邊)':>14}{'總成本':>12}")
            for r in self.cost_rows:
                res = r.result
                L.append(
                    f"  {r.cost_model:<16}{res.total_return:>10.2%}{res.cagr:>9.2%}"
                    f"{res.sharpe:>8.2f}{res.round_trips:>6}"
                    f"{res.turnover:>14.2f}{res.cost_total:>12,.0f}")
        else:
            L.append("  （未執行成本敏感度）")
        L.append("")
        L.append("[6] 不同市場期間表現（策略 vs 基準）")
        if self.periods:
            L.append(f"  {'期間':<26}{'策略':>10}{'基準':>10}{'超額':>10}{'筆數':>6}")
            for p in self.periods:
                bench = f"{p.benchmark_return:+.2%}" if self.strategy_benchmark_available \
                    else "N/A"
                excess = f"{p.excess():+.2%}" if self.strategy_benchmark_available else "N/A"
                L.append(f"  {p.label:<26}{p.strategy_return:>10.2%}{bench:>10}"
                         f"{excess:>10}{p.n_trades:>6}")
            L.append(
                "  註：每個子期間皆為「獨立起始」的單獨回測（各自重新投入初始資金、"
                "無跨期間持倉）。")
            L.append(
                "      因此跨子期間界線、無法在該子期間內完成持有期的部位會被排除，"
                "本表『筆數』會低於 [4] 的完整測試段筆數，")
            L.append(
                "      各列報酬也不可相加或複合回 [4] 的測試段總報酬；本表僅供各"
                "子期間內與同期基準的方向性對照。")
        L.append("")
        L.append("[7] 資料完整度與測試限制")
        for n in self.data_notes:
            L.append("  - " + n)
        if self.test_guard_violations:
            L.append("  - ⚠️ 測試集重複使用：" + "；".join(self.test_guard_violations))
        else:
            L.append("  - 測試集僅用於最終評估（每個策略於本執行中各評估一次）")
        L.append("")
        L.append("[8] 指標與成本假設定義（可稽核）")
        L.append("  週轉率：")
        L.append("    turnover（累積、雙邊）= (買進名目 + 賣出名目) / 初始資金；"
                 "買賣各計一次、未年化，故一次完整進出會計入約 2 倍名目金額。")
        L.append("    turnover_one_way = turnover / 2；"
                 "turnover_annualized = turnover / 年數。")
        L.append("    portfolio_turnover_annualized = (名目/2) / 平均權益 / 年數"
                 "（與資金規模無關，適合跨策略比較）。")
        L.append("    週轉率≠交易次數；交易次數另列於『筆數』欄。")
        L.append("  成本：[成本情境] 為券商手續費（牌告 0.1425%，折扣依券商而異、")
        L.append("    低消可設定）＋證交稅（一般股票賣出 0.3%；本策略持有 >= 1 日，"
                 "不適用當沖 0.15%）＋雙邊滑價；severe 另限制單筆成交量佔比 1%。")
        L.append("    實際費率並非所有券商通用，請以自身帳戶條件為準。")
        L.append("  其他：Sharpe 以日報酬年化 252 交易日計算；無風險利率未扣除；"
                 "價格為已還原序列。")
        L.append("")
        L.append("[9] 結論限制")
        L.append("  - 本報告不宣稱策略具備實際獲利能力，也不代表未來報酬。")
        L.append("  - 股票池為寫死的 20 檔，存在 survivorship bias，"
                 "結果僅適用於此樣本。")
        L.append("  - 價格已套用分割與現金股利還原；與價格序列不符而捨棄的事件採"
                 "保守處理（略微低估報酬）。")
        L.append("  - 樣本期間僅約 2 年且屬單一市況（多頭），未涵蓋空頭與高波動"
                 "期間，穩健性無法確認。")
        L.append("=" * 78)
        return "\n".join(L)


def evaluate_strategies(scores: pd.DataFrame,
                        price_panel: dict[str, pd.DataFrame],
                        split: TimeSplit,
                        strategies: list[StrategySpec],
                        *, benchmark: pd.DataFrame | None = None,
                        cost_model: CostModel | None = None,
                        segments: tuple[str, ...] = ("train", "valid", "test"),
                        guard: HoldoutGuard | None = None,
                        initial_capital: float = 1_000_000.0) -> list[SegmentResult]:
    """在指定區段上跑策略並比較（同一成本、同一風險條件）。"""
    cm = cost_model or get_cost_model("typical")
    out: list[SegmentResult] = []
    for seg_name in segments:
        seg = getattr(split, seg_name)
        seg_scores = subset_by_dates(scores, seg.start, seg.end)
        seg_panel = subset_panel(price_panel, seg.start, seg.end)
        seg_bench = subset_by_dates(benchmark, seg.start, seg.end) \
            if benchmark is not None and not benchmark.empty else None
        for spec in strategies:
            if seg_name == "test" and guard is not None:
                guard.record(f"{spec.name}@test")
            res = run_backtest(seg_scores, seg_panel, strategy=spec,
                               benchmark=seg_bench, cost_model=cm,
                               initial_capital=initial_capital)
            note = ""
            if not seg_panel:
                note = "該區段無價格資料"
            elif res.round_trips == 0:
                note = "該區段無成交（樣本不足）"
            out.append(SegmentResult(seg_name, spec.name, cm.name, res,
                                     len(seg_scores), note))
    return out


def market_periods(scores: pd.DataFrame, price_panel: dict[str, pd.DataFrame],
                   split: TimeSplit, spec: StrategySpec,
                   benchmark: pd.DataFrame | None, *, n_periods: int = 4,
                   cost_model: CostModel | None = None,
                   initial_capital: float = 1_000_000.0) -> list[MarketPeriodResult]:
    """把測試段等分為數個子期間，檢視不同市況下的表現。"""
    cm = cost_model or get_cost_model("typical")
    dates = sorted(pd.to_datetime(scores["date"]).dt.normalize().unique())
    dates = [d for d in dates if split.test.start <= d <= split.test.end]
    if len(dates) < n_periods * 5:
        return []
    chunks = np.array_split(np.array(dates), n_periods)
    out: list[MarketPeriodResult] = []
    for c in chunks:
        if len(c) == 0:
            continue
        lo, hi = pd.Timestamp(c[0]), pd.Timestamp(c[-1])
        s = subset_by_dates(scores, lo, hi)
        p = subset_panel(price_panel, lo, hi)
        b = subset_by_dates(benchmark, lo, hi) \
            if benchmark is not None and not benchmark.empty else None
        res = run_backtest(s, p, strategy=spec, benchmark=b, cost_model=cm,
                           initial_capital=initial_capital)
        if len(res.benchmark_equity) > 1:
            bret = res.benchmark_total_return
        else:
            bret = 0.0
        out.append(MarketPeriodResult(
            f"{lo:%Y-%m-%d}~{hi:%Y-%m-%d}", res.total_return, bret,
            res.round_trips))
    return out


def cost_sensitivity(scores: pd.DataFrame, price_panel: dict[str, pd.DataFrame],
                     seg: Segment, spec: StrategySpec,
                     benchmark: pd.DataFrame | None,
                     scenario_names: list[str] | None = None,
                     initial_capital: float = 1_000_000.0) -> list[SegmentResult]:
    """同一策略在多組成本情境下的表現。"""
    names = scenario_names or ["zero", "ideal", "typical", "conservative",
                               "adverse", "severe"]
    s = subset_by_dates(scores, seg.start, seg.end)
    p = subset_panel(price_panel, seg.start, seg.end)
    b = subset_by_dates(benchmark, seg.start, seg.end) \
        if benchmark is not None and not benchmark.empty else None
    out: list[SegmentResult] = []
    for name in names:
        res = run_backtest(s, p, strategy=spec, benchmark=b,
                           cost_model=get_cost_model(name),
                           initial_capital=initial_capital)
        out.append(SegmentResult(seg.name, spec.name, name, res, len(s)))
    return out


# --------------------------------------------------------------------------
# 資料載入與 CLI
# --------------------------------------------------------------------------
def loader_stats() -> dict:
    """建立載入統計計數器。"""
    from .data import loader
    return loader.new_stats()


def load_db_inputs(benchmark_id: str | None = None,
                   stats: dict | None = None,
                   ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """從本機 DB 載入特徵面板、價格面板與基準。"""
    from .data import loader

    feature_panel, _ = _load_feature_panel(stats=stats)
    # 已清理無效列並套用除權息／分割還原（策略與基準處理方式一致）
    price = loader.load_price_panel(adjusted=True)
    price_panel = loader.panel_dict(price)
    bid = benchmark_id or CONFIG.benchmark
    benchmark = loader.benchmark_series(price, bid)
    return feature_panel, price_panel, benchmark


def _load_feature_panel(stats: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .pipeline import load_panel
    return load_panel(stats=stats)


def run_validation(*, strategies: list[StrategySpec] | None = None,
                   cost_scenarios: list[str] | None = None,
                   train_frac: float = 0.5, valid_frac: float = 0.25,
                   n_folds: int = 3,
                   benchmark_id: str | None = None) -> ValidationReport:
    """完整驗證流程：載入資料 → 切分 → 比較 → 成本敏感度 → 市場期間。"""
    qa = loader_stats()
    panel, price_panel, benchmark = load_db_inputs(benchmark_id, stats=qa)
    notes: list[str] = []
    if panel.empty:
        raise RuntimeError("特徵面板為空，請先執行 ingest 與 run")

    scores = build_scores(panel)
    if scores.empty:
        raise RuntimeError("無法產生分數，請確認資料完整度")

    dates = sorted(pd.to_datetime(scores["date"]).dt.normalize().unique())
    hold = baseline().hold_days
    split = make_split(dates, train_frac=train_frac, valid_frac=valid_frac,
                       embargo_days=hold)
    folds = walk_forward(dates, n_folds=n_folds, train_frac=train_frac,
                         embargo_days=hold)

    bench_status = check_benchmark(benchmark, split.train.start, split.test.end)

    specs = strategies or comparison_variants()
    guard = HoldoutGuard()
    rows = evaluate_strategies(scores, price_panel, split, specs,
                              benchmark=benchmark if bench_status.ok else None,
                              guard=guard)

    cost_rows = cost_sensitivity(scores, price_panel, split.test,
                                 baseline(),
                                 benchmark if bench_status.ok else None,
                                 cost_scenarios)
    periods = market_periods(scores, price_panel, split, baseline(),
                             benchmark if bench_status.ok else None)

    # 資料完整度 / 限制
    notes.append(f"股票池 {len(set(panel['stock_id']))} 檔、"
                 f"特徵列 {len(panel):,}、分數列 {len(scores):,}")
    notes.append(f"分數期間 {dates[0]:%Y-%m-%d} ~ {dates[-1]:%Y-%m-%d}"
                 f"（{len(dates)} 交易日，約 {len(dates) / 252:.2f} 年）")
    notes.append("股票池為寫死的 20 檔大型權值股，未含歷史成分與下市股票，"
                 "存在 survivorship bias；結果不代表整個台股市場。")
    notes.append(f"價格已套用除權息／分割還原（策略與基準一致）：共 {qa['stocks']} 檔、"
                 f"套用 {qa['actions_applied']} 個調整、捨棄 {qa['actions_skipped']} 個"
                 f"與價格序列不符的事件、移除 {qa['rows_dropped']} 列無效價格。"
                 f"被捨棄的事件採保守處理（略微低估報酬）。")
    if len(dates) < 252 * 2:
        notes.append(f"⚠️ 樣本期間僅 {len(dates) / 252:.2f} 年，"
                     "不足以評估長期穩定性。")
    notes.append("⚠️ 樣本僅涵蓋單一市況（本期間大盤為上漲走勢），"
                 "未包含空頭／高波動期間，無法確認策略在逆境的表現。")
    notes.append("參數選擇僅在 train/valid 上進行；test 只用於最終評估一次；"
                 "報告中的變體比較不得回過頭用來挑選最佳參數。")
    if split.test.n_dates < MIN_TEST_DAYS:
        notes.append(f"⚠️ 測試段僅 {split.test.n_dates} 交易日，"
                     f"低於建議的 {MIN_TEST_DAYS} 日，結論不具代表性。")
    if not bench_status.ok:
        notes.append(f"⚠️ {bench_status.reason}")
    notes.append("ML／融合策略未納入比較：lightgbm 未安裝且未驗證，"
                 "強行比較會產生不公平排名。")

    return ValidationReport(
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
        split=split, folds=folds, benchmark=bench_status, rows=rows,
        cost_rows=cost_rows, periods=periods,
        strategy_benchmark_available=bench_status.ok,
        test_guard_violations=guard.violations, data_notes=notes,
        fusion_available=False)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="樣本外驗證（P2.1）")
    ap.add_argument("--quick", action="store_true", help="只比較基準策略")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--cost", nargs="*", default=None, help="成本情境名稱")
    args = ap.parse_args(argv)

    specs = [baseline()] if args.quick else None
    report = run_validation(strategies=specs, cost_scenarios=args.cost,
                            n_folds=args.folds)
    text = report.to_text()
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"validation_{dt.datetime.now():%Y%m%d_%H%M%S}.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n（報告已寫入：{out}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
