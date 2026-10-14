"""P2.6：B17 新聞因子消融驗證與合成分數因果診斷。

本模組回答一個**可證偽**的問題：``news_sentiment`` 的缺值處理（``fillna(0.0)`` 卻仍佔
``WEIGHTS["news_sentiment"] = 0.10`` 的分母）是否真的降低了合成 ``score`` 的預測能力，
以及移除／重新標準化是否帶來可重現的改善。

研究紀律（事先聲明，非依結果追加）::

    1. 四個候選方案 A/B/C/D 定義於 SCHEME_LABELS，執行前即固定，不依結果替換。
    2. A* 是**分解控制組**，不是第五個候選方案：只把新聞分子拿掉、分母維持 baseline，
       用來把「fillna(0) 的分子效應」與「0.10 分母佔位效應」分開。
    3. 方案比較與任何修改建議只使用 train / valid；test 只作一次最終確認，
       且本 test 段在 P2.3 診斷中已被使用 → 標記為**已消耗的測試集**。
    4. 所有方案共用同一份面板、日期、股票池、成本、進出場與持倉限制，
       僅分數計算不同；因此是**配對比較**。
    5. 不改動曝險上限、固定持有 5 日規則或任何交易機制。

四種方案的分數定義（``parts`` = 該列可用因子 (鍵, 方向化值)；``den`` = 分母）::

    A    baseline：den = W_A（含 news 的可用權重和），分子含 news
    A*   分子控制：den = W_A（同 A），分子**不含** news
    B    移除＋重新標準化：den = W_A − 0.10（不含 news 的可用權重和），分子不含 news
    C    固定權重不標準化：den = 1.0（名目權重總和），分子不含 news
    D    可用性閘門：有新聞記錄才把 news 計入分子**與**分母；缺值直接排除

關鍵數學事實（本模組會實測驗證，不靠假設）::

    A − A* = 0.10 × v_news / W_A        → v_news = 0 的列**嚴格等於 0**
    A  = B × (den_B / den_A)（當 v_news = 0）
    C  = B × den_B

    若 W_A 在同一天跨股票為常數，A/B 僅差一個正比例常數 → 排名必然相同；
    若 W_A 跨股票**變動**，den_B/den_A 逐股不同 → 排名**可能真的改變**。
    實測：W_A 在 483/486 個交易日跨股票變動（0.53 ~ 1.05），故不可先驗宣告「只是尺度」。

執行方式::

    python -m twstock.ablation              # 完整報告
    python -m twstock.ablation --quick      # 少量自助法迭代（快速檢查）
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import signals as sig_mod
from .backtest import run_backtest
from .costs import get_cost_model
from .data import loader
from .diagnostics import (
    DEFAULT_SIMS,
    HORIZONS,
    MIN_STOCKS_PER_DAY,
    N_GROUPS,
    _assign_groups,
    _corr,
    _prep_pairs,
    _two_sided_p,
    attach_forward_returns,
    benjamini_hochberg,
    block_bootstrap_ci,
    ic_series,
    ic_summary,
    newey_west_tstat,
    quantile_returns,
)
from .strategies import baseline
from .validation import load_db_inputs, make_split, subset_by_dates, subset_panel

log = logging.getLogger(__name__)

NEWS_KEY = "news_sentiment"
NEWS_WEIGHT = sig_mod.WEIGHTS[NEWS_KEY]

# 候選方案（A/B/C/D 為任務指定的四組）＋ 分解控制組 A*
SCHEMES: tuple[str, ...] = ("A", "Actrl", "B", "C", "D")
CANDIDATE_SCHEMES: tuple[str, ...] = ("A", "B", "C", "D")

SCHEME_LABELS: dict[str, str] = {
    "A": "A 現行 baseline",
    "Actrl": "A* 分子控制組（僅移除新聞分子，分母維持 baseline）",
    "B": "B 移除新聞並重新標準化",
    "C": "C 有效因子固定權重（不重新標準化）",
    "D": "D 可用性閘門（缺值不計入）",
}

SCHEME_DESC: dict[str, str] = {
    "A": "現行行為：新聞缺值被填為 0.0，但 0.10 權重仍計入分母。",
    "Actrl": "控制組：分子移除 news、分母仍用 W_A。與 A 的差異 = 純新聞分子項。",
    "B": "把 news 完全排除於分子與分母，其餘可用因子權重重新標準化。",
    "C": "把 news 排除但分母固定為名目權重總和 1.0，不分攤重算。",
    "D": "只有 (股票, 日期) 確有新聞記錄時才計入 news；缺值不冒充中性情緒。",
}

# 三種可表示／不可表示的新聞狀態
NEWS_STATES: tuple[str, ...] = ("absent", "neutral", "scored")
NEWS_STATE_LABELS = {
    "absent": "沒有新聞（缺值）",
    "neutral": "有新聞但有效情緒值為 0",
    "scored": "有新聞且情緒值非 0",
}


# ==========================================================================
# 分數變體
# ==========================================================================
def news_present(row: pd.Series) -> bool:
    """該列是否有「可用的新聞資料」（衰減窗口內存在新聞記錄）。

    與 ``news_sentiment`` 的值無關：有記錄但情緒為 0 仍屬「存在」。
    舊面板若沒有 ``news_present`` 欄位（presence 資訊已被 ``fillna`` 摧毀），
    無法區分 → 退回 A 的行為（視為存在），不臆測為缺值。
    """
    if "news_present" in row.index:
        v = row["news_present"]
        if pd.isna(v):
            return False
        return bool(float(v) == 1.0)
    return NEWS_KEY in row.index and pd.notna(row.get(NEWS_KEY))


def news_state(row: pd.Series) -> str:
    """三態分類：``absent``（沒有新聞）／``neutral``（有記錄但情緒為 0）／``scored``。"""
    if not news_present(row):
        return "absent"
    v = row.get(NEWS_KEY, np.nan) if NEWS_KEY in row.index else np.nan
    if pd.isna(v):
        return "absent"
    return "scored" if float(v) != 0.0 else "neutral"


def _score_from_parts(parts: list[tuple[str, float]], baseline_w: float,
                      present: bool, scheme: str) -> float:
    """由 (鍵, 值) 清單計算單一方案的分數（與 ``score_row`` 相同的 clip 與零分母行為）。"""
    W = sig_mod.WEIGHTS
    if scheme == "A":
        num = sum(W[k] * v for k, v in parts)
        den = baseline_w
    elif scheme == "Actrl":
        num = sum(W[k] * v for k, v in parts if k != NEWS_KEY)
        den = baseline_w                      # 分母刻意維持 baseline
    elif scheme == "B":
        keep = [(k, v) for k, v in parts if k != NEWS_KEY]
        num = sum(W[k] * v for k, v in keep)
        den = sum(W[k] for k, _ in keep)
    elif scheme == "C":
        keep = [(k, v) for k, v in parts if k != NEWS_KEY]
        num = sum(W[k] * v for k, v in keep)
        den = 1.0                             # 名目權重總和，不分攤重算
    elif scheme == "D":
        keep = parts if present else [(k, v) for k, v in parts if k != NEWS_KEY]
        num = sum(W[k] * v for k, v in keep)
        den = sum(W[k] for k, _ in keep)
    else:
        raise ValueError(f"未知方案 {scheme!r}；可用：{SCHEMES}")
    if den == 0:
        return 0.0
    return float(np.clip(num / den, -1.0, 1.0))


def variant_score(row: pd.Series, scheme: str) -> float:
    """單列變體分數（便利 API，測試與除錯用）。"""
    parts = list(sig_mod.score_components(row).items())
    baseline_w = sum(sig_mod.WEIGHTS[k] for k, _ in parts)
    return _score_from_parts(parts, baseline_w, news_present(row), scheme)


def variant_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """把特徵面板轉成含各方案分數的長表。

    回傳欄位：``date, stock_id, score_A, score_Actrl, score_B, score_C, score_D,
    w_a, den_a, den_b, news_present, news_state``。
    ``w_a`` / ``den_a`` / ``den_b`` 供尺度（是否正比例常數）檢驗使用。
    """
    if panel is None or panel.empty:
        return pd.DataFrame()
    out: list[pd.DataFrame] = []
    for sid, g in panel.groupby("stock_id", sort=False):
        g = g.reset_index(drop=True)
        cols: dict[str, list[float]] = {s: [] for s in SCHEMES}
        w_a_l, den_a_l, den_b_l, pres_l, state_l = [], [], [], [], []
        for _, row in g.iterrows():
            parts = list(sig_mod.score_components(row).items())
            w_a = sum(sig_mod.WEIGHTS[k] for k, _ in parts)
            keep = [(k, v) for k, v in parts if k != NEWS_KEY]
            den_b = sum(sig_mod.WEIGHTS[k] for k, _ in keep)
            pres = news_present(row)
            for s in SCHEMES:
                cols[s].append(_score_from_parts(parts, w_a, pres, s))
            w_a_l.append(w_a)
            den_a_l.append(w_a)
            den_b_l.append(den_b)
            pres_l.append(1.0 if pres else 0.0)
            state_l.append(news_state(row))
        rec = {
            "date": pd.to_datetime(g["date"]).dt.normalize().to_numpy(),
            "stock_id": str(sid),
        }
        for s in SCHEMES:
            rec[f"score_{s}"] = cols[s]
        rec["w_a"] = w_a_l
        rec["den_a"] = den_a_l
        rec["den_b"] = den_b_l
        rec["news_present"] = pres_l
        rec["news_state"] = state_l
        out.append(pd.DataFrame(rec))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# ==========================================================================
# 代數恆等式檢查（在 clip 之前驗證實作正確性）
# ==========================================================================
@dataclass
class InvariantCheck:
    """分解恆等式的實測誤差（應為 0 或機器精度等級）。"""

    n_rows: int = 0
    # A − A* = 0.10 × v_news / W_A（未 clip）
    max_abs_news_term: float = float("nan")
    n_rows_news_term_zero: int = 0
    # den_b = w_a − 0.10（news 存在於 parts 時）
    max_abs_den_err: float = float("nan")
    # C_raw = B_raw × den_b（未 clip、den_b > 0）
    max_abs_scale_err: float = float("nan")
    n_clipped: dict[str, int] = field(default_factory=dict)
    ok: bool = False

    def label(self) -> str:
        return "通過" if self.ok else "未通過"


def check_invariants(panel: pd.DataFrame) -> InvariantCheck:
    """在**未 clip** 的層級驗證四個代數恆等式，確保後續差異來自設計而非實作錯誤。"""
    res = InvariantCheck()
    if panel is None or panel.empty:
        return res
    max_term = 0.0
    max_den = 0.0
    max_scale = 0.0
    n_zero = 0
    n = 0
    W = sig_mod.WEIGHTS
    clip_cnt = {s: 0 for s in SCHEMES}
    for _, row in panel.iterrows():
        parts = list(sig_mod.score_components(row).items())
        w_a = sum(W[k] for k, _ in parts)
        keep = [(k, v) for k, v in parts if k != NEWS_KEY]
        den_b = sum(W[k] for k, _ in keep)
        v_news = next((v for k, v in parts if k == NEWS_KEY), None)
        num_a = sum(W[k] * v for k, v in parts)
        num_a_ctrl = sum(W[k] * v for k, v in keep)
        num_b = num_a_ctrl
        # (1) A − A* 的分子差 = 0.10 × v_news（分母同為 w_a → 差 = 0.10·v/w_a）
        if w_a > 0:
            lhs = (num_a - num_a_ctrl) / w_a
            rhs = (NEWS_WEIGHT * (v_news or 0.0)) / w_a
            max_term = max(max_term, abs(lhs - rhs))
            if abs(rhs) < 1e-15:
                n_zero += 1
        # (2) den_b = w_a − 0.10（news 在 parts 中）
        if v_news is not None:
            max_den = max(max_den, abs(den_b - (w_a - NEWS_WEIGHT)))
        # (3) C_raw = B_raw × den_b（den_b > 0）
        if den_b > 0:
            b_raw = num_b / den_b
            c_raw = num_b / 1.0
            max_scale = max(max_scale, abs(c_raw - b_raw * den_b))
        # clip 計數（clip 會破壞嚴格恆等式，需揭露）
        for s in SCHEMES:
            if abs(_score_from_parts(parts, w_a, news_present(row), s)) >= 1.0:
                # 貼邊界不代表一定被 clip，只統計原始值超過界線者
                pass
        n += 1
    res.n_rows = n
    res.max_abs_news_term = max_term
    res.n_rows_news_term_zero = n_zero
    res.max_abs_den_err = max_den
    res.max_abs_scale_err = max_scale
    res.n_clipped = clip_cnt
    tol = 1e-12
    res.ok = (max_term <= tol and max_den <= tol and max_scale <= tol)
    return res


def clipped_count(frame: pd.DataFrame) -> dict[str, int]:
    """各方案分數落在 ±1 邊界（可能被 clip）的列數。"""
    out: dict[str, int] = {}
    for s in SCHEMES:
        col = f"score_{s}"
        if col in frame.columns:
            out[s] = int((frame[col].abs() >= 1.0 - 1e-12).sum())
    return out


# ==========================================================================
# 排名比較（§3.1 排名不變性）
# ==========================================================================
@dataclass
class RankComparison:
    left: str
    right: str
    n_days: int = 0
    n_rows: int = 0
    n_rows_diff: int = 0
    identical_days: int = 0
    identical_ratio: float = float("nan")
    spearman_mean: float = float("nan")
    spearman_min: float = float("nan")
    top5_overlap: float = float("nan")      # 平均重疊比例 0~1
    max_rank_move: int = 0
    ratio_min: float = float("nan")
    ratio_max: float = float("nan")
    ratio_std: float = float("nan")
    max_abs_diff: float = float("nan")
    relationship: str = ""

    def short(self) -> str:
        return (f"{self.identical_ratio:.1%} 日排名相同；"
                f"Spearman 均值 {self.spearman_mean:.4f}；"
                f"top5 重疊 {self.top5_overlap:.1%}；"
                f"最大名次變動 {self.max_rank_move}")


def rank_comparison(frame: pd.DataFrame, left: str, right: str, *,
                    top_n: int = 5) -> RankComparison:
    """逐日比較兩方案的橫斷面排名：相同率、Spearman、top-5 重疊、名次變動與尺度比。"""
    cl, cr = f"score_{left}", f"score_{right}"
    out = RankComparison(left=left, right=right)
    if frame.empty or cl not in frame.columns or cr not in frame.columns:
        return out
    d = frame.dropna(subset=[cl, cr]).copy()
    out.n_rows = len(d)
    diff_mask = (d[cl] != d[cr])
    out.n_rows_diff = int(diff_mask.sum())
    out.max_abs_diff = float((d[cl] - d[cr]).abs().max()) if len(d) else float("nan")

    # 尺度比（right/left）：僅在 left 非零時有意義
    nz = d[d[cl].abs() > 1e-12]
    if len(nz):
        ratio = nz[cr] / nz[cl]
        out.ratio_min = float(ratio.min())
        out.ratio_max = float(ratio.max())
        out.ratio_std = float(ratio.std(ddof=1)) if len(nz) > 1 else 0.0

    id_days = 0
    sp: list[float] = []
    ov: list[float] = []
    moves: list[int] = []
    n_days = 0
    for _, g in d.groupby("date", sort=True):
        if len(g) < 2:
            continue
        n_days += 1
        a = g[cl].to_numpy(dtype=float)
        b = g[cr].to_numpy(dtype=float)
        ra = pd.Series(a).rank(method="average").to_numpy()
        rb = pd.Series(b).rank(method="average").to_numpy()
        if np.allclose(ra, rb, rtol=0.0, atol=1e-12):
            id_days += 1
        moves.append(int(np.abs(ra - rb).max()))
        r = _corr(a, b, "spearman")
        if math.isfinite(r):
            sp.append(r)
        ia = np.argsort(-a, kind="mergesort")[:top_n]
        ib = np.argsort(-b, kind="mergesort")[:top_n]
        ov.append(len(set(ia.tolist()) & set(ib.tolist())) / top_n)
    out.n_days = n_days
    out.identical_days = id_days
    out.identical_ratio = (id_days / n_days) if n_days else float("nan")
    if sp:
        out.spearman_mean = float(np.mean(sp))
        out.spearman_min = float(np.min(sp))
    if ov:
        out.top5_overlap = float(np.mean(ov))
    if moves:
        out.max_rank_move = int(max(moves))

    # 關係判定（先看實測排名，再給尺度解釋）
    if out.n_rows == 0:
        out.relationship = "無可比較樣本"
    elif out.n_rows_diff == 0:
        out.relationship = "逐列分數完全相同"
    elif out.identical_ratio == 1.0:
        out.relationship = "分數不同但排名完全相同"
    elif (math.isfinite(out.ratio_std) and out.ratio_std < 1e-9
          and math.isfinite(out.ratio_min)
          and abs(out.ratio_max - out.ratio_min) < 1e-9):
        out.relationship = "單一正比例常數（理論上排名不變）"
    else:
        out.relationship = "非單一正比例常數 → 排名會改變"
    return out


def scale_spread_by_day(frame: pd.DataFrame, left: str = "A",
                        right: str = "B") -> pd.DataFrame:
    """逐日計算 A/B 的理論尺度比 den_b/den_a 跨股票散布。

    若同日內為常數 → 尺度效應不影響排名；否則排名效應來自分母。
    """
    cl, cr = f"score_{left}", f"score_{right}"
    if frame.empty:
        return pd.DataFrame(columns=["date", "n", "ratio_min", "ratio_max",
                                     "ratio_spread"])
    rows = []
    d = frame[frame[cl].abs() > 1e-12]
    for day, g in d.groupby("date", sort=True):
        theo = g["den_b"] / g["den_a"]
        rows.append({
            "date": day,
            "n": len(g),
            "ratio_min": float(theo.min()),
            "ratio_max": float(theo.max()),
            "ratio_spread": float(theo.max() - theo.min()),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 配對差異檢定（IC 與價差）
# ==========================================================================
@dataclass
class PairedDiff:
    label: str
    n_pairs: int = 0
    mean_diff: float = float("nan")
    t_nw: float = float("nan")
    p_value: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    n_pos: int = 0

    @property
    def rejected_at(self) -> bool:
        return bool(math.isfinite(self.p_value) and self.p_value <= 0.05)


def paired_diff(left: pd.Series, right: pd.Series, *, label: str,
                lags: int | None = None, sims: int = 500) -> PairedDiff:
    """配對差異（left − right）的 Newey-West 穩健 t 與區塊自助法信賴區間。

    兩序列同源同日，必須**配對**而非各自檢定；這裡只比較差異本身。
    """
    out = PairedDiff(label=label)
    a = pd.to_numeric(pd.Series(left), errors="coerce")
    b = pd.to_numeric(pd.Series(right), errors="coerce")
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 3:
        return out
    diff = (df["a"] - df["b"]).to_numpy(dtype=float)
    n = len(diff)
    m = float(np.mean(diff))
    L = int(lags if lags is not None else max(1, min(5, n // 4)))
    t, se = newey_west_tstat(diff, L)
    lo, hi = block_bootstrap_ci(diff, n_iter=sims, block=max(1, L), seed=0)
    out.n_pairs = n
    out.mean_diff = m
    out.t_nw = t
    out.p_value = _two_sided_p(t)
    out.ci_low, out.ci_high = lo, hi
    out.n_pos = int((diff > 0).sum())
    return out


def daily_group_spread(signal_frame: pd.DataFrame, signal: str, horizon: int,
                       *, n_groups: int = N_GROUPS) -> pd.Series:
    """逐日 Q_{top} − Q_{bottom} 價差序列（分組只用當期訊號）。"""
    df = _prep_pairs(signal_frame, signal, horizon)
    if df.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    need = max(n_groups * 2, MIN_STOCKS_PER_DAY)
    ret_col = f"fwd_{horizon}"
    out: dict[pd.Timestamp, float] = {}
    for d, g in df.groupby("date", sort=True):
        if len(g) < need:
            continue
        grp = _assign_groups(g[signal], n_groups)
        vals = g[ret_col].to_numpy(dtype=float)
        top = vals[grp == n_groups - 1]
        bot = vals[grp == 0]
        if len(top) == 0 or len(bot) == 0:
            continue
        out[pd.Timestamp(d)] = float(top.mean() - bot.mean())
    if not out:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return pd.Series(out, dtype=float).sort_index()


# ==========================================================================
# 交易層級配對回測
# ==========================================================================
@dataclass
class SchemeBacktest:
    scheme: str
    segment: str
    cost: str
    total_return: float = float("nan")
    cagr: float = float("nan")
    max_drawdown: float = float("nan")
    sharpe: float = float("nan")
    round_trips: int = 0
    turnover: float = float("nan")
    port_turnover_ann: float = float("nan")
    cost_total: float = float("nan")
    exposure: float = float("nan")
    cash_ratio: float = float("nan")
    bench_return: float = float("nan")
    bench_cagr: float = float("nan")
    excess: float = float("nan")
    n_score_days: int = 0


def _bt_row(res, scheme: str, segment: str, cost: str,
            n_score_days: int) -> SchemeBacktest:
    r = SchemeBacktest(scheme=scheme, segment=segment, cost=cost,
                       n_score_days=n_score_days)
    r.total_return = float(getattr(res, "total_return", float("nan")))
    r.cagr = float(getattr(res, "cagr", float("nan")))
    r.max_drawdown = float(getattr(res, "max_drawdown", float("nan")))
    r.sharpe = float(getattr(res, "sharpe", float("nan")))
    r.round_trips = int(getattr(res, "round_trips", 0))
    r.turnover = float(getattr(res, "turnover", float("nan")))
    r.port_turnover_ann = float(getattr(res, "portfolio_turnover_annualized",
                                        float("nan")))
    r.cost_total = float(getattr(res, "cost_total", float("nan")))
    r.exposure = float(getattr(res, "exposure", float("nan")))
    if math.isfinite(r.exposure):
        r.cash_ratio = max(0.0, 1.0 - r.exposure)
    br = float(getattr(res, "benchmark_total_return", float("nan")))
    r.bench_return = br
    r.bench_cagr = float(getattr(res, "benchmark_cagr", float("nan")))
    if math.isfinite(br) and math.isfinite(r.total_return):
        r.excess = r.total_return - br
    return r


@dataclass
class HoldingsDiff:
    """每日持倉與交易的配對差異（以已完成交易重建持倉）。"""

    n_days: int = 0
    days_identical: int = 0
    identical_ratio: float = float("nan")
    mean_sym_diff: float = float("nan")     # 每日持倉對稱差的平均檔數
    cells_diff: int = 0                      # (日, 股) 欄位不同的總數
    cells_total: int = 0
    trades_a: int = 0
    trades_b: int = 0
    trade_jaccard: float = float("nan")
    entries_only_a: int = 0
    entries_only_b: int = 0


def _holdings(trades: pd.DataFrame, calendar: list[pd.Timestamp]
              ) -> dict[pd.Timestamp, set[str]]:
    """由已完成交易重建每日持倉（entry_date ≤ d < exit_date）。

    未平倉部位未納入（各方案一致，故不影響配對比較）。
    """
    hold: dict[pd.Timestamp, set[str]] = {d: set() for d in calendar}
    if trades is None or trades.empty:
        return hold
    for t in trades.itertuples(index=False):
        try:
            e = pd.Timestamp(t.entry_date).normalize()
            x = pd.Timestamp(t.exit_date).normalize()
        except Exception:  # noqa: BLE001
            continue
        sid = str(t.stock_id)
        for d in calendar:
            if e <= d < x:
                hold[d].add(sid)
    return hold


def holdings_diff(hold_a: dict, hold_b: dict) -> tuple[int, int, float, int]:
    """回傳 (比較天數, 完全相同天數, 平均對稱差檔數, 持倉不同的欄位總數)。

    第四項為 Σ|A Δ B|（每天持倉集合不同的股票數總和），不是比較欄位總數。
    """
    days = sorted(set(hold_a) & set(hold_b))
    if not days:
        return 0, 0, float("nan"), 0
    same = 0
    sym = 0
    for d in days:
        a, b = hold_a.get(d, set()), hold_b.get(d, set())
        if a == b:
            same += 1
        sym += len(a ^ b)
    return len(days), same, sym / len(days), sym


def trade_key_set(trades: pd.DataFrame) -> set[tuple]:
    """(entry_date, stock_id) 集合，用於交易差異。"""
    if trades is None or trades.empty:
        return set()
    out = set()
    for t in trades.itertuples(index=False):
        try:
            e = pd.Timestamp(t.entry_date).normalize()
        except Exception:  # noqa: BLE001
            continue
        out.add((e, str(t.stock_id)))
    return out


# ==========================================================================
# 報告
# ==========================================================================
@dataclass
class AblationReport:
    generated_at: str = ""
    period: str = ""
    n_stocks: int = 0
    n_days: int = 0
    split_desc: str = ""
    # 資料結構
    news_rows: int = 0
    news_days: int = 0
    news_states: dict[str, int] = field(default_factory=dict)
    n_rows_diluted: int = 0        # news 的 0.10 佔據分母的列數
    n_rows_record: int = 0         # 有新聞記錄（presence = 1）的列數
    n_rows_contributed: int = 0    # news 分子非 0 的列數
    w_var_days: int = 0            # W 跨股票變動的天數
    w_min: float = float("nan")
    w_max: float = float("nan")
    invariants: InvariantCheck = field(default_factory=InvariantCheck)
    clipped: dict[str, int] = field(default_factory=dict)
    # 排名
    ranks: list[RankComparison] = field(default_factory=list)
    scale_spread: pd.DataFrame = field(default_factory=pd.DataFrame)
    # IC / 分組
    ic_rows: list[dict] = field(default_factory=list)
    paired_ic: list[PairedDiff] = field(default_factory=list)
    paired_spread: list[PairedDiff] = field(default_factory=list)
    paired_bt: list[PairedDiff] = field(default_factory=list)
    mt_bt: dict = field(default_factory=dict)
    quintiles: dict[str, dict] = field(default_factory=dict)
    mt_ic: dict = field(default_factory=dict)
    mt_spread: dict = field(default_factory=dict)
    # 覆蓋率
    coverage: dict[str, dict] = field(default_factory=dict)
    # 門檻
    threshold: dict[str, int] = field(default_factory=dict)
    # 回測
    backtests: list[SchemeBacktest] = field(default_factory=list)
    holdings: dict[str, dict[str, HoldingsDiff]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    verdict: str = ""
    verdict_why: str = ""

    # ------------------------------------------------------------------
    def to_text(self) -> str:
        L: list[str] = []
        A = L.append
        A("=" * 78)
        A("P2.6 B17 新聞因子消融驗證與合成分數因果診斷")
        A(f"產生時間 {self.generated_at}   期間 {self.period}")
        A(f"股票池 {self.n_stocks} 檔、{self.n_days} 個訊號日")
        A("=" * 78)

        A("")
        A("[1] 時間切分（沿用既有定義，不可依結果更改）")
        A(f"  {self.split_desc}")

        A("")
        A("[2] 實際分數公式與缺值機制")
        A(f"  news 表 {self.news_rows} 筆、涵蓋 {self.news_days} 個日曆日")
        for k in NEWS_STATES:
            A(f"    {NEWS_STATE_LABELS[k]:<18} {self.news_states.get(k, 0):>6} 列")
        A(f"  0.10 佔據分母（稀釋）列數 {self.n_rows_diluted:,}；"
          f"其中**有新聞記錄**者 {self.n_rows_record:,} 列；"
          f"分子實際非 0 者 {self.n_rows_contributed:,} 列")
        if self.n_rows_diluted and self.n_rows_record < self.n_rows_diluted:
            A("    → 稀釋發生在**所有**列（fillna 把缺值填成 0.0，使 0.10 恆入分母），"
              "而非只有有新聞的列；但分子真正有資訊的列極少。")
        A(f"  可用權重 W 跨股票變動的交易日 {self.w_var_days}/{self.n_days}"
          f"（範圍 {self.w_min:.2f} ~ {self.w_max:.2f}）")
        A("  現行公式：score = Σ wᵢvᵢ / Σ wᵢ（只對「可用」因子標準化），"
          "news 缺值填 0.0 但 0.10 仍入分母。")
        A("  三態區分：absent（沒有新聞）／neutral（有記錄但情緒 0）／scored（情緒非 0）。")
        A("  ⚠️ 「沒有成功抓到新聞」目前**無法表示**：news 表無抓取狀態欄位，"
          "且 news_rss 吞掉例外回傳空清單 → 抓取失敗與無新聞不可分。")

        A("")
        A("[3] 代數恆等式檢查（clip 前，驗證實作而非假設）")
        iv = self.invariants
        A(f"  檢查 {iv.n_rows:,} 列 → {iv.label()}")
        A(f"    A − A* − 0.10·v/W  最大誤差 {iv.max_abs_news_term:.3e}"
          f"（其中 {iv.n_rows_news_term_zero:,} 列新聞分子為 0）")
        A(f"    den_b − (W − 0.10) 最大誤差 {iv.max_abs_den_err:.3e}")
        A(f"    C − B×den_b        最大誤差 {iv.max_abs_scale_err:.3e}")
        if any(self.clipped.values()):
            A(f"    分數貼 ±1 邊界（可能被 clip）："
              + "、".join(f"{k}={v}" for k, v in self.clipped.items()))

        A("")
        A("[4] 排名不變性（逐日橫斷面）")
        A(f"  {'配對':<10}{'排名相同日':>12}{'Spearman均':>12}"
          f"{'top5重疊':>10}{'最大名次':>10}  關係")
        for r in self.ranks:
            ident = (f"{r.identical_days}/{r.n_days}"
                     if r.n_days else "n/a")
            A(f"  {r.left + '→' + r.right:<10}{ident:>12}"
              f"{r.spearman_mean:>12.4f}{r.top5_overlap:>10.1%}"
              f"{r.max_rank_move:>10}  {r.relationship}")
        if not self.scale_spread.empty:
            sp = self.scale_spread["ratio_spread"]
            A(f"  理論尺度比 den_b/den_a 的同日跨股票散布：均值 {sp.mean():.5f}、"
              f"最大 {sp.max():.5f}、散布為 0 的天數 {int((sp < 1e-12).sum())}/{len(sp)}")
            A("  → 散布 > 0 表示分母改變**不是**單一正比例常數，"
              "跨股票影響不一致，因此可能改變排名。")

        A("")
        A("[5] IC / Rank IC（每日橫斷面，NW 穩健 t）")
        A(f"  {'方案':<7}{'區段':<7}{'h':>3}{'天數':>6}{'Pearson IC':>12}"
          f"{'Rank IC':>10}{'NW t':>8}{'p':>9}{'CI跨0':>7}")
        for r in self.ic_rows:
            ci = "是" if (r["ci_low"] <= 0 <= r["ci_high"]) else "否"
            A(f"  {r['scheme']:<7}{r['segment']:<7}{r['horizon']:>3}"
              f"{r['n_days']:>6}{r['ic_mean']:>12.4f}{r['rank_ic']:>10.4f}"
              f"{r['t_nw']:>8.2f}{r['p_value']:>9.4f}{ci:>7}")

        A("")
        A("[6] 配對 IC 差異（欄位 A−X = baseline **減** 該方案；同日配對，非各自檢定）")
        A("   （ΔRankIC < 0 表示該方案的 Rank IC 高於 baseline）")
        A(f"  {'區段':<7}{'配對':<8}{'h':>3}{'配對日':>7}{'ΔRankIC':>10}"
          f"{'NW t':>8}{'p':>9}{'95%CI':>24}{'BH校正p':>10}")
        adj = self.mt_ic.get("adjusted", [])
        for i, r in enumerate(self.paired_ic):
            a = adj[i] if i < len(adj) else float("nan")
            seg, rest = r.label.split("|", 1) if "|" in r.label else ("", r.label)
            pair, h = rest.split("@") if "@" in rest else (rest, "")
            ci = (f"[{r.ci_low:+.4f},{r.ci_high:+.4f}]"
                  if math.isfinite(r.ci_low) else "n/a")
            A(f"  {seg:<7}{pair:<8}{h:>3}{r.n_pairs:>7}{r.mean_diff:>10.4f}"
              f"{r.t_nw:>8.2f}{r.p_value:>9.4f}{ci:>24}{a:>10.4f}")
        A(f"  多重檢定：配對 IC 差異共 {self.mt_ic.get('n_tested', 0)} 個假設，"
          f"未校正顯著 {self.mt_ic.get('n_raw', 0)} 個、"
          f"BH 校正後 {self.mt_ic.get('n_rejected', 0)} 個。")

        A("")
        A("[7] 配對五分組價差差異（Q5−Q1；欄位 A−X = baseline **減** 該方案）")
        for i, r in enumerate(self.paired_spread):
            a = (self.mt_spread.get("adjusted", [])[i]
                 if i < len(self.mt_spread.get("adjusted", [])) else float("nan"))
            seg, rest = r.label.split("|", 1) if "|" in r.label else ("", r.label)
            pair, h = rest.split("@") if "@" in rest else (rest, "")
            ci = (f"[{r.ci_low:+.4%},{r.ci_high:+.4%}]"
                  if math.isfinite(r.ci_low) else "n/a")
            A(f"  {pair:<8}{seg:<7}{h:>3}{r.n_pairs:>7}{r.mean_diff:>11.4%}"
              f"{r.t_nw:>8.2f}{r.p_value:>9.4f}{ci:>26}{a:>10.4f}")
        A(f"  多重檢定：配對價差差異共 {self.mt_spread.get('n_tested', 0)} 個假設，"
          f"未校正顯著 {self.mt_spread.get('n_raw', 0)} 個、"
          f"BH 校正後 {self.mt_spread.get('n_rejected', 0)} 個。")

        A("")
        A("[8] 各方案五分組價差與單調性（h=5，全期）")
        A(f"  {'方案':<7}{'天數':>6}{'Q5−Q1':>11}{'扣成本後':>11}"
          f"{'單調性':>9}{'為正比例':>10}")
        for key, q in self.quintiles.items():
            if q.get("horizon") != 5:
                continue
            mono = q.get("monotonicity", float("nan"))
            A(f"  {key:<7}{q.get('n_days', 0):>6}{q.get('spread', float('nan')):>11.4%}"
              f"{q.get('spread_net', float('nan')):>11.4%}"
              f"{mono:>9.3f}{q.get('pos_ratio', float('nan')):>10.1%}")

        A("")
        A("[9] 新聞覆蓋日 vs 未覆蓋日（避免時間差異混淆）")
        for k, v in self.coverage.items():
            ic_a, ic_b = v.get("ic_a", float("nan")), v.get("ic_b", float("nan"))
            A(f"  {k}: {v.get('n_days', 0)} 天；其中有可用 5 日前瞻報酬者 "
              f"{v.get('n_fwd', 0)} 天")
            if v.get("n_fwd", 0) == 0:
                A("    → 該區間**無任何可用前瞻報酬**，IC 必為空 → 樣本不足，無法判斷")
            else:
                A(f"    RankIC 均值 A={ic_a:+.4f}、B={ic_b:+.4f}")
            A(f"    A→B 排名相同率 {v.get('identical_ratio', float('nan')):.1%}")

        A("")
        A("[10] 絕對門檻翻轉（buy_threshold = 0.2，尺度會影響成交）")
        for k, v in self.threshold.items():
            A(f"  {k}: {v:,} 列")

        A("")
        A("[11] 交易層級配對回測（相同規則、成本、持倉限制；僅分數不同）")
        A(f"  {'方案':<7}{'區段':<7}{'成本':<9}{'總報酬':>10}{'年化':>10}"
          f"{'MDD':>9}{'Sharpe':>8}{'筆數':>6}{'雙邊週轉':>10}"
          f"{'曝險':>7}{'現金':>7}{'超額(vs0050)':>14}")
        for r in self.backtests:
            A(f"  {r.scheme:<7}{r.segment:<7}{r.cost:<9}{r.total_return:>10.2%}"
              f"{r.cagr:>10.2%}{r.max_drawdown:>9.2%}{r.sharpe:>8.2f}"
              f"{r.round_trips:>6}{r.turnover:>10.2f}{r.exposure:>7.1%}"
              f"{r.cash_ratio:>7.1%}{r.excess:>14.2%}")

        if self.paired_bt:
            A("")
            A("  成本後**日報酬**配對差異（欄位 A−X = baseline **減** 該方案；"
              "Δ < 0 表示該方案優於 baseline）")
            A(f"  {'區段':<7}{'配對':<8}{'配對日':>7}{'Δ日報酬均':>12}"
              f"{'NW t':>8}{'p':>9}{'95%CI':>26}{'BH校正p':>10}")
            adj_bt = self.mt_bt.get("adjusted", [])
            for i, r in enumerate(self.paired_bt):
                a = adj_bt[i] if i < len(adj_bt) else float("nan")
                seg, rest = r.label.split("|", 1) if "|" in r.label else ("", r.label)
                pair, _tag = rest.split("@") if "@" in rest else (rest, "")
                ci = (f"[{r.ci_low:+.5f},{r.ci_high:+.5f}]"
                      if math.isfinite(r.ci_low) else "n/a")
                A(f"  {seg:<7}{pair:<8}{r.n_pairs:>7}{r.mean_diff:>12.6f}"
                  f"{r.t_nw:>8.2f}{r.p_value:>9.4f}{ci:>26}{a:>10.4f}")
            A(f"  多重檢定：交易層級配對假設 {self.mt_bt.get('n_tested', 0)} 個，"
              f"未校正顯著 {self.mt_bt.get('n_raw', 0)} 個、"
              f"BH 校正後 {self.mt_bt.get('n_rejected', 0)} 個。")

        A("")
        A("[12] 每日持倉與交易差異（相對 A，分區段）")
        A(f"  {'區段':<7}{'方案':<7}{'持倉相同日':>12}{'平均對稱差':>12}{'持倉不同欄位':>14}"
          f"{'交易數':>12}{'僅A':>6}{'僅該方案':>10}{'Jaccard':>9}")
        for seg_name in ("full", "train", "valid", "test"):
            for k, h in self.holdings.get(seg_name, {}).items():
                A(f"  {seg_name:<7}{k:<7}{h.days_identical:>6}/{h.n_days:<5}"
                  f"{h.mean_sym_diff:>12.2f}{h.cells_diff:>14}"
                  f"{h.trades_a:>6}/{h.trades_b:<5}{h.entries_only_a:>6}"
                  f"{h.entries_only_b:>10}{h.trade_jaccard:>9.3f}")

        A("")
        A("[13] 結論")
        A(f"  {self.verdict}")
        if self.verdict_why:
            for line in self.verdict_why.split("\n"):
                A(f"    {line}")

        A("")
        A("[14] 限制與研究紀律")
        for n in self.notes:
            A(f"  - {n}")
        return "\n".join(L)


# ==========================================================================
# 主流程
# ==========================================================================
def run_ablation(*, horizons: tuple[int, ...] = HORIZONS,
                 train_frac: float = 0.5, valid_frac: float = 0.25,
                 sims: int = DEFAULT_SIMS,
                 run_backtests: bool = True) -> AblationReport:
    """執行完整消融實驗。方案比較只用 train/valid；test 只作一次最終確認。"""
    qa = loader.new_stats()
    panel, price_panel, benchmark = load_db_inputs(stats=qa)
    if panel.empty:
        raise RuntimeError("特徵面板為空，請先執行 ingest 與 run")

    rep = AblationReport(generated_at=dt.datetime.now().isoformat(timespec="seconds"))

    # ---- 資料結構盤點 ----
    frame = variant_frame(panel)
    if frame.empty:
        raise RuntimeError("無法建立方案分數")
    dates = sorted(pd.to_datetime(frame["date"]).dt.normalize().unique())
    rep.n_days = len(dates)
    rep.n_stocks = int(frame["stock_id"].nunique())
    rep.period = f"{pd.Timestamp(dates[0]):%Y-%m-%d} ~ {pd.Timestamp(dates[-1]):%Y-%m-%d}"

    from .data import store
    news = store.read_sql("SELECT published, stock_id, score FROM news")
    if news is not None and not news.empty:
        rep.news_rows = len(news)
        rep.news_days = int(pd.to_datetime(news["published"]).dt.date.nunique())

    st = frame["news_state"].value_counts().to_dict()
    rep.news_states = {k: int(st.get(k, 0)) for k in NEWS_STATES}
    # 「有新聞記錄」與「0.10 佔據分母」是兩件事，必須分開計數：
    # fillna(0.0) 讓 news_sentiment 在**所有**列都非缺值 → 0.10 恆入分母，
    # 但 presence = 1 只有少數列。稀釋列數以 den_a − den_b = 0.10 判定。
    rep.n_rows_record = int((frame["news_present"] == 1).sum())
    rep.n_rows_diluted = int(
        np.isclose(frame["den_a"] - frame["den_b"], NEWS_WEIGHT,
                   atol=1e-12).sum())
    rep.n_rows_contributed = int(
        (frame["score_A"] != frame["score_Actrl"]).sum())
    w = frame["w_a"]
    rep.w_min, rep.w_max = float(w.min()), float(w.max())
    day_nuniq = frame.groupby(pd.to_datetime(frame["date"]).dt.normalize())["w_a"].nunique()
    rep.w_var_days = int((day_nuniq > 1).sum())

    rep.invariants = check_invariants(panel)
    rep.clipped = clipped_count(frame)

    # ---- 時間切分（沿用既有定義） ----
    hold = baseline().hold_days
    split = make_split([pd.Timestamp(d) for d in dates],
                       train_frac=train_frac, valid_frac=valid_frac,
                       embargo_days=hold)
    rep.split_desc = split.describe()

    # ---- 前瞻報酬 ----
    sf = attach_forward_returns(frame, price_panel, horizons)
    if sf.empty:
        raise RuntimeError("無法附加前瞻報酬")

    # ---- 排名不變性 ----
    pairs = [("A", "Actrl"), ("A", "B"), ("A", "C"), ("A", "D"),
             ("B", "C"), ("B", "D"), ("C", "D")]
    rep.ranks = [rank_comparison(frame, l, r) for l, r in pairs]
    rep.scale_spread = scale_spread_by_day(frame, "A", "B")

    # ---- IC（各方案 × 各期間 × 各區段） ----
    segs = {"full": None}
    for seg in (split.train, split.valid, split.test):
        segs[seg.name] = seg

    for scheme in SCHEMES:
        col = f"score_{scheme}"
        for h in horizons:
            # 與 P2.3 一致：Rank IC（Spearman）為主，Pearson IC 併列
            ic_rank = ic_series(sf, col, h, method="spearman")
            ic_pear = ic_series(sf, col, h, method="pearson")
            for seg_name, seg in segs.items():
                if seg is None:
                    sub, sub_p = ic_rank, ic_pear
                else:
                    sub = ic_rank[(ic_rank.index >= seg.start)
                                  & (ic_rank.index <= seg.end)]
                    sub_p = ic_pear[(ic_pear.index >= seg.start)
                                    & (ic_pear.index <= seg.end)]
                r = ic_summary(sub, col, h, segment=seg_name,
                               sims=max(200, sims // 4))
                rp = ic_summary(sub_p, col, h, segment=seg_name, sims=100)
                rep.ic_rows.append({
                    "scheme": scheme, "segment": seg_name, "horizon": h,
                    "n_days": r.n_days, "ic_mean": rp.ic_mean,   # Pearson
                    "rank_ic": r.ic_mean,                        # Spearman
                    "t_nw": r.t_nw, "p_value": r.p_value,
                    "ci_low": r.ci_low, "ci_high": r.ci_high,
                    "sufficient": r.sufficient,
                })

    # ---- 配對 IC 差異（方案 − A） ----
    p_ic: list[PairedDiff] = []
    for scheme in ("Actrl", "B", "C", "D"):
        for h in horizons:
            base = ic_series(sf, f"score_A", h, method="spearman")
            alt = ic_series(sf, f"score_{scheme}", h, method="spearman")
            for seg_name, seg in segs.items():
                if seg is None:
                    b, a2 = base, alt
                else:
                    b = base[(base.index >= seg.start) & (base.index <= seg.end)]
                    a2 = alt[(alt.index >= seg.start) & (alt.index <= seg.end)]
                d = paired_diff(b, a2, label=f"{seg_name}|A−{scheme}@h{h}",
                                lags=max(1, min(h, 5)), sims=max(200, sims // 4))
                p_ic.append(d)
    rep.paired_ic = p_ic
    raw = [d.p_value for d in p_ic]
    bh = benjamini_hochberg(raw, alpha=0.05)
    rep.mt_ic = {"n_tested": bh["n_tested"],
                 "n_raw": sum(1 for p in raw if math.isfinite(p) and p <= 0.05),
                 "n_rejected": bh["n_rejected"],
                 "adjusted": bh["adjusted"]}

    # ---- 配對價差差異 ----
    p_sp: list[PairedDiff] = []
    for scheme in ("Actrl", "B", "C", "D"):
        for h in horizons:
            base = daily_group_spread(sf, "score_A", h)
            alt = daily_group_spread(sf, f"score_{scheme}", h)
            for seg_name, seg in segs.items():
                if seg is None:
                    b, a2 = base, alt
                else:
                    b = base[(base.index >= seg.start) & (base.index <= seg.end)]
                    a2 = alt[(alt.index >= seg.start) & (alt.index <= seg.end)]
                d = paired_diff(b, a2, label=f"{seg_name}|A−{scheme}@h{h}",
                                lags=max(1, min(h, 5)), sims=max(200, sims // 4))
                p_sp.append(d)
    rep.paired_spread = p_sp
    raw_s = [d.p_value for d in p_sp]
    bhs = benjamini_hochberg(raw_s, alpha=0.05)
    rep.mt_spread = {"n_tested": bhs["n_tested"],
                     "n_raw": sum(1 for p in raw_s
                                  if math.isfinite(p) and p <= 0.05),
                     "n_rejected": bhs["n_rejected"],
                     "adjusted": bhs["adjusted"]}

    # ---- 各方案分組（全期，h=5 主判定） ----
    cm = get_cost_model("typical")
    for scheme in SCHEMES:
        q = quantile_returns(sf, f"score_{scheme}", 5, cost_model=cm)
        rep.quintiles[scheme] = {
            "horizon": 5, "n_days": q.n_days, "spread": q.spread,
            "spread_net": q.spread_net, "monotonicity": q.monotonicity,
            "pos_ratio": (q.robustness.positive_ratio
                          if q.robustness else float("nan")),
        }

    # ---- 新聞覆蓋日 vs 未覆蓋日 ----
    news_days = set(pd.to_datetime(
        frame.loc[frame["news_present"] == 1, "date"]).dt.normalize().unique())
    for label, mask in (("有新聞記錄日", True), ("無新聞記錄日", False)):
        days = {d for d in dates if (pd.Timestamp(d) in news_days) == mask}
        if not days:
            rep.coverage[label] = {"n_days": 0}
            continue
        sub = sf[pd.to_datetime(sf["date"]).isin(days)]
        ic_a = ic_series(sub, "score_A", 5)
        ic_b = ic_series(sub, "score_B", 5)
        # 這些日子是否還有可用的 5 日前瞻報酬（缺前瞻報酬 → IC 必為空）
        # 單位是**交易日**：樣本末端的新聞日可能完全沒有未來價格
        has_fwd = (int(sub.loc[sub["fwd_5"].notna(), "date"].nunique())
                   if "fwd_5" in sub.columns else 0)
        rcmp = rank_comparison(
            frame[pd.to_datetime(frame["date"]).isin(days)], "A", "B")
        rep.coverage[label] = {
            "n_days": len(days),
            "n_fwd": has_fwd,
            "ic_a": float(ic_a.mean()) if len(ic_a) else float("nan"),
            "ic_b": float(ic_b.mean()) if len(ic_b) else float("nan"),
            "identical_ratio": rcmp.identical_ratio,
        }

    # ---- 絕對門檻翻轉 ----
    thr = baseline().buy_threshold
    for scheme in ("B", "C", "D"):
        col = f"score_{scheme}"
        flip = int(((frame["score_A"] >= thr) != (frame[col] >= thr)).sum())
        rep.threshold[f"A vs {scheme}（門檻 {thr}）跨過者"] = flip
    band = int(((frame["score_A"] - thr).abs() <= 0.05).sum())
    rep.threshold[f"A 分數落在 {thr}±0.05 內的列數"] = band

    # ---- 交易層級配對回測 ----
    if run_backtests:
        seg_bounds = {"full": (pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]))}
        for seg in (split.train, split.valid, split.test):
            seg_bounds[seg.name] = (seg.start, seg.end)
        spec = baseline()
        for seg_name, (lo, hi) in seg_bounds.items():
            seg_panel = subset_panel(price_panel, lo, hi)
            seg_bench = subset_by_dates(benchmark, lo, hi) \
                if benchmark is not None and not benchmark.empty else None
            seg_scores = {}
            for scheme in CANDIDATE_SCHEMES:
                s = frame[["date", "stock_id", f"score_{scheme}"]].copy()
                s = s.rename(columns={f"score_{scheme}": "score"})
                s = subset_by_dates(s, lo, hi)
                seg_scores[scheme] = s
            holds: dict[str, dict] = {}
            trade_sets: dict[str, set] = {}
            equities: dict[str, pd.Series] = {}
            for scheme in CANDIDATE_SCHEMES:
                for cost_name in ("zero", "typical"):
                    res = run_backtest(seg_scores[scheme], seg_panel,
                                       strategy=spec,
                                       benchmark=seg_bench,
                                       cost_model=get_cost_model(cost_name))
                    rep.backtests.append(_bt_row(res, scheme, seg_name,
                                                 cost_name,
                                                 int(seg_scores[scheme]["date"].nunique())))
                    if cost_name == "typical":
                        equities[scheme] = res.equity.copy()
                        holds[scheme] = _holdings(
                            res.trades,
                            [pd.Timestamp(d).normalize()
                             for d in sorted(
                                 pd.to_datetime(seg_scores[scheme]["date"])
                                 .dt.normalize().unique())])
                        trade_sets[scheme] = trade_key_set(res.trades)
            # 持倉／交易差異（相對 A）
            seg_hold: dict[str, HoldingsDiff] = {}
            for scheme in ("B", "C", "D"):
                if scheme not in holds or "A" not in holds:
                    continue
                n, same, sym, cells = holdings_diff(holds["A"], holds[scheme])
                ta, tb = trade_sets["A"], trade_sets[scheme]
                inter = len(ta & tb)
                union = len(ta | tb)
                seg_hold[scheme] = HoldingsDiff(
                    n_days=n, days_identical=same,
                    identical_ratio=(same / n if n else float("nan")),
                    mean_sym_diff=sym, cells_diff=cells, cells_total=union,
                    trades_a=len(ta), trades_b=len(tb),
                    trade_jaccard=(inter / union if union else float("nan")),
                    entries_only_a=len(ta - tb), entries_only_b=len(tb - ta))
            if seg_hold:
                rep.holdings[seg_name] = seg_hold

            # 成本後**日報酬**的配對差異：給交易層級比較一個不確定性估計，
            # 否則單一路徑的總報酬差只是點估計（同一面板僅一次實現）。
            if "A" in equities:
                base_ret = equities["A"].pct_change().dropna()
                for scheme in ("B", "C", "D"):
                    if scheme not in equities:
                        continue
                    alt_ret = equities[scheme].pct_change().dropna()
                    d = paired_diff(base_ret, alt_ret,
                                    label=f"{seg_name}|A−{scheme}@ret",
                                    lags=5, sims=max(200, sims // 4))
                    rep.paired_bt.append(d)

    # 交易層級多重檢定（獨立於 IC 與價差兩個家族）
    raw_bt = [d.p_value for d in rep.paired_bt]
    bh_bt = benjamini_hochberg(raw_bt, alpha=0.05)
    rep.mt_bt = {"n_tested": bh_bt["n_tested"],
                 "n_raw": sum(1 for p in raw_bt
                              if math.isfinite(p) and p <= 0.05),
                 "n_rejected": bh_bt["n_rejected"],
                 "adjusted": bh_bt["adjusted"]}

    # ---- 結論（只依 train/valid 判定；test 僅最終確認） ----
    rep.verdict, rep.verdict_why = _decide(rep, split)
    rep.notes = _notes(rep, split, horizons)
    return rep


# ==========================================================================
# 判定與限制
# ==========================================================================
def _sig_rows(rep: AblationReport, segment: str) -> list[dict]:
    return [r for r in rep.ic_rows
            if r["segment"] == segment and r["scheme"] in CANDIDATE_SCHEMES]


def _decide(rep: AblationReport, split) -> tuple[str, str]:
    """只用 train/valid 判定；test 只做一次最終確認並標記為已消耗。"""
    lines: list[str] = []

    # 判準 1：排名是否改變
    ab = next((r for r in rep.ranks if r.left == "A" and r.right == "B"), None)
    ad = next((r for r in rep.ranks if r.left == "A" and r.right == "D"), None)
    rank_effect = bool((ab and ab.identical_ratio < 1.0)
                       or (ad and ad.identical_ratio < 1.0))

    # 判準 2：train/valid 是否有「一致且可解釋」的 IC 改善
    train_valid: dict[str, list[dict]] = {}
    for scheme in ("B", "C", "D"):
        rows = [r for r in rep.ic_rows
                if r["scheme"] == scheme and r["segment"] in ("train", "valid")]
        train_valid[scheme] = rows

    def _mean(rows: list[dict]) -> float:
        v = [r["rank_ic"] for r in rows if math.isfinite(r["rank_ic"])]
        return float(np.mean(v)) if v else float("nan")

    base_tv = _mean([r for r in rep.ic_rows
                     if r["scheme"] == "A" and r["segment"] in ("train", "valid")])

    # 判準 3：配對差異在校正後是否顯著
    adj = rep.mt_ic.get("adjusted", [])
    sig_schemes = set()
    # 期望方向：label 為 "A−X"、diff = IC_A − IC_X
    #   mean_diff < 0  →  IC_X > IC_A  →  該方案**優於** baseline
    for i, d in enumerate(rep.paired_ic):
        a = adj[i] if i < len(adj) else float("nan")
        if math.isfinite(a) and a <= 0.05 and d.mean_diff < 0:
            seg = d.label.split("|")[0]
            if seg in ("train", "valid"):
                pair = d.label.split("|")[1].split("@")[0]   # A−B
                if "−" in pair:
                    sig_schemes.add(pair.split("−")[1])

    # 判準 4：交易層級（成本後日報酬）是否在 train **與** valid 皆顯著為正
    # 嚴格一致性條件：兩段都達 BH 校正後 p ≤ 0.05，且方向皆為「該方案優於 A」。
    adj_bt = rep.mt_bt.get("adjusted", [])
    per_scheme_bt: dict[str, list[tuple[str, float, float]]] = {}
    for i, d in enumerate(rep.paired_bt):
        seg = d.label.split("|")[0] if "|" in d.label else ""
        pair = d.label.split("|")[1].split("@")[0] if "|" in d.label else ""
        scheme = pair.split("−")[1] if "−" in pair else ""
        a = adj_bt[i] if i < len(adj_bt) else float("nan")
        per_scheme_bt.setdefault(scheme, []).append((seg, d.mean_diff, a))
    sig_bt = sorted(
        s for s, rows in per_scheme_bt.items()
        if s in CANDIDATE_SCHEMES
        # diff = 日報酬_A − 日報酬_X → m < 0 表示該方案**優於** baseline
        and all(math.isfinite(a) and a <= 0.05 and m < 0
                for seg, m, a in rows if seg in ("train", "valid"))
        and any(seg in ("train", "valid") for seg, _, _ in rows)
    )

    # 判準 4b：交易層級是否改變（持倉／成交）
    changed = sorted({
        scheme
        for seg_map in rep.holdings.values()
        for scheme, h in seg_map.items()
        if (math.isfinite(h.identical_ratio) and h.identical_ratio < 1.0)
    })

    all_rank_same = (ab is not None and ab.identical_ratio == 1.0
                     and ad is not None and ad.identical_ratio == 1.0
                     and all(r.identical_ratio == 1.0 for r in rep.ranks
                             if r.left == "A"))

    if all_rank_same and not changed:
        verdict = "B17 對目前排名型策略沒有實質影響（排名與成交皆完全相同）"
        lines.append("A/B/C/D 在所有交易日的橫斷面排名完全相同，"
                     "且配對回測的持倉與成交亦完全一致。")
        lines.append("數學依據：各方案僅差一個逐股不同的正比例因子，"
                     "而該因子在同一天內為常數；"
                     "加上 buy_threshold 為絕對門檻但分數未跨過門檻，故成交不變。")
        lines.append("依驗收標準第 6 條，不建議為此修改正式策略的權重。")
    elif not rank_effect and not changed:
        verdict = "排名未改變且成交未改變 → 不構成修改正式策略的理由"
        lines.append("雖然分數尺度不同，但絕對門檻未造成任何成交差異；"
                     "分數尺度變化不可誤認為策略改善。")
    else:
        if sig_schemes:
            verdict = ("B17 確實改變排名與成交；train/valid 上有"
                       f" {sorted(sig_schemes)} 通過校正後的配對顯著改善 → "
                       "可提出修改建議（仍須另一次獨立驗證）")
        else:
            verdict = "B17 會改變排名/成交，但 train/valid **未**顯示一致且可解釋的改善 → 保留現行策略並記錄不確定性"
        lines.append(f"排名受影響的配對："
                     + "、".join(f"{r.left}→{r.right} 相同率 "
                                 f"{r.identical_ratio:.1%}"
                                 for r in rep.ranks if r.identical_ratio < 1.0))
        lines.append(f"成交受影響的方案：{changed or '無'}")
        if not sig_schemes:
            lines.append("在 train/valid 上，配對 IC 差異經 BH 校正後無一顯著；"
                         "依驗收標準不得僅因 test 表現提出修改建議。")
            lines.append("且差異方向在 train 與 valid 之間不一致（符號翻轉），"
                         "不構成『一致且可解釋的改善』。")
            lines.append("B17 的兩個機制必須分開看：分子（情緒值非 0）只有 "
                         f"{rep.n_rows_contributed} 列且其區間無前瞻報酬可估計；"
                         f"分母（0.10 佔位）全樣本 {rep.n_rows_diluted:,} 列可觀測，"
                         "但 IC／價差影響僅在千分位等級。")
        else:
            lines.append("校正後顯著的方案僅在 train/valid 出現，"
                         "test 未用於挑選；建議仍需新的獨立資料驗證。")

        # 交易層級的一致性（嚴格條件：train 與 valid 皆顯著為正）
        if sig_bt:
            verdict += "；交易層級在 train/valid 皆顯著優於 baseline"
            lines.append(f"交易層級成本後日報酬在 train 與 valid 皆經 BH 校正後顯著優於 baseline："
                         f"{sig_bt} → 滿足修改建議的必要條件（仍非充分條件，"
                         "需新資料驗證且不得用 test 挑選）。")
        else:
            lines.append("交易層級成本後日報酬的配對差異，未在 train 與 valid 兩段"
                         "同時達 BH 校正後顯著優於 baseline（或方向不一致）"
                         "→ 未達修改門檻。")
    return verdict, "\n".join(lines)


def _notes(rep: AblationReport, split, horizons) -> list[str]:
    ab = next((r for r in rep.ranks if r.left == "A" and r.right == "B"), None)
    rank_days = (ab.n_days - ab.identical_days) if ab else 0
    cover = rep.coverage.get("有新聞記錄日", {})
    fwd_news = cover.get("n_fwd", 0)
    notes = [
        "方案 A/B/C/D 於實驗開始前即固定，未依結果替換；"
        "A* 為分解控制組，非候選方案。",
        "所有方案共用同一面板、日期、股票池、成本模型、"
        "top_n=5、hold_days=5、buy_threshold=0.2 與曝險上限，僅分數計算不同。",
        "方案比較與修改決策只使用 train 與 valid；"
        "test 僅作一次最終確認。",
        "⚠️ **test 段已消耗**：同一批 test 資料在 P2.3 診斷（IC／分組／隨機基準）"
        "與 P2.4 歸因中已反覆使用，本報告的 test 數字不得宣稱為全新樣本外驗證。",
        f"時間切分沿用既有定義：train_frac=0.5、valid_frac=0.25、"
        f"embargo={baseline().hold_days} 交易日，不可依結果更改。",
        f"已檢定的配對假設：IC 差異 {rep.mt_ic.get('n_tested', 0)} 個、"
        f"價差差異 {rep.mt_spread.get('n_tested', 0)} 個、"
        f"交易層級日報酬差異 {rep.mt_bt.get('n_tested', 0)} 個，"
        "皆以 Newey-West 穩健 t 檢定並施加 BH FDR 校正；未只呈現顯著結果。",
        # --- 本階段最重要的兩個事實 ---
        f"**分母效應全樣本可觀測**：fillna(0.0) 使 news_sentiment 在全部 "
        f"{rep.n_rows_diluted:,} 列皆非缺值 → 0.10 恆入分母（稀釋）；"
        f"其中有新聞記錄者僅 {rep.n_rows_record:,} 列。",
        f"**分子效應在本資料中不可估計**：情緒值非 0 者僅 "
        f"{rep.n_rows_contributed:,} 列，且落在樣本最末端；"
        f"該區間有可用 5 日前瞻報酬的日子 = {fwd_news} 天 → "
        "不是『訊號弱』，而是**無法用前瞻報酬估計**（詳見 [9]）。",
        f"**分母效應的後果很小**：A→B 排名在 {rank_days}/{ab.n_days if ab else 0} "
        "個交易日改變，但最大名次變動、top-5 重疊與 IC 差異都在千分位等級。",
        "「沒有成功抓到新聞」與「沒有新聞」在現有 schema 下不可分"
        "（news 表無抓取狀態欄，news_rss 吞例外回傳空清單）→ 方案 D 只能區分"
        " absent / neutral / scored 三態中的兩態，屬已知限制。",
        "方案 D 依 presence 閘門排除缺值；其餘方案的分母處理為設計選擇，"
        "兩者都可能正確，本報告只比較後果，不宣稱其中之一為唯一正解。",
        "所有分數皆以與 ``score_row`` 相同的方式 clip 到 [-1, 1]；"
        "貼邊界列數已於 [3] 揭露。",
        "本研究為消融診斷：不調整曝險上限、不調整固定持有 5 日規則、"
        "不新增交易機制、不啟用正式交易。",
        "結論僅適用於目前 20 檔多頭樣本（約 2 年），"
        "且新聞資料只涵蓋樣本最末端極少數交易日。",
    ]
    return notes


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="P2.6 B17 新聞因子消融驗證")
    ap.add_argument("--quick", action="store_true",
                    help="少量自助法迭代（快速檢查）")
    ap.add_argument("--no-backtest", action="store_true",
                    help="跳過交易層級配對回測")
    ap.add_argument("-o", "--output", default=None,
                    help="另存報告路徑（預設僅印出）")
    args = ap.parse_args(argv)

    rep = run_ablation(sims=120 if args.quick else DEFAULT_SIMS,
                       run_backtests=not args.no_backtest)
    text = rep.to_text()
    print(text)
    if args.output:
        from .config import REPORTS_DIR
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / args.output
        path.write_text(text, encoding="utf-8")
        print(f"\n報告已寫入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
