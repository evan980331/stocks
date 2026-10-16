"""訊號資訊量診斷（P2.3）：訊號清單、前瞻報酬、IC / Rank IC、分組測試。

本模組回答的問題只有一個：**現有訊號到底有沒有可重現的預測資訊？**
它不調參數、不挑最佳組合，也不產生「最佳策略」。

時間對齊（最重要，錯了整份報告就沒意義）
  訊號於 **T 日收盤**計算完成 → 第一個可成交價格是 **T+1 開盤**。
  因此前瞻報酬一律定義為::

      fwd_h(T) = open(T+1+h) / open(T+1) - 1

  絕不使用 T 日（含）之前的任何價格當作報酬起點；T→T+1 的隔夜跳空
  **不計入**，因為在 T 日收盤時無法成交。

統計處理（避免高估顯著性）
  1. **同日橫斷面相關**：同一天所有股票共享市場因子，不是獨立觀察值。
     因此以「每日 IC」為觀察單位（每日一個值），而非把所有股票日資料
     混在一起算單一相關係數。
  2. **重疊窗口自相關**：持有 h 日的前瞻報酬會使相鄰日的 IC 序列自我相關，
     故採用 **Newey-West（Bartlett 核）** 穩健標準誤，並另以
     **移動區塊自助法** 對 IC 平均值建立信賴區間。
  3. **多重檢定**：同時測試「多個訊號 × 多個持有期間」會製造假陽性，
     故以 **Benjamini-Hochberg FDR** 校正，並同時列出未校正結果供對照。
  4. **隨機排序基準**：以「同日內隨機置換訊號」建立虛無分布，保留股票池、
     時間結構與橫斷面大小，避免不公平比較。

指令：
  python -m twstock.diagnostics            # 讀本機 DB，輸出診斷報告
  python -m twstock.diagnostics --quick    # 只跑主要持有期間、較少模擬次數
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
from .config import CONFIG, REPORTS_DIR
from .costs import get_cost_model
from .validation import TimeSplit, make_split, subset_by_dates

log = logging.getLogger(__name__)

HORIZONS: tuple[int, ...] = (1, 5, 10)

# 持有期間的選擇理由（**事先決定，未依結果調整**；任務要求事前說明）
HORIZON_RATIONALE = (
    "持有期間選擇理由（事先決定、未依結果調整）：1 日 = 最短可交易間隔，"
    "檢驗訊號是否只有隔日效果；5 日 = 現行策略的實際持有期（CONFIG.hold_days），"
    "與交易決策直接相關；10 日 = 現行持有期的兩倍，檢驗資訊是否在更長期間仍存在"
    "或已反轉。刻意不測更多期間，以避免擴大多重檢定的假陽性風險。")
N_GROUPS = 5                 # 五分組（任務允許 5 或 10；樣本僅 20 檔 → 用 5）
MIN_DAYS_FOR_IC = 20         # 少於此交易日不宣稱 IC 有結論
MIN_STOCKS_PER_DAY = 5       # 單日有效檔數下限
DEFAULT_SIMS = 500           # 隨機基準／自助法迭代次數
DEFAULT_BLOCK = 5            # 移動區塊自助法的區塊長度（交易日）

# --------------------------------------------------------------------------
# 訊號清單（任務 3.1：逐一記錄公式、觀察期間、產生時間、方向、缺值處理）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SignalSpec:
    key: str            # 對應 signals.WEIGHTS 的因子鍵
    cn_name: str
    weight: float
    formula: str
    lookback: str
    available_at: str
    missing: str

    @property
    def direction(self) -> str:
        return "正 = 偏多（分數越高越看多）；IC 為正代表訊號方向正確"


SIGNAL_REGISTRY: tuple[SignalSpec, ...] = (
    SignalSpec(
        "kline", "K線綜合（KD+MACD+型態）", sig_mod.WEIGHTS["kline"],
        "kline_score = 0.40*clip((K-D)/40) + 0.35*(0.6*sign(DIF-DEA) "
        "+ 0.4*sign(OSC-OSC_prev)) + 0.25*型態分數",
        "KD(9,3,3)、MACD(12,26,9)、20 日內型態", "T 日收盤",
        "K 線欄位全缺時該因子不計入，權重重新標準化（不以 0 冒充中性）"),
    SignalSpec(
        "breakout", "20 日突破 / 破線", sig_mod.WEIGHTS["breakout"],
        "close > max(high, 20)[shift 1] → +1；close < min(low, 20)[shift 1] → -1",
        "20 交易日", "T 日收盤", "未突破則該因子不計入"),
    SignalSpec(
        "mom_20", "20 日動能", sig_mod.WEIGHTS["mom_20"],
        "clip(close.pct_change(20) * 5, -1, 1)", "20 交易日", "T 日收盤",
        "前 20 日不足時為 NaN → 不計入"),
    SignalSpec(
        "mom_60", "60 日動能", sig_mod.WEIGHTS["mom_60"],
        "clip(close.pct_change(60) * 2.5, -1, 1)", "60 交易日", "T 日收盤",
        "前 60 日不足時為 NaN → 以 0.0 計入（原始程式碼行為，已標示）"),
    SignalSpec(
        "ma_alignment", "均線多頭排列", sig_mod.WEIGHTS["ma_alignment"],
        "ma5 > ma20 > ma60 → +1，否則 -0.3", "60 交易日", "T 日收盤",
        "均線不足時比較結果為 False → -0.3"),
    SignalSpec(
        "rsi_rev", "RSI 逆勢", sig_mod.WEIGHTS["rsi_rev"],
        "RSI(14) < 30 → +0.8；> 70 → -0.6；其餘 0.0", "14 交易日", "T 日收盤",
        "RSI 缺值時該因子不計入"),
    SignalSpec(
        "vol_ratio", "量能方向", sig_mod.WEIGHTS["vol_ratio"],
        "vol_ratio > 1.5 時：當日漲 → +0.6，跌 → -0.6；否則不計入",
        "20 交易日（量均）", "T 日收盤", "量能不足時該因子不計入"),
    SignalSpec(
        "inst_net_sum3", "三大法人近 3 日淨買超", sig_mod.WEIGHTS["inst_net_sum3"],
        "clip(inst_net_sum3 / max(1e6, |inst_net_sum3|) * 1e6 / 5e7, -1, 1)",
        "3 交易日", "T 日收盤後（法人資料當日公布）",
        "無法人資料時該因子不計入"),
    SignalSpec(
        "streak_inst_buy", "法人連續買超天數", sig_mod.WEIGHTS["streak_inst_buy"],
        "連續買超 >= 3 日 → min(1, 天數/8)，否則不計入", "最長至反轉為止",
        "T 日收盤後", "無法人資料時該因子不計入"),
    SignalSpec(
        "margin_ratio", "券資比", sig_mod.WEIGHTS["margin_ratio"],
        "clip((0.35 - 券資比) * 4, -1, 1)（券資比低 → 偏多）", "當日餘額",
        "T 日收盤後", "無融資券資料時該因子不計入"),
    SignalSpec(
        "rev_yoy", "月營收 YoY", sig_mod.WEIGHTS["rev_yoy"],
        "clip(YoY(%) / 30, -1, 1)", "近 12 個月（YoY 需同期）",
        "公告日（以公告日近似，非精確 point-in-time）",
        "無營收資料時該因子不計入；**公告時間為近似值，為已知限制**"),
    SignalSpec(
        "news_sentiment", "新聞情緒（衰減）", sig_mod.WEIGHTS["news_sentiment"],
        "clip(加權情緒 * 1.5, -1, 1)；權重 當日 1.0 / +1 日 0.66 / +2 日 0.40",
        "近 7 日新聞", "新聞發布日", "無新聞時以 0.0 計入（原始程式碼行為）"),
)

SIGNAL_KEYS: tuple[str, ...] = tuple(s.key for s in SIGNAL_REGISTRY)
SIGNAL_BY_KEY: dict[str, SignalSpec] = {s.key: s for s in SIGNAL_REGISTRY}


# --------------------------------------------------------------------------
# 前瞻報酬（時間對齊）
# --------------------------------------------------------------------------
def forward_returns(price_panel: dict[str, pd.DataFrame],
                    horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """由訊號日 T 計算前瞻報酬：T+1 開盤進場、持有 h 日後以開盤出場。

    回傳長表 [date, stock_id, fwd_1, fwd_5, fwd_10]；``date`` 為**訊號日**。
    資料不足以完成持有期者為缺值（不是 0）。
    """
    rows: list[pd.DataFrame] = []
    for sid, df in (price_panel or {}).items():
        if df is None or df.empty or "open" not in df.columns:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        d = (d.sort_values("date").drop_duplicates("date", keep="last")
             .reset_index(drop=True))
        op = pd.to_numeric(d["open"], errors="coerce").to_numpy(dtype=float)
        op = np.where(op > 0, op, np.nan)      # 非正數價格視為缺值
        n = len(d)
        rec: dict[str, object] = {"date": d["date"], "stock_id": sid}
        for h in horizons:
            entry = np.full(n, np.nan)
            exitp = np.full(n, np.nan)
            if n > h + 1:
                m = n - h - 1
                entry[:m] = op[1:n - h]         # T+1 開盤
                exitp[:m] = op[1 + h:n]         # T+1+h 開盤
            rec[f"fwd_{h}"] = exitp / entry - 1.0
        rows.append(pd.DataFrame(rec))
    if not rows:
        return pd.DataFrame(columns=["date", "stock_id"]
                            + [f"fwd_{h}" for h in horizons])
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# 訊號框架
# --------------------------------------------------------------------------
def build_signal_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """由特徵面板產生逐因子方向化訊號 + 合成規則分數（長表）。

    使用 ``signals.score_components``（與 ``score_row`` 完全相同來源），
    確保診斷對象是策略**實際使用**的訊號。缺值因子保持 NaN，不以 0 冒充中性。
    """
    if panel is None or panel.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []

    def _one(r: pd.Series) -> pd.Series:
        parts = sig_mod.score_components(r)
        parts["_score"] = sig_mod.score_row(r)[0]
        return pd.Series(parts)

    for sid, g in panel.groupby("stock_id", sort=False):
        g = g.reset_index(drop=True)
        comp = g.apply(_one, axis=1).reset_index(drop=True)
        if comp.empty:
            continue
        out = pd.DataFrame({
            "date": pd.to_datetime(g["date"]).dt.normalize().to_numpy(),
            "stock_id": str(sid),
        })
        for key in SIGNAL_KEYS:
            out[key] = (comp[key].to_numpy(dtype=float)
                        if key in comp.columns else np.nan)
        out["score"] = (comp["_score"].to_numpy(dtype=float)
                        if "_score" in comp.columns else np.nan)
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def attach_forward_returns(signal_frame: pd.DataFrame,
                           price_panel: dict[str, pd.DataFrame],
                           horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """把前瞻報酬併入訊號框架（以 date + stock_id 為鍵）。"""
    if signal_frame is None or signal_frame.empty:
        return pd.DataFrame()
    fwd = forward_returns(price_panel, horizons)
    if fwd.empty:
        return signal_frame.copy()
    keys = ["date", "stock_id"]
    a = signal_frame.copy()
    b = fwd.copy()
    a["date"] = pd.to_datetime(a["date"]).dt.normalize()
    b["date"] = pd.to_datetime(b["date"]).dt.normalize()
    return a.merge(b, on=keys, how="left")


# --------------------------------------------------------------------------
# 統計工具
# --------------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩（不依賴 scipy）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _two_sided_p(t: float | None) -> float:
    if t is None or not math.isfinite(t):
        return float("nan")
    return math.erfc(abs(t) / math.sqrt(2.0))


def newey_west_tstat(x, lags: int | None = None) -> tuple[float, float]:
    """Newey-West（Bartlett 核）穩健 t 統計量。回傳 (t, 標準誤)。

    用於重疊窗口造成自相關的序列（如 h 日前瞻報酬的每日 IC）。
    """
    a = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(),
                   dtype=float)
    n = len(a)
    if n < 3:
        return float("nan"), float("nan")
    e = a - a.mean()
    g0 = float(e @ e) / n
    s = g0
    L = int(max(0, min(int(lags or 0), n - 1)))
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * w * gk
    if s <= 1e-18:
        return float("nan"), float("nan")
    se = math.sqrt(s / n)
    if not math.isfinite(se) or se == 0:
        return float("nan"), float("nan")
    return float(a.mean() / se), se


def block_bootstrap_ci(x, *, n_iter: int = 2000, block: int = DEFAULT_BLOCK,
                       alpha: float = 0.05, seed: int = 0
                       ) -> tuple[float, float]:
    """移動區塊自助法（circular block bootstrap）對平均值的 (1-alpha) 信賴區間。

    同時處理自相關與非常態；不假設 IC 序列獨立同分布。
    """
    a = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(),
                   dtype=float)
    n = len(a)
    if n < 4:
        return float("nan"), float("nan")
    b = int(max(1, min(block, n)))
    n_blocks = int(math.ceil(n / b))
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(b)[None, :]).ravel() % n
        means[i] = a[idx[:n]].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR 校正（回傳校正後 p 值與否決結果）。"""
    p = np.asarray([np.nan if v is None else float(v) for v in pvalues],
                   dtype=float)
    adj = np.full(len(p), np.nan)
    rej = np.zeros(len(p), dtype=bool)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if n == 0:
        return {"adjusted": adj.tolist(), "rejected": rej.tolist(),
                "n_tested": 0, "n_rejected": 0, "alpha": alpha}
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx], kind="mergesort")]
    ranked = p[order] * n / np.arange(1, n + 1)
    # 由大到小取累積最小，確保單調
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0.0, 1.0)
    adj[order] = ranked
    rej[order] = ranked <= alpha
    return {"adjusted": adj.tolist(), "rejected": rej.tolist(),
            "n_tested": n, "n_rejected": int(rej.sum()), "alpha": alpha}


# --------------------------------------------------------------------------
# IC / Rank IC
# --------------------------------------------------------------------------
def _prep_pairs(signal_frame: pd.DataFrame, signal: str, horizon: int
                ) -> pd.DataFrame:
    ret_col = f"fwd_{horizon}"
    if signal not in signal_frame.columns or ret_col not in signal_frame.columns:
        return pd.DataFrame(columns=["date", signal, ret_col])
    df = signal_frame[["date", "stock_id", signal, ret_col]].copy()
    df[signal] = pd.to_numeric(df[signal], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.dropna(subset=[signal, ret_col])


def ic_series(signal_frame: pd.DataFrame, signal: str, horizon: int, *,
              method: str = "spearman",
              min_stocks: int = MIN_STOCKS_PER_DAY) -> pd.Series:
    """每日橫斷面 IC 序列（觀察單位 = 交易日，不是股票日）。"""
    df = _prep_pairs(signal_frame, signal, horizon)
    if df.empty:
        # 明確回傳「空的 DatetimeIndex 序列」，讓呼叫端可直接切片而不會型別錯誤
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    ret_col = f"fwd_{horizon}"
    out: dict[pd.Timestamp, float] = {}
    for d, g in df.groupby("date", sort=True):
        if len(g) < min_stocks:
            continue
        x = g[signal].to_numpy(dtype=float)
        y = g[ret_col].to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            continue                       # 該日訊號或報酬無變化 → 無法算相關
        out[pd.Timestamp(d)] = _corr(x, y, method)
    if not out:
        # 例如整天訊號都無變化（常數）→ 無法算 IC；仍必須回傳 DatetimeIndex，
        # 否則呼叫端對 index 做時間比較時會退化成 RangeIndex 而型別錯誤。
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return pd.Series(out, dtype=float).sort_index()


def _corr(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> float:
    """安全相關：任一側變異為 0 時回 NaN（不使用會發出警告的 corrcoef）。"""
    xa = _rankdata(x) if method != "pearson" else np.asarray(x, dtype=float)
    ya = _rankdata(y) if method != "pearson" else np.asarray(y, dtype=float)
    if np.std(xa) == 0 or np.std(ya) == 0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


@dataclass
class ICResult:
    signal: str
    horizon: int
    segment: str = "all"
    n_days: int = 0
    ic_mean: float = float("nan")
    ic_std: float = float("nan")
    icir: float = float("nan")
    pos_ratio: float = float("nan")
    t_naive: float = float("nan")
    t_nw: float = float("nan")
    p_value: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    autocorr1: float = float("nan")
    nw_lags: int = 0
    sufficient: bool = False

    def rejected_at(self, alpha: float = 0.05) -> bool:
        return bool(math.isfinite(self.p_value) and self.p_value <= alpha)


def ic_summary(ic: pd.Series, signal: str, horizon: int, *, segment: str = "all",
               nw_lags: int | None = None, sims: int = 2000) -> ICResult:
    """由每日 IC 序列彙總統計（含 NW 穩健 t 與區塊自助法信賴區間）。"""
    x = pd.to_numeric(pd.Series(ic), errors="coerce").dropna()
    n = len(x)
    if n < 2:
        return ICResult(signal=signal, horizon=horizon, segment=segment, n_days=n)
    m = float(x.mean())
    s = float(x.std(ddof=1))
    icir = (m / s) if s > 0 else float("nan")
    pos = float((x > 0).mean())
    t_naive = (m / (s / math.sqrt(n))) if s > 0 else float("nan")
    L = int(nw_lags if nw_lags is not None else max(1, min(horizon, n // 4)))
    t_nw, _se = newey_west_tstat(x.to_numpy(dtype=float), L)
    lo, hi = block_bootstrap_ci(x.to_numpy(dtype=float), n_iter=sims,
                                block=max(1, L), seed=0)
    # 常數序列無法算自相關（corrcoef 會除以 0）→ 明確回 NaN，不發出警告
    ac1 = float(x.autocorr(1)) if (n > 3 and s > 0) else float("nan")
    return ICResult(
        signal=signal, horizon=horizon, segment=segment, n_days=n,
        ic_mean=m, ic_std=s, icir=icir, pos_ratio=pos, t_naive=t_naive,
        t_nw=t_nw, p_value=_two_sided_p(t_nw), ci_low=lo, ci_high=hi,
        autocorr1=ac1, nw_lags=L, sufficient=n >= MIN_DAYS_FOR_IC)


def per_stock_ic(signal_frame: pd.DataFrame, signal: str, horizon: int, *,
                 min_obs: int = 60,
                 method: str = "spearman") -> pd.Series:
    """個股層級的時間序列 IC（檢查是否只由少數股票驅動）。"""
    df = _prep_pairs(signal_frame, signal, horizon)
    if df.empty:
        return pd.Series(dtype=float)
    ret_col = f"fwd_{horizon}"
    out: dict[str, float] = {}
    for sid, g in df.groupby("stock_id", sort=True):
        if len(g) < min_obs:
            continue
        r = _corr(g[signal].to_numpy(dtype=float),
                  g[ret_col].to_numpy(dtype=float), method)
        if math.isfinite(r):
            out[str(sid)] = r
    return pd.Series(out, dtype=float).sort_index()


# --------------------------------------------------------------------------
# 分組測試
# --------------------------------------------------------------------------
@dataclass
class SpreadRobustness:
    """Q5-Q1 價差的穩健性：是否只由少數交易日主導。"""

    n_days: int = 0
    mean: float = float("nan")
    trimmed_mean: float = float("nan")
    top_k_share: float = float("nan")
    positive_ratio: float = float("nan")

    def label(self) -> str:
        if self.n_days == 0:
            return "無交易日可檢驗"
        share = (f"{self.top_k_share:.1%}" if math.isfinite(self.top_k_share)
                 else "N/A")
        trim = (f"{self.trimmed_mean:+.3%}"
                if math.isfinite(self.trimmed_mean) else "N/A")
        return (f"{self.n_days} 日；去極端值後平均 {trim}；"
                f"最佳 5 日佔價差總和 {share}；為正比例 {self.positive_ratio:.1%}")


def spread_robustness(spread: pd.Series, *, top_k: int = 5,
                      trim: float = 0.05) -> SpreadRobustness:
    """檢驗 Q5-Q1 價差是否只由少數交易日主導。

    - ``trimmed_mean``：去除最高與最低各 ``trim`` 比例的交易日後的平均價差；
      若價差集中在少數幾天，去極端值後會大幅縮水。
    - ``top_k_share``：最好的 ``top_k`` 個交易日佔價差總和的比例。
      總和 <= 0 時回 NaN —— 總和為負時「佔比」沒有意義，不可硬算。
    - ``positive_ratio``：價差為正的交易日比例。
    """
    x = np.asarray(pd.to_numeric(pd.Series(spread), errors="coerce").dropna(),
                   dtype=float)
    out = SpreadRobustness(n_days=int(len(x)))
    if len(x) == 0:
        return out
    out.mean = float(x.mean())
    out.positive_ratio = float((x > 0).mean())
    k = int(max(1, round(len(x) * trim)))
    if len(x) > 2 * k:
        s = np.sort(x)
        out.trimmed_mean = float(s[k:len(s) - k].mean())
    else:
        out.trimmed_mean = out.mean
    total = float(x.sum())
    if total > 0 and len(x) >= top_k:
        out.top_k_share = float(np.sort(x)[-top_k:].sum() / total)
    return out


@dataclass
class QuantileResult:
    signal: str
    horizon: int
    n_groups: int = N_GROUPS
    n_days: int = 0
    group_means: list[float] = field(default_factory=list)
    group_counts: list[int] = field(default_factory=list)
    spread: float = float("nan")           # 最高組 - 最低組（單次持有期）
    spread_t: float = float("nan")
    spread_p: float = float("nan")
    cost_rate: float = float("nan")
    spread_net: float = float("nan")
    monotonicity: float = float("nan")     # 分組序與平均報酬的 Spearman
    robustness: SpreadRobustness = field(default_factory=SpreadRobustness)

    @property
    def ordered(self) -> bool:
        """分組平均報酬是否單調遞增（允許相等）。"""
        m = [v for v in self.group_means if math.isfinite(v)]
        return len(m) == self.n_groups and all(
            b >= a - 1e-12 for a, b in zip(m, m[1:]))


def _assign_groups(values: pd.Series, n_groups: int) -> np.ndarray:
    """依訊號強度分成等大小的 n_groups 組（僅用當期可得的訊號，不看未來報酬）。"""
    r = np.asarray(values.rank(method="first").to_numpy(dtype=float))
    k = len(r)
    return np.minimum(((r - 1) * n_groups / k).astype(int), n_groups - 1)


def quantile_returns(signal_frame: pd.DataFrame, signal: str, horizon: int, *,
                     n_groups: int = N_GROUPS,
                     cost_model=None,
                     min_stocks_per_group: int = 2) -> QuantileResult:
    """依訊號強度分組，觀察各組未來報酬（分組只用當期訊號，不含未來資訊）。"""
    df = _prep_pairs(signal_frame, signal, horizon)
    res = QuantileResult(signal=signal, horizon=horizon, n_groups=n_groups)
    if df.empty:
        return res
    need = max(n_groups * min_stocks_per_group, MIN_STOCKS_PER_DAY)
    ret_col = f"fwd_{horizon}"
    means_rows: list[np.ndarray] = []
    spread_series: dict[pd.Timestamp, float] = {}
    counts = np.zeros(n_groups, dtype=int)
    for d, g in df.groupby("date", sort=True):
        if len(g) < need:
            continue
        grp = _assign_groups(g[signal], n_groups)
        vals = g[ret_col].to_numpy(dtype=float)
        means = np.empty(n_groups, dtype=float)
        for j in range(n_groups):
            sel = vals[grp == j]
            means[j] = sel.mean() if len(sel) else np.nan
            counts[j] += len(sel)
        if not np.isfinite(means).all():
            continue
        means_rows.append(means)
        spread_series[pd.Timestamp(d)] = float(means[-1] - means[0])
    res.n_days = len(means_rows)
    if not means_rows:
        return res
    avg = np.vstack(means_rows).mean(axis=0)
    res.group_means = [float(v) for v in avg]
    res.group_counts = [int(c) for c in counts]
    res.spread = float(avg[-1] - avg[0])
    sp = pd.Series(spread_series, dtype=float)
    if len(sp) >= 3:
        t, _ = newey_west_tstat(sp.to_numpy(dtype=float),
                                max(1, min(horizon, len(sp) // 4)))
        res.spread_t = t
        res.spread_p = _two_sided_p(t)
    if len(avg) >= 2:
        m = _corr(np.arange(len(avg), dtype=float), avg, "spearman")
        if math.isfinite(m):
            res.monotonicity = m
    cm = cost_model or get_cost_model("typical")
    res.cost_rate = cm.roundtrip_cost_rate(100.0, 10_000)
    res.spread_net = res.spread - res.cost_rate
    res.robustness = spread_robustness(sp)
    return res


def quantile_stability(signal_frame: pd.DataFrame, signal: str, horizon: int,
                       segments, *, n_groups: int = N_GROUPS,
                       cost_model=None) -> dict[str, QuantileResult]:
    """在各時間區段分別做分組測試，檢查**排序是否在不同時期維持**。

    只看全期的單調性不夠：若排序只在某一段成立，就不是穩定的關係。
    """
    out: dict[str, QuantileResult] = {}
    for seg in segments:
        sub = subset_by_dates(signal_frame, seg.start, seg.end)
        out[seg.name] = quantile_returns(sub, signal, horizon,
                                         n_groups=n_groups, cost_model=cost_model)
    return out


# --------------------------------------------------------------------------
# 隨機基準（同日內置換）
# --------------------------------------------------------------------------
@dataclass
class RandomBaseline:
    signal: str
    horizon: int
    n_iter: int
    n_days: int
    observed: float = float("nan")
    null_mean: float = float("nan")
    null_std: float = float("nan")
    p_value: float = float("nan")
    null_q05: float = float("nan")
    null_q95: float = float("nan")


def random_ic_null(signal_frame: pd.DataFrame, signal: str, horizon: int, *,
                   n_iter: int = DEFAULT_SIMS, seed: int = 0,
                   min_stocks: int = MIN_STOCKS_PER_DAY) -> RandomBaseline:
    """同日內隨機置換訊號，建立 Rank IC 的虛無分布。

    保留股票池、交易日曆與每日橫斷面大小，只破壞「訊號與股票的對應」。
    經驗雙尾 p 值 = P(|虛無 IC| >= |觀察 IC|)。
    """
    df = _prep_pairs(signal_frame, signal, horizon)
    out = RandomBaseline(signal=signal, horizon=horizon, n_iter=n_iter, n_days=0)
    if df.empty:
        return out
    ret_col = f"fwd_{horizon}"
    A_rows: list[np.ndarray] = []
    B_rows: list[np.ndarray] = []
    for _d, g in df.groupby("date", sort=True):
        if len(g) < min_stocks:
            continue
        a = _rankdata(g[signal].to_numpy(dtype=float))
        b = _rankdata(g[ret_col].to_numpy(dtype=float))
        a = a - a.mean()
        b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        A_rows.append(a / na)
        B_rows.append(b / nb)
    if not A_rows:
        return out
    out.n_days = len(A_rows)
    rng = np.random.default_rng(seed)
    null = np.zeros(n_iter, dtype=float)
    for a, b in zip(A_rows, B_rows):
        m = len(a)
        key = rng.random((n_iter, m))
        order = np.argsort(key, axis=1, kind="stable")
        null += a[order] @ b
    null /= len(A_rows)
    out.null_mean = float(null.mean())
    out.null_std = float(null.std(ddof=1)) if n_iter > 1 else float("nan")
    out.null_q05 = float(np.quantile(null, 0.05))
    out.null_q95 = float(np.quantile(null, 0.95))
    out.observed = float(np.mean([float(a @ b) for a, b in zip(A_rows, B_rows)]))
    out.p_value = float((int((np.abs(null) >= abs(out.observed)).sum()) + 1)
                        / (n_iter + 1))
    return out


# --------------------------------------------------------------------------
# 報告
# --------------------------------------------------------------------------
@dataclass
class DiagnosticsReport:
    generated_at: str
    horizons: tuple[int, ...]
    n_stocks: int
    n_days: int
    period: str
    ic_all: list[ICResult]
    ic_segments: list[ICResult]
    quantiles: list[QuantileResult]
    segment_quantiles: dict[str, dict[str, QuantileResult]] = field(
        default_factory=dict)
    randoms: list[RandomBaseline] = field(default_factory=list)
    per_stock: dict[str, pd.Series] = field(default_factory=dict)
    multiple_testing: dict = field(default_factory=dict)
    data_notes: list[str] = field(default_factory=list)
    decisions: list[tuple[str, str, str]] = field(default_factory=list)

    def to_text(self) -> str:
        L: list[str] = []
        L.append("=" * 84)
        L.append("訊號資訊量診斷報告（P2.3）")
        L.append(f"產生時間：{self.generated_at}")
        L.append("=" * 84)
        L.append("")
        L.append("[1] 研究對象與定義")
        L.append(f"  股票 {self.n_stocks} 檔、{self.n_days} 個交易日（{self.period}）")
        L.append("  前瞻報酬定義：fwd_h(T) = open(T+1+h)/open(T+1) - 1")
        L.append("    訊號於 T 日收盤可得 → T+1 開盤為第一個可成交價；"
                 "T→T+1 隔夜跳空不計入。")
        L.append(f"  持有期間：{', '.join(str(h) for h in self.horizons)} 個交易日")
        L.append("  " + HORIZON_RATIONALE
                 + ("" if set(HORIZONS) <= set(self.horizons) else
                    f"（本次僅執行 {', '.join(str(h) for h in self.horizons)}）"))
        L.append("  統計單位：**每日橫斷面 IC**（非股票日）；"
                 "同日股票共享市場因子，不是獨立觀察值。")
        L.append("  顯著性：Newey-West 穩健 t（處理重疊窗口自相關）"
                 "+ 移動區塊自助法信賴區間。")
        L.append("")
        L.append("  訊號清單：")
        L.append(f"    {'因子':<22}{'權重':>7}  公式（摘要）")
        for s in SIGNAL_REGISTRY:
            L.append(f"    {s.cn_name:<22}{s.weight:>7.2f}  "
                     f"{s.formula[:58]}")
        L.append("")
        L.append("[2] IC 與 Rank IC（全期）")
        L.append(f"  {'訊號':<22}{'h':>3}{'天數':>6}{'平均IC':>10}{'IC標準差':>10}"
                 f"{'ICIR':>8}{'IC>0比例':>10}{'t(naive)':>10}{'t(NW)':>9}"
                 f"{'p值':>9}")
        for r in self.ic_all:
            spec = SIGNAL_BY_KEY.get(r.signal)
            name = spec.cn_name if spec else r.signal
            L.append(f"  {name:<22}{r.horizon:>3}{r.n_days:>6}"
                     f"{r.ic_mean:>10.4f}{r.ic_std:>10.4f}{r.icir:>8.3f}"
                     f"{r.pos_ratio:>10.2%}{r.t_naive:>10.2f}{r.t_nw:>9.2f}"
                     f"{r.p_value:>9.4f}")
        L.append("")
        L.append("[3] IC 穩健性（區塊自助法 95% 信賴區間）")
        L.append(f"  {'訊號':<22}{'h':>3}{'下界':>10}{'上界':>10}"
                 f"{'跨越0':>8}{'IC自相關(lag1)':>16}{'NW lags':>9}")
        for r in self.ic_all:
            spec = SIGNAL_BY_KEY.get(r.signal)
            name = spec.cn_name if spec else r.signal
            crosses = "是" if (math.isfinite(r.ci_low) and r.ci_low <= 0 <= r.ci_high) \
                else "否"
            L.append(f"  {name:<22}{r.horizon:>3}{r.ci_low:>10.4f}"
                     f"{r.ci_high:>10.4f}{crosses:>8}{r.autocorr1:>16.3f}"
                     f"{r.nw_lags:>9}")
        L.append("")
        L.append("[4] 分組測試（五分組，每組約 4 檔；僅用當期訊號分組）")
        L.append(f"  {'訊號':<22}{'h':>3}{'Q1(低)':>9}{'Q2':>9}{'Q3':>9}"
                 f"{'Q4':>9}{'Q5(高)':>9}{'Q5-Q1':>9}{'t':>8}{'單調':>6}")
        for q in self.quantiles:
            spec = SIGNAL_BY_KEY.get(q.signal)
            name = spec.cn_name if spec else q.signal
            gm = q.group_means + [float("nan")] * (5 - len(q.group_means))
            L.append(f"  {name:<22}{q.horizon:>3}"
                     + "".join(f"{v:>9.3%}" for v in gm[:5])
                     + f"{q.spread:>9.3%}{q.spread_t:>8.2f}"
                     + f"{('是' if q.ordered else '否'):>6}")
        L.append("")
        L.append("  扣成本後的 Q5-Q1（單次持有期，typical 成本 = "
                 f"{self.quantiles[0].cost_rate:.4%}）"
                 if self.quantiles else "  扣成本：無資料")
        L.append(f"    {'訊號':<22}{'h':>3}{'毛價差':>10}{'成本':>9}{'淨價差':>10}"
                 f"{'淨價差>0':>10}")
        for q in self.quantiles:
            spec = SIGNAL_BY_KEY.get(q.signal)
            name = spec.cn_name if spec else q.signal
            L.append(f"    {name:<22}{q.horizon:>3}{q.spread:>10.3%}"
                     f"{q.cost_rate:>9.3%}{q.spread_net:>10.3%}"
                     f"{('是' if q.spread_net > 0 else '否'):>10}")
        L.append("")
        L.append("  穩健性：價差是否只由少數交易日主導（去極端值 = 去除最高／最低各 5%）")
        L.append(f"    {'訊號':<22}{'h':>3}{'毛價差':>10}{'去極端值':>11}"
                 f"{'最佳5日佔比':>12}{'為正比例':>10}")
        for q in self.quantiles:
            spec = SIGNAL_BY_KEY.get(q.signal)
            name = spec.cn_name if spec else q.signal
            rb = q.robustness
            L.append(f"    {name:<22}{q.horizon:>3}{q.spread:>10.3%}"
                     f"{_fmt_pct(rb.trimmed_mean):>11}"
                     f"{_fmt_pct(rb.top_k_share, signed=False):>12}"
                     f"{_fmt_pct(rb.positive_ratio, signed=False):>10}")
        L.append("    註：最佳 5 日佔價差總和接近 100% 表示價差幾乎全來自少數幾天；"
                 "**超過 100% 表示少數幾天不只貢獻全部、還被其他天抵銷掉一部分**；"
                 "價差總和為負時該欄為 N/A（負總和的佔比沒有意義）。")
        L.append("")
        L.append("  分期間一致性（h=5；各段獨立分組，檢查排序是否在不同時期維持）")
        if self.segment_quantiles:
            L.append(f"    {'訊號':<22}{'區段':>7}{'交易日':>7}{'Q5-Q1':>10}"
                     f"{'單調':>6}{'扣成本後':>10}")
            for key, segs in self.segment_quantiles.items():
                spec = SIGNAL_BY_KEY.get(key)
                name = spec.cn_name if spec else key
                for seg_name, q in segs.items():
                    L.append(f"    {name:<22}{seg_name:>7}{q.n_days:>7}"
                             f"{_fmt_pct(q.spread):>10}"
                             f"{('是' if q.ordered else '否'):>6}"
                             f"{_fmt_pct(q.spread_net):>10}")
        else:
            L.append("    （資料量不足以切分區段）")
        L.append("")
        L.append("[5] 隨機排序基準（同日內隨機置換，保留股票池與時間結構）")
        L.append(f"  {'訊號':<22}{'h':>3}{'觀察IC':>10}{'虛無均值':>10}"
                 f"{'虛無5%':>10}{'虛無95%':>10}{'經驗p值':>10}{'迭代':>7}")
        for r in self.randoms:
            spec = SIGNAL_BY_KEY.get(r.signal)
            name = spec.cn_name if spec else r.signal
            L.append(f"  {name:<22}{r.horizon:>3}{r.observed:>10.4f}"
                     f"{r.null_mean:>10.4f}{r.null_q05:>10.4f}"
                     f"{r.null_q95:>10.4f}{r.p_value:>10.4f}{r.n_iter:>7}")
        L.append("")
        L.append("[6] 樣本外分段穩健性（train / valid / test）")
        L.append(f"  {'訊號':<22}{'h':>3}{'區段':>7}{'天數':>6}{'平均IC':>10}"
                 f"{'t(NW)':>9}{'p值':>9}")
        for r in self.ic_segments:
            spec = SIGNAL_BY_KEY.get(r.signal)
            name = spec.cn_name if spec else r.signal
            L.append(f"  {name:<22}{r.horizon:>3}{r.segment:>7}{r.n_days:>6}"
                     f"{r.ic_mean:>10.4f}{r.t_nw:>9.2f}{r.p_value:>9.4f}")
        L.append("")
        L.append("[7] 個股層級 IC（是否只由少數股票驅動）")
        for key, s in self.per_stock.items():
            if s.empty:
                continue
            spec = SIGNAL_BY_KEY.get(key)
            name = spec.cn_name if spec else key
            pos = int((s > 0).sum())
            L.append(f"  {name:<22} 有|IC|>0.05 的股票數 "
                     f"{int((s.abs() > 0.05).sum())}/{len(s)}；正 IC {pos} 檔；"
                     f"中位數 {s.median():+.4f}；最大 {s.max():+.3f}；"
                     f"最小 {s.min():+.3f}")
        L.append("")
        mt = self.multiple_testing
        L.append("[8] 多重檢定校正（Benjamini-Hochberg FDR）")
        L.append(f"  同時檢定 {mt.get('n_tested', 0)} 個假設"
                 f"（訊號 × 持有期間）；FDR 水準 alpha = {mt.get('alpha', 0.05)}")
        L.append(f"  未校正即顯著：{mt.get('n_raw_sig', 0)} 個；"
                 f"BH 校正後仍顯著：{mt.get('n_rejected', 0)} 個")
        if mt.get("n_raw_sig", 0) > mt.get("n_rejected", 0):
            L.append("  ⚠️ 有假設在未校正時看似顯著、校正後消失 → "
                     "屬多重檢定造成的假陽性，不得單獨採信。")
        if mt.get("by_signal"):
            L.append(f"  {'訊號':<22}{'校正後最小p值':>16}{'通過FDR':>10}")
            for k, v in mt["by_signal"].items():
                spec = SIGNAL_BY_KEY.get(k)
                name = spec.cn_name if spec else k
                L.append(f"  {name:<22}{v['min_adj_p']:>16.4f}"
                         f"{('是' if v['passes'] else '否'):>10}")
        L.append("")
        L.append("[9] 資料完整度與研究限制")
        for n in self.data_notes:
            L.append("  - " + n)
        L.append("  - " + HORIZON_RATIONALE)
        L.append("  - 價差穩健性與分期間一致性均以**事先決定**的準則呈現："
                 "不因某個訊號在某段表現好就保留它。")
        L.append("")
        L.append("[10] 訊號判定（依本報告證據，非最終定論）")
        L.append(f"  {'訊號':<22}{'判定':<12}理由")
        for key, verdict, why in self.decisions:
            spec = SIGNAL_BY_KEY.get(key)
            name = spec.cn_name if spec else key
            L.append(f"  {name:<22}{verdict:<12}{why}")
        L.append("")
        L.append("[11] 結論限制")
        L.append("  - IC 只衡量「同期横斷面排序與未來報酬的關聯」，**不證明因果**，"
                 "也不保證可交易。")
        L.append("  - 期間僅約 2 年；IC 在此期間為正不代表其他市況成立"
                 "（空頭期間的 IC 另見 P2.9 壓力測試：python -m twstock.cli stress）。")
        L.append("  - 股票池為寫死的 20 檔大型權值股，存在 survivorship bias，"
                 "不代表整個台股市場。")
        L.append("  - 本報告**不挑選最佳參數**，也不宣稱任何策略具備獲利能力。")
        L.append("=" * 84)
        return "\n".join(L)


def _fmt_pct(x: float, *, signed: bool = True, digits: int = 3) -> str:
    """百分比格式化；無法計算時明確標示 N/A，不輸出 nan%。"""
    if x is None or not math.isfinite(float(x)):
        return "N/A"
    return f"{float(x):+.{digits}%}" if signed else f"{float(x):.{digits}%}"


def decide_signal(spec: SignalSpec, ic: ICResult, q: QuantileResult | None,
                  rand: RandomBaseline | None, adj_p: float
                  ) -> tuple[str, str]:
    """依證據判定訊號：獲支持 / 未獲支持 / 樣本不足。"""
    if not ic.sufficient:
        return "樣本不足", f"僅 {ic.n_days} 個交易日（< {MIN_DAYS_FOR_IC}）"
    supported = (math.isfinite(adj_p) and adj_p <= 0.05
                 and rand is not None and rand.p_value <= 0.05
                 and q is not None and math.isfinite(q.spread_net)
                 and q.spread_net > 0)
    if supported:
        return "初步獲支持", (f"平均 IC {ic.ic_mean:+.4f}、FDR 校正後 p="
                              f"{adj_p:.4f}、經驗 p={rand.p_value:.4f}、"
                              f"扣成本 Q5-Q1 {q.spread_net:+.3%}")
    reasons = []
    if not (math.isfinite(adj_p) and adj_p <= 0.05):
        reasons.append(f"FDR 校正後不顯著（p={adj_p:.3f}）"
                       if math.isfinite(adj_p) else "無有效 p 值")
    if rand is not None and rand.p_value > 0.05:
        reasons.append(f"未優於隨機排序（經驗 p={rand.p_value:.3f}）")
    if q is not None and math.isfinite(q.spread_net) and q.spread_net <= 0:
        reasons.append(f"扣成本後 Q5-Q1 {q.spread_net:+.3%} 為非正")
    return "未獲支持", "；".join(reasons) or f"平均 IC {ic.ic_mean:+.4f}"


def run_diagnostics(*, horizons: tuple[int, ...] = HORIZONS,
                    sims: int = DEFAULT_SIMS,
                    train_frac: float = 0.5, valid_frac: float = 0.25
                    ) -> DiagnosticsReport:
    """完整診斷流程：載入資料 → 前瞻報酬 → IC → 分組 → 隨機基準 → 校正。"""
    from .validation import load_db_inputs

    qa = _new_stats()
    panel, price_panel, _benchmark = load_db_inputs(stats=qa)
    if panel.empty:
        raise RuntimeError("特徵面板為空，請先執行 ingest 與 run")

    sf = attach_forward_returns(build_signal_frame(panel), price_panel, horizons)
    if sf.empty:
        raise RuntimeError("無法建立訊號框架")
    dates = sorted(pd.to_datetime(sf["date"]).dt.normalize().unique())
    n_stocks = int(sf["stock_id"].nunique())
    period = f"{pd.Timestamp(dates[0]):%Y-%m-%d} ~ {pd.Timestamp(dates[-1]):%Y-%m-%d}"

    signals = list(SIGNAL_KEYS) + ["score"]
    cm = get_cost_model("typical")

    # ---- IC（全期） ----
    ic_all: list[ICResult] = []
    for key in signals:
        for h in horizons:
            ic_all.append(ic_summary(ic_series(sf, key, h), key, h, sims=sims))

    # ---- 分組 ----
    quantiles = [quantile_returns(sf, key, h, cost_model=cm)
                 for key in signals for h in horizons]

    # ---- 隨機基準 ----
    randoms = [random_ic_null(sf, key, h, n_iter=sims, seed=7)
               for key in signals for h in horizons]

    # ---- 多重檢定 ----
    bh = benjamini_hochberg([r.p_value for r in ic_all], alpha=0.05)
    n_raw = sum(1 for r in ic_all if r.rejected_at(0.05))
    by_signal: dict[str, dict] = {}
    for r, adj in zip(ic_all, bh["adjusted"]):
        cur = by_signal.setdefault(r.signal, {"min_adj_p": 1.0, "passes": False})
        if math.isfinite(adj):
            cur["min_adj_p"] = min(cur["min_adj_p"], adj)
        if math.isfinite(adj) and adj <= 0.05:
            cur["passes"] = True
    mt = {"n_tested": bh["n_tested"], "n_rejected": bh["n_rejected"],
          "n_raw_sig": n_raw, "alpha": bh["alpha"], "by_signal": by_signal}
    adj_by_key = {(r.signal, r.horizon): a for r, a in
                  zip(ic_all, bh["adjusted"])}

    # ---- 樣本外分段 ----
    hold = 5
    split = None
    seg_ic: list[ICResult] = []
    try:
        split = make_split([pd.Timestamp(d) for d in dates],
                           train_frac=train_frac, valid_frac=valid_frac,
                           embargo_days=hold)
        for key in signals:
            for h in horizons:
                ic = ic_series(sf, key, h)
                for seg in (split.train, split.valid, split.test):
                    sub = (ic if ic.empty
                           else ic[(ic.index >= seg.start) & (ic.index <= seg.end)])
                    seg_ic.append(ic_summary(sub, key, h, segment=seg.name,
                                             sims=max(200, sims // 4)))
    except ValueError as e:
        log.warning("無法切分樣本外區段：%s", e)
        split = None
        seg_ic = []

    # ---- 分期間分組一致性（排序是否在不同時期維持） ----
    seg_q: dict[str, dict[str, QuantileResult]] = {}
    if split is not None:
        segs = (split.train, split.valid, split.test)
        for key in signals:
            seg_q[key] = quantile_stability(sf, key, 5, segs, cost_model=cm)

    # ---- 個股層級 ----
    per_stock = {key: per_stock_ic(sf, key, 5) for key in signals}

    # ---- 判定 ----
    q_by_key = {(q.signal, q.horizon): q for q in quantiles}
    r_by_key = {(r.signal, r.horizon): r for r in randoms}
    decisions = []
    for key in signals:
        # 以 5 日為主判定（與策略 hold_days 一致），1/10 日僅供對照
        ic5 = next((r for r in ic_all if r.signal == key and r.horizon == 5), None)
        if ic5 is None:
            continue
        verdict, why = decide_signal(
            SIGNAL_BY_KEY.get(key, SignalSpec(key, key, 0.0, "", "", "", "")),
            ic5, q_by_key.get((key, 5)), r_by_key.get((key, 5)),
            adj_by_key.get((key, 5), float("nan")))
        decisions.append((key, verdict, why))

    notes = [
        f"股票池 {n_stocks} 檔、訊號列 {len(sf):,}；價格已套用除權息／分割還原。",
        "前瞻報酬以 T+1 開盤起算，與回測的成交假設（T+1 開盤進場）一致。",
        "缺值因子保持 NaN 並自相關計算中剔除，不以 0 冒充中性。",
        "持有期間與顯著性方法均**事先決定**，未依結果調整；已測試的假設數量"
        "與多重檢定校正均完整揭露。",
        f"多重檢定：同時檢定 {bh['n_tested']} 個（訊號 × 持有期間）假設，"
        f"未校正 {n_raw} 個看似顯著、BH 校正後 {bh['n_rejected']} 個。",
        "⚠️ 樣本僅約 2 年，IC 在此期間的表現不保證其他市況成立"
        "（空頭期間的 IC 另見 P2.9 壓力測試：python -m twstock.cli stress）。",
        "⚠️ 股票池為寫死的 20 檔大型權值股，未含歷史成分與下市股 → survivorship bias。",
        "⚠️ 月營收以『公告日』近似，非精確 point-in-time；法人與融資券為當日公布，"
        "但未驗證實際公告時間戳。",
        "本研究為探索性分析：已嘗試的假設（IC、分組、隨機基準）皆在本報告列出，"
        "未以測試結果反覆修改假設後再宣稱獨立驗證。",
        "本報告不挑選最佳參數，也不代表任何策略具備獲利能力。",
    ]
    return DiagnosticsReport(
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
        horizons=horizons, n_stocks=n_stocks, n_days=len(dates), period=period,
        ic_all=ic_all, ic_segments=seg_ic, quantiles=quantiles,
        segment_quantiles=seg_q, randoms=randoms, per_stock=per_stock,
        multiple_testing=mt, data_notes=notes, decisions=decisions)


def _new_stats() -> dict:
    from .data import loader
    return loader.new_stats()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="訊號資訊量診斷（P2.3）")
    ap.add_argument("--quick", action="store_true",
                    help="只跑主要持有期間與較少模擬次數")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    args = ap.parse_args(argv)
    horizons = (5,) if args.quick else HORIZONS
    sims = 200 if args.quick else args.sims
    report = run_diagnostics(horizons=horizons, sims=sims)
    text = report.to_text()
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"diagnostics_{dt.datetime.now():%Y%m%d_%H%M%S}.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n（報告已寫入：{out}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
