"""特徵計算：技術面 + K線型態 + 籌碼面。輸入個股日K DataFrame，輸出逐日特徵表。"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SENTIMENT_WEIGHTS = {0: 1.0, 1: 0.66, 2: 0.4}


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean())
    loss = ((-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean())
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    out[gain == loss] = 100.0  # 全漲無跌時 RSI=100
    return out


def kd_stochastic(close: pd.Series, high: pd.Series, low: pd.Series,
                  n: int = 9, m1: int = 3, m2: int = 3):
    """台股標準 KD（9,3,3），回傳 (K, D)。"""
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    return k, d


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD（DIF, DEA, OSC），台灣慣例 OSC 乘 2。"""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    osc = (dif - dea) * 2
    return dif, dea, osc


def bollinger(close: pd.Series, n: int = 20, k: float = 2):
    """布林通道，回傳 (上軌, 中軌, 下軌, %B, 帶寬)。"""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, pctb, bandwidth


def candle_patterns(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series):
    """K 線型態辨識，回傳 (score[-1,1], 中文標籤)。"""
    body = c - o
    rng = (h - l).replace(0, np.nan)
    abs_body = body.abs()
    hi_body = pd.concat([o, c], axis=1).max(axis=1)
    lo_body = pd.concat([o, c], axis=1).min(axis=1)
    upper_sh = h - hi_body
    lower_sh = lo_body - l
    bull = body > 0

    score = pd.Series(0.0, index=o.index)
    label = pd.Series("", index=o.index, dtype=object)

    prev_body = body.shift(1)
    prev_bull = bull.shift(1).astype("boolean").fillna(False).astype(bool)
    prev_rng = rng.shift(1)

    # --- 吞噬 ---
    engulf_bull = (bull & (~prev_bull) & (abs_body > prev_body.abs())
                   & (prev_body.abs() / prev_rng <= 0.6))
    engulf_bear = ((~bull) & prev_bull & (abs_body > prev_body.abs())
                   & (prev_body.abs() / prev_rng <= 0.6))
    # --- 鎚子線 / 射擊之星 ---
    small_body = abs_body / rng <= 0.35
    hammer = (lower_sh >= 2 * abs_body) & small_body & (upper_sh <= 0.5 * abs_body)
    shooting = (upper_sh >= 2 * abs_body) & small_body & (lower_sh <= 0.5 * abs_body)
    # --- 十字線 ---
    doji = abs_body / rng <= 0.10
    # --- 紅三兵 / 黑三兵 ---
    up_step = bull & (c > c.shift(1))
    soldiers = up_step & up_step.shift(1) & up_step.shift(2)
    dn_step = (~bull) & (c < c.shift(1))
    crows = dn_step & dn_step.shift(1) & dn_step.shift(2)
    # --- 晨星 / 暮星（三天型態）---
    d1, d2, d3 = body.shift(2), body.shift(1), body
    morning = ((d1 < 0) & (d1.abs() / rng.shift(2) >= 0.5)
               & (d2.abs() / rng.shift(1) <= 0.3)
               & (d3 > 0) & (d3.abs() / rng >= 0.5)
               & (c > (o.shift(2) + c.shift(2)) / 2))
    evening = ((d1 > 0) & (d1.abs() / rng.shift(2) >= 0.5)
               & (d2.abs() / rng.shift(1) <= 0.3)
               & (d3 < 0) & (d3.abs() / rng >= 0.5)
               & (c < (o.shift(2) + c.shift(2)) / 2))

    # 由弱到強排序，強型態覆蓋弱型態
    signals = [
        (doji, 0.0, "十字線"),
        (hammer, 0.7, "鎚子線"),
        (shooting, -0.7, "射擊之星"),
        (engulf_bull, 0.8, "多方吞噬"),
        (engulf_bear, -0.8, "空方吞噬"),
        (morning, 0.9, "晨星"),
        (evening, -0.9, "暮星"),
        (soldiers, 1.0, "紅三兵"),
        (crows, -1.0, "黑三兵"),
    ]
    for mask, val, name in signals:
        m = mask.fillna(False)
        score = score.mask(m, float(val))
        label = label.mask(m, name)
    return score.clip(-1, 1), label


def build_features(price: pd.DataFrame,
                   inst: pd.DataFrame | None = None,
                   margin: pd.DataFrame | None = None,
                   revenue: pd.DataFrame | None = None,
                   news_daily: pd.DataFrame | None = None,
                   momentum_windows: tuple[int, ...] = (20, 60),
                   rsi_window: int = 14,
                   vol_ma_window: int = 20) -> pd.DataFrame:
    df = price.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # 非正數價格視為缺值（缺值不可填 0，否則 pct_change 會變成 ±inf）
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= 0, col] = np.nan
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    close = df["close"]
    for w in momentum_windows:
        df[f"mom_{w}"] = close.pct_change(w)
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()
    df["ma_alignment"] = ((df["ma5"] > df["ma20"]) & (df["ma20"] > df["ma60"])).astype(float)
    df["rsi"] = rsi(close, rsi_window)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(vol_ma_window).mean()
    df["ret_1d"] = close.pct_change()
    df["volatility_20"] = close.pct_change().rolling(20).std() * (252 ** 0.5)

    # ===== K 線技術指標 =====
    if {"high", "low", "open"}.issubset(df.columns):
        hi, lo, op = df["high"], df["low"], df["open"]

        k, d = kd_stochastic(close, hi, lo)
        df["kd_k"], df["kd_d"] = k, d
        df["kd_gold"] = ((k > d) & (k.shift(1) <= d.shift(1))).astype(float)
        df["kd_death"] = ((k < d) & (k.shift(1) >= d.shift(1))).astype(float)
        df["kd_ob"] = (k > 80).astype(float)
        df["kd_os"] = (k < 20).astype(float)

        dif, dea, osc = macd(close)
        df["macd_dif"], df["macd_dea"], df["macd_osc"] = dif, dea, osc
        df["macd_gold"] = ((dif > dea) & (dif.shift(1) <= dea.shift(1))).astype(float)
        df["macd_death"] = ((dif < dea) & (dif.shift(1) >= dea.shift(1))).astype(float)

        _u, _m, _l, pctb, bw = bollinger(close)
        df["bb_pctb"], df["bb_bandwidth"] = pctb, bw

        # 20 日突破 / 破線
        prior_high = hi.rolling(20).max().shift(1)
        prior_low = lo.rolling(20).min().shift(1)
        df["breakout20"] = (close > prior_high).astype(float)
        df["breakdown20"] = (close < prior_low).astype(float)

        pat_score, pat_label = candle_patterns(op, hi, lo, close)
        df["pat_score"], df["pat_label"] = pat_score, pat_label

        # 綜合 K 線分數：KD 40% + MACD 35% + 型態 25%
        kd_c = ((k - d) / 40).clip(-1, 1)
        macd_c = (0.6 * np.sign(dif - dea)
                  + 0.4 * np.sign(osc - osc.shift(1).fillna(0))).clip(-1, 1)
        df["kline_score"] = (0.40 * kd_c + 0.35 * macd_c + 0.25 * pat_score).clip(-1, 1)


    if inst is not None and not inst.empty:
        inst = inst.copy()
        inst["date"] = pd.to_datetime(inst["date"])
        inst = inst.sort_values("date")
        # FinMind 原生 schema: date, stock_id, buy, name, sell（含外資/投信/自營商）
        if {"buy", "sell", "name"}.issubset(inst.columns):
            daily = (inst.assign(net=inst["buy"].astype(float) - inst["sell"].astype(float))
                        .groupby("date")["net"].sum().reset_index(name="inst_net"))
            df = df.merge(daily, on="date", how="left")
        elif any(c.endswith("Net") for c in inst.columns):
            net_cols = [c for c in inst.columns if c.endswith("Net")]
            inst["inst_net"] = inst[net_cols].astype(float).sum(axis=1)
            df = df.merge(inst[["date", "inst_net"]], on="date", how="left")
        else:
            return df
        df["inst_net_sum3"] = df["inst_net"].rolling(3).sum()
        buy = (df["inst_net"] > 0).astype(float)
        grp = (df["inst_net"] <= 0).cumsum()
        df["streak_inst_buy"] = buy.groupby(grp).cumsum()

    if margin is not None and not margin.empty:
        mg = margin.copy()
        mg["date"] = pd.to_datetime(mg["date"])
        mg = mg.sort_values("date")
        mp_col = next((c for c in ("MarginPurchaseBalance", "MarginPurchaseTodayBalance")
                       if c in mg.columns), None)
        ss_col = next((c for c in ("ShortSaleBalance", "ShortSaleTodayBalance")
                       if c in mg.columns), None)
        if mp_col and ss_col:
            mg["margin_ratio"] = (
                mg[ss_col].astype(float) / mg[mp_col].astype(float).replace(0, np.nan)
            )
            df = df.merge(mg[["date", "margin_ratio"]], on="date", how="left")

    if revenue is not None and not revenue.empty:
        rev = revenue.copy()
        rev["date"] = pd.to_datetime(rev["date"])
        rev = rev.sort_values("date").drop_duplicates("date", keep="last")
        if "YoY" not in rev.columns and {"revenue", "revenue_year", "revenue_month"}.issubset(rev.columns):
            rev = rev.sort_values(["stock_id", "revenue_year", "revenue_month"], ignore_index=True)
            prev_yoy = rev.groupby("stock_id")["revenue"].shift(12)
            rev["YoY"] = (rev["revenue"] - prev_yoy) / prev_yoy.abs().replace(0, np.nan) * 100
        if "YoY" in rev.columns:
            df = pd.merge_asof(df.sort_values("date"),
                               rev[["date", "YoY"]].drop_duplicates("date", keep="last").sort_values("date"),
                               on="date", direction="backward")
            df = df.rename(columns={"YoY": "rev_yoy"})

    if news_daily is not None and not news_daily.empty:
        nd = news_daily.copy()
        nd["date"] = pd.to_datetime(nd["date"])
        decayed = []
        for _, row in nd.iterrows():
            for lag, w in SENTIMENT_WEIGHTS.items():
                decayed.append({"date": row["date"] + pd.Timedelta(days=lag),
                                "w_score": float(row["score"]) * w})
        dec = (pd.DataFrame(decayed).groupby("date")["w_score"].mean().reset_index())
        df = df.merge(dec, on="date", how="left")
        df["news_sentiment"] = df["w_score"].fillna(0.0)
        df = df.drop(columns=["w_score"])

    return df
