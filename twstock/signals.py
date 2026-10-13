"""規則式多因子打分（baseline 訊號）。

每檔股票每天得到一個 -1 ~ +1 的分數，分項：
  K線（KD、MACD、型態、突破）、趨勢（動能、均線、RSI、量能）、
  籌碼（法人、融券）、基本面（月營收）、消息面（新聞情緒）。

理由欄位以中文輸出，直接供儀表板與通知顯示。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import CONFIG

log = logging.getLogger(__name__)

# 因子權重（總和 = 1.0，打分時會再依可用因子標準化）
WEIGHTS = {
    "kline": 0.18,        # K 線綜合（KD + MACD + 型態）
    "breakout": 0.07,     # 20 日突破 / 破線
    "mom_20": 0.12,       # 20 日動能
    "mom_60": 0.06,       # 60 日動能
    "ma_alignment": 0.09, # 均線多頭排列
    "rsi_rev": 0.05,      # RSI 逆勢（超賣買）
    "vol_ratio": 0.05,    # 量能
    "inst_net_sum3": 0.12,  # 法人近 3 日淨買超
    "streak_inst_buy": 0.05,
    "margin_ratio": 0.05,   # 券資比
    "rev_yoy": 0.11,        # 月營收 YoY
    "news_sentiment": 0.10, # 新聞情緒
}

# 因子名稱 → 中文註解（顯示在「理由」欄）
NAME_CN = {
    "kline_up": "K線偏多",
    "kline_dn": "K線偏空",
    "kd_gold": "KD黃金交叉",
    "kd_death": "KD死亡交叉",
    "macd_up": "MACD翻多",
    "macd_dn": "MACD轉弱",
    "engulf_up": "多方吞噬",
    "engulf_dn": "空方吞噬",
    "hammer": "鎚子線止跌",
    "shooting": "射擊之星",
    "morning": "晨星",
    "evening": "暮星",
    "soldiers": "紅三兵",
    "crows": "黑三兵",
    "doji": "十字變盤",
    "breakout": "突破20日高點",
    "breakdown": "跌破20日低點",
    "mom20": "20日動能強",
    "mom60": "60日趨勢多",
    "ma": "均線多頭排列",
    "rsi_os": "RSI超賣",
    "rsi_ob": "RSI過熱",
    "rsi": "RSI中性",
    "volume": "量能放大",
    "inst3": "法人買超",
    "inst_streak": "法人連續買超",
    "margin": "券資比偏低",
    "rev_yoy": "月營收成長",
    "news": "新聞面偏多",
    "neutral": "持平",
    "no-data": "資料不足",
}


# 型態方向分類（供理由挑選時與分數方向對齊）
BULL_PATS = {"多方吞噬", "鎚子線", "晨星", "紅三兵"}
BEAR_PATS = {"空方吞噬", "射擊之星", "暮星", "黑三兵"}


def _z(x: pd.Series) -> pd.Series:
    s = x.astype(float)
    std = s.std()
    return (s - s.mean()) / std if std and std > 0 else s * 0.0


def _factor_parts(row: pd.Series) -> list[tuple[str, float, str]]:
    """各因子的 (因子鍵, 方向化數值, 顯示名稱)。

    方向化數值：**正 = 偏多、負 = 偏空**，-1 ~ +1。此函式是唯一的因子計算來源，
    ``score_row``（合成總分）與 ``score_components``（P2.3 逐因子診斷）共用它，
    確保診斷分析的就是策略真正採用的訊號，而非另寫一份近似版本。
    """
    parts: list[tuple[str, float, str]] = []

    # ===== K 線綜合（KD + MACD + 型態）=====
    kl = row.get("kline_score", np.nan)
    if pd.notna(kl):
        lab = row.get("pat_label", "") or ""
        kg = row.get("kd_gold", 0) or 0
        kdth = row.get("kd_death", 0) or 0
        mg = row.get("macd_gold", 0) or 0
        md = row.get("macd_death", 0) or 0
        if kl >= 0:
            # 偏多：只挑方向一致的標籤，避免「分數正但理由寫死亡交叉」
            if lab in BULL_PATS:
                name = lab
            elif kg:
                name = "kd_gold"
            elif mg:
                name = "macd_up"
            else:
                name = "kline_up"
        else:
            if lab in BEAR_PATS:
                name = lab
            elif kdth:
                name = "kd_death"
            elif md:
                name = "macd_dn"
            else:
                name = "kline_dn"
        parts.append(("kline", _clip(float(kl)), name))

    # ===== 20 日突破 / 破線 =====
    bo = row.get("breakout20", 0) or 0
    bd = row.get("breakdown20", 0) or 0
    if bo:
        parts.append(("breakout", 1.0, "breakout"))
    elif bd:
        parts.append(("breakout", -1.0, "breakdown"))

    mom20 = row.get("mom_20", np.nan)
    mom60 = row.get("mom_60", np.nan)
    parts.append(("mom_20", _clip(mom20 * 5), "mom20"))
    parts.append(("mom_60", _clip((mom60 or 0) * 2.5), "mom60"))

    parts.append(("ma_alignment",
                  1.0 if row.get("ma_alignment", 0) == 1 else -0.3, "ma"))

    rsi_v = row.get("rsi", np.nan)
    if pd.notna(rsi_v):
        if rsi_v < 30:
            parts.append(("rsi_rev", 0.8, "rsi_os"))
        elif rsi_v > 70:
            parts.append(("rsi_rev", -0.6, "rsi_ob"))
        else:
            parts.append(("rsi_rev", 0.0, "rsi"))

    vr = row.get("vol_ratio", np.nan)
    if pd.notna(vr) and vr > 1.5:
        # 放量方向跟當日漲跌
        ret = row.get("ret_1d", 0) or 0
        parts.append(("vol_ratio", 0.6 if ret > 0 else -0.6, "volume"))

    inst3 = row.get("inst_net_sum3", np.nan)
    if pd.notna(inst3):
        parts.append(("inst_net_sum3",
                      _clip(inst3 / max(1e6, abs(inst3)) * 1e6 / 5e7), "inst3"))
    streak = row.get("streak_inst_buy", 0) or 0
    if streak >= 3:
        parts.append(("streak_inst_buy", min(1.0, streak / 8), "inst_streak"))

    mr = row.get("margin_ratio", np.nan)
    if pd.notna(mr):
        # 券資比 < 0.35 偏多（軋空空間），> 0.6 偏空
        parts.append(("margin_ratio", _clip((0.35 - mr) * 4), "margin"))

    yoy = row.get("rev_yoy", np.nan)
    if pd.notna(yoy):
        parts.append(("rev_yoy", _clip(float(yoy) / 30.0), "rev_yoy"))

    ns = row.get("news_sentiment", np.nan) if "news_sentiment" in row else np.nan
    if pd.notna(ns):
        parts.append(("news_sentiment", _clip(float(ns) * 1.5), "news"))

    return parts


def score_components(row: pd.Series) -> dict[str, float]:
    """逐因子方向化數值（正 = 偏多），鍵為 ``WEIGHTS`` 的因子名。

    供 P2.3 訊號資訊量診斷使用；與 ``score_row`` 共用 ``_factor_parts``，
    因此分析對象就是策略實際採用的訊號。缺值因子**不會**出現在回傳字典中，
    交由呼叫端明確視為缺值，不以 0 冒充「中性」。
    """
    return {key: float(value) for key, value, _name in _factor_parts(row)}


def score_row(row: pd.Series) -> tuple[float, str]:
    parts = _factor_parts(row)
    total_w = sum(WEIGHTS[key] for key, _, _ in parts)
    if total_w == 0:
        return 0.0, "資料不足"
    score = sum(WEIGHTS[key] * v for key, v, _ in parts) / total_w
    # 只有非零權重的因子才進 reasons，且與原本邏輯一致（v > 0.15）
    reasons = [NAME_CN.get(name, name) for _, v, name in parts if v > 0.15]
    return float(np.clip(score, -1, 1)), "、".join(reasons) or "持平"



def _clip(x: float, limit: float = 1.0) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(np.clip(x, -limit, limit))


def score_stocks(features_by_stock: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """對每檔股票逐日打分，回傳長表 date, stock_id, score, reason, close。"""
    rows = []
    for sid, df in features_by_stock.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            s, reason = score_row(row)
            rows.append({
                "date": row["date"].date().isoformat(),
                "stock_id": sid,
                "score": round(s, 4),
                "reason": reason,
                "close": row.get("close"),
            })
    return pd.DataFrame(rows)


# 建議動作的中英對照（DB 存英文代碼，顯示時轉中文）
ACTION_CN = {"BUY": "買進", "HOLD": "持有", "AVOID": "迴避"}


def _rank_pct(s: pd.Series) -> pd.Series:
    """當日橫斷面排名百分位，映射到 [-1, 1]（尺度無關）。

    只有一檔股票時無法比較，回傳 0（中性），不臆測方向。
    """
    v = pd.to_numeric(s, errors="coerce")
    n = int(v.notna().sum())
    if n <= 1:
        return pd.Series(0.0, index=s.index)
    r = v.rank(method="average")
    return (r - 1) / (n - 1) * 2 - 1


def fuse_scores(rule: pd.DataFrame, ml: pd.DataFrame | None = None,
                *, weight_rule: float | None = None,
                weight_ml: float | None = None) -> pd.DataFrame:
    """融合規則分數與 ML 預測分數。

    問題背景：規則分數定義於 [-1, 1]，而 ML 輸出是「預測未來 N 日超額報酬率」
    （量級約 ±0.03）。兩者尺度不同，直接加權相加會讓 ML 形同被忽略、
    且缺值填 0 會冒充「中性」。

    作法：先把每個訊號做**當日橫斷面排名百分位**映射到 [-1, 1]，再加權平均。
    排名化的好處是尺度不變、對離群值穩健，適合跨模型融合。

    - 沒有 ML 資料（或全缺）時，明確退化為規則分數，不假裝有 ML 訊號。
    - 回傳欄位：date, stock_id, score, rule_score, ml_score, reason, close
    """
    w_r = CONFIG.fusion_weight_rule if weight_rule is None else weight_rule
    w_m = CONFIG.fusion_weight_ml if weight_ml is None else weight_ml

    out = rule.copy()
    out["rule_score"] = pd.to_numeric(out["score"], errors="coerce")
    out["ml_score"] = np.nan

    has_ml = ml is not None and not ml.empty and "ml_score" in ml.columns
    if not has_ml:
        out["score"] = out["rule_score"]
        return out.drop(columns=["rule_score"])

    keys = ["date", "stock_id"]
    merged = out.merge(ml[keys + ["ml_score"]], on=keys, how="left",
                       suffixes=("", "_ml"))
    # 若來源本身已帶 ml_score 欄位，避免重複
    if "ml_score_ml" in merged.columns:
        merged["ml_score"] = merged["ml_score_ml"].combine_first(merged["ml_score"])
        merged = merged.drop(columns=["ml_score_ml"])

    # 若 ML 欄位存在但**全部**為缺值，視同沒有 ML：直接回規則分數，
    # 不可用排名或 0 冒充「模型已判讀為中性」。
    if not merged["ml_score"].notna().any():
        out["score"] = out["rule_score"]
        return out.drop(columns=["rule_score"])

    merged["rank_rule"] = (merged.groupby("date")["rule_score"]
                           .transform(_rank_pct))
    ml_present = merged["ml_score"].notna()
    merged["rank_ml"] = merged.groupby("date")["ml_score"].transform(_rank_pct)

    total = w_r + w_m or 1.0
    fused = (merged["rank_rule"] * w_r + merged["rank_ml"] * w_m) / total
    # 沒有 ML 的列：明確只用規則分數（避免用 0 冒充模型判讀）
    merged["score"] = np.where(ml_present, fused, merged["rank_rule"])
    return merged.drop(columns=["rank_rule", "rank_ml"])


def pick_top_n(scores: pd.DataFrame, date: str, top_n: int = 5) -> pd.DataFrame:
    df = scores[scores["date"] == date].copy()
    if df.empty:
        return df
    df = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["action"] = np.where(df["score"] >= 0.2, "BUY",
                            np.where(df["score"] <= -0.2, "AVOID", "HOLD"))
    return df
