"""LightGBM walk-forward 模型（可選，未安裝 lightgbm 時明確停用而非靜默失敗）。

label：未來 hold_days 日超額報酬（相對基準 0050）。

修正重點（W5）：
  舊版 ``_make_labels`` 直接從 panel 找 `stock_id == 0050`，但 panel 由股票池組成、
  從不含基準，導致 label 全為 NaN、模型永遠訓練不出東西卻不報錯。
  現改為由呼叫端明確傳入基準日K序列，並在資料不足時清楚回報原因。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import CONFIG

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "mom_20", "mom_60", "ma_alignment", "rsi", "vol_ratio", "ret_1d",
    "volatility_20", "inst_net_sum3", "streak_inst_buy", "margin_ratio",
    "rev_yoy", "news_sentiment",
]


def make_labels(panel: pd.DataFrame, hold_days: int,
                benchmark: pd.DataFrame | None) -> pd.DataFrame:
    """為 panel 產生 label = 個股未來 hold_days 報酬 − 基準同期報酬。

    benchmark 為基準（0050）日K（欄位 date, close）。若缺失則 label 全為 NaN，
    呼叫端可據此判斷 ML 不可用。
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"])
    df["fwd_ret"] = df.groupby("stock_id")["close"].pct_change(hold_days).shift(-hold_days)

    if benchmark is None or benchmark.empty:
        df["fwd_ret_bench"] = np.nan
        df["label"] = np.nan
        return df

    bench = benchmark.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    bench = (bench[["date", "close"]]
             .sort_values("date")
             .drop_duplicates("date", keep="last")
             .rename(columns={"close": "bench_close"}))
    df = df.merge(bench, on="date", how="left")
    df["bench_close"] = df["bench_close"].ffill()
    # 基準的未來報酬與個股同一交易日對齊
    bench_fwd = (bench.assign(
        fwd_ret_bench=bench["bench_close"].pct_change(hold_days).shift(-hold_days))
        [["date", "fwd_ret_bench"]])
    df = df.drop(columns=[c for c in ("fwd_ret_bench",) if c in df.columns])
    df = df.merge(bench_fwd, on="date", how="left")
    df["label"] = df["fwd_ret"] - df["fwd_ret_bench"]
    return df


# 向後相容舊名稱
_make_labels = make_labels


def try_import_lgbm():
    try:
        import lightgbm as lgb  # noqa: F401
        return lgb
    except ImportError:
        return None


def train_walk_forward(panel: pd.DataFrame, benchmark: pd.DataFrame | None = None,
                       ) -> tuple[object | None, dict]:
    """輸入長表（每檔特徵逐日）與基準日K，回傳 (model, meta)。

    meta 一律包含 enabled / reason / rows，讓上層能誠實顯示「ML 未啟用」的原因。
    """
    lgb = try_import_lgbm()
    if lgb is None:
        log.info("lightgbm 未安裝，ML 層停用")
        return None, {"enabled": False, "reason": "lightgbm_not_installed"}
    if panel is None or panel.empty:
        return None, {"enabled": False, "reason": "no_panel"}
    if benchmark is None or benchmark.empty:
        log.warning("缺少基準（%s）日K，無法計算超額報酬標籤", CONFIG.benchmark)
        return None, {"enabled": False, "reason": "missing_benchmark"}

    df = make_labels(panel, CONFIG.hold_days, benchmark)
    df = df.dropna(subset=["label"])
    if df.empty:
        return None, {"enabled": False, "reason": "no_labels"}

    feature = [c for c in FEATURE_COLS if c in df.columns]
    cutoff = df["date"].max() - pd.Timedelta(days=90)
    train, valid = df[df["date"] < cutoff], df[df["date"] >= cutoff]
    if train.empty or valid.empty:
        return None, {"enabled": False, "reason": "insufficient_history",
                      "rows": len(df)}

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(train[feature], train["label"],
              eval_set=[(valid[feature], valid["label"])],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    meta = {
        "enabled": True,
        "reason": "",
        "feature": feature,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "rows": len(df),
        "best_iteration": model.best_iteration_,
    }
    log.info("LightGBM trained: %s", meta)
    return model, meta


def predict_scores(model, panel: pd.DataFrame,
                   benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    """對每檔最新一天的特徵預測未來超額報酬，回傳 date, stock_id, ml_score。"""
    if model is None or panel is None or panel.empty:
        return pd.DataFrame(columns=["date", "stock_id", "ml_score"])
    feature = [c for c in FEATURE_COLS if c in panel.columns]
    latest = panel.copy()
    latest["date"] = pd.to_datetime(latest["date"])
    latest = latest.sort_values("date").groupby("stock_id").tail(1)
    latest = latest.dropna(subset=feature, how="all")
    if latest.empty:
        return pd.DataFrame(columns=["date", "stock_id", "ml_score"])
    latest = latest.copy()
    latest["ml_score"] = model.predict(latest[feature])
    out = latest[latest["stock_id"] != CONFIG.benchmark]
    return out[["date", "stock_id", "ml_score"]].copy()


def ml_status() -> dict:
    """目前 ML 可用性（供健康檢查與儀表板誠實顯示）。"""
    if try_import_lgbm() is None:
        return {"available": False, "reason": "lightgbm_not_installed"}
    return {"available": True, "reason": ""}
