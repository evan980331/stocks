"""LightGBM walk-forward 模型（可選，未安裝 lightgbm 時自動停用）。

label：未來 hold_days 日超額報酬（相對 0050）。
訓練：時間序 walk-forward，最後 3 個月做驗證。
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


def _make_labels(panel: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    df = panel.copy()
    df = df.sort_values(["stock_id", "date"])
    df["fwd_ret"] = df.groupby("stock_id")["close"].pct_change(hold_days).shift(-hold_days)
    bench = df[df["stock_id"] == CONFIG.benchmark][["date", "close"]].rename(
        columns={"close": "bench_close"})
    df = df.merge(bench, on="date", how="left")
    df["bench_close"] = df["bench_close"].ffill()
    df["fwd_ret_bench"] = df.groupby("stock_id")["bench_close"].pct_change(hold_days).shift(-hold_days)
    df["label"] = df["fwd_ret"] - df["fwd_ret_bench"]
    return df


def try_import_lgbm():
    try:
        import lightgbm as lgb  # noqa: F401
        return lgb
    except ImportError:
        return None


def train_walk_forward(panel: pd.DataFrame) -> tuple[object | None, dict]:
    """輸入長表（每檔特徵逐日），回傳訓練好的 model 與 meta。"""
    lgb = try_import_lgbm()
    if lgb is None:
        log.info("lightgbm 未安裝，跳過 ML 訓練")
        return None, {"enabled": False}

    df = _make_labels(panel, CONFIG.hold_days)
    df = df.dropna(subset=["label"])
    feature = [c for c in FEATURE_COLS if c in df.columns]
    train = df[df["date"] < df["date"].max() - pd.Timedelta(days=90)]
    valid = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]

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
        "feature": feature,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "best_iteration": model.best_iteration_,
    }
    log.info("LightGBM trained: %s", meta)
    return model, meta


def predict_scores(model, panel: pd.DataFrame) -> pd.DataFrame:
    """對每檔最新一天的特徵預測未來超額報酬，回傳 date, stock_id, ml_score。"""
    if model is None:
        return pd.DataFrame()
    feature = [c for c in FEATURE_COLS if c in panel.columns]
    latest = (panel.sort_values("date")
                   .groupby("stock_id").tail(1))
    latest = latest.dropna(subset=feature, how="all")
    latest["ml_score"] = model.predict(latest[feature])
    guess_bench = latest[latest["stock_id"] != CONFIG.benchmark].copy()
    return guess_bench[["date", "stock_id", "ml_score"]].copy()
