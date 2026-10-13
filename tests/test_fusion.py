"""W6：規則分數（[-1,1]）與 ML 預測報酬率（±0.03）尺度不同，不得直接相加。

融合必須先轉成當日橫斷面排名百分位，做到尺度無關，且缺 ML 時不得假裝有 ML 訊號。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from twstock.signals import fuse_scores


def _rule() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-10-12"] * 3,
        "stock_id": ["A", "B", "C"],
        "score": [0.5, 0.1, -0.2],
        "reason": ["", "", ""],
        "close": [100.0, 200.0, 300.0],
    })


def _ml() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-10-12"] * 3,
        "stock_id": ["A", "B", "C"],
        "ml_score": [0.01, 0.02, -0.03],
    })


def test_scale_invariance():
    """把 ML 預測值放大一百萬倍，排序與融合結果必須完全不變（舊版會被完全改變）。"""
    base = fuse_scores(_rule(), _ml())
    scaled_ml = _ml()
    scaled_ml["ml_score"] = scaled_ml["ml_score"] * 1_000_000
    scaled = fuse_scores(_rule(), scaled_ml)
    assert np.allclose(base["score"].to_numpy(), scaled["score"].to_numpy())


def test_ml_not_ignored_by_scale():
    """ML 必須真的影響結果：weight_ml=0 與預設權重結果應不同。"""
    with_ml = fuse_scores(_rule(), _ml())
    ml_off = fuse_scores(_rule(), _ml(), weight_ml=0.0)
    assert not np.allclose(with_ml["score"].to_numpy(), ml_off["score"].to_numpy())


def test_no_ml_falls_back_to_rule_only():
    out = fuse_scores(_rule(), None)
    assert np.allclose(out["score"].to_numpy(), [0.5, 0.1, -0.2])


def test_all_nan_ml_falls_back_to_rule_only():
    ml = _ml()
    ml["ml_score"] = np.nan
    out = fuse_scores(_rule(), ml)
    assert np.allclose(out["score"].to_numpy(), [0.5, 0.1, -0.2])


def test_partial_ml_uses_rule_rank_for_missing():
    ml = _ml()
    ml.loc[ml["stock_id"] == "A", "ml_score"] = np.nan   # A 缺 ML
    out = fuse_scores(_rule(), ml).set_index("stock_id")
    # A 的規則分數最高 → 排名百分位 +1
    assert abs(out.loc["A", "score"] - 1.0) < 1e-9


def test_fusion_keeps_rule_ordering_for_single_stock_day():
    rule = _rule().head(1)          # 當日只有一檔 → 無法比較，保守回中性
    out = fuse_scores(rule, _ml().head(1))
    assert abs(out["score"].iloc[0]) < 1e-9
