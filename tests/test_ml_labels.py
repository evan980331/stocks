"""W5：ML 標籤必須真的能用基準算出超額報酬；缺基準時要明確標記不可用。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from twstock.ml_model import make_labels, ml_status, train_walk_forward


def _panel(days: int = 60, stocks=("2330", "2317")) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=days)
    rows = []
    for i, sid in enumerate(stocks):
        base = 100.0 + i * 50
        rows.append(pd.DataFrame({
            "date": dates, "stock_id": sid,
            "close": np.linspace(base, base * 1.2, days),
            "open": np.linspace(base, base * 1.2, days), "high": 1.0, "low": 1.0,
            "volume": 1e6,
        }))
    return pd.concat(rows, ignore_index=True)


def _benchmark(days: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=days)
    return pd.DataFrame({"date": dates, "close": np.linspace(200, 210, days)})


def test_labels_present_with_benchmark():
    out = make_labels(_panel(), hold_days=5, benchmark=_benchmark())
    assert out["label"].notna().any(), "有基準時應能算出超額報酬標籤"
    assert out["fwd_ret"].notna().any()
    assert out["fwd_ret_bench"].notna().any()


def test_labels_all_nan_without_benchmark():
    out = make_labels(_panel(), hold_days=5, benchmark=None)
    assert out["label"].isna().all(), "缺基準時應標記為不可用，而非硬算"


def test_labels_are_excess_return():
    panel = _panel(days=40, stocks=("2330",))
    bench = _benchmark(days=40)
    out = make_labels(panel, hold_days=5, benchmark=bench).dropna(subset=["label"])
    assert len(out) > 0
    # label 應等於個股未來報酬減基準未來報酬
    assert np.allclose(out["label"], out["fwd_ret"] - out["fwd_ret_bench"])


def test_train_requires_benchmark(monkeypatch):
    """即使 lightgbm 已安裝，缺基準也必須明確回報不可用。"""
    import twstock.ml_model as ml
    monkeypatch.setattr(ml, "try_import_lgbm", lambda: object())
    model, meta = ml.train_walk_forward(_panel(), benchmark=None)
    assert model is None
    assert meta["enabled"] is False and meta["reason"] == "missing_benchmark"


def test_ml_status_is_honest_without_lightgbm():
    st = ml_status()
    assert "available" in st
    if not st["available"]:
        assert st["reason"] == "lightgbm_not_installed"
