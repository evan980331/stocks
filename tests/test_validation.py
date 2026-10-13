"""P2.1 樣本外驗證：時間切分、purge/embargo、轉換不洩漏、基準處理。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twstock.validation import (FeatureScaler, HoldoutGuard, check_benchmark,
                                evaluate_strategies, make_split,
                                purge_boundary, walk_forward)


def _dates(n=400, start="2024-01-01"):
    return list(pd.bdate_range(start, periods=n))


# --------------------------------------------------------------------------
# 切分與 purge
# --------------------------------------------------------------------------
def test_split_is_chronological_and_disjoint():
    ds = _dates(400)
    sp = make_split(ds, embargo_days=5)
    assert sp.train.end < sp.valid.start
    assert sp.valid.end < sp.test.start
    assert sp.test.end == ds[-1]


def test_split_embargo_creates_gap():
    ds = _dates(400)
    sp = make_split(ds, embargo_days=5)
    all_dates = sorted(ds)
    gaps = [d for d in all_dates if sp.train.end < d < sp.test.start]
    assert len(gaps) >= 1, "train 與 test 之間應存在被 purge 的空隙"


def test_no_training_label_window_overlaps_test():
    """核心洩漏檢查：任何訓練樣本的標籤期間都不得跨入測試段。"""
    ds = _dates(400)
    hold = 5
    sp = make_split(ds, embargo_days=hold)
    idx = {d: i for i, d in enumerate(ds)}
    train = [d for d in ds if sp.train.start <= d <= sp.train.end]
    for d in train:
        i = idx[d]
        assert i + hold < len(ds)
        label_end = ds[i + hold]           # 標籤觀察期間最後一天
        assert label_end < sp.test.start, f"{d} 的標籤期間跨入測試段"


def test_purge_boundary_removes_tail_samples():
    ds = _dates(50)
    boundary = ds[40]
    kept = purge_boundary(ds[:40], hold_days=5, boundary=boundary)
    assert len(kept) == 35, "尾端 5 個樣本的標籤會跨過邊界，必須移除"
    assert kept[-1] == ds[34]


def test_purge_boundary_hold_zero_keeps_all():
    ds = _dates(20)
    assert purge_boundary(ds, 0, ds[-1]) == list(ds)


def test_make_split_rejects_insufficient_data():
    with pytest.raises(ValueError):
        make_split(_dates(10), embargo_days=5)


def test_walk_forward_records_windows_and_skips_short_folds():
    ds = _dates(500)
    folds = walk_forward(ds, n_folds=3, train_frac=0.5, embargo_days=5)
    assert folds, "應至少產生一個 fold"
    for f in folds:
        assert f.test_start <= f.test_end
        assert f.n_test >= 20, "過短的測試窗應被略過"
        assert f.train_end < f.test_start, "訓練資料不得晚於測試段"


def test_walk_forward_is_expanding():
    ds = _dates(500)
    folds = walk_forward(ds, n_folds=3, train_frac=0.5, embargo_days=5)
    if len(folds) > 1:
        assert folds[1].n_train >= folds[0].n_train


# --------------------------------------------------------------------------
# 轉換：只用訓練資料擬合
# --------------------------------------------------------------------------
def test_scaler_fit_uses_training_statistics_only():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})       # mean 2, std 1.58
    test = pd.DataFrame({"x": [100.0, 200.0]})                   # 極端值
    sc = FeatureScaler().fit(train, ["x"])
    assert sc.mean_["x"] == pytest.approx(2.0)
    out = sc.transform(test)
    expected = (np.array([100.0, 200.0]) - 2.0) / sc.std_["x"]
    assert np.allclose(out["x"].to_numpy(), expected)


def test_scaler_output_independent_of_test_values():
    """同樣的 fit，不同測試集不得改變任何參數（否則就是用了測試集資訊）。"""
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    sc_a = FeatureScaler().fit(train, ["x"])
    params_a = dict(sc_a.mean_), dict(sc_a.std_)
    FeatureScaler().fit(train, ["x"]).transform(
        pd.DataFrame({"x": [1e9, -1e9]}))
    sc_b = FeatureScaler().fit(train, ["x"])
    assert params_a == (dict(sc_b.mean_), dict(sc_b.std_))


def test_scaler_requires_fit_before_transform():
    with pytest.raises(RuntimeError):
        FeatureScaler().transform(pd.DataFrame({"x": [1.0]}))


# --------------------------------------------------------------------------
# 基準檢查
# --------------------------------------------------------------------------
def test_benchmark_missing_is_explicit():
    st = check_benchmark(None)
    assert st.ok is False and "缺少基準" in st.reason
    st2 = check_benchmark(pd.DataFrame())
    assert st2.ok is False


def test_benchmark_missing_columns():
    st = check_benchmark(pd.DataFrame({"date": ["2026-01-01"], "px": [1.0]}))
    assert st.ok is False and "date/close" in st.reason


def test_benchmark_range_mismatch_is_rejected():
    bench = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=10),
                          "close": np.linspace(100, 110, 10)})
    st = check_benchmark(bench, start=pd.Timestamp("2025-01-01"),
                         end=pd.Timestamp("2026-12-31"))
    assert st.ok is False
    assert "時間範圍不一致" in st.reason


def test_benchmark_ok_when_covering_range():
    d = pd.bdate_range("2025-06-01", periods=200)
    bench = pd.DataFrame({"date": d, "close": np.linspace(100, 130, 200)})
    st = check_benchmark(bench, start=d[0], end=d[-1])
    assert st.ok is True and st.rows == 200


# --------------------------------------------------------------------------
# 測試集使用守門
# --------------------------------------------------------------------------
def test_test_set_guard_flags_reuse():
    g = HoldoutGuard()
    assert g.record("A@test") == 1
    assert g.violations == []
    assert g.record("A@test") == 2
    assert g.violations and "A@test" in g.violations[0]


# --------------------------------------------------------------------------
# 端到端（合成資料、無基準）
# --------------------------------------------------------------------------
def _synthetic_panel_and_prices(days=260, stocks=("AAA", "BBB")):
    dates = pd.bdate_range("2025-01-01", periods=days)
    rng = np.random.default_rng(7)
    panel_rows, price_panel = [], {}
    for k, sid in enumerate(stocks):
        price = 100 + k * 10 + np.cumsum(rng.normal(0.05, 1.0, days))
        price = np.maximum(price, 1.0)
        price_panel[sid] = pd.DataFrame({
            "date": dates, "open": price, "high": price * 1.01,
            "low": price * 0.99, "close": price, "volume": 1_000_000.0})
        panel_rows.append(pd.DataFrame({
            "date": dates, "stock_id": sid, "close": price,
            "mom_20": pd.Series(price).pct_change(20).to_numpy(),
            "mom_60": pd.Series(price).pct_change(60).to_numpy(),
            "ma_alignment": 1.0, "rsi": 50.0, "vol_ratio": 1.0,
            "ret_1d": 0.0,
        }))
    return pd.concat(panel_rows, ignore_index=True), price_panel


def test_evaluate_strategies_runs_on_synthetic_data():
    from twstock.strategies import baseline
    from twstock.validation import build_scores
    panel, prices = _synthetic_panel_and_prices()
    scores = build_scores(panel)
    assert not scores.empty
    dates = sorted(pd.to_datetime(scores["date"]).dt.normalize().unique())
    split = make_split(dates, embargo_days=5)
    rows = evaluate_strategies(scores, prices, split, [baseline()],
                               benchmark=None, segments=("train", "test"))
    assert len(rows) == 2
    assert all(r.result.strategy_name == baseline().name for r in rows)


def test_report_marks_benchmark_unavailable_without_fabricating():
    from twstock.validation import ValidationReport
    ds = _dates(300)
    sp = make_split(ds, embargo_days=5)
    rep = ValidationReport(generated_at="now", split=sp, folds=[],
                           benchmark=check_benchmark(None), rows=[],
                           strategy_benchmark_available=False)
    text = rep.to_text()
    assert "基準不可用" in text
    assert "N/A" in text
    assert "0.00%" not in text.split("基準檢查")[1].split("[4]")[0]
