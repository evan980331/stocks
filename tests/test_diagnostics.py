"""P2.3 訊號資訊量診斷測試：時間對齊、IC 正確性、分組、隨機基準、多重檢定。

全部使用合成資料，不依賴網路。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from twstock import signals as sig_mod
from twstock.diagnostics import (HORIZONS, N_GROUPS, SIGNAL_KEYS, MIN_DAYS_FOR_IC,
                                 SignalSpec, attach_forward_returns,
                                 benjamini_hochberg, block_bootstrap_ci,
                                 build_signal_frame, decide_signal,
                                 forward_returns, ic_series, ic_summary,
                                 newey_west_tstat, per_stock_ic,
                                 quantile_returns, random_ic_null)


# --------------------------------------------------------------------------
# 合成資料
# --------------------------------------------------------------------------
def _price(dates, opens, closes=None) -> pd.DataFrame:
    closes = closes if closes is not None else opens
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": np.asarray(opens, dtype=float),
        "high": np.asarray(opens, dtype=float) * 1.01,
        "low": np.asarray(opens, dtype=float) * 0.99,
        "close": np.asarray(closes, dtype=float),
        "volume": 1_000_000.0,
    })


def _signal_frame(rows) -> pd.DataFrame:
    """rows: list of (date, stock_id, signal_value, fwd_5)。"""
    df = pd.DataFrame(rows, columns=["date", "stock_id", "sig", "fwd_5"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------
# 1. 訊號時間與前瞻報酬的對齊
# --------------------------------------------------------------------------
def test_forward_return_starts_at_next_open_not_signal_close():
    """T+1 開盤起算；若誤用 T 日收盤，答案會完全不同。"""
    dates = pd.bdate_range("2025-01-01", periods=7)
    opens = [10, 20, 30, 40, 50, 60, 70]
    closes = [100, 200, 300, 400, 500, 600, 700]     # 刻意與 open 脫鉤
    fwd = forward_returns({"AAA": _price(dates, opens, closes)},
                          horizons=(1, 5))
    r = fwd.set_index("date")
    i0 = dates[0]
    # fwd_1 = open[2]/open[1] - 1 = 30/20 - 1
    assert r.loc[i0, "fwd_1"] == pytest.approx(30 / 20 - 1)
    # 若誤用 close(T)/close(T+1) 會得到 200/100-1 = 1.0
    assert r.loc[i0, "fwd_1"] != pytest.approx(1.0)
    # fwd_5 = open[6]/open[1] - 1 = 70/20 - 1
    assert r.loc[i0, "fwd_5"] == pytest.approx(70 / 20 - 1)


def test_forward_return_tail_is_missing_not_fabricated():
    dates = pd.bdate_range("2025-01-01", periods=7)
    fwd = forward_returns({"AAA": _price(dates, np.arange(10, 80, 10),
                                         np.arange(10, 80, 10))},
                          horizons=(1, 5))
    r = fwd.set_index("date")
    # h=1 需要 i+2 <= 6 → i <= 4；i=5、6 應為缺值
    assert r.loc[dates[5], "fwd_1"] != r.loc[dates[5], "fwd_1"]   # NaN
    assert r.loc[dates[6], "fwd_1"] != r.loc[dates[6], "fwd_1"]
    # h=5 只有 i=0 有效
    assert np.isfinite(r.loc[dates[0], "fwd_5"])
    assert not np.isfinite(r.loc[dates[1], "fwd_5"])


def test_forward_return_horizons_are_distinct():
    dates = pd.bdate_range("2025-01-01", periods=12)
    opens = np.linspace(100, 210, 12)                 # 線性趨勢
    fwd = forward_returns({"AAA": _price(dates, opens, opens)}, horizons=HORIZONS)
    r = fwd.set_index("date")
    a, b, c = r.loc[dates[0], "fwd_1"], r.loc[dates[0], "fwd_5"], r.loc[dates[0], "fwd_10"]
    assert a < b < c                                   # 上漲趨勢下持有越久報酬越大


# --------------------------------------------------------------------------
# 2. 缺值與不可交易股票的排除
# --------------------------------------------------------------------------
def test_non_positive_price_is_missing_not_zero():
    """價格 0 必須變缺值；受影響的窗口一律 NaN，不能變成 0% 報酬。"""
    dates = pd.bdate_range("2025-01-01", periods=5)
    opens = [100.0, 100.0, 0.0, 100.0, 100.0]          # 第 3 列無效
    fwd = forward_returns({"AAA": _price(dates, opens, opens)}, horizons=(1,))
    vals = fwd["fwd_1"].to_numpy(dtype=float)
    assert not np.isfinite(vals[0])     # 出場價落在無效列 → 缺值
    assert not np.isfinite(vals[1])     # 進場價落在無效列 → 缺值
    assert np.isfinite(vals[2]) and vals[2] == pytest.approx(0.0)
    # 若把 0 當成價格，第 0 列會變成假的 -100% 報酬、第 1 列會變成除以 0
    assert vals[0] != pytest.approx(-1.0)
    assert not np.any(np.isinf(vals))


def test_missing_pairs_are_dropped_from_ic():
    rows = [
        ("2025-01-02", "AAA", 1.0, 0.01),
        ("2025-01-02", "BBB", 2.0, 0.02),
        ("2025-01-02", "CCC", 3.0, np.nan),            # 缺未來報酬
        ("2025-01-02", "DDD", np.nan, 0.04),           # 缺訊號
        ("2025-01-02", "EEE", 5.0, 0.05),
        ("2025-01-02", "FFF", 6.0, 0.06),
    ]
    ic = ic_series(_signal_frame(rows), "sig", 5, min_stocks=1)
    assert len(ic) == 1
    assert np.isfinite(ic.iloc[0])


def test_insufficient_names_on_a_day_are_skipped():
    rows = [(f"2025-01-{d:02d}", f"S{i}", float(i), float(i) / 100)
            for d in (2, 3) for i in range(4)]          # 每日僅 4 檔
    ic = ic_series(_signal_frame(rows), "sig", 5, min_stocks=5)
    assert ic.empty


# --------------------------------------------------------------------------
# 3. IC 與 Rank IC 計算正確性
# --------------------------------------------------------------------------
def test_rank_ic_is_exactly_one_for_monotone_relation():
    rows = [("2025-01-02", f"S{i}", float(i), float(i) / 1000) for i in range(6)]
    ic = ic_series(_signal_frame(rows), "sig", 5, method="spearman")
    assert ic.iloc[0] == pytest.approx(1.0)


def test_pearson_ic_is_distorted_by_outlier_but_rank_ic_is_not():
    sig = [1.0, 2.0, 3.0, 4.0, 5.0]
    ret = [0.01, 0.02, 0.03, 0.04, 50.0]               # 極端離群值
    rows = [("2025-01-02", f"S{i}", sig[i], 0.5 + ret[i]) for i in range(5)]
    sf = _signal_frame(rows)
    rank_ic = ic_series(sf, "sig", 5, method="spearman").iloc[0]
    pear_ic = ic_series(sf, "sig", 5, method="pearson").iloc[0]
    assert rank_ic == pytest.approx(1.0)
    assert pear_ic < 0.9


def test_ic_is_negative_when_signal_is_inverted():
    rows = [("2025-01-02", f"S{i}", float(5 - i), float(i) / 1000)
            for i in range(6)]
    ic = ic_series(_signal_frame(rows), "sig", 5)
    assert ic.iloc[0] == pytest.approx(-1.0)


def test_ic_observation_unit_is_the_day_not_the_stock():
    """同一天 3 檔重複 → 觀察單位仍是『一天』，不是 3 個獨立觀察值。"""
    rows = [("2025-01-02", f"S{i}", float(i), float(i) / 1000) for i in range(5)]
    ic = ic_series(_signal_frame(rows), "sig", 5)
    assert len(ic) == 1


def test_ic_summary_reports_insufficiency_below_threshold():
    ic = pd.Series([0.1, 0.2, 0.3], index=pd.bdate_range("2025-01-01", periods=3))
    res = ic_summary(ic, "sig", 5)
    assert res.sufficient is False
    assert res.n_days == 3


def test_ic_summary_detects_positive_mean_with_enough_days():
    rng = np.random.default_rng(3)
    ic = pd.Series(rng.normal(0.2, 0.1, 120),
                   index=pd.bdate_range("2025-01-01", periods=120))
    res = ic_summary(ic, "sig", 1, sims=300)
    assert res.sufficient is True
    assert res.ic_mean == pytest.approx(0.2, abs=0.03)
    assert res.pos_ratio > 0.9
    assert res.t_nw > 5


# --------------------------------------------------------------------------
# 4. 統計工具
# --------------------------------------------------------------------------
def test_newey_west_lag0_matches_naive_t():
    rng = np.random.default_rng(11)
    x = rng.normal(0.05, 1.0, 500)
    t_nw, _ = newey_west_tstat(x, lags=0)
    naive = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    assert t_nw == pytest.approx(naive, rel=0.05)


def test_newey_west_returns_nan_for_tiny_input():
    t, se = newey_west_tstat([1.0, 2.0])
    assert math.isnan(t) and math.isnan(se)


def test_newey_west_widens_error_for_autocorrelated_series():
    rng = np.random.default_rng(5)
    n = 600
    e = rng.normal(0, 1, n)
    # 建立高度自相關序列（AR(1)）
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + e[i]
    x = x + 1.0
    t_naive = x.mean() / (x.std(ddof=1) / math.sqrt(n))
    t_nw, _ = newey_west_tstat(x, lags=20)
    assert abs(t_nw) < abs(t_naive) * 0.6      # 自相關會讓 naive t 嚴重高估


def test_block_bootstrap_ci_brackets_constant_series():
    lo, hi = block_bootstrap_ci([0.3] * 50, n_iter=200, block=5)
    assert lo == pytest.approx(0.3)
    assert hi == pytest.approx(0.3)


def test_block_bootstrap_ci_contains_zero_for_noise():
    rng = np.random.default_rng(2)
    lo, hi = block_bootstrap_ci(rng.normal(0, 1, 400), n_iter=400, block=5)
    assert lo < 0 < hi


def test_benjamini_hochberg_adjustment_is_monotone_and_bounded():
    p = [0.001, 0.01, 0.02, 0.5, 0.9]
    bh = benjamini_hochberg(p, alpha=0.05)
    adj = bh["adjusted"]
    assert all(a >= raw - 1e-12 for a, raw in zip(adj, p))
    assert all(0 <= a <= 1 for a in adj)
    assert all(b >= a - 1e-12 for a, b in zip(adj, adj[1:]))
    assert adj[0] == pytest.approx(0.005)
    assert bh["n_rejected"] == 3


def test_benjamini_hochberg_handles_missing_pvalues():
    bh = benjamini_hochberg([0.01, float("nan"), 0.2])
    assert bh["n_tested"] == 2
    assert math.isnan(bh["adjusted"][1])


# --------------------------------------------------------------------------
# 5. 分組測試
# --------------------------------------------------------------------------
def test_quantile_returns_are_ordered_for_monotone_signal():
    rows = [("2025-01-02", f"S{i:02d}", float(i), float(i) / 100)
            for i in range(20)]
    q = quantile_returns(_signal_frame(rows), "sig", 5, n_groups=N_GROUPS)
    assert q.ordered is True
    assert q.monotonicity == pytest.approx(1.0)
    # Q5-Q1 = (17.5 - 1.5)/100
    assert q.spread == pytest.approx((17.5 - 1.5) / 100)


def test_quantile_groups_are_equal_size():
    rows = [("2025-01-02", f"S{i:02d}", float(i), float(i) / 100)
            for i in range(20)]
    q = quantile_returns(_signal_frame(rows), "sig", 5, n_groups=N_GROUPS)
    assert q.group_counts == [4, 4, 4, 4, 4]


def test_quantile_grouping_uses_only_current_day_signal():
    """分組只用當期訊號；把某日訊號整體平移不改變分組結構。"""
    rows = [("2025-01-02", f"S{i:02d}", float(i), float(i) / 100)
            for i in range(20)]
    q1 = quantile_returns(_signal_frame(rows), "sig", 5)
    shifted = [(d, s, v + 1000.0, f) for d, s, v, f in rows]
    q2 = quantile_returns(_signal_frame(shifted), "sig", 5)
    assert q1.group_means == pytest.approx(q2.group_means)


def test_quantile_spread_net_subtracts_roundtrip_cost():
    rows = [("2025-01-02", f"S{i:02d}", float(i), float(i) / 100)
            for i in range(20)]
    q = quantile_returns(_signal_frame(rows), "sig", 5)
    assert q.cost_rate > 0
    assert q.spread_net == pytest.approx(q.spread - q.cost_rate)


def test_quantile_reports_no_result_when_too_few_stocks():
    rows = [("2025-01-02", f"S{i}", float(i), float(i) / 100) for i in range(4)]
    q = quantile_returns(_signal_frame(rows), "sig", 5, n_groups=N_GROUPS)
    assert q.n_days == 0
    assert not q.group_means


# --------------------------------------------------------------------------
# 6. 隨機基準
# --------------------------------------------------------------------------
def _panel_with_signal(n_dates: int = 40, n_stocks: int = 20, seed: int = 1):
    """建立『未來報酬由某隱藏因子決定』的合成資料。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_dates)
    rows = []
    for d in dates:
        latent = rng.normal(0, 1, n_stocks)
        for i in range(n_stocks):
            rows.append((d, f"S{i:02d}", float(latent[i]),
                         float(latent[i]) * 0.01))
    return _signal_frame(rows)


def test_random_null_is_centred_near_zero():
    sf = _panel_with_signal()
    rb = random_ic_null(sf, "sig", 5, n_iter=200, seed=3)
    assert rb.n_days == 40
    assert abs(rb.null_mean) < 0.02


def test_random_null_observed_matches_ic_series_mean():
    sf = _panel_with_signal()
    rb = random_ic_null(sf, "sig", 5, n_iter=100, seed=3)
    ic = ic_series(sf, "sig", 5).mean()
    assert rb.observed == pytest.approx(ic, abs=1e-9)


def test_random_null_rejects_a_perfectly_predictive_signal():
    sf = _panel_with_signal()
    rb = random_ic_null(sf, "sig", 5, n_iter=300, seed=3)
    assert rb.p_value < 0.01            # 完全預測 → 遠在虛無分布之外


def test_random_null_is_calibrated_on_pure_noise():
    """虛無分布必須近似校準：α=5% 時假陽性率應接近 5%，而非大量誤報。"""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-02", periods=40)
    n_panels, rejects = 40, 0
    for k in range(n_panels):
        rows = []
        for d in dates:
            for i in range(20):
                rows.append((d, f"S{i:02d}", float(rng.normal()),
                             float(rng.normal()) * 0.01))
        rb = random_ic_null(_signal_frame(rows), "sig", 5, n_iter=100, seed=k)
        assert abs(rb.null_mean) < 0.05          # 虛無分布中心必須在 0 附近
        if rb.p_value <= 0.05:
            rejects += 1
    # 期望約 2 個（5%）；門檻寬鬆但足以揭發「幾乎全部誤報」的失準
    assert rejects <= 10, f"虛無分布失準：{rejects}/{n_panels} 被誤判為顯著"


# --------------------------------------------------------------------------
# 7. 個股層級 IC
# --------------------------------------------------------------------------
def test_per_stock_ic_distinguishes_stock_level_signs():
    dates = pd.bdate_range("2025-01-02", periods=80)
    rows = []
    for k, d in enumerate(dates):
        for i in range(6):
            v = float(i + k)
            rows.append((d, "UP", v, v / 1000))
            rows.append((d, "DOWN", v, -v / 1000))
    s = per_stock_ic(_signal_frame(rows), "sig", 5, min_obs=60)
    assert s["UP"] > 0.9
    assert s["DOWN"] < -0.9


# --------------------------------------------------------------------------
# 8. 訊號框架與因子一致性
# --------------------------------------------------------------------------
def _panel_row(date, sid, **over):
    base = dict(
        date=date, stock_id=sid, close=100.0, volume=1_000_000.0,
        kline_score=0.2, pat_label="", kd_gold=0, kd_death=0,
        macd_gold=0, macd_death=0, breakout20=0, breakdown20=0,
        mom_20=0.01, mom_60=0.02, ma_alignment=1.0, rsi=50.0,
        vol_ratio=1.0, ret_1d=0.0, inst_net_sum3=1e6, streak_inst_buy=3.0,
        margin_ratio=0.3, rev_yoy=10.0, news_sentiment=0.1,
    )
    base.update(over)
    return base


def test_score_components_reproduce_score_row():
    rows = [_panel_row("2025-01-02", "AAA"),
            _panel_row("2025-01-02", "BBB", kline_score=-0.5, ma_alignment=0.0,
                       rsi=20.0, rev_yoy=-40.0)]
    panel = pd.DataFrame(rows)
    sf = build_signal_frame(panel)
    for _, r in panel.iterrows():
        comp = sig_mod.score_components(r)
        w = sum(sig_mod.WEIGHTS[k] for k in comp)
        manual = sum(sig_mod.WEIGHTS[k] * v for k, v in comp.items()) / w
        expected = sig_mod.score_row(r)[0]
        assert manual == pytest.approx(expected, abs=1e-12)
    assert set(sf.columns) >= set(SIGNAL_KEYS) | {"score"}


def test_missing_factor_column_stays_nan_not_zero():
    panel = pd.DataFrame([_panel_row("2025-01-02", "AAA")]) \
        .drop(columns=["margin_ratio"])
    sf = build_signal_frame(panel)
    assert sf["margin_ratio"].isna().all()


def test_attach_forward_returns_does_not_leak_future_rows():
    dates = pd.bdate_range("2025-01-02", periods=8)
    price = {"AAA": _price(dates, np.arange(10, 90, 10), np.arange(10, 90, 10))}
    sf = pd.DataFrame({"date": dates, "stock_id": "AAA",
                       "sig": np.arange(8, dtype=float)})
    out = attach_forward_returns(sf, price, horizons=(1,))
    # 每一列仍對應自己的訊號日，沒有被對齊到別的日期
    assert out["date"].tolist() == list(dates)
    assert out["sig"].tolist() == list(np.arange(8, dtype=float))
    assert out["fwd_1"].notna().sum() == 6


# --------------------------------------------------------------------------
# 9. 降級與判定
# --------------------------------------------------------------------------
def test_decide_signal_marks_insufficient_sample():
    spec = SignalSpec("sig", "測試", 0.1, "", "", "", "")
    ic = ic_summary(pd.Series([0.1] * 5), "sig", 5)
    verdict, why = decide_signal(spec, ic, None, None, float("nan"))
    assert verdict == "樣本不足"
    assert str(MIN_DAYS_FOR_IC) in why


def test_decide_signal_unfavorable_when_costs_kill_the_spread():
    spec = SignalSpec("sig", "測試", 0.1, "", "", "", "")
    ic = ic_summary(pd.Series([0.05] * 120), "sig", 5, sims=100)
    q = quantile_returns(_signal_frame(
        [("2025-01-02", f"S{i:02d}", float(i), float(i) / 1e6)
         for i in range(20)]), "sig", 5)
    rb = random_ic_null(_panel_with_signal(), "sig", 5, n_iter=50, seed=1)
    verdict, why = decide_signal(spec, ic, q, rb, 0.5)
    assert verdict in ("未獲支持", "樣本不足")
    assert "扣成本" in why or "FDR" in why


# --------------------------------------------------------------------------
# 10. 端到端（合成資料，monkeypatch 載入路徑）
# --------------------------------------------------------------------------
def _e2e_inputs(n_dates: int = 120, n_stocks: int = 8, seed: int = 4):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_dates)
    price_panel, rows = {}, []
    latent = rng.normal(0, 0.01, (n_dates, n_stocks))
    for i in range(n_stocks):
        sid = f"S{i:02d}"
        opens = 100 * np.cumprod(1 + rng.normal(0, 0.01, n_dates))
        price_panel[sid] = _price(dates, opens, opens)
        for k, d in enumerate(dates):
            rows.append(_panel_row(
                d, sid, mom_20=float(latent[k, i]),
                kline_score=float(np.clip(latent[k, i] * 20, -1, 1)),
                news_sentiment=float(latent[k, i])))
    return pd.DataFrame(rows), price_panel


def test_run_diagnostics_end_to_end_on_synthetic_data(monkeypatch, tmp_path):
    from twstock import diagnostics as dg
    from twstock import validation as val

    panel, price_panel = _e2e_inputs()
    monkeypatch.setattr(val, "load_db_inputs",
                        lambda *a, **k: (panel, price_panel, pd.DataFrame()))
    rep = dg.run_diagnostics(horizons=(5,), sims=50)
    assert rep.n_stocks == 8
    assert len(rep.ic_all) == len(SIGNAL_KEYS) + 1      # 含合成 score
    text = rep.to_text()
    for section in ("[1] 研究對象與定義", "[2] IC 與 Rank IC", "[4] 分組測試",
                    "[5] 隨機排序基準", "[8] 多重檢定校正", "[11] 結論限制"):
        assert section in text
    assert "不證明因果" in text


def test_report_marks_no_conclusion_when_data_insufficient(monkeypatch):
    from twstock import diagnostics as dg
    from twstock import validation as val

    panel, price_panel = _e2e_inputs(n_dates=12, n_stocks=6, seed=8)
    monkeypatch.setattr(val, "load_db_inputs",
                        lambda *a, **k: (panel, price_panel, pd.DataFrame()))
    rep = dg.run_diagnostics(horizons=(5,), sims=30)
    assert all(not r.sufficient for r in rep.ic_all)
    assert all(v == "樣本不足" for _k, v, _w in rep.decisions)
