"""P2.9 空頭市況壓力測試：期間辨識、對照基準、時間對齊與隔離要求。

測試全程**不連外、不讀寫正式資料庫**：所有資料都由本檔自行合成並直接傳入，
`test_run_stress_never_touches_network_or_real_db` 另外以 monkeypatch 確保
壓力測試流程不會發出任何 HTTP 請求。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twstock import stress
from twstock.config import CONFIG
from twstock.costs import get_cost_model
from twstock.strategies import baseline

START = "2024-01-01"


def _dates(n: int, start: str = START):
    return pd.bdate_range(start, periods=n)


def _series(values, start: str = START) -> pd.Series:
    v = np.asarray(values, dtype=float)
    return pd.Series(v, index=pd.DatetimeIndex(_dates(len(v), start)))


def _wide_series(values, start: str = START) -> pd.Series:
    """稀疏日期序列（每個值間隔 7 個日曆日），用來檢查期距計算。"""
    v = np.asarray(values, dtype=float)
    idx = pd.DatetimeIndex([pd.Timestamp(start) + pd.Timedelta(days=7 * i)
                            for i in range(len(v))])
    return pd.Series(v, index=idx)


def _price_frame(close, start: str = START) -> pd.DataFrame:
    """合成日 K：``open[i] = close[i-1]``，讓 T+1 開盤進場價格可被獨立驗算。"""
    c = np.asarray(close, dtype=float)
    open_ = np.concatenate([[c[0]], c[:-1]])
    return _price_frame_from(c, open_, start)


def _price_frame_from(close, open_, start: str = START) -> pd.DataFrame:
    c = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "date": _dates(len(c), start), "open": np.asarray(open_, dtype=float),
        "high": c * 1.01, "low": c * 0.99, "close": c,
        "volume": np.full(len(c), 1_000_000.0),
    })


# --------------------------------------------------------------------------
# 期間辨識
# --------------------------------------------------------------------------
def test_drawdown_episode_peak_trough_and_recovery():
    close = np.concatenate([np.linspace(100, 120, 40),   # 前高 120
                            np.linspace(120, 96, 12),    # 下跌 20%
                            np.linspace(96, 121, 20),    # 復原到 121
                            np.linspace(121, 130, 10)])
    eps = stress.drawdown_episodes(_series(close))
    assert len(eps) == 1, "單一非重疊回撤事件"
    p = eps[0]
    assert p.peak_value == pytest.approx(120.0)
    assert p.trough_value == pytest.approx(96.0)
    assert p.max_drawdown == pytest.approx(-0.20, abs=1e-9)
    assert p.peak_date < p.trough_date < p.recovery_date
    assert p.severe is True


def test_drawdown_tier_boundary_is_exclusive_of_severe():
    close = np.concatenate([np.linspace(100, 100, 20), np.linspace(100, 88, 6),
                            np.linspace(88, 101, 10)])
    eps = stress.drawdown_episodes(_series(close))
    assert len(eps) == 1
    p = eps[0]
    assert p.max_drawdown < -0.10
    assert p.severe is False, "12% 回撤屬 ≥10% 但不屬 ≥15%"
    assert stress.drawdown_episodes(_series(close), min_tier=0.15) == []


def test_drawdown_episodes_are_disjoint():
    close = np.concatenate([np.linspace(100, 130, 40), np.linspace(130, 100, 15),
                            np.linspace(100, 140, 25), np.linspace(140, 110, 12),
                            np.linspace(110, 150, 20)])
    eps = stress.drawdown_episodes(_series(close))
    assert len(eps) >= 2
    for a, b in zip(eps, eps[1:]):
        assert a.trough_date < b.peak_date, "非重疊：前一段低點必早於後一段前高"


def test_detection_is_prefix_stable_with_future_data():
    """手動加入未來資料不得改變既有期間（期間辨識不得使用未來資訊）。"""
    base = np.concatenate([np.linspace(100, 120, 30), np.linspace(120, 90, 10),
                           np.linspace(90, 95, 10)])
    first = stress.drawdown_episodes(_series(base))
    fut = np.concatenate([base, np.linspace(95, 400, 30)])
    after = stress.drawdown_episodes(_series(fut))
    assert len(first) >= 1
    assert [(p.peak_date, p.trough_date, p.max_drawdown) for p in first] == \
        [(p.peak_date, p.trough_date, p.max_drawdown)
         for p in after[:len(first)]], "既有期間不得因後來的大漲而被改寫"


def test_unrecovered_drawdown_is_flagged():
    close = np.concatenate([np.linspace(100, 120, 20), np.linspace(120, 90, 10)])
    p = stress.drawdown_episodes(_series(close))[0]
    assert p.recovery_date is None
    assert "尚未回到前高" in p.note
    assert p.episode_end == _series(close).index[-1]


def test_detection_handles_degenerate_input():
    assert stress.drawdown_episodes(pd.Series(dtype=float)) == []
    assert stress.drawdown_episodes(_series([100.0, 100.0])) == []
    flat = stress.drawdown_episodes(_series(np.full(50, 100.0)))
    assert flat == [], "完全無波動時沒有回撤事件"
    assert stress.drawdown_episodes(_series([np.nan, 100.0, np.nan])) == []


def test_mask_episodes_filters_short_runs_and_requires_alignment():
    idx = pd.DatetimeIndex(_dates(20))
    mask = np.zeros(20, dtype=bool)
    mask[2:3] = True           # 1 天 → 太短
    mask[5:12] = True          # 7 天 → 保留
    got = stress.mask_episodes(idx, mask, kind="volatility", label="v")
    assert len(got) == 1
    assert got[0].n_dates == 7
    assert got[0].start == idx[5] and got[0].end == idx[11]
    with pytest.raises(ValueError):
        stress.mask_episodes(idx, mask[:5], kind="volatility", label="v")


def test_mask_episodes_severity_is_max_within_run():
    idx = pd.DatetimeIndex(_dates(10))
    mask = np.zeros(10, dtype=bool)
    mask[2:8] = True           # 6 天 ≥ MIN_MASK_DAYS
    sev = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.1, 0.1, 0.1, 0.1])
    got = stress.mask_episodes(idx, mask, kind="volatility", label="v",
                              severity=sev)
    assert len(got) == 1 and got[0].n_dates == 6
    assert got[0].severity == pytest.approx(0.9)


def test_volatility_and_regime_series_math():
    close = _series(np.full(30, 100.0))
    vol = stress.volatility_series(close, window=5)
    assert vol.iloc[:5].isna().all()
    assert vol.iloc[-1] == pytest.approx(0.0, abs=1e-12), "無波動時波動率為 0"
    moving = _series(np.linspace(100, 130, 30))
    rf = stress.regime_frame(moving, windows=(20,))
    assert rf["ret20"].iloc[-1] == pytest.approx(130 / moving.iloc[-21] - 1)


def test_identify_periods_reports_all_four_kinds_and_nested_tiers():
    close = np.concatenate([np.linspace(100, 150, 60), np.linspace(150, 100, 20),
                            np.linspace(100, 160, 40)])
    bench = pd.DataFrame({"date": _dates(len(close)), "close": close})
    got = stress.identify_periods(bench)
    assert set(got) == {"drawdown", "drawdown_severe", "volatility", "bear_regime"}
    keys = {(p.peak_date, p.trough_date) for p in got["drawdown_severe"]}
    wider = {(p.peak_date, p.trough_date) for p in got["drawdown"]}
    assert keys <= wider, "≥15% 必須是 ≥10% 的子集合（不得當成獨立樣本）"


# --------------------------------------------------------------------------
# 基準拆解
# --------------------------------------------------------------------------
def test_bench_buy_hold_computes_return_and_drawdown():
    b = pd.DataFrame({"date": _dates(3), "close": [100.0, 80.0, 110.0]})
    ret, mdd = stress.bench_buy_hold(b)
    assert ret == pytest.approx(0.10)
    assert mdd == pytest.approx(-0.20)


def test_bench_buy_hold_degenerate_returns_nan():
    for bad in (None, pd.DataFrame(), pd.DataFrame({"date": _dates(2), "close": [1.0, np.nan]})):
        ret, mdd = stress.bench_buy_hold(bad)
        assert np.isnan(ret) and np.isnan(mdd)


def test_equal_exposure_matches_constant_weight_replication():
    eq = _series([100.0, 110.0, 105.0, 120.0])
    x = 0.4
    expected = float(np.prod(1 + x * eq.pct_change().dropna().to_numpy()) - 1)
    assert stress.equal_exposure_return(eq, x) == pytest.approx(expected)


def test_equal_exposure_limits():
    eq = _series([100.0, 110.0, 90.0, 120.0])
    assert stress.equal_exposure_return(eq, 0.0) == 0.0, "曝險 0 = 現金基準"
    assert stress.equal_exposure_return(eq, 1.0) == pytest.approx(
        eq.iloc[-1] / eq.iloc[0] - 1), "曝險 100% = 基準買進持有"
    assert np.isnan(stress.equal_exposure_return(_series([100.0]), 0.5))


def test_cagr_is_withheld_for_short_windows():
    one_year = 0.10
    assert stress._cagr(one_year, 1.0) == pytest.approx(0.10)
    assert np.isnan(stress._cagr(0.05, 0.2)), "2.4 個月的報酬不得年化"
    assert np.isnan(stress._cagr(float("nan"), 2.0))
    assert np.isnan(stress._cagr(-1.5, 2.0)), "總虧損超過 100% 不得年化"


def test_worst_k_uses_realised_equity_only():
    eq = _series([100.0, 110.0, 99.0, 90.0, 120.0, 130.0])
    assert stress._worst_k(eq, 1) == pytest.approx(99 / 110 - 1)
    assert stress._worst_k(eq, 2) == pytest.approx(90 / 110 - 1)
    assert np.isnan(stress._worst_k(eq, 10)), "天數不足回 N/A，不以部分期間冒充"


# --------------------------------------------------------------------------
# 視窗評估：時間對齊、前視偏差、覆蓋與 N/A
# --------------------------------------------------------------------------
def _synth_scores(price_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """以實際收盤動能造出每日都有變化的分數（只用 T 日（含）以前的價格）。"""
    rows = []
    for sid, df in price_panel.items():
        close = df["close"].to_numpy(dtype=float)
        mom = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
        rows.append(pd.DataFrame({"date": df["date"], "stock_id": sid,
                                  "score": mom * 100.0}))
    return pd.concat(rows, ignore_index=True)


def _synthetic_market(n_days=90, n_stocks=8, seed=0, drop=(30, 50)):
    rng = np.random.default_rng(seed)
    panel = {}
    for k in range(n_stocks):
        r = rng.normal(0.0004, 0.012, n_days)
        r[drop[0]:drop[1]] -= 0.02
        panel[f"9{k:03d}"] = _price_frame(100 * np.cumprod(1 + r))
    bench_close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.010, n_days))
    return panel, pd.DataFrame({"date": _dates(n_days), "close": bench_close})


def test_window_evaluation_is_immune_to_future_data():
    panel, bench = _synthetic_market()
    scores = _synth_scores(panel)
    spec, cm = baseline(), get_cost_model("typical")
    lo, hi = _dates(90)[5], _dates(90)[55]
    before = stress.evaluate_window(scores, panel, bench, spec, cm, lo, hi,
                                    n_sims=0)
    # 視窗之後把價格乘以 5：若回測偷看未來，結果必然改變
    mutated = {sid: df.copy() for sid, df in panel.items()}
    for sid, df in mutated.items():
        df.loc[df["date"] > hi, ["open", "close", "high", "low"]] *= 5.0
    after = stress.evaluate_window(scores, mutated, bench, spec, cm, lo, hi,
                                   n_sims=0)
    assert after.strat_total == pytest.approx(before.strat_total)
    assert after.strat_mdd == pytest.approx(before.strat_mdd)
    assert after.n_trades == before.n_trades


def test_entry_uses_next_open_not_signal_day_close():
    panel, bench = _synthetic_market(n_stocks=2)
    scores = _synth_scores(panel)
    # 分數只在第一個交易日觸發，方便定位唯一一筆交易
    first = _dates(90)[0]
    one = scores[scores["date"] == first].copy()
    one["score"] = 1.0
    cm = get_cost_model("zero")
    res = stress.evaluate_window(one, panel, bench, baseline(), cm,
                                 first, _dates(90)[40], n_sims=0,
                                 run_gross=False)
    assert res.n_trades >= 1
    # 驗證方式：以相同分數直接跑一次回測，檢查進場價等於 T+1 開盤
    from twstock.backtest import run_backtest
    raw = run_backtest(one, panel, strategy=baseline(), cost_model=cm)
    assert not raw.trades.empty
    for t in raw.trades.itertuples(index=False):
        sid = t.stock_id
        df = panel[sid]
        entry_open = float(df.loc[df["date"] == t.entry_date, "open"].iloc[0])
        assert t.entry_price == pytest.approx(entry_open)
        assert t.entry_date > t.signal_date, "T 日收盤訊號不得當日成交"


def test_window_without_stock_data_is_na_and_keeps_benchmark():
    panel, bench = _synthetic_market()
    scores = _synth_scores(panel)
    early_lo, early_hi = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")
    # 基準補上該段資料，模擬「基準歷史比個股長」
    b = pd.concat([pd.DataFrame({"date": pd.DatetimeIndex(_dates(20, "2020-01-01")),
                                 "close": np.linspace(100, 90, 20)}), bench],
                  ignore_index=True)
    b = b.sort_values("date").reset_index(drop=True)
    w = stress.evaluate_window(scores, panel, b, baseline(),
                               get_cost_model("typical"), early_lo, early_hi,
                               n_sims=0, run_gross=False)
    assert np.isnan(w.strat_total), "無資料不得回報 0% 報酬"
    assert w.bench_total == pytest.approx(-0.10, abs=0.02), "基準仍應如實計算"
    assert w.n_trades == 0 and w.sufficient is False
    assert "未涵蓋" in w.note


def test_tail_signals_are_excluded_not_silently_traded():
    n = 60
    close = 100 * np.cumprod(1 + np.full(n, 0.001))
    panel = {"9001": _price_frame(close)}
    bench = pd.DataFrame({"date": _dates(n), "close": close})
    scores = pd.DataFrame({"date": _dates(n), "stock_id": "9001",
                           "score": np.full(n, 1.0)})
    spec = baseline()
    w = stress.evaluate_window(scores, panel, bench, spec,
                               get_cost_model("typical"), _dates(n)[0],
                               _dates(n)[-1], n_sims=0, run_gross=False)
    assert w.n_score_rows == n
    # 持有 hold_days 日：最後 hold_days+1 個訊號無法在視窗內完成一回合
    assert w.n_excluded_tail == spec.hold_days + 1
    assert w.n_evaluable_signals == n - (spec.hold_days + 1)
    assert w.n_stocks == 1, "基準不得被計入股票檔數"


def test_random_control_keeps_rule_and_is_reproducible():
    panel, bench = _synthetic_market()
    scores = _synth_scores(panel)
    args = dict(n_sims=8, seed=3, run_gross=False)
    a = stress.evaluate_window(scores, panel, bench, baseline(),
                               get_cost_model("typical"), _dates(90)[5],
                               _dates(90)[55], **args)
    b = stress.evaluate_window(scores, panel, bench, baseline(),
                               get_cost_model("typical"), _dates(90)[5],
                               _dates(90)[55], **args)
    assert a.rand_median == pytest.approx(b.rand_median), "同 seed 必須可重現"
    assert a.n_rand == 8 and 0 <= a.rand_percentile <= 100


# --------------------------------------------------------------------------
# 訊號檢驗
# --------------------------------------------------------------------------
def _signal_frame(n_days=120, n_stocks=8, seed=7):
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    rows = []
    for k in range(n_stocks):
        rows.append(pd.DataFrame({
            "date": dates, "stock_id": f"9{k:03d}",
            "score": rng.normal(0, 1, n_days),
            "rev_yoy": rng.normal(0, 1, n_days),
            "mom_20": rng.normal(0, 1, n_days),
            "news_sentiment": np.zeros(n_days),
            "fwd_5": rng.normal(0, 0.02, n_days),
        }))
    return pd.concat(rows, ignore_index=True)


def test_signal_checks_respect_date_mask_not_span():
    sf = _signal_frame()
    cal = pd.DatetimeIndex(sorted(sf["date"].unique()))
    mask = stress.date_mask(cal, lo=cal[20], hi=cal[59])
    got = stress.signal_checks(sf, [("片段", mask)], signals=("score",))
    assert len(got) == 1
    assert got[0].n_days == 40, "遮罩選 40 天就只能用 40 天，不可取整段區間"
    # 非連續遮罩：只挑第 20~29 與 40~49 天
    jump = stress.date_mask(cal, lo=cal[20], hi=cal[29]) | \
        stress.date_mask(cal, lo=cal[40], hi=cal[49])
    sparse = stress.signal_checks(sf, [("跳著", jump)], signals=("score",))[0]
    assert sparse.n_days == 20, "期間之間的正常交易日不得被算進壓力日"


def test_date_mask_and_empty_segments_are_safe():
    cal = pd.DatetimeIndex(_dates(10))
    assert int(stress.date_mask(cal).sum()) == 10
    assert int(stress.date_mask(cal, lo=cal[5]).sum()) == 5
    empty = pd.Series(False, index=cal)
    got = stress.signal_checks(_signal_frame(30), [("空白", empty)],
                               signals=("score",))
    assert got[0].n_days == 0
    assert np.isnan(got[0].ic_mean)


def test_missing_signal_column_reports_na_not_zero():
    sf = _signal_frame(40)[["date", "stock_id", "score", "fwd_5"]]
    got = stress.signal_checks(sf, [("全部", stress.date_mask(
        pd.DatetimeIndex(sorted(sf["date"].unique()))))], signals=("rev_yoy",))
    assert got[0].n_days == 0 and np.isnan(got[0].ic_mean)


def test_news_factor_status_reports_zero_coverage():
    sf = _signal_frame(30)
    cal = pd.DatetimeIndex(sorted(sf["date"].unique()))
    txt = stress.news_factor_status(sf, [("全部", stress.date_mask(cal))])
    assert "非零 0/240" in txt


# --------------------------------------------------------------------------
# 覆蓋報告與整體流程
# --------------------------------------------------------------------------
def test_coverage_excludes_benchmark_from_stock_dates():
    panel, bench = _synthetic_market(n_days=90)
    panel[CONFIG.benchmark] = _price_frame(np.linspace(100, 120, 90))
    # 所有個股都晚 10 天才開始；基準（0050）從第一天就有 → 不得算進個股起始日
    panel = {sid: (df.loc[10:].reset_index(drop=True)
                   if sid != CONFIG.benchmark else df)
             for sid, df in panel.items()}
    long_panel = pd.concat(
        [df.assign(stock_id=sid) for sid, df in panel.items()
         if sid != CONFIG.benchmark], ignore_index=True)
    cov = stress.coverage_report(panel, long_panel, bench)
    assert cov.pool_size == len(CONFIG.pool)
    assert cov.stock_start == _dates(90)[10], "基準較長的歷史不得算進個股起始日"
    assert cov.stock_end == _dates(90)[-1]
    # 基準領先的那 10 個交易日只能靠基準交易 → 必須如實列為歷史缺口
    assert cov.n_bench_only_dates == 10
    assert cov.min_stocks_per_day == 8, "每日有效檔數應為實際股票數"


def test_coverage_detects_bad_prices():
    df = _price_frame(np.linspace(100, 110, 20))
    df.loc[3, "high"] = 1.0            # high < low
    df.loc[5, "volume"] = 0.0
    df.loc[7, "close"] = 0.0
    cov = stress.coverage_report({"9001": df}, pd.DataFrame(), pd.DataFrame())
    assert cov.n_high_lt_low >= 1
    assert cov.n_zero_volume >= 1
    assert cov.n_close_outside + cov.n_nonpositive >= 1


def test_run_stress_never_touches_network_or_real_db(monkeypatch):
    """壓力測試不得連外，也不得對正式資料庫發出寫入。"""
    import requests

    def _no_network(*a, **kw):
        raise AssertionError("壓力測試不得發出 HTTP 請求")

    monkeypatch.setattr(requests.sessions.Session, "request", _no_network)

    from twstock.data import store
    from twstock import validation

    writes = []

    def _guard_write(*a, **kw):
        writes.append((a, kw))
        raise AssertionError("壓力測試不得寫入資料庫")

    for name in ("write_df", "set_meta", "upsert", "insert_df"):
        if hasattr(store, name):
            monkeypatch.setattr(store, name, _guard_write)

    n = 120
    rng = np.random.default_rng(11)
    panel = {}
    for k in range(8):
        r = rng.normal(0.0003, 0.012, n)
        r[30:45] -= 0.02
        panel[f"9{k:03d}"] = _price_frame(100 * np.cumprod(1 + r))
    bench_close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.010, n))
    bench = pd.DataFrame({"date": _dates(n), "close": bench_close})
    price_all = dict(panel)
    price_all[CONFIG.benchmark] = _price_frame(bench_close)
    feat = _synth_scores(panel)
    feat["mom_60"] = np.nan
    sig_frame = feat.assign(news_sentiment=0.0)

    monkeypatch.setattr(validation, "load_db_inputs",
                        lambda *a, **kw: (feat, price_all, bench))
    # 特徵面板 → 訊號框架的轉換在此替換掉（本測試只驗證 stress 模組本身）
    monkeypatch.setattr(stress, "build_signal_frame",
                        lambda p: sig_frame.copy())

    rep = stress.run_stress(n_sims=5)
    text = rep.to_text()
    assert writes == []
    assert "空頭市況壓力測試報告（P2.9）" in text
    for section in ("[1] 資料可行性", "[2] 壓力期間辨識", "[3] 策略 vs 基準",
                    "[5] 回撤、波動與交易統計", "[6] 空頭期間訊號檢驗",
                    "[7] 已確認結論", "[8] 下一步建議"):
        assert section in text, f"缺少段落 {section}"
    assert "nan" not in text.lower(), "報告不得輸出 nan"
    assert "nan%" not in text.lower()
    assert rep.conclusions and rep.next_steps and rep.notes
    assert any(w.kind == "drawdown" for w in rep.windows)
    assert all(w.n_stocks <= 8 for w in rep.windows)


def test_report_to_text_never_emits_nan_on_empty_analysis():
    rep = stress.StressReport(
        generated_at="2026-01-01T00:00:00", spec=baseline(),
        cost_model_name="typical", coverage=stress.Coverage(), periods={},
        windows=[stress.WindowResult(
            label="x", kind="drawdown", start=None, end=None, n_dates=0,
            years=float("nan"), strat_total=float("nan"),
            bench_total=float("nan"), excess=float("nan"))],
        signals=[stress.SignalCheck(signal="score", segment="s", horizon=5)])
    rep.notes = ["n"]
    rep.conclusions = ["c"]
    rep.next_steps = ["s"]
    text = rep.to_text()
    assert "nan" not in text.lower()


def test_cli_module_entrypoint_has_main():
    assert callable(stress.main)
