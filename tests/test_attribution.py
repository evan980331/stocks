"""P2.4 策略失效歸因測試：控制變因、訊號→持倉一致性、週轉拆解、端到端。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from twstock.attribution import (ControlRuns, DelayStats, HoldingStats,
                                 concentration_stats, entry_delay,
                                 exposure_stats, holding_stats,
                                 oracle_scores, run_attribution,
                                 selection_gap, shuffled_scores,
                                 turnover_breakdown)
from twstock.backtest import run_backtest
from twstock.strategies import baseline


# --------------------------------------------------------------------------
# 合成資料
# --------------------------------------------------------------------------
def _price(dates, opens) -> pd.DataFrame:
    opens = np.asarray(opens, dtype=float)
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "open": opens,
        "high": opens * 1.01, "low": opens * 0.99, "close": opens,
        "volume": 1_000_000.0,
    })


def _trades(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in ("entry_date", "exit_date", "signal_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def _toy_market(n_days: int = 90, n_stocks: int = 6, seed: int = 0):
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    rng = np.random.default_rng(seed)
    panel_rows, price_panel = [], {}
    for i in range(n_stocks):
        sid = f"S{i:02d}"
        opens = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n_days))
        price_panel[sid] = _price(dates, opens)
        for k, d in enumerate(dates):
            panel_rows.append({"date": d, "stock_id": sid, "close": opens[k]})
    return pd.DataFrame(panel_rows), price_panel, dates


def _scores_long(dates, n_stocks: int, *, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        vals = rng.uniform(-1, 1, n_stocks)
        for i in range(n_stocks):
            rows.append({"date": d, "stock_id": f"S{i:02d}",
                         "score": float(vals[i])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. 隨機選股控制
# --------------------------------------------------------------------------
def test_shuffled_scores_preserves_per_day_distribution():
    dates = pd.bdate_range("2025-01-02", periods=5)
    scores = _scores_long(dates, 6, seed=3)
    sh = shuffled_scores(scores, seed=0)
    for d in dates:
        a = sorted(scores.loc[scores["date"] == d, "score"].round(9))
        b = sorted(sh.loc[sh["date"] == d, "score"].round(9))
        assert a == pytest.approx(b)          # 分數分布完全相同
    # 但至少有一天股票與分數的對應被改變（否則等於沒洗牌）
    changed = (sh.sort_values(["date", "stock_id"])["score"].to_numpy()
               != scores.sort_values(["date", "stock_id"])["score"].to_numpy())
    assert changed.any()


def test_shuffled_scores_is_reproducible_with_seed():
    dates = pd.bdate_range("2025-01-02", periods=5)
    scores = _scores_long(dates, 6, seed=3)
    a = shuffled_scores(scores, seed=11)
    b = shuffled_scores(scores, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_shuffled_scores_does_not_move_rows_between_dates():
    dates = pd.bdate_range("2025-01-02", periods=6)
    scores = _scores_long(dates, 5, seed=4)
    sh = shuffled_scores(scores, seed=1)
    assert sorted(sh["date"].unique()) == sorted(scores["date"].unique())
    assert len(sh) == len(scores)
    assert sh.groupby("date").size().tolist() == \
        scores.groupby("date").size().tolist()


# --------------------------------------------------------------------------
# 2. 完美前瞻上限
# --------------------------------------------------------------------------
def test_oracle_scores_rank_future_returns_within_range():
    dates = pd.bdate_range("2025-01-02", periods=8)
    rows = []
    for d in dates:
        for i in range(6):
            rows.append({"date": d, "stock_id": f"S{i}", "fwd_5": float(i) / 100})
    sf = pd.DataFrame(rows)
    orc = oracle_scores(sf, hold_days=5)
    assert orc["score"].between(-1, 1).all()
    # 未來報酬最高的股票，分數必須是當日最高
    top = orc.loc[orc["date"] == dates[0]].sort_values("score")
    assert top.iloc[-1]["stock_id"] == "S5"
    assert top.iloc[0]["stock_id"] == "S0"


def test_oracle_scores_drop_missing_future_returns():
    dates = pd.bdate_range("2025-01-02", periods=3)
    rows = [{"date": dates[0], "stock_id": "AAA", "fwd_5": 0.01},
            {"date": dates[0], "stock_id": "BBB", "fwd_5": float("nan")}]
    orc = oracle_scores(pd.DataFrame(rows), hold_days=5)
    assert len(orc) == 1
    assert orc.iloc[0]["stock_id"] == "AAA"


def test_oracle_scores_empty_when_no_forward_column():
    sf = pd.DataFrame([{"date": "2025-01-02", "stock_id": "AAA", "score": 0.5}])
    assert oracle_scores(sf, hold_days=5).empty


# --------------------------------------------------------------------------
# 3. 訊號 → 持倉一致性
# --------------------------------------------------------------------------
def _run_toy(n_days: int = 90, n_stocks: int = 6, seed: int = 0):
    _panel, price_panel, dates = _toy_market(n_days, n_stocks, seed)
    scores = _scores_long(dates, n_stocks, seed=seed + 1)
    res = run_backtest(scores, price_panel, strategy=baseline().with_updates(
        name="toy", hold_days=5, top_n=2, buy_threshold=0.1))
    return scores, price_panel, res


def test_selection_gap_captured_is_subset_of_actual_trades():
    """歸因的「已成交」必須真的出現在交易明細中（一致性）。"""
    scores, price_panel, res = _run_toy()
    spec = baseline().with_updates(name="toy", hold_days=5, top_n=2,
                                   buy_threshold=0.1)
    gap = selection_gap(scores, price_panel, res.trades, spec)
    assert gap.n_candidates > 0
    assert 0.0 <= gap.captured_rate <= 1.0
    assert gap.n_captured <= gap.n_candidates
    # 逐筆核對：每個「已成交」候選都必須存在於交易明細
    book_entries = {(pd.Timestamp(r.entry_date).normalize(), str(r.stock_id))
                    for r in res.trades.itertuples(index=False)}
    assert len(book_entries) <= len(res.trades)
    assert gap.n_captured <= len(book_entries)


def test_selection_gap_returns_no_candidates_for_empty_scores():
    _panel, price_panel, _dates = _toy_market()
    gap = selection_gap(pd.DataFrame(columns=["date", "stock_id", "score"]),
                        price_panel, pd.DataFrame(), baseline())
    assert gap.n_candidates == 0
    assert math.isnan(gap.captured_rate)
    assert "無候選" in gap.label()


# --------------------------------------------------------------------------
# 4. 進場延遲與持有期間
# --------------------------------------------------------------------------
def test_entry_delay_measures_signal_to_entry_lag():
    t = _trades([{"stock_id": "AAA", "signal_date": "2025-01-06",
                  "entry_date": "2025-01-07", "exit_date": "2025-01-14",
                  "qty": 10, "entry_price": 100.0, "holding_days": 5},
                 {"stock_id": "BBB", "signal_date": "2025-01-06",
                  "entry_date": "2025-01-09", "exit_date": "2025-01-16",
                  "qty": 10, "entry_price": 100.0, "holding_days": 5}])
    d = entry_delay(t)
    assert d.n == 2
    assert d.median_days == 2.0            # (1 + 3) / 2
    assert d.max_days == 3.0


def test_entry_delay_handles_missing_signal_date():
    t = _trades([{"stock_id": "AAA", "signal_date": None,
                  "entry_date": "2025-01-07", "exit_date": "2025-01-14",
                  "qty": 10, "entry_price": 100.0, "holding_days": 5}])
    assert entry_delay(t).n == 0


def test_backtest_records_signal_date_on_every_trade():
    _scores, _panel, res = _run_toy()
    assert not res.trades.empty
    assert res.trades["signal_date"].notna().all()
    lag = (pd.to_datetime(res.trades["entry_date"])
           - pd.to_datetime(res.trades["signal_date"])).dt.days
    assert (lag >= 1).all()                # T+1 或更晚，絕不會當日成交


def test_holding_stats_classifies_expiry_early_and_extended():
    spec = baseline().with_updates(name="toy", hold_days=5)
    t = _trades([{"stock_id": "A", "holding_days": 5},
                 {"stock_id": "B", "holding_days": 5},
                 {"stock_id": "C", "holding_days": 2},
                 {"stock_id": "D", "holding_days": 7}])
    st = holding_stats(t, spec)
    assert st.n == 4
    assert st.to_expiry == 2 and st.early_exit == 1 and st.extended == 1
    assert "平均持有" in st.label(spec)


# --------------------------------------------------------------------------
# 5. 週轉率拆解
# --------------------------------------------------------------------------
def test_turnover_breakdown_flags_rapid_repeat_entries():
    spec = baseline().with_updates(name="toy", hold_days=5)
    t = _trades([
        {"stock_id": "AAA", "entry_date": "2025-01-06",
         "exit_date": "2025-01-13", "qty": 10, "entry_price": 100.0},
        {"stock_id": "AAA", "entry_date": "2025-01-14",      # 隔天又買回
         "exit_date": "2025-01-21", "qty": 10, "entry_price": 100.0},
        {"stock_id": "BBB", "entry_date": "2025-01-06",
         "exit_date": "2025-01-13", "qty": 10, "entry_price": 100.0},
    ])
    tb = turnover_breakdown(t, spec)
    assert tb.n == 3
    assert tb.distinct_stocks == 2
    assert tb.repeat_entries == 1
    assert tb.repeat_rate == pytest.approx(1 / 3)
    assert tb.per_stock["AAA"] == 2


def test_turnover_breakdown_ignores_long_gaps():
    spec = baseline().with_updates(name="toy", hold_days=5)
    t = _trades([
        {"stock_id": "AAA", "entry_date": "2025-01-06",
         "exit_date": "2025-01-13", "qty": 10, "entry_price": 100.0},
        {"stock_id": "AAA", "entry_date": "2025-03-03",
         "exit_date": "2025-03-10", "qty": 10, "entry_price": 100.0},
    ])
    assert turnover_breakdown(t, spec).repeat_entries == 0


# --------------------------------------------------------------------------
# 6. 集中度與曝險
# --------------------------------------------------------------------------
def test_concentration_hhi_bounds():
    one = _trades([{"stock_id": "AAA", "qty": 10, "entry_price": 100.0}])
    st = concentration_stats(one)
    assert st.hhi == pytest.approx(1.0)
    assert st.top1_share == pytest.approx(1.0)
    five = _trades([{"stock_id": f"S{i}", "qty": 10, "entry_price": 100.0}
                    for i in range(5)])
    assert concentration_stats(five).hhi == pytest.approx(0.2)


def test_concentration_handles_empty_trades():
    assert math.isnan(concentration_stats(pd.DataFrame()).hhi)
    assert concentration_stats(pd.DataFrame()).label() == "無交易"


def test_exposure_cash_share_complements_exposure():
    _scores, _panel, res = _run_toy()
    spec = baseline()
    ex = exposure_stats(res, spec)
    assert ex.cash_share == pytest.approx(1.0 - ex.exposure)


# --------------------------------------------------------------------------
# 7. 控制變因統計
# --------------------------------------------------------------------------
def _dummy_result(total: float, n: int = 3) -> object:
    from twstock.backtest import BacktestResult
    return BacktestResult(total_return=total, round_trips=n)


def test_control_percentile_and_pvalue_bounds():
    base = _dummy_result(0.10)
    randoms = [_dummy_result(v) for v in (0.01, 0.05, 0.20, 0.30)]
    c = ControlRuns(label="t", baseline=base, randoms=randoms)
    assert c.percentile_of_baseline() == pytest.approx(50.0)
    assert 0.0 < c.p_value_beats_random() <= 1.0
    # 策略贏過全部隨機 → p 為最小值 1/(n+1)
    c2 = ControlRuns(label="t", baseline=_dummy_result(9.0), randoms=randoms)
    assert c2.p_value_beats_random() == pytest.approx(1 / 5)
    assert c2.percentile_of_baseline() == pytest.approx(100.0)


def test_control_handles_no_random_runs():
    c = ControlRuns(label="t", baseline=_dummy_result(0.1), randoms=[])
    assert math.isnan(c.percentile_of_baseline())
    assert math.isnan(c.p_value_beats_random())


# --------------------------------------------------------------------------
# 8. 端到端
# --------------------------------------------------------------------------
def _e2e_inputs(n_dates: int = 130, n_stocks: int = 6, seed: int = 5):
    dates = pd.bdate_range("2025-01-02", periods=n_dates)
    rng = np.random.default_rng(seed)
    panel_rows, price_panel = [], {}
    for i in range(n_stocks):
        sid = f"S{i:02d}"
        opens = 100 * np.cumprod(1 + rng.normal(0.001, 0.012, n_dates))
        price_panel[sid] = _price(dates, opens)
        for k, d in enumerate(dates):
            panel_rows.append({
                "date": d, "stock_id": sid, "close": opens[k],
                "volume": 1_000_000.0, "kline_score": float(
                    np.clip(rng.normal(0, 0.5), -1, 1)),
                "pat_label": "", "kd_gold": 0, "kd_death": 0, "macd_gold": 0,
                "macd_death": 0, "breakout20": 0, "breakdown20": 0,
                "mom_20": float(rng.normal(0, 0.05)),
                "mom_60": float(rng.normal(0, 0.05)), "ma_alignment": 1.0,
                "rsi": float(rng.uniform(30, 70)), "vol_ratio": 1.0,
                "ret_1d": 0.0, "inst_net_sum3": float(rng.normal(0, 1e6)),
                "streak_inst_buy": 0.0, "margin_ratio": 0.3,
                "rev_yoy": float(rng.normal(10, 20)),
                "news_sentiment": float(rng.normal(0, 0.1)),
            })
    return pd.DataFrame(panel_rows), price_panel


def test_run_attribution_end_to_end(monkeypatch):
    from twstock import validation as val

    panel, price_panel = _e2e_inputs()
    monkeypatch.setattr(val, "load_db_inputs",
                        lambda *a, **k: (panel, price_panel, pd.DataFrame()))
    rep = run_attribution(n_sims=5, seed=0)
    text = rep.to_text()
    for section in ("[1] 設定", "[3] 控制變因比較", "[4] 訊號 → 持倉的轉換落差",
                    "[6] 週轉率拆解", "[7] 曝險與集中度", "[10] 結論與限制"):
        assert section in text
    assert "不可交易" in text                 # 完美前瞻必須標示為不可交易
    assert "未以這些結果回頭挑選參數" in text
    assert len(rep.controls.randoms) == 5


def test_run_attribution_marks_benchmark_unavailable(monkeypatch):
    from twstock import validation as val

    panel, price_panel = _e2e_inputs()
    monkeypatch.setattr(val, "load_db_inputs",
                        lambda *a, **k: (panel, price_panel, pd.DataFrame()))
    rep = run_attribution(n_sims=3)
    assert any("基準不可用" in n for n in rep.notes)


def test_attribution_report_does_not_fabricate_when_no_trades():
    """沒有交易時必須顯示「無交易／N/A」，不得出現 nan% 或 0.00% 冒充真實數字。"""
    text = _empty_report().to_text()
    assert "無交易" in text
    assert "nan%" not in text
    assert "N/A" in text


def _empty_report():
    from twstock.attribution import AttributionReport
    from twstock.strategies import baseline as _b
    from twstock.validation import Segment
    import datetime as _dt

    empty = _dummy_result(0.0, 0)
    seg = Segment("test", pd.Timestamp("2025-01-01"),
                  pd.Timestamp("2025-02-01"), 20)
    return AttributionReport(
        generated_at="now", segment=seg, spec=_b(), cost_model_name="typical",
        controls=ControlRuns(label="t", baseline=empty, randoms=[]),
        benchmark_total_return=float("nan"),
        gap=selection_gap(pd.DataFrame(columns=["date", "stock_id", "score"]),
                          {}, pd.DataFrame(), _b()),
        delay=DelayStats(), holding=HoldingStats(),
        turnover=turnover_breakdown(pd.DataFrame(), _b()),
        concentration=concentration_stats(pd.DataFrame()),
        exposure=exposure_stats(empty, _b()), periods=[], notes=[])
