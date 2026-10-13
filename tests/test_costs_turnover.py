"""P2.2：交易成本模型、週轉率定義、成本敏感度與降低換手的策略變體。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twstock.backtest import run_backtest, run_cost_scenarios
from twstock.costs import COST_SCENARIOS, CostModel, get_cost_model
from twstock.strategies import StrategySpec, baseline, comparison_variants


# --------------------------------------------------------------------------
# 成本模型
# --------------------------------------------------------------------------
def test_tax_applies_only_on_sell():
    cm = CostModel(commission_rate=0.0, commission_discount=1.0, tax_rate=0.003,
                   slippage=0.0, min_fee=0.0)
    assert cm.buy_cost(100_000) == pytest.approx(0.0)
    assert cm.sell_cost(100_000) == pytest.approx(300.0)


def test_commission_charged_on_both_sides():
    cm = CostModel(commission_rate=0.001425, commission_discount=1.0,
                   tax_rate=0.0, slippage=0.0, min_fee=0.0)
    assert cm.buy_cost(100_000) == pytest.approx(142.5)
    assert cm.sell_cost(100_000) == pytest.approx(142.5)


def test_minimum_fee_applies_to_small_orders():
    cm = CostModel(commission_rate=0.001425, commission_discount=0.28,
                   tax_rate=0.0, slippage=0.0, min_fee=20.0)
    # 1000 元 * 0.0399% = 0.4 元 → 套用低消 20 元
    assert cm.buy_cost(1_000) == pytest.approx(20.0)


def test_discount_reduces_commission():
    full = CostModel(commission_rate=0.001425, commission_discount=1.0,
                     tax_rate=0.0, slippage=0.0, min_fee=0.0)
    disc = CostModel(commission_rate=0.001425, commission_discount=0.28,
                     tax_rate=0.0, slippage=0.0, min_fee=0.0)
    assert disc.buy_cost(1_000_000) < full.buy_cost(1_000_000)


def test_participation_rate_caps_fill_and_zero_volume_blocks():
    cm = CostModel(participation_rate=0.01, min_fee=0.0)
    assert cm.max_fill_qty(volume=1_000_000, price=10.0, desired_qty=50_000) == 10_000
    assert cm.max_fill_qty(volume=1_000_000, price=10.0, desired_qty=500) == 500
    assert cm.max_fill_qty(volume=0, price=10.0, desired_qty=500) == 0
    assert cm.max_fill_qty(volume=None, price=10.0, desired_qty=500) == 0


def test_cost_scenarios_are_ordered_by_severity():
    order = ["zero", "ideal", "typical", "conservative", "adverse", "severe"]
    costs = [COST_SCENARIOS[n].buy_cost(1_000_000) for n in order]
    assert costs == sorted(costs)
    assert COST_SCENARIOS["zero"].buy_cost(1_000_000) == 0
    assert get_cost_model(None).name == "typical"


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        get_cost_model("nope")


# --------------------------------------------------------------------------
# 合成回測資料
# --------------------------------------------------------------------------
def _prices(days=90, price=100.0, drift=0.2, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    close = np.maximum(price + np.cumsum(rng.normal(drift, 1.0, days)), 1.0)
    return pd.DataFrame({"date": dates, "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 2_000_000.0})


def _scores(dates, sid="AAA", score=0.9):
    return pd.DataFrame({"date": [str(pd.Timestamp(d).date()) for d in dates],
                         "stock_id": sid, "score": score})


# --------------------------------------------------------------------------
# 週轉率定義
# --------------------------------------------------------------------------
def test_turnover_matches_its_documented_formula():
    """turnover = (買進名目 + 賣出名目) / 初始資金，且買賣各計一次。"""
    price = {"AAA": _prices(days=60)}
    dates = price["AAA"]["date"]
    res = run_backtest(_scores(dates), price, top_n=1, hold_days=5,
                       cost_model=COST_SCENARIOS["zero"],
                       initial_capital=1_000_000.0)

    assert not res.trades.empty
    expected_notional = float(
        (res.trades["entry_price"] * res.trades["qty"]
         + res.trades["exit_price"] * res.trades["qty"]).sum())
    assert res.notional_traded == pytest.approx(expected_notional)
    assert res.turnover == pytest.approx(expected_notional / 1_000_000.0)
    assert res.turnover_one_way == pytest.approx(res.turnover / 2)
    assert res.round_trips == len(res.trades)
    assert res.turnover_annualized == pytest.approx(res.turnover / res.years)


def test_turnover_is_not_trade_count():
    """週轉率是名目金額比率，不是交易次數（避免混淆）。"""
    price = {"AAA": _prices(days=60)}
    dates = price["AAA"]["date"]
    res = run_backtest(_scores(dates), price, top_n=1, hold_days=5,
                       cost_model=COST_SCENARIOS["zero"])
    assert res.turnover != pytest.approx(float(res.round_trips))


def test_portfolio_turnover_is_annualized_and_asset_scaled():
    price = {"AAA": _prices(days=120)}
    dates = price["AAA"]["date"]
    res = run_backtest(_scores(dates), price, top_n=1, hold_days=5,
                       cost_model=COST_SCENARIOS["zero"])
    assert res.portfolio_turnover_annualized > 0


# --------------------------------------------------------------------------
# 成本敏感度
# --------------------------------------------------------------------------
def test_higher_costs_reduce_performance_monotonically():
    price = {"AAA": _prices(days=120)}
    dates = price["AAA"]["date"]
    scenarios = ["zero", "ideal", "typical", "conservative", "adverse"]
    results = run_cost_scenarios(_scores(dates), price, baseline(), scenarios,
                                 initial_capital=1_000_000.0)
    returns = [results[s].total_return for s in scenarios]
    costs = [results[s].cost_total for s in scenarios]
    assert returns == sorted(returns, reverse=True), f"成本越高報酬應越低：{returns}"
    assert costs == sorted(costs), f"成本總額應遞增：{costs}"
    assert returns[0] > returns[-1]


def test_liquidity_constraint_limits_order_size():
    """嚴苛情境限制單筆成交量佔比，成交張數（部位）應不大於無限制情境。"""
    p = _prices(days=90)
    p["volume"] = 100_000.0                    # 成交量小 → 1% = 1000 股
    price = {"AAA": p}
    dates = p["date"]
    loose = run_backtest(_scores(dates), price, top_n=1, hold_days=5,
                         cost_model=COST_SCENARIOS["typical"])
    tight = run_backtest(_scores(dates), price, top_n=1, hold_days=5,
                         cost_model=COST_SCENARIOS["severe"])
    if not tight.trades.empty and not loose.trades.empty:
        assert tight.trades["qty"].max() <= loose.trades["qty"].max()


def test_zero_cost_scenario_has_no_cost():
    price = {"AAA": _prices(days=60)}
    dates = price["AAA"]["date"]
    res = run_backtest(_scores(dates), price, top_n=1, hold_days=5,
                       cost_model=COST_SCENARIOS["zero"])
    assert res.cost_total == 0.0


# --------------------------------------------------------------------------
# 降低換手
# --------------------------------------------------------------------------
def test_variants_validate_and_cover_required_comparisons():
    specs = comparison_variants()
    names = {s.name for s in specs}
    assert "baseline_rule_h5" in names
    assert any(s.tags and "hold_period" in s.tags for s in specs)
    assert any(s.tags and "lower_turnover" in s.tags for s in specs)
    for s in specs:
        s.validate()


def test_hysteresis_spec_requires_buffer():
    with pytest.raises(ValueError):
        StrategySpec(name="bad", buy_threshold=0.2, exit_threshold=0.3).validate()
    with pytest.raises(ValueError):
        StrategySpec(name="bad2", hold_days=5, min_hold_days=10,
                     exit_threshold=0.0).validate()


def test_rebalance_cadence_reduces_trades():
    price = {"AAA": _prices(days=150)}
    dates = price["AAA"]["date"]
    s = _scores(dates)
    base = run_backtest(s, price, strategy=baseline(),
                        cost_model=COST_SCENARIOS["typical"])
    slow = run_backtest(s, price, strategy=baseline().with_updates(
        rebalance_days=20), cost_model=COST_SCENARIOS["typical"])
    assert slow.round_trips <= base.round_trips


def test_hysteresis_exits_earlier_when_score_collapses():
    """分數跌破門檻時應提早出場，而不是硬抱到最長持有期。"""
    days = 40
    dates = pd.bdate_range("2025-01-01", periods=days)
    price = {"AAA": pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 5_000_000.0})}
    scores = pd.DataFrame({
        "date": [str(pd.Timestamp(d).date()) for d in dates],
        "stock_id": "AAA",
        "score": [0.9] * 10 + [-0.5] * (days - 10),
    })
    fixed = run_backtest(
        scores, price, strategy=baseline().with_updates(
            name="fixed20", hold_days=20),
        cost_model=COST_SCENARIOS["zero"])
    hyst = run_backtest(
        scores, price, strategy=baseline().with_updates(
            name="hyst", hold_days=20, exit_threshold=0.0, min_hold_days=5,
            rebalance_days=1),
        cost_model=COST_SCENARIOS["zero"])

    assert not fixed.trades.empty and not hyst.trades.empty
    assert hyst.trades["holding_days"].iloc[0] < fixed.trades["holding_days"].iloc[0]


def test_hysteresis_respects_min_hold():
    days = 40
    dates = pd.bdate_range("2025-01-01", periods=days)
    price = {"AAA": pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 5_000_000.0})}
    # 進場後立刻轉負，但 min_hold_days=7 之前不得出場
    scores = pd.DataFrame({
        "date": [str(pd.Timestamp(d).date()) for d in dates],
        "stock_id": "AAA",
        "score": [0.9] + [-0.9] * (days - 1),
    })
    res = run_backtest(
        scores, price, strategy=baseline().with_updates(
            name="minhold", hold_days=20, exit_threshold=0.0, min_hold_days=7),
        cost_model=COST_SCENARIOS["zero"])
    assert not res.trades.empty
    assert res.trades["holding_days"].iloc[0] >= 7


# --------------------------------------------------------------------------
# 回歸：無效價格不得污染權益曲線
# --------------------------------------------------------------------------
def test_zero_price_row_does_not_corrupt_equity():
    p = _prices(days=40)
    p.loc[20, ["open", "high", "low", "close"]] = 0.0      # 資料錯誤列
    price = {"AAA": p}
    dates = p["date"]
    res = run_backtest(_scores(dates, score=0.9), price, top_n=1, hold_days=5)
    eq = res.equity.dropna()
    assert np.isfinite(eq.to_numpy()).all()
    assert (eq > 0).all(), "0 元價格不得讓權益歸零"
    assert res.max_drawdown > -1.0
