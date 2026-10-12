import numpy as np
import pandas as pd

from twstock.backtest import run_backtest


def _mk_price(dates, base=100.0):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": [base] * len(dates),
        "high": [base * 1.02] * len(dates),
        "low": [base * 0.98] * len(dates),
        "close": [base] * len(dates),
        "volume": [1e6] * len(dates),
    })


def test_backtest_simple_rising():
    dates = pd.bdate_range("2025-06-01", periods=20)
    price = {"2330": _mk_price(dates)}
    scores = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "stock_id": "2330",
        "score": [0.9] * len(dates),
        "close": [100.0] * len(dates),
    })
    res = run_backtest(scores, price, top_n=1, hold_days=5)
    assert len(res.equity) > 0
    # 等價格進出 → 扣手續費後應為小負報酬
    assert res.total_return <= 0


def test_backtest_positive_trend():
    dates = pd.bdate_range("2025-06-01", periods=60)
    n = len(dates)
    price = {"2330": pd.DataFrame({
        "date": dates,
        "open": np.linspace(100, 130, n), "high": np.linspace(102, 132, n),
        "low": np.linspace(98, 128, n), "close": np.linspace(101, 131, n),
        "volume": [1e6] * n,
    })}
    scores = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "stock_id": "2330",
        "score": [0.9] * n,
    })
    res = run_backtest(scores, price, top_n=1, hold_days=5)
    assert res.total_return > 0


def test_backtest_win_rate():
    dates = pd.bdate_range("2025-06-01", periods=30)
    price = {"2330": _mk_price(dates)}
    scores = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "stock_id": "2330",
        "score": [0.5] * len(dates),
    })
    res = run_backtest(scores, price, top_n=1, hold_days=5)
    if len(res.trades) > 0:
        assert 0 <= res.win_rate <= 1
