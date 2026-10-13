"""W7：回測不得使用未來資訊。

訊號以 T 日收盤算出，必須在 T+1 開盤成交。此測試直接檢查成交價等於次一日開盤。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from twstock.backtest import run_backtest


def _price(dates, opens, closes):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": opens, "high": np.array(opens) * 1.01,
        "low": np.array(opens) * 0.99, "close": closes, "volume": 1e6,
    })


def _scores(dates, sid="2330", score=0.9):
    return pd.DataFrame({
        "date": [str(d.date()) for d in pd.to_datetime(dates)],
        "stock_id": sid, "score": score,
    })


def test_entry_uses_next_day_open_not_signal_day():
    dates = pd.bdate_range("2026-06-01", periods=10)
    opens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    closes = [100.0] * 10
    price = {"2330": _price(dates, opens, closes)}
    scores = _scores(dates[:1])          # 只有第一天有訊號

    res = run_backtest(scores, price, top_n=1, hold_days=2, initial_capital=1_000_000)

    assert len(res.trades) == 1, f"應剛好成交一筆，實際 {len(res.trades)}"
    t = res.trades.iloc[0]
    assert pd.Timestamp(t["entry_date"]) == dates[1], "進場日必須是 T+1"
    assert t["entry_price"] == opens[1], "進場價必須是 T+1 開盤（不可用 T 日價格）"
    assert pd.Timestamp(t["exit_date"]) == dates[3], "持有 2 日後於 T+3 出場"
    assert t["exit_price"] == opens[3]


def test_entry_does_not_use_signal_day_close():
    """訊號日的收盤價不得被用來當成進場價。"""
    dates = pd.bdate_range("2026-06-01", periods=10)
    opens = [100, 105, 105, 105, 105, 105, 105, 105, 105, 105]
    closes = [100, 50, 50, 50, 50, 50, 50, 50, 50, 50]   # 進場日盤中重挫
    price = {"2330": _price(dates, opens, closes)}
    res = run_backtest(_scores(dates[:1]), price, top_n=1, hold_days=1,
                       initial_capital=1_000_000)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["entry_price"] == 105.0
    assert res.trades.iloc[0]["entry_price"] != closes[1]


def test_limit_up_blocks_entry():
    """T+1 開盤即漲停時無法成交，應放棄該訊號。"""
    dates = pd.bdate_range("2026-06-01", periods=10)
    closes = [100.0] * 10
    opens = [100.0] * 10
    opens[1] = 110.0                     # 漲停 (10%)
    price = {"2330": _price(dates, opens, closes)}
    res = run_backtest(_scores(dates[:1]), price, top_n=1, hold_days=2,
                       initial_capital=1_000_000)
    assert len(res.trades) == 0, "漲停不應成交"


def test_no_trades_when_score_below_threshold():
    dates = pd.bdate_range("2026-06-01", periods=10)
    price = {"2330": _price(dates, [10] * 10, [10] * 10)}
    res = run_backtest(_scores(dates, score=0.05), price, top_n=1, hold_days=2)
    assert len(res.trades) == 0


def test_benchmark_comparison_available():
    dates = pd.bdate_range("2026-06-01", periods=20)
    price = {"2330": _price(dates, [10] * 20, [10] * 20)}
    bench = pd.DataFrame({"date": dates, "close": np.linspace(100, 130, 20)})
    res = run_backtest(_scores(dates, score=0.05), price, top_n=1, hold_days=2,
                       benchmark=bench)
    assert res.benchmark_total_return > 0
    assert "0050" in res.summary()


def test_metrics_are_finite():
    dates = pd.bdate_range("2026-06-01", periods=40)
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 40))
    price = {"2330": _price(dates, closes, closes)}
    res = run_backtest(_scores(dates, score=0.9), price, top_n=1, hold_days=5)
    assert len(res.equity) == len(dates)
    assert np.isfinite(res.cagr) and np.isfinite(res.sharpe)
    assert -1 <= res.max_drawdown <= 0
