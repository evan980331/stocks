"""以合成資料測試 feature 計算正確性。"""
import numpy as np
import pandas as pd

from twstock.features import build_features, rsi


def _synth_price(days: int = 120, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=days)
    close = base + np.cumsum(rng.normal(0, 1, days))
    return pd.DataFrame({
        "date": dates, "open": close * (1 + rng.normal(0, 0.002, days)),
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.integers(1e5, 1e6, days), "amount": 1e8,
    })


def test_rsi_range():
    close = pd.Series(np.linspace(100, 110, 50))
    r = rsi(close)
    assert r.iloc[-1] > 90


def test_build_features_shape():
    p = _synth_price()
    f = build_features(p)
    assert len(f) == len(p)
    for col in ["mom_20", "mom_60", "ma_alignment", "rsi", "vol_ratio", "volatility_20"]:
        assert col in f.columns


def test_inst_merge_and_streak():
    p = _synth_price()
    inst = pd.DataFrame({
        "date": p["date"], "stock_id": "2330",
        "Foreign_Investors_Net": np.where(np.arange(len(p)) % 3 == 0, 5e6, -1e6),
        "Investment_Trust_Net": 0.0, "Dealer_Self_Net": 0.0, "Dealer_Hedging_Net": 0.0,
        "Foreign_Investors_Buy": 1e7, "Foreign_Investors_Sell": 5e6,
        "Investment_Trust_Buy": 0, "Investment_Trust_Sell": 0,
        "Dealer_Self_Buy": 0, "Dealer_Self_Sell": 0,
        "Dealer_Hedging_Buy": 0, "Dealer_Hedging_Sell": 0,
    })
    f = build_features(p, inst=inst)
    assert "inst_net_sum3" in f.columns
    assert "streak_inst_buy" in f.columns


def test_news_sentiment_decay():
    p = _synth_price()
    news = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
        "stock_id": ["2330", "2330"],
        "score": [0.5, -0.3],
    })
    f = build_features(p, news_daily=news)
    assert "news_sentiment" in f.columns
    # 第一個交易日的情緒應為加權平均
    first = f.dropna(subset=["news_sentiment"]).head(1)
    assert abs(first["news_sentiment"].iloc[0] - (0.5 * 1.0 + (-0.3) * 1.0) / 2) < 1e-9
