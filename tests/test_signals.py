import numpy as np
import pandas as pd

from twstock.signals import score_row, pick_top_n, score_stocks


def _row(**kw) -> pd.Series:
    base = dict(
        mom_20=0.05, mom_60=0.10, ma_alignment=1.0, rsi=50, vol_ratio=1.0,
        ret_1d=0.01, inst_net_sum3=np.nan, streak_inst_buy=0,
        margin_ratio=np.nan, rev_yoy=np.nan, news_sentiment=np.nan,
        close=100.0,
    )
    base.update(kw)
    return pd.Series(base)


def test_score_row_neutral():
    s, _ = score_row(_row())
    assert -1 <= s <= 1


def test_score_row_positive_when_bullish():
    bull = _row(mom_20=0.15, ma_alignment=1.0, rsi=45,
                inst_net_sum3=6e7, news_sentiment=0.8)
    bear = _row(mom_20=-0.15, ma_alignment=0.0, rsi=75,
                inst_net_sum3=-6e7, news_sentiment=-0.8)
    s_bull, _ = score_row(bull)
    s_bear, _ = score_row(bear)
    assert s_bull > s_bear


def test_pick_top_n_sorted():
    df = score_stocks({
        "A": pd.DataFrame({
            "date": pd.to_datetime(["2025-06-01"]),
            "close": [100.0],
            "mom_20": [0.2], "mom_60": [0.3], "ma_alignment": [1.0],
            "rsi": [40], "vol_ratio": [1.0], "ret_1d": [0.01],
        }),
        "B": pd.DataFrame({
            "date": pd.to_datetime(["2025-06-01"]),
            "close": [50.0],
            "mom_20": [-0.1], "mom_60": [-0.2], "ma_alignment": [0.0],
            "rsi": [80], "vol_ratio": [0.5], "ret_1d": [-0.01],
        }),
    })
    top = pick_top_n(df, "2025-06-01", top_n=1)
    assert top.iloc[0]["stock_id"] == "A"
