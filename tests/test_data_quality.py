"""資料品質：無效價格清理與除權息／分割還原（合成資料，不連網）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twstock.data import quality as q


def _series(closes, start="2026-01-05"):
    dates = pd.bdate_range(start, periods=len(closes))
    c = np.array(closes, dtype=float)
    return pd.DataFrame({"date": dates, "stock_id": "X",
                         "open": c, "high": c, "low": c, "close": c,
                         "volume": 1_000_000.0})


# ---- 無效值 ----
def test_find_invalid_rows_detects_zero_price():
    df = _series([100, 0, 101, 102])
    bad = q.find_invalid_price_rows(df)
    assert len(bad) == 1
    assert pd.Timestamp(bad.iloc[0]["date"]) == pd.Timestamp("2026-01-06")


def test_clean_prices_treats_zero_as_missing_not_zero_value():
    df = _series([100, 0, 101])
    clean, dropped = q.clean_prices(df)
    assert len(clean) == 2 and len(dropped) == 1
    assert 0.0 not in set(clean["close"])          # 不可退化成 0 值
    ret = clean["close"].pct_change().dropna()
    assert np.isfinite(ret).all()                  # 不會產生 ±inf


def test_audit_reports_invalid_and_jumps():
    # 100 -> 47 為 -53%，明顯超出漲跌停 → 疑似未調整公司行動
    df = _series([100, 100, 47, 47.5])
    rep = q.audit_prices(df, "X")
    assert rep.invalid_rows == 0
    assert len(rep.detected_jumps) == 1


def test_negative_and_nan_prices_are_invalid():
    df = _series([100, -5, 101])
    df.loc[1, "close"] = np.nan
    assert len(q.find_invalid_price_rows(df)) == 1


# ---- 因子計算 ----
def test_split_factor():
    assert q.split_factor(200.0, 50.0) == pytest.approx(0.25)
    assert q.split_factor(50.0, 200.0) == pytest.approx(4.0)   # 反分割
    with pytest.raises(ValueError):
        q.split_factor(0.0, 50.0)


def test_cash_dividend_factor():
    # 除息前 100、配 5 元現金 → 因子 0.95
    assert q.cash_dividend_factor(100.0, 5.0) == pytest.approx(0.95)
    with pytest.raises(ValueError):
        q.cash_dividend_factor(3.0, 5.0)           # 股利大於股價不合理
    with pytest.raises(ValueError):
        q.cash_dividend_factor(0.0, 1.0)


def test_stock_dividend_factor():
    # 每股配 0.25 股 → 股數變 1.25 倍 → 因子 0.8
    assert q.stock_dividend_factor(0.25) == pytest.approx(0.8)


# ---- 還原 ----
def test_apply_split_makes_series_continuous():
    # 1 拆 4：188 → 47（原始序列 -75% 假崩盤）
    df = _series([188.0, 189.0, 47.0, 47.5])
    action = q.CorporateAction("X", pd.Timestamp("2026-01-07"), "split",
                               q.split_factor(189.0, 47.25), True)
    adj = q.apply_corporate_actions(df, [action])
    ret = adj.sort_values("date")["close"].pct_change().dropna()
    assert (ret.abs() < 0.05).all(), f"還原後仍有假跳動：{ret.tolist()}"


def test_adjustment_keeps_latest_prices_unchanged():
    df = _series([188.0, 47.0, 47.5])
    action = q.CorporateAction("X", pd.Timestamp("2026-01-06"), "split",
                               q.split_factor(188.0, 47.0), True)
    adj = q.apply_corporate_actions(df, [action])
    assert adj.sort_values("date")["close"].iloc[-1] == pytest.approx(47.5)


def test_apply_no_actions_is_noop():
    df = _series([100, 101, 102])
    pd.testing.assert_frame_equal(q.apply_corporate_actions(df, []), df)


def test_split_adjusts_volume_inversely():
    df = _series([100.0, 25.0])
    action = q.CorporateAction("X", pd.Timestamp("2026-01-06"), "split", 0.25, True)
    adj = q.apply_corporate_actions(df, [action])
    first_vol = adj.sort_values("date")["volume"].iloc[0]
    assert first_vol == pytest.approx(1_000_000.0 / 0.25)


# ---- 驗證守門：只在能降低不連續時才調整 ----
def test_prune_rejects_event_not_reflected_in_prices():
    # 事件宣稱分割（因子 0.8），但序列當日並未下跌 → 套用會製造假跳動
    df = _series([100.0, 99.0, 100.0])
    action = q.CorporateAction("X", pd.Timestamp("2026-01-06"), "stock_dividend",
                               0.8, True)
    kept, skipped = q.prune_actions_by_observed_move(df, [action])
    assert kept == []
    assert skipped and "未反映" in skipped[0]


def test_prune_keeps_event_reflected_in_prices():
    # 除息：前一日 100、除息日 95 → 應該保留
    df = _series([100.0, 95.0, 95.5])
    action = q.CorporateAction("X", pd.Timestamp("2026-01-06"), "cash_dividend",
                               q.cash_dividend_factor(100.0, 5.0), False)
    kept, skipped = q.prune_actions_by_observed_move(df, [action])
    assert len(kept) == 1 and not skipped


def test_prune_skips_when_no_price_at_event_date():
    df = _series([100.0, 101.0])
    action = q.CorporateAction("X", pd.Timestamp("2026-02-01"), "split", 0.5, True)
    kept, skipped = q.prune_actions_by_observed_move(df, [action])
    assert kept == [] and skipped


def test_build_actions_from_finmind_like_frames():
    price = _series([100.0, 95.0, 95.5])
    divs = pd.DataFrame([{
        "date": "2026-01-06", "stock_id": "X", "year": "114",
        "CashEarningsDistribution": 5.0, "CashStatutorySurplus": 0.0,
        "StockEarningsDistribution": 0.0, "StockStatutorySurplus": 0.0,
        "CashExDividendTradingDate": "2026-01-06",
        "StockExDividendTradingDate": "",
    }])
    actions, warnings = q.build_corporate_actions(price, "X", splits=None,
                                                  dividends=divs)
    assert len(actions) == 1
    assert actions[0].kind == "cash_dividend"
    assert actions[0].factor == pytest.approx(0.95)


def test_build_actions_warns_without_prior_close():
    price = _series([100.0, 95.0])          # 事件日在第一筆 → 無前一日可比較
    divs = pd.DataFrame([{
        "date": "2026-01-05", "stock_id": "X", "year": "114",
        "CashEarningsDistribution": 5.0, "CashStatutorySurplus": 0.0,
        "StockEarningsDistribution": 0.0, "StockStatutorySurplus": 0.0,
        "CashExDividendTradingDate": "2026-01-05",
        "StockExDividendTradingDate": "",
    }])
    actions, warnings = q.build_corporate_actions(price, "X", None, divs)
    assert actions == []
    assert any("無除息前收盤價" in w for w in warnings)
