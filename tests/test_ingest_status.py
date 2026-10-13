"""W4：抓取失敗不得被記為成功，也不得更新 last_ingest（避免誤判資料新鮮）。"""
from __future__ import annotations

import pandas as pd
import pytest

from twstock import pipeline
from twstock.data import store


def _fake_price(sid: str, days: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2026-10-01", periods=days)
    return pd.DataFrame({
        "date": dates, "stock_id": sid,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6,
    })


@pytest.fixture()
def isolated(temp_db, monkeypatch):
    """把 pipeline 的所有外部抓取換成可控假函式，並停用等待。"""
    monkeypatch.setattr(pipeline.fm, "polite_sleep", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.sent, "update_news_for", lambda *a, **k: 0)
    monkeypatch.setattr(pipeline.fm, "fetch_institutional", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.fm, "fetch_margin", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.fm, "fetch_revenue", lambda *a, **k: None)
    # 單元測試不得連外：公司行動（分割／股利）一律以假函式取代
    monkeypatch.setattr(pipeline.fm, "fetch_splits", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.fm, "fetch_dividends", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "CONFIG",
                        type("C", (), {"pool": {"2330": "台積電", "2317": "鴻海"},
                                       "benchmark": "0050",
                                       "lookback_years": 1.0})())
    return monkeypatch


def test_all_fetch_failure_does_not_mark_fresh(isolated):
    isolated.setattr(pipeline.fm, "fetch_benchmark", lambda *a, **k: None)
    isolated.setattr(pipeline.fm, "fetch_price", lambda *a, **k: None)   # 全部失敗

    pipeline.ingest()
    assert store.get_meta("last_ingest") is None, "全數失敗竟然被記為成功"
    failed = store.get_meta("last_ingest_failed_ids")
    assert failed and "2330" in failed and "2317" in failed


def test_partial_failure_records_ids_and_still_marks_fresh(isolated):
    isolated.setattr(pipeline.fm, "fetch_benchmark", lambda *a, **k: _fake_price("0050"))
    isolated.setattr(pipeline.fm, "fetch_price",
                     lambda sid, *a, **k: _fake_price(sid) if sid == "2330" else None)

    pipeline.ingest()
    assert store.get_meta("last_ingest") is not None
    assert store.get_meta("last_ingest_failed_ids") == "2317"
    # 成功的那檔真的寫進去了
    assert store.read_sql("SELECT COUNT(*) AS n FROM price WHERE stock_id='2330'").iloc[0]["n"] == 3


def test_empty_data_is_not_a_failure(isolated):
    isolated.setattr(pipeline.fm, "fetch_benchmark", lambda *a, **k: None)
    isolated.setattr(pipeline.fm, "fetch_price",
                     lambda *a, **k: pd.DataFrame())   # 成功但真的沒資料

    pipeline.ingest()
    assert store.get_meta("last_ingest_failed_ids") == ""
    log = store.read_sql(
        "SELECT dataset, ok, error FROM ingest_log "
        "WHERE stock_id='2330' AND dataset='price'")
    assert len(log) == 1 and log.iloc[0]["error"] == "empty" and log.iloc[0]["ok"] == 1


def test_ingest_log_records_failure(isolated):
    isolated.setattr(pipeline.fm, "fetch_benchmark", lambda *a, **k: None)
    isolated.setattr(pipeline.fm, "fetch_price", lambda *a, **k: None)
    pipeline.ingest()
    log = store.read_sql("SELECT ok, error FROM ingest_log WHERE dataset='price'")
    assert (log["ok"] == 0).all()
    assert (log["error"] == "fetch_failed").all()
