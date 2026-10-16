"""P2.8 §4：新聞品質與覆蓋率報告的擴充（合成資料，不連外）。

驗證重點：
  1. 「當日新增新聞」與「當日發布新聞」是兩種不同日期，不得混用。
  2. 發布時間與取得時間各有**獨立**的涵蓋區間，不可互相代替。
  3. 各新聞來源的資料量、失敗率與最近成功時間可被產出。
  4. 股票關聯覆蓋率區分「標題可驗證」與「僅查詢關聯」，後者不算已驗證。
  5. 尚未遷移的舊 DB（缺溯源欄位）不擲錯，且缺失一律視為不可追溯。
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from twstock.data import report, store


def _news(**over) -> dict:
    row = {
        "published": "2026-10-13", "stock_id": "2330", "title": "t",
        "link": "l", "score": 0.5, "source": "google_news",
        "published_at": "2026-10-13T09:00:00",
        "fetched_at": "2026-10-13T18:00:00",
        "timestamp_source": "source", "score_kind": "scored",
        "provenance": "recorded", "news_uid": "u",
        "content_hash": "h", "run_id": "r1", "link_basis": "title_match",
        "source_version": "v1",
    }
    row.update(over)
    return row


def _run(**over) -> dict:
    row = {
        "run_id": "r1", "dataset": "news", "stock_id": "2330",
        "run_at": "2026-10-13T18:00:00", "source": "google_news", "task": "news",
        "started_at": "2026-10-13T17:59:00", "ended_at": "2026-10-13T18:00:00",
        "status": "success_with_data", "error_class": "none", "error": "",
        "rows_fetched": 1, "rows_parsed": 1, "rows_validated": 1,
        "rows_written": 1, "data_start": "", "data_end": "", "attempts": 1,
        "source_version": "v1", "config_id": "c", "provenance": "recorded",
        "notes": "",
    }
    row.update(over)
    return row


# ==========================================================================
# §4：當日新增 vs 當日發布
# ==========================================================================
def test_added_and_published_dates_are_separate(temp_db):
    """一則「週五發布、週一才抓到」的新聞：發布日週五、新增日週一。"""
    store.write_df(pd.DataFrame([
        _news(published="2026-10-09", fetched_at="2026-10-12T18:00:00",
              link="a", news_uid="a"),
    ]), "news")
    cov = report.news_coverage()

    published = report.daily_news_counts(cov)
    added = report.daily_added_counts(cov)
    assert list(published["date"]) == ["2026-10-09"]
    assert list(added["date"]) == ["2026-10-12"]
    assert added.iloc[0]["added"] == 1


def test_undated_legacy_is_not_counted_as_any_day_added(temp_db):
    """舊資料沒有 fetched_at → 歸入「取得時間不明」，不得當成任何一天的新增。"""
    store.write_df(pd.DataFrame([
        _news(published="2026-10-09", fetched_at=None, news_uid=None,
              link_basis=None, provenance="unknown",
              timestamp_source="unknown", link="old"),
    ]), "news")
    added = report.daily_added_counts()
    assert list(added["date"]) == [report.ADDED_UNDATED_LABEL]
    assert report.news_coverage().n_fetched_unknown == 1


# ==========================================================================
# §4：發布／取得雙涵蓋區間
# ==========================================================================
def test_coverage_ranges_distinguish_published_and_fetched(temp_db):
    store.write_df(pd.DataFrame([
        _news(published="2026-10-06", fetched_at="2026-10-12T09:00:00",
              link="a", news_uid="a"),
        _news(published="2026-10-13", fetched_at="2026-10-13T09:00:00",
              link="b", news_uid="b"),
    ]), "news")
    cov = report.news_coverage()
    assert (cov.date_min, cov.date_max) == ("2026-10-06", "2026-10-13")
    assert (cov.fetched_min, cov.fetched_max) == ("2026-10-12", "2026-10-13")
    text = report.build_report_text()
    assert "依發布時間 published" in text
    assert "依取得時間 fetched_at" in text


def test_unknown_timestamp_rate_is_reported(temp_db):
    store.write_df(pd.DataFrame([
        _news(link="a", news_uid="a"),
        _news(link="b", news_uid="b", published=None,
              timestamp_source="unknown", score_kind="neutral", score=0.0),
    ]), "news")
    cov = report.news_coverage()
    assert cov.n_unknown_date == 1
    assert cov.unknown_timestamp_rate == pytest.approx(0.5)
    assert "發布時間未知率" in report.build_report_text()


# ==========================================================================
# §4：去重後總數
# ==========================================================================
def test_distinct_news_total_uses_stable_uid(temp_db):
    """同一則新聞關聯到兩檔股票 → 兩列、但去重後總數為 1。"""
    store.write_df(pd.DataFrame([
        _news(stock_id="2330", link="a", news_uid="same"),
        _news(stock_id="2317", link="b", news_uid="same"),
    ]), "news")
    cov = report.news_coverage()
    assert cov.n_rows == 2
    assert cov.n_distinct_uid == 1


# ==========================================================================
# §4：股票關聯覆蓋率
# ==========================================================================
def test_link_basis_separates_verified_and_query_only(temp_db):
    store.write_df(pd.DataFrame([
        _news(link="a", news_uid="a", link_basis="title_match"),
        _news(link="b", news_uid="b", link_basis="rss_query"),
        _news(link="c", news_uid="c", link_basis=None, provenance="unknown",
              fetched_at=None),
    ]), "news")
    cov = report.news_coverage()
    assert cov.n_link_verified == 1
    assert cov.n_link_unverifiable == 2          # 僅查詢關聯 + 未記錄
    assert cov.link_verified_rate == pytest.approx(1 / 3)
    text = report.build_report_text()
    assert "股票關聯依據" in text and "無法確認關聯筆數" in text


# ==========================================================================
# §4：各新聞來源
# ==========================================================================
def test_source_volume_is_per_publisher(temp_db):
    """資料量依 news.source（發布媒體）統計，與抓取管道無關。"""
    store.write_df(pd.DataFrame([
        _news(link="a", news_uid="a", source="UDN"),
        _news(link="b", news_uid="b", source="UDN",
              score_kind="neutral", score=0.0),
        _news(link="c", news_uid="c", source="中央社"),
    ]), "news")
    vol = report.news_source_volume().set_index("source")
    assert vol.loc["UDN", "n_news"] == 2
    assert vol.loc["UDN", "n_scored"] == 1
    assert vol.loc["中央社", "n_news"] == 1


def test_fetch_source_stats_failure_rate_and_last_success(temp_db):
    store.write_df(pd.DataFrame([
        _run(stock_id="2330", source="google_news",
             status="success_with_data", ended_at="2026-10-13T18:00:00"),
        _run(stock_id="2331", source="google_news",
             status="failed", ended_at="2026-10-14T18:00:00"),
        _run(stock_id="2454", source="ctee", status="success_empty",
             ended_at="2026-10-12T18:00:00"),
    ]), "ingest_run")

    df = report.news_fetch_source_stats().set_index("source")
    assert df.loc["google_news", "n_failed"] == 1
    assert df.loc["google_news", "n_decided"] == 2
    assert df.loc["google_news", "failure_rate"] == pytest.approx(0.5)
    assert df.loc["google_news", "last_success"] == "2026-10-13T18:00:00"
    assert df.loc["ctee", "last_success"] == "2026-10-12T18:00:00"
    # 失敗時間不得被當成最近成功時間
    assert df.loc["google_news", "last_success"] < "2026-10-14T18:00:00"
    text = report.build_report_text()
    assert "擷取來源成敗" in text and "最近成功" in text


def test_source_without_decided_records_has_no_failure_rate(temp_db):
    """只有不可追溯或空紀錄的來源不得產生一個看似 0% 的失敗率。"""
    store.write_df(pd.DataFrame([
        _run(stock_id="2330", source="x", status="unknown"),
    ]), "ingest_run")
    df = report.news_fetch_source_stats().set_index("source")
    assert df.loc["x", "n_decided"] == 0
    assert df.loc["x", "failure_rate"] is None
    assert "無可判定" in report.build_report_text()


# ==========================================================================
# §4：欄位不齊／空表不得擲錯（唯讀分析尚未遷移的舊 DB）
# ==========================================================================
def test_unmigrated_news_table_does_not_crash_report(tmp_path, monkeypatch):
    """尚未遷移的舊 DB 缺溯源欄位 → 一律視為不可追溯，而不是擲錯。"""
    db = tmp_path / "unmigrated.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
    CREATE TABLE news (published DATE, stock_id TEXT, title TEXT, link TEXT,
        score REAL, source TEXT, PRIMARY KEY (stock_id, link));
    INSERT INTO news VALUES ('2026-10-07','2330','舊','l1',0.5,'x');
    """)
    conn.commit()
    conn.close()

    monkeypatch.setenv("TWSTOCK_READONLY", "1")
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    cov = report.news_coverage()
    assert cov.n_rows == 1
    assert cov.n_fetched_known == 0
    assert cov.n_uid_missing == 1
    assert cov.n_link_unverifiable == 1
    assert report.daily_added_counts().iloc[0]["date"] == \
        report.ADDED_UNDATED_LABEL
    # 不得擲錯
    assert isinstance(report.build_report_text(), str)


def test_empty_report_renders_without_crash(temp_db):
    text = report.build_report_text()
    assert "新聞資料品質" in text
    assert "每日新聞筆數：無資料" in text
    assert "每日新增新聞筆數：無資料" in text
    assert report.news_source_volume().empty
    assert report.news_fetch_source_stats().empty
