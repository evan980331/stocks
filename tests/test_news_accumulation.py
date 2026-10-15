"""P2.8 每日新聞累積：穩定識別碼、內容雜湊、股票關聯依據、冪等累積與遷移。

全部測試以注入的假 getter 驅動，**不連外、不依賴外部 API**。
"""
from __future__ import annotations

import datetime as dt
import email.utils
import sqlite3

import pandas as pd
import pytest

from twstock.data import status as st, store
from twstock.sentiment import news_rss, sentiment

UTC = dt.timezone.utc
FIXED_NOW = dt.datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# 測試替身
# --------------------------------------------------------------------------
class _Resp:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code


def _rss(entries) -> bytes:
    """entries: [(title, link, pub_datetime|None, guid|None)]"""
    items = []
    for e in entries:
        title, link, when = e[0], e[1], e[2]
        guid = e[3] if len(e) > 3 else None
        pub = (f"<pubDate>{email.utils.format_datetime(when)}</pubDate>"
               if when is not None else "")
        g = f"<guid>{guid}</guid>" if guid else ""
        items.append(f"<item><title>{title}</title><link>{link}</link>"
                     f"{pub}{g}<description>x</description></item>")
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<rss version="2.0"><channel><title>q</title>'
            '<link>https://news.google.com/</link>'
            + "".join(items) + "</channel></rss>").encode("utf-8")


def _getter(content: bytes, status_code: int = 200):
    def g(url, timeout=None, headers=None):
        return _Resp(content, status_code)
    return g


def _raiser(exc: Exception):
    def g(url, timeout=None, headers=None):
        raise exc
    return g


PUB = FIXED_NOW - dt.timedelta(hours=2)


# ==========================================================================
# §2：識別碼、內容雜湊、關聯依據
# ==========================================================================
def test_normalize_link_drops_tracking_but_keeps_meaningful_params():
    """只移除不改變文章身分的參數；不得把不同文章誤併成同一則。"""
    a = "https://Example.com/a?id=1&utm_source=rss&fbclid=xyz&oc=5"
    b = "https://example.com/a?id=1#section"
    assert news_rss.normalize_link(a) == news_rss.normalize_link(b)

    # 意義不同的查詢參數必須保留（否則會錯誤合併兩篇不同文章）
    assert news_rss.normalize_link("https://x.com/a?id=1") != \
        news_rss.normalize_link("https://x.com/a?id=2")


def test_content_hash_is_reproducible_and_input_sensitive():
    h1 = news_rss.content_hash("標題", "https://a.com/x?utm_source=y", None)
    h2 = news_rss.content_hash("標題", "https://a.com/x", None)
    assert h1 == h2, "正規化後相同內容必須得到相同雜湊"
    assert h1 != news_rss.content_hash("標題2", "https://a.com/x", None)
    assert h1 != news_rss.content_hash("標題", "https://a.com/x",
                                       "2026-10-14T10:00:00")


def test_news_uid_prefers_guid_then_link_then_content():
    with_guid = news_rss.news_uid(guid="G1", link="https://a.com/1", title="t")
    without_guid = news_rss.news_uid(link="https://a.com/1", title="t")
    no_link = news_rss.news_uid(link="", title="t",
                                published_at="2026-10-14T10:00:00")

    assert with_guid.startswith("guid:") and without_guid.startswith("link:")
    assert no_link.startswith("content:")
    # guid 優先：同一文章若有 guid，就以 guid 為準
    assert with_guid != without_guid
    # 連結正規化後同一文章得到同一 uid
    assert news_rss.news_uid(link="https://a.com/1?utm_source=z") == without_guid


def test_link_basis_distinguishes_verified_and_query_only():
    assert news_rss.link_basis_for("2330", "台積電",
                                   "台積電營收創新高") == "title_match"
    assert news_rss.link_basis_for("2330", "台積電", "2330 上漲") == "title_match"
    # 只因為查詢關聯、標題未出現代號或公司名 → 不可宣稱已驗證
    assert news_rss.link_basis_for("2330", "台積電", "半導體類股走強") == "rss_query"
    assert news_rss.link_basis_for("", "", "任意標題") == "unknown"


def test_fetched_item_carries_provenance_fields(temp_db):
    got = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW, run_id="RUN1",
        getter=_getter(_rss([("台積電營收創新高", "https://a.com/1", PUB, "G1")])))
    assert got.rows_written == 1
    row = store.read_sql("SELECT * FROM news").iloc[0]
    assert row["news_uid"].startswith("guid:")
    assert row["content_hash"] and len(str(row["content_hash"])) == 64
    assert row["link_basis"] == "title_match"
    assert row["run_id"] == "RUN1"
    assert row["source_version"] == news_rss.NEWS_RULESET_VERSION


def test_link_basis_recorded_as_unverified_when_title_lacks_identity(temp_db):
    sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW,
        getter=_getter(_rss([("半導體類股走強", "https://a.com/9", PUB)])))
    row = store.read_sql("SELECT link_basis FROM news").iloc[0]
    assert row["link_basis"] == "rss_query", "查詢關聯不得冒充內容驗證"


# ==========================================================================
# §3.2/§3.3：每日累積與冪等性
# ==========================================================================
def test_refetch_is_idempotent_and_does_not_rewrite_first_seen(temp_db):
    """重複抓取同一新聞：不重複計入，且**不得**改寫首次見到的 fetched_at。

    ``fetched_at`` 是 point-in-time 的可用起點；若被後來的抓取覆寫，
    「當初什麼時候看得到這則新聞」就永久不可重建（任務 §5.2）。
    """
    feed = _rss([("台積電營收創新高", "https://a.com/1", PUB, "G1")])

    first = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW, run_id="RUN1", getter=_getter(feed))
    assert first.rows_written == 1 and first.n_existing == 0
    original = store.read_sql("SELECT * FROM news").iloc[0]

    # 第二天重抓：來源時間不同、抓取時間不同
    later_now = FIXED_NOW + dt.timedelta(days=1)
    second = sentiment.update_news_for(
        "2330", "台積電", now=later_now, run_id="RUN2", getter=_getter(feed))
    assert second.status == st.IngestStatus.SUCCESS_WITH_DATA.value, \
        "來源仍正常提供資料，不得因為沒有新增就誤記為 success_empty"
    assert second.rows_written == 0 and second.n_existing == 1

    df = store.read_sql("SELECT * FROM news")
    assert len(df) == 1, "重複抓取產生了重複資料"
    row = df.iloc[0]
    assert row["fetched_at"] == original["fetched_at"], "首次抓取時間被覆寫了"
    assert row["run_id"] == original["run_id"], "首次任務識別被覆寫了"
    assert row["provenance"] == original["provenance"]


def test_same_article_across_stocks_shares_uid_but_not_rows(temp_db):
    """同一篇文章關聯到兩檔股票：各留一列（關聯不同），但 uid 相同。

    報表因此可以區分「列數（每股一條）」與「去重後總數（實際幾則新聞）」。
    """
    feed = _rss([("台積電鴻海同步擴產", "https://a.com/same", PUB, "GSAME")])
    sentiment.update_news_for("2330", "台積電", now=FIXED_NOW,
                              getter=_getter(feed))
    sentiment.update_news_for("2317", "鴻海", now=FIXED_NOW,
                              getter=_getter(feed))

    df = store.read_sql("SELECT stock_id, news_uid FROM news ORDER BY stock_id")
    assert len(df) == 2
    assert set(df["stock_id"]) == {"2330", "2317"}
    assert df["news_uid"].nunique() == 1, "同一文章的識別碼應跨股票一致"


def test_same_title_different_link_is_not_merged(temp_db):
    """不得因去重而錯誤合併不同新聞。"""
    sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW,
        getter=_getter(_rss([("公司聲明", "https://a.com/1", PUB, "G1")])))
    sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW,
        getter=_getter(_rss([("公司聲明", "https://a.com/2", PUB, "G2")])))
    assert store.read_sql("SELECT COUNT(*) AS n FROM news").iloc[0]["n"] == 2


def test_second_run_of_same_day_adds_only_new_articles(temp_db):
    """當日第二次擷取只新增來源新出現的文章。"""
    sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW, run_id="R1",
        getter=_getter(_rss([("舊新聞", "https://a.com/old", PUB, "GO")])))

    later = FIXED_NOW + dt.timedelta(hours=3)
    res = sentiment.update_news_for(
        "2330", "台積電", now=later, run_id="R2",
        getter=_getter(_rss([("舊新聞", "https://a.com/old", PUB, "GO"),
                             ("突發利多", "https://a.com/new", later, "GN")])))
    assert res.rows_fetched == 2
    assert res.rows_written == 1 and res.n_existing == 1
    df = store.read_sql("SELECT title FROM news ORDER BY title")
    assert set(df["title"]) == {"舊新聞", "突發利多"}
    assert store.read_sql(
        "SELECT run_id FROM news WHERE title='突發利多'").iloc[0][
            "run_id"] == "R2"


def test_fetch_failure_never_touches_existing_rows(temp_db):
    """新聞來源失敗不得覆蓋既有資料，也不得偽裝成沒有新聞。"""
    sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW,
        getter=_getter(_rss([("既有新聞", "https://a.com/1", PUB, "G1")])))

    bad = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW + dt.timedelta(days=1),
        getter=_raiser(ConnectionError("網路中斷")))
    assert bad.status == st.IngestStatus.FAILED.value
    assert bad.error_class == st.ErrorClass.NETWORK.value
    assert bad.rows_written == 0

    df = store.read_sql("SELECT title, provenance FROM news")
    assert len(df) == 1 and df.iloc[0]["title"] == "既有新聞"
    assert df.iloc[0]["provenance"] == "recorded"


def test_ingest_records_repeated_fetch_as_success_with_data(temp_db):
    """重複抓取（全部命中既有列）必須在 ingest_run 中記為成功有資料。"""
    feed = _rss([("台積電營收創新高", "https://a.com/1", PUB, "G1")])
    sentiment.update_news_for("2330", "台積電", now=FIXED_NOW, run_id="R1",
                              getter=_getter(feed))
    second = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW + dt.timedelta(days=1), run_id="R2",
        getter=_getter(feed))

    assert second.record is not None
    assert second.record.status == st.IngestStatus.SUCCESS_WITH_DATA.value
    assert second.record.rows_written == 0
    assert "existing=1" in second.record.notes


def test_repeated_fetch_notes_capture_dedup_counts(temp_db):
    feed = _rss([("台積電營收創新高", "https://a.com/1", PUB, "G1")])
    sentiment.update_news_for("2330", "台積電", now=FIXED_NOW,
                              getter=_getter(feed))
    second = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW + dt.timedelta(days=1),
        getter=_getter(feed))
    assert second.n_existing == 1 and second.rows_written == 0
    assert second.record.notes.startswith("entities=1 stored=1")


# ==========================================================================
# §4/§7：schema 遷移與舊資料
# ==========================================================================
def _legacy_news_db(tmp_path) -> object:
    db = tmp_path / "legacy_news.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
    CREATE TABLE news (
        published DATE, stock_id TEXT, title TEXT, link TEXT,
        score REAL, source TEXT,
        PRIMARY KEY (stock_id, link));
    """)
    conn.execute("INSERT INTO news VALUES "
                 "('2026-10-06','2330','舊新聞','l1',0.5,'x')")
    conn.commit()
    conn.close()
    return db


def test_migration_adds_provenance_columns_without_touching_legacy(tmp_path,
                                                                   monkeypatch):
    db = _legacy_news_db(tmp_path)
    monkeypatch.delenv("TWSTOCK_READONLY", raising=False)
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    df = store.read_sql("SELECT * FROM news")
    assert len(df) == 1
    for col in ("news_uid", "content_hash", "run_id", "link_basis",
                "source_version", "fetched_at", "provenance"):
        assert col in df.columns, f"遷移未補上欄位 {col}"
    row = df.iloc[0]
    assert pd.isna(row["news_uid"]) or row["news_uid"] is None, \
        "舊資料不得被臆測出識別碼"
    assert row["provenance"] == st.UNKNOWN_PROVENANCE
    assert row["title"] == "舊新聞" and row["score"] == 0.5, "舊資料不得被改寫"


def test_migration_is_idempotent_and_creates_indexes(tmp_path, monkeypatch):
    db = _legacy_news_db(tmp_path)
    monkeypatch.delenv("TWSTOCK_READONLY", raising=False)
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    store.read_sql("SELECT 1")
    store.read_sql("SELECT 1")     # 再次觸發遷移
    conn = sqlite3.connect(db)
    idx = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    conn.close()
    assert {"idx_news_uid", "idx_news_fetched", "idx_news_published",
            "idx_news_run"} <= idx, f"新聞索引未建立：{idx}"


def test_legacy_row_is_not_auto_upgraded_on_refetch(temp_db):
    """舊資料（unknown）即使被重抓命中，也不得自動改寫為已確認。"""
    # 模擬本階段之前寫入的一列：無 news_uid / run_id / link_basis
    store.write_df(pd.DataFrame([{
        "published": "2026-10-06", "stock_id": "2330", "title": "舊新聞",
        "link": "https://a.com/1", "score": 0.0, "source": "x",
        "published_at": None, "fetched_at": None,
        "timestamp_source": "unknown", "score_kind": "neutral",
        "provenance": "unknown",
    }]), "news")

    res = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW,
        getter=_getter(_rss([("舊新聞", "https://a.com/1",
                              FIXED_NOW - dt.timedelta(days=8), "G1")])))
    # 舊文章超出 days=7 窗口 → 不會被重抓到；改用窗口內同連結
    res2 = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW, days=30,
        getter=_getter(_rss([("舊新聞", "https://a.com/1", PUB, "G1")])))
    assert res.rows_written == 0 and res2.rows_written == 0, \
        "重抓命中既有列時不得新增或覆寫"

    row = store.read_sql("SELECT * FROM news").iloc[0]
    assert row["provenance"] == "unknown", "unknown 被自動升級了"
    assert row["timestamp_source"] == "unknown"
    assert pd.isna(row["fetched_at"]) or row["fetched_at"] is None
