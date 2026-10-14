"""P2.7 擷取狀態治理：以合成資料驗證狀態區分、失敗不可偽裝為成功、可追溯性。

全部測試皆以注入的假 getter／假抓取函式驅動，**不連外、不付費、不依賴外部 API**。
"""
from __future__ import annotations

import datetime as dt
import email.utils
import sqlite3

import pandas as pd
import pytest
import requests

from twstock import pipeline, worker
from twstock.data import report, status as st, store
from twstock.features import build_features
from twstock.sentiment import news_rss, sentiment

UTC = dt.timezone.utc
FIXED_NOW = dt.datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# 測試替身：HTTP 回應與 RSS
# --------------------------------------------------------------------------
class _Resp:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code


def _rss(entries) -> bytes:
    """entries: [(title, link, pub_datetime|None)]"""
    items = []
    for title, link, when in entries:
        pub = (f"<pubDate>{email.utils.format_datetime(when)}</pubDate>"
               if when is not None else "")
        items.append(f"<item><title>{title}</title><link>{link}</link>"
                     f"{pub}<description>x</description></item>")
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


def _news(sid: str, title: str = "台積電營收創新高", link: str | None = None,
          published: str | None = "2026-10-14", **extra) -> pd.DataFrame:
    row = {"published": published, "stock_id": sid, "title": title,
           "link": link or f"https://example.com/{sid}", "score": 0.33,
           "source": "test", "fetched_at": "2026-10-14T12:00:00",
           "timestamp_source": "source", "score_kind": "scored",
           "provenance": "recorded"}
    row.update(extra)
    return pd.DataFrame([row])


# ==========================================================================
# 狀態詞彙與分類
# ==========================================================================
@pytest.mark.parametrize("cls", ["network", "timeout", "http_status", "parse",
                                 "validation", "write"])
def test_hard_failure_never_classified_as_success_empty(cls):
    """失敗絕不可被降級成「成功但沒有資料」。"""
    got = st.classify(error_class=cls)
    assert got == st.IngestStatus.FAILED.value
    assert got != st.IngestStatus.SUCCESS_EMPTY.value


def test_empty_error_class_is_success_empty_not_failed():
    assert st.classify(error_class="empty") == st.IngestStatus.SUCCESS_EMPTY.value


def test_classify_partial_and_success_with_data():
    assert st.classify(n_ok=2, n_fail=1) == st.IngestStatus.PARTIAL_SUCCESS.value
    assert st.classify(written=3) == st.IngestStatus.SUCCESS_WITH_DATA.value
    assert st.classify(written=0) == st.IngestStatus.SUCCESS_EMPTY.value


def test_secrets_are_redacted_in_records():
    """不得記錄 API 金鑰／密碼。"""
    rec = st.IngestRecord(
        run_id="R", dataset="price", run_at="t",
        error="GET https://api.finmindtrade.com/api/v4/data?token=SECRET123 "
              "failed; password=hunter2",
        notes="apikey=ABCDEF")
    row = rec.as_row()
    for field in ("error", "notes"):
        assert "SECRET123" not in row[field]
        assert "hunter2" not in row[field]
        assert "ABCDEF" not in row[field]
        assert "redacted" in row[field]


# ==========================================================================
# §5-1／§5-2：正常取得資料 vs 正常回應但無資料
# ==========================================================================
def test_feed_success_with_data():
    r = news_rss.fetch_feed(
        "u", getter=_getter(_rss([("利多", "l1", FIXED_NOW - dt.timedelta(hours=2))])))
    assert r.status == news_rss.SUCCESS_WITH_DATA
    assert r.n_entries == 1 and r.http_status == 200


def test_feed_normal_response_without_entries_is_empty_not_failed():
    r = news_rss.fetch_feed("u", getter=_getter(_rss([])))
    assert r.status == news_rss.SUCCESS_EMPTY
    assert r.ok is True


def test_news_empty_result_status_is_success_empty(temp_db):
    res = sentiment.update_news_for("2330", "台積電", getter=_getter(_rss([])),
                                   now=FIXED_NOW)
    assert res.status == st.IngestStatus.SUCCESS_EMPTY.value
    assert res.error_class == st.ErrorClass.EMPTY.value
    assert res.is_failure is False
    log = store.read_sql("SELECT status FROM ingest_run")
    assert log.iloc[0]["status"] == "success_empty"


# ==========================================================================
# §5-3：網路失敗與逾時
# ==========================================================================
def test_feed_network_failure_is_failed_not_empty():
    r = news_rss.fetch_feed("u", getter=_raiser(requests.exceptions.ConnectionError("boom")))
    assert r.status == news_rss.FAILED
    assert r.error_class == news_rss.ERR_NETWORK
    assert r.status != news_rss.SUCCESS_EMPTY


def test_feed_timeout_is_failed_timeout():
    r = news_rss.fetch_feed(
        "u", getter=_raiser(requests.exceptions.ConnectTimeout("timed out")))
    assert r.status == news_rss.FAILED
    assert r.error_class == news_rss.ERR_TIMEOUT


def test_news_fetch_failure_is_recorded_as_failed(temp_db):
    """過去新聞失敗一律被寫成 ok=True；現在必須是 failed。"""
    res = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW,
        getter=_raiser(requests.exceptions.ConnectionError("no route")))
    assert res.status == st.IngestStatus.FAILED.value
    assert res.error_class == st.ErrorClass.NETWORK.value
    assert res.status != st.IngestStatus.SUCCESS_EMPTY.value
    row = store.read_sql("SELECT status, error_class FROM ingest_run").iloc[0]
    assert row["status"] == "failed" and row["error_class"] == "network"
    assert store.read_sql("SELECT COUNT(*) AS n FROM news").iloc[0]["n"] == 0


# ==========================================================================
# §5-4：HTTP 成功但解析失敗
# ==========================================================================
def test_http_200_but_unparseable_body_is_parse_failure():
    r = news_rss.fetch_feed("u", getter=_getter(b"<<< not a feed >>>"))
    assert r.status == news_rss.FAILED
    assert r.error_class == news_rss.ERR_PARSE
    assert r.http_status == 200


def test_http_error_status_is_classified():
    r = news_rss.fetch_feed("u", getter=_getter(b"", status_code=503))
    assert r.status == news_rss.FAILED
    assert r.error_class == news_rss.ERR_HTTP
    assert r.http_status == 503


# ==========================================================================
# §5-5：部分成功
# ==========================================================================
def test_ingest_summary_partial_success_overall():
    recs = [
        st.IngestRecord(run_id="R", dataset="price", run_at="t", stock_id="2330",
                        status=st.IngestStatus.SUCCESS_WITH_DATA.value),
        st.IngestRecord(run_id="R", dataset="price", run_at="t", stock_id="2317",
                        status=st.IngestStatus.FAILED.value),
    ]
    s = st.IngestSummary(run_id="R", statuses={"2330": "success_with_data",
                                               "2317": "failed"}, records=recs)
    assert s.overall_status == st.IngestStatus.PARTIAL_SUCCESS.value
    # 根資料集（價量）未全數失敗 → 不算整輪失敗
    assert s.root_all_failed() is False
    assert s.failed_count == 1 and s.usable_count == 1


def test_ingest_summary_root_all_failed_for_price_failures():
    recs = [st.IngestRecord(run_id="R", dataset="price", run_at="t",
                            stock_id="2317", status="failed")]
    s = st.IngestSummary(run_id="R", statuses={"2317": "failed"}, records=recs)
    assert s.root_all_failed() is True
    assert s.overall_status == st.IngestStatus.FAILED.value


def test_summary_empty_is_not_root_failure():
    """來源正常回應但確實無資料：**不是**失敗（任務第 2 節）。"""
    recs = [st.IngestRecord(run_id="R", dataset="price", run_at="t",
                            stock_id="2330", status="success_empty")]
    s = st.IngestSummary(run_id="R", statuses={"2330": "success_empty"},
                         records=recs)
    assert s.root_all_failed() is False
    assert s.overall_status == st.IngestStatus.SUCCESS_EMPTY.value


# ==========================================================================
# §5-6：寫入失敗與重試
# ==========================================================================
def test_write_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(df, table, key_cols=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is locked")
        return len(df)

    import twstock.pipeline as p
    orig, p.store.write_df = p.store.write_df, flaky
    try:
        df = pd.DataFrame([{"stock_id": "2330", "date": "2026-10-14"}])
        n, cls, err, attempts, wrote = p._write_with_retry(
            df, "price", sleep=lambda s: None)
    finally:
        p.store.write_df = orig
    assert wrote is True and n == 1 and attempts == 2
    assert cls == st.ErrorClass.NONE.value


def test_write_failure_after_retries_is_failed_write():
    def always_fail(df, table, key_cols=None):
        raise RuntimeError("disk full")

    import twstock.pipeline as p
    orig, p.store.write_df = p.store.write_df, always_fail
    try:
        df = pd.DataFrame([{"stock_id": "2330", "date": "2026-10-14"}])
        n, cls, err, attempts, wrote = p._write_with_retry(
            df, "price", sleep=lambda s: None)
    finally:
        p.store.write_df = orig
    assert wrote is False and n == 0
    assert cls == st.ErrorClass.WRITE.value and attempts >= 2
    assert "disk full" in err


def test_write_df_rejects_no_intersecting_columns(temp_db):
    """有資料但欄位無交集時必須拒絕，不可靜默回 0 讓呼叫端記成功。"""
    bad = pd.DataFrame([{"not_a_column": 1}])
    with pytest.raises(store.WriteRejected):
        store.write_df(bad, "price")


# ==========================================================================
# §5-7：重複執行與冪等性
# ==========================================================================
def test_news_write_is_idempotent(temp_db):
    df = _news("2330")
    store.write_df(df, "news")
    store.write_df(df, "news")
    assert store.read_sql("SELECT COUNT(*) AS n FROM news").iloc[0]["n"] == 1


def test_ingest_run_records_do_not_duplicate_on_rerun(temp_db):
    rec = st.IngestRecord(run_id="R1", dataset="news", run_at="t",
                          stock_id="2330", status="success_with_data")
    st.write_records([rec])
    st.write_records([rec, rec])
    assert store.read_sql(
        "SELECT COUNT(*) AS n FROM ingest_run").iloc[0]["n"] == 1


# ==========================================================================
# §5-8：舊資料與 unknown 狀態
# ==========================================================================
def _legacy_db(tmp_path) -> object:
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
    CREATE TABLE ingest_log (run_at TEXT, stock_id TEXT, dataset TEXT,
        ok INTEGER, rows INTEGER, error TEXT,
        PRIMARY KEY (run_at, stock_id, dataset));
    CREATE TABLE news (published DATE, stock_id TEXT, title TEXT, link TEXT,
        score REAL, source TEXT, PRIMARY KEY (stock_id, link));
    """)
    conn.execute("INSERT INTO ingest_log VALUES "
                 "('2026-01-02T00:00:00','2330','price',1,100,'')")
    conn.execute("INSERT INTO ingest_log VALUES "
                 "('2026-01-02T00:00:00','2317','price',0,0,'fetch_failed')")
    conn.execute("INSERT INTO news VALUES "
                 "('2026-01-03','2330','利多','l1',0.5,'x')")
    conn.execute("INSERT INTO news VALUES "
                 "('2026-01-04','2330','中性','l2',0.0,'x')")
    conn.execute("INSERT INTO news VALUES "
                 "('2026-01-05','2330','缺分數','l3',NULL,'x')")
    conn.commit()
    conn.close()
    return db


def test_legacy_ingest_log_backfilled_as_unknown(tmp_path, monkeypatch):
    db = _legacy_db(tmp_path)
    monkeypatch.delenv("TWSTOCK_READONLY", raising=False)
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    df = store.read_sql("SELECT status, error_class, provenance FROM ingest_run")
    assert len(df) == 2
    assert set(df["status"]) == {"unknown"}, "舊紀錄不可被補猜成功"
    assert set(df["provenance"]) == {"unknown"}

    # 遷移必須冪等：再讀一次不會新增紀錄
    store.read_sql("SELECT 1")
    assert store.read_sql(
        "SELECT COUNT(*) AS n FROM ingest_run").iloc[0]["n"] == 2


def test_legacy_news_marked_untraceable_but_score_kind_derived(tmp_path, monkeypatch):
    """舊資料以可靠證據（score 欄位）重建情緒型態，但抓取成敗標為不可追溯。"""
    db = _legacy_db(tmp_path)
    monkeypatch.delenv("TWSTOCK_READONLY", raising=False)
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    df = store.read_sql(
        "SELECT title, score_kind, timestamp_source, provenance FROM news "
        "ORDER BY title")
    assert set(df["provenance"]) == {"unknown"}, "舊新聞不得標為可追溯"
    assert set(df["timestamp_source"]) == {"unknown"}
    assert set(df["score_kind"]) == {"scored", "neutral", "missing"}


def test_legacy_rows_not_counted_as_fetch_success(tmp_path, monkeypatch):
    db = _legacy_db(tmp_path)
    monkeypatch.delenv("TWSTOCK_READONLY", raising=False)
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    rates = report.fetch_rates(st.DATASET_PRICE)
    assert rates["n_unknown"] == 2
    assert rates["n_decided"] == 0
    assert rates["success_rate"] is None, "不可追溯的舊資料不得計入成功率"
    assert rates["failure_rate"] is None


def test_unknown_status_is_not_a_claimed_success():
    s = st.IngestSummary(statuses={"2330": "unknown"},
                         records=[st.IngestRecord(run_id="R", dataset="news",
                                                  run_at="t", stock_id="2330",
                                                  status="unknown")])
    assert s.overall_status == st.IngestStatus.UNKNOWN.value
    assert s.root_all_failed() is True   # 沒有可信結論時不得宣稱成功


# ==========================================================================
# §5-9：新聞中性、缺值、無新聞、抓取失敗的區別
# ==========================================================================
def test_neutral_news_distinguished_from_no_news(temp_db):
    neutral = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW,
        getter=_getter(_rss([("公司發布聲明", "n1", FIXED_NOW - dt.timedelta(hours=1))])))
    assert neutral.status == st.IngestStatus.SUCCESS_WITH_DATA.value
    assert neutral.n_neutral == 1 and neutral.n_scored == 0
    assert neutral.rows_written == 1

    empty = sentiment.update_news_for("2330", "台積電",
                                      getter=_getter(_rss([])), now=FIXED_NOW)
    assert empty.status == st.IngestStatus.SUCCESS_EMPTY.value
    assert empty.rows_written == 0
    # 中性與無新聞狀態不同、資料列也不同
    kinds = store.read_sql("SELECT score_kind FROM news")["score_kind"].tolist()
    assert kinds == ["neutral"]


def test_scored_news_has_valid_sentiment(temp_db):
    got = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW,
        getter=_getter(_rss([("台積電營收創新高法說利多", "s1",
                             FIXED_NOW - dt.timedelta(hours=1))])))
    assert got.status == st.IngestStatus.SUCCESS_WITH_DATA.value
    assert got.n_scored == 1 and got.n_missing_timestamp == 0
    row = store.read_sql("SELECT score, score_kind, provenance FROM news").iloc[0]
    assert row["score"] > 0 and row["score_kind"] == "scored"
    assert row["provenance"] == "recorded"


def test_missing_timestamp_is_not_fabricated(temp_db):
    """沒有發布時間時不得以今天代替（過去就是這樣偽造資料）。"""
    got = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW,
        getter=_getter(_rss([("無時間新聞", "u1", None)])),
        run_at="2026-10-14T12:00:00")
    assert got.n_missing_timestamp == 1
    row = store.read_sql(
        "SELECT published, published_at, timestamp_source FROM news").iloc[0]
    assert row["published"] is None or pd.isna(row["published"])
    assert row["published_at"] is None or pd.isna(row["published_at"])
    assert row["timestamp_source"] == "unknown"
    assert "2026-10-14" not in str(row["published_at"])


def test_failed_fetch_writes_nothing_and_is_distinguishable(temp_db):
    got = sentiment.update_news_for(
        "2330", "台積電", now=FIXED_NOW,
        getter=_raiser(requests.exceptions.ConnectTimeout("timed out")))
    assert got.status == st.IngestStatus.FAILED.value
    assert got.error_class == st.ErrorClass.TIMEOUT.value
    assert store.read_sql("SELECT COUNT(*) AS n FROM news").iloc[0]["n"] == 0
    # 失敗與「沒有新聞」在紀錄上必須不同
    assert got.status != st.IngestStatus.SUCCESS_EMPTY.value


# ==========================================================================
# §5-10：未來新聞不得洩漏至過去訊號
# ==========================================================================
def test_future_dated_news_is_not_stored(temp_db):
    """晚於抓取時間的發布日無法驗證，不得入庫、不得影響訊號。

    時間以 ``now=FIXED_NOW`` 注入，結果不隨真實日期漂移。
    """
    feed = _rss([("未來新聞", "f1", FIXED_NOW + dt.timedelta(days=5))])
    direct = news_rss.fetch_google_news("2330", "台積電",
                                       getter=_getter(feed), now=FIXED_NOW)
    assert direct.n_future_skipped == 1 and direct.n_stored == 0

    got = sentiment.update_news_for("2330", "台積電", now=FIXED_NOW,
                                    getter=_getter(feed),
                                    run_at="2026-10-14T12:00:00")
    assert got.rows_written == 0
    assert got.status == st.IngestStatus.SUCCESS_EMPTY.value
    assert store.read_sql("SELECT COUNT(*) AS n FROM news").iloc[0]["n"] == 0


def test_news_never_affects_dates_before_publication():
    """情緒衰減只往未來擴散；發布日之前的交易日不可看到該新聞。"""
    dates = pd.date_range("2026-10-01", periods=8, freq="D")
    price = pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1e6, "amount": 1e8, "turnover": 0.01})
    pub = dates[4]
    news = pd.DataFrame([{"date": pub, "stock_id": "X", "score": 1.0}])
    f = build_features(price, news_daily=news)
    before = f[f["date"] < pub]
    assert (before["news_present"] == 0).all(), "新聞洩漏到發布前的交易日"
    assert f.loc[f["date"] == pub, "news_present"].iloc[0] == 1
    # 衰減窗口（0/1/2 日）之後不再有影響
    assert f.loc[f["date"] == pub + pd.Timedelta(days=3),
                 "news_present"].iloc[0] == 0


def test_unknown_timestamp_news_is_excluded_from_signals():
    """時間不明的列不可進入情緒窗口（不得被當成「今天的新聞」）。"""
    dates = pd.date_range("2026-10-01", periods=5, freq="D")
    price = pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1e6, "amount": 1e8, "turnover": 0.01})
    news = pd.DataFrame([{"date": None, "stock_id": "X", "score": 1.0}])
    f = build_features(price, news_daily=news)
    assert (f["news_present"] == 0).all()


# ==========================================================================
# §5-11：任務失敗不得被 worker 標記為完整成功
# ==========================================================================
def _summary(statuses: dict) -> st.IngestSummary:
    recs = [st.IngestRecord(run_id="R", dataset="price", run_at="t",
                            stock_id=sid, status=sv)
            for sid, sv in statuses.items()]
    return st.IngestSummary(run_id="R", run_at="t", started_at="t",
                            ended_at="t",
                            counts={k: 1 for k in statuses},
                            statuses=statuses, records=recs)


@pytest.fixture()
def worker_env(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "STATE_DIR", tmp_path)
    monkeypatch.setattr(worker, "_should_run", lambda force=False: (True, "forced"))
    monkeypatch.setattr(pipeline, "run_scores", lambda *a, **k: pd.DataFrame())
    return monkeypatch


def test_worker_raises_when_every_dataset_failed(worker_env):
    """全數失敗不得回報成功（退出碼／心跳都必須是失敗）。"""
    worker_env.setattr(pipeline, "ingest",
                       lambda *a, **k: _summary({"2330": "failed",
                                                 "2317": "failed"}))
    with pytest.raises(RuntimeError):
        worker.run_cycle(force=True)


def test_worker_partial_failure_is_visible_not_full_success(worker_env):
    worker_env.setattr(
        pipeline, "ingest",
        lambda *a, **k: _summary({"2330": "success_with_data",
                                  "2317": "failed"}))
    result = worker.run_cycle(force=True)
    assert result == "partial_success"
    hb = worker.read_heartbeat()
    assert hb["ingest_status"] == "partial_success"
    assert hb["ingest_failed_records"] == 1
    assert hb["ingest_usable_records"] == 1


def test_worker_accepts_normal_empty_source_as_not_failure(worker_env):
    """來源正常回應但當日無資料，不是失敗。"""
    worker_env.setattr(pipeline, "ingest",
                       lambda *a, **k: _summary({"2330": "success_empty"}))
    assert worker.run_cycle(force=True) == "success_empty"
    assert worker.read_heartbeat()["ingest_status"] == "success_empty"


def test_worker_reports_success_with_data_as_ok(worker_env):
    worker_env.setattr(pipeline, "ingest",
                       lambda *a, **k: _summary({"2330": "success_with_data"}))
    assert worker.run_cycle(force=True) == "ok"


def test_worker_raises_even_when_only_auxiliary_datasets_succeeded(worker_env):
    """價量（根資料集）全失敗、但新聞成功 → **仍必須**視為整輪失敗。

    否則會用陳舊價格繼續打分，產生看似正常的訊號（回歸測試）。
    """
    recs = [
        st.IngestRecord(run_id="R", dataset="price", run_at="t",
                        stock_id="2330", status="failed"),
        st.IngestRecord(run_id="R", dataset="price", run_at="t",
                        stock_id="2317", status="failed"),
        st.IngestRecord(run_id="R", dataset="news", run_at="t",
                        stock_id="2330", status="success_with_data"),
    ]
    s = st.IngestSummary(run_id="R", statuses={"2330": "failed",
                                                "2317": "failed"}, records=recs)
    assert s.usable_count > 0            # 確實有其他資料集成功
    assert s.root_all_failed() is True   # 但根資料集全失敗 → 整輪失敗
    worker_env.setattr(pipeline, "ingest", lambda *a, **k: s)
    with pytest.raises(RuntimeError):
        worker.run_cycle(force=True)


def test_worker_runs_when_price_source_responds_empty(worker_env):
    """價量來源正常但當日無資料 → 不應中斷打分（不是失敗）。"""
    s = _summary({"2330": "success_empty"})
    assert s.root_all_failed() is False
    worker_env.setattr(pipeline, "ingest", lambda *a, **k: s)
    assert worker.run_cycle(force=True) == "success_empty"


# ==========================================================================
# 紀錄欄位完整性（§2：每次執行應保留的資訊）
# ==========================================================================
def test_ingest_record_keeps_required_fields():
    rec = st.IngestRecord(
        run_id="R", dataset="news", run_at="2026-10-14T12:00:00",
        source="google_news_rss", task="fetch_google_news", stock_id="2330",
        started_at="2026-10-14T12:00:00", ended_at="2026-10-14T12:00:05",
        status="success_with_data", error_class="none",
        rows_fetched=5, rows_parsed=4, rows_validated=3, rows_written=4,
        data_start="2026-10-10", data_end="2026-10-14", attempts=2,
        source_version="news_rss/v1", config_id="cfg:abc", provenance="recorded")
    row = rec.as_row()
    for key in ("run_id", "dataset", "source", "task", "stock_id",
                "started_at", "ended_at", "status", "error_class", "error",
                "rows_fetched", "rows_parsed", "rows_validated",
                "rows_written", "data_start", "data_end", "attempts",
                "source_version", "config_id", "provenance", "notes"):
        assert key in row, f"紀錄缺少欄位 {key}"
    assert rec.duration_s == pytest.approx(5.0)


# ==========================================================================
# 報告：修復前後的覆蓋率統計可被產出
# ==========================================================================
def test_news_coverage_separates_neutral_missing_and_untraceable(temp_db):
    store.write_df(pd.DataFrame([
        {"published": "2026-10-13", "stock_id": "2330", "title": "a",
         "link": "l1", "score": 0.5, "source": "t",
         "timestamp_source": "source", "score_kind": "scored",
         "provenance": "recorded"},
        {"published": "2026-10-13", "stock_id": "2330", "title": "b",
         "link": "l2", "score": 0.0, "source": "t",
         "timestamp_source": "source", "score_kind": "neutral",
         "provenance": "recorded"},
        {"published": None, "stock_id": "2330", "title": "c", "link": "l3",
         "score": 0.0, "source": "t", "timestamp_source": "unknown",
         "score_kind": "neutral", "provenance": "recorded"},
    ]), "news")
    cov = report.news_coverage()
    assert cov.n_rows == 3
    assert cov.n_scored == 1 and cov.n_neutral == 2
    assert cov.n_with_known_date == 2 and cov.n_unknown_date == 1
    assert cov.scored_rate_all == pytest.approx(1 / 3)
    assert cov.scored_rate_dated == pytest.approx(1 / 2)
    assert cov.n_untraceable == 0


def test_report_text_does_not_claim_trading_edge():
    text = report.build_report_text()
    assert "可交易優勢" in text          # 明確聲明不宣稱優勢
    assert "有效情緒覆蓋率" in text
    assert "抓取失敗率" in text


def test_report_renders_daily_news_counts(temp_db):
    """§3.6 要求明確顯示「每日新聞筆數」——不能只算不顯示。"""
    store.write_df(pd.DataFrame([
        {"published": "2026-10-13", "stock_id": "2330", "title": "a",
         "link": "d1", "score": 0.5, "source": "t",
         "timestamp_source": "source", "score_kind": "scored",
         "provenance": "recorded"},
        {"published": "2026-10-13", "stock_id": "2330", "title": "b",
         "link": "d2", "score": 0.0, "source": "t",
         "timestamp_source": "source", "score_kind": "neutral",
         "provenance": "recorded"},
        {"published": None, "stock_id": "2330", "title": "c", "link": "d3",
         "score": 0.0, "source": "t", "timestamp_source": "unknown",
         "score_kind": "neutral", "provenance": "recorded"},
    ]), "news")
    text = report.build_report_text()
    assert "每日新聞筆數" in text
    assert "2026-10-13 | 2 | 1 | 1 | 0 | 0" in text      # 兩則，其中一則有分數
    assert "(時間不明) | 1 | 0 | 1 | 0 | 1" in text
    per = report.daily_news_counts()
    assert set(per["date"]) == {"2026-10-13", "(時間不明)"}
    assert per[per["date"] == "2026-10-13"].iloc[0]["rows"] == 2


# ==========================================================================
# §2：真實 pipeline.ingest() 是否真的寫出完整可重建的紀錄
# ==========================================================================
def _fake_price(sid: str, days: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2026-10-01", periods=days)
    return pd.DataFrame({"date": dates, "stock_id": sid, "open": 100.0,
                         "high": 101.0, "low": 99.0, "close": 100.0,
                         "volume": 1e6})


def _structured_news(sid: str) -> sentiment.NewsIngestResult:
    """結構化新聞結果（不連外的替身），模擬成功取得並寫入一則。"""
    return sentiment.NewsIngestResult(
        stock_id=sid, status=st.IngestStatus.SUCCESS_WITH_DATA.value,
        rows_fetched=2, rows_parsed=1, rows_written=1, n_scored=1,
        data_start="2026-10-13", data_end="2026-10-13", attempts=1,
        source_version=news_rss.NEWS_RULESET_VERSION)


@pytest.fixture()
def ingest_env(temp_db, monkeypatch):
    """把 pipeline 的所有外部抓取換成可控假函式（**不連外**）。"""
    monkeypatch.setattr(pipeline.fm, "polite_sleep", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.fm, "fetch_institutional",
                        lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pipeline.fm, "fetch_margin", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.fm, "fetch_revenue", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.fm, "fetch_splits", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.fm, "fetch_dividends", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.sent, "update_news_for",
                        lambda sid, name, **k: _structured_news(sid))
    monkeypatch.setattr(pipeline, "CONFIG",
                        type("C", (), {"pool": {"2330": "台積電", "2317": "鴻海"},
                                       "benchmark": "0050",
                                       "lookback_years": 1.0})())
    return monkeypatch


def _run_partial_ingest(env) -> st.IngestSummary:
    env.setattr(pipeline.fm, "fetch_benchmark",
                lambda *a, **k: _fake_price("0050"))
    env.setattr(pipeline.fm, "fetch_price",
                lambda sid, *a, **k: _fake_price(sid) if sid == "2330" else None)
    return pipeline.ingest()


def test_pipeline_ingest_records_all_required_fields(ingest_env):
    """§2：來源、任務、起訖時間、狀態、錯誤分類、各階段筆數、日期範圍、
    重試次數、設定識別都必須真的寫進 DB（不是只有 dataclass 有欄位）。"""
    summary = _run_partial_ingest(ingest_env)
    assert summary.overall_status == st.IngestStatus.PARTIAL_SUCCESS.value
    assert summary.statuses == {"2330": "success_with_data", "2317": "failed"}

    ok = store.read_sql("SELECT * FROM ingest_run WHERE dataset='price' "
                        "AND stock_id='2330'").iloc[0]
    assert ok["status"] == "success_with_data"
    assert ok["source"] == "finmind" and ok["task"] == "fetch_price"
    assert ok["started_at"] and ok["ended_at"] and ok["run_at"]
    assert int(ok["rows_fetched"]) == 3 and int(ok["rows_written"]) == 3
    assert ok["data_start"] == "2026-10-01" and ok["data_end"] == "2026-10-05"
    assert int(ok["attempts"]) >= 1
    assert ok["config_id"] and ok["config_id"] != ""
    assert ok["provenance"] == "recorded"

    bad = store.read_sql("SELECT * FROM ingest_run WHERE dataset='price' "
                         "AND stock_id='2317'").iloc[0]
    assert bad["status"] == "failed" and bad["error_class"] == "network"
    assert bad["error"] and int(bad["rows_written"]) == 0
    # 部分失敗必須能從紀錄辨識，而不是只有一個整體退出碼
    sts = set(store.read_sql("SELECT status FROM ingest_run WHERE dataset='price"
                             "'")["status"])
    assert sts == {"success_with_data", "failed"}


def test_pipeline_records_unstructured_news_as_unknown(ingest_env):
    """非結構化回傳不可冒充成功，只能記 unknown。"""
    ingest_env.setattr(pipeline.sent, "update_news_for", lambda *a, **k: 0)
    _run_partial_ingest(ingest_env)
    news = store.read_sql("SELECT status, error_class FROM ingest_run "
                          "WHERE dataset='news'")
    assert len(news) == 2
    assert set(news["status"]) == {"unknown"}
    assert set(news["error_class"]) == {"unknown"}


def test_pipeline_repeated_run_does_not_duplicate_records(ingest_env, monkeypatch):
    """同一 run_id 重跑不得產生重複紀錄（幂等）。"""
    monkeypatch.setattr(st, "new_run_id", lambda now=None: "FIXED_RUN")
    _run_partial_ingest(ingest_env)
    first = store.read_sql("SELECT COUNT(*) AS n FROM ingest_run").iloc[0]["n"]
    _run_partial_ingest(ingest_env)
    second = store.read_sql("SELECT COUNT(*) AS n FROM ingest_run").iloc[0]["n"]
    assert first == second, "重跑同一任務產生了重複紀錄"


def test_pipeline_never_claims_success_when_everything_fails(ingest_env):
    """全數失敗時：不得更新 last_ingest、紀錄必須是 failed、不可冒充成功。"""
    ingest_env.setattr(pipeline.fm, "fetch_benchmark", lambda *a, **k: None)
    ingest_env.setattr(pipeline.fm, "fetch_price", lambda *a, **k: None)
    summary = pipeline.ingest()
    assert summary.root_all_failed() is True
    assert summary.overall_status == st.IngestStatus.FAILED.value
    assert store.get_meta("last_ingest") is None
    assert store.get_meta("last_ingest_status") == "failed"
    assert store.get_meta("last_ingest_failed_ids") == "2330,2317"
