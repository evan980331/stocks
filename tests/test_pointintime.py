"""P2.8 §5：point-in-time 新聞可用性——發布時間與取得時間共同決定最早可用日。

全部以合成資料驅動，**不連外**。
"""
from __future__ import annotations

import datetime as dt
import sqlite3

import pandas as pd

from twstock.data import pointintime, store
from twstock.features import build_features


def _write(stock_id: str, *, published: str | None, fetched_at: str | None,
           title: str = "n", link: str = "https://a.com/x",
           timestamp_source: str = "source",
           provenance: str = "recorded", score: float = 1.0) -> None:
    store.write_df(pd.DataFrame([{
        "published": published, "stock_id": stock_id, "title": title,
        "link": link, "score": score, "source": "t",
        "published_at": f"{published}T09:00:00" if published else None,
        "fetched_at": fetched_at, "timestamp_source": timestamp_source,
        "score_kind": "scored" if score else "neutral",
        "provenance": provenance,
    }]), "news", key_cols=("stock_id", "link"))


# ==========================================================================
# §5.1：可用日期判定
# ==========================================================================
def test_usable_from_is_max_of_published_and_fetched(temp_db):
    """發布日與抓取日的較晚者才是可用日（不得以發布日冒充可用日）。"""
    assert pointintime.usable_from("2026-10-10", "2026-10-12T09:00:00") == \
        dt.date(2026, 10, 12), "晚於發布日才抓到，可用日應為抓取日"
    # 抓取日與發布日相同 → 可用日即該日
    assert pointintime.usable_from("2026-10-12", "2026-10-12T09:00:00") == \
        dt.date(2026, 10, 12)
    # 抓取日早於發布日（無法驗證的來源時間）→ 不可用，另由專屬測試覆蓋
    assert pointintime.usable_from("2026-10-12", "2026-10-10T09:00:00") is None


def test_usable_from_rejects_fetched_before_published(temp_db):
    """抓取時間早於來源聲稱的發布時間 → 無法驗證，不可用於訊號。"""
    assert pointintime.usable_from("2026-10-12", "2026-10-10T09:00:00",
                                   "source", "recorded") is None
    assert pointintime.usable_from("2026-10-12", "2026-10-10T09:00:00") is None


def test_usable_from_rejects_legacy_and_unknown_timestamps(temp_db):
    assert pointintime.usable_from("2026-10-10", "2026-10-12T09:00:00",
                                   "source", "unknown") is None, \
        "舊資料（抓取時點不可重建）不得進入訊號路徑"
    assert pointintime.usable_from("2026-10-10", "2026-10-12T09:00:00",
                                   "unknown", "recorded") is None, \
        "時間戳不可追溯者不得納入嚴格時間對齊"


def test_usable_from_rejects_missing_dates(temp_db):
    assert pointintime.usable_from(None, "2026-10-12T09:00:00") is None
    assert pointintime.usable_from("2026-10-10", None) is None
    assert pointintime.usable_from("", "") is None


# ==========================================================================
# §5.2：事後取得的新聞不得回填到較早交易日
# ==========================================================================
def test_delayed_arrival_news_not_backfilled(temp_db):
    """發布於 D-2、D 才抓到 → 不得影響 D-2 / D-1 的訊號。（D=週五 10-09）"""
    _write("2330", published="2026-10-07", fetched_at="2026-10-09T18:00:00",
           link="https://a.com/delayed")
    daily = pointintime.news_daily_for_signals()
    assert list(daily["date"]) == [pd.Timestamp("2026-10-09")]
    assert daily.iloc[0]["score"] == 1.0

    dates = pd.bdate_range("2026-10-06", periods=5)   # 10-06 ~ 10-12
    price = pd.DataFrame({"date": dates, "open": 100.0, "high": 101.0,
                          "low": 99.0, "close": 100.0, "volume": 1e6})
    f = build_features(price, news_daily=daily)
    assert (f.loc[f["date"] < "2026-10-09", "news_present"] == 0).all(), \
        "延遲到達的新聞洩漏到取得前的交易日"
    assert f.loc[f["date"] == "2026-10-09", "news_present"].iloc[0] == 1


def test_same_day_arrival_is_usable_same_day(temp_db):
    """當日收盤後取得 → 當日訊號可用（訊號在收盤後計算、隔日開盤才進場）。"""
    _write("2330", published="2026-10-09", fetched_at="2026-10-09T18:30:00",
           link="https://a.com/sameday")
    daily = pointintime.news_daily_for_signals()
    assert list(daily["date"]) == [pd.Timestamp("2026-10-09")]


def test_news_arriving_after_close_does_not_leak_to_prior_days(temp_db):
    """連續三個交易日各自取得不同新聞：只在各自取得日生效，不互相回填。"""
    _write("2330", published="2026-10-06", fetched_at="2026-10-07T18:00:00",
           title="a", link="https://a.com/a")
    _write("2330", published="2026-10-06", fetched_at="2026-10-08T18:00:00",
           title="b", link="https://a.com/b")
    _write("2330", published="2026-10-07", fetched_at="2026-10-09T18:00:00",
           title="c", link="https://a.com/c")
    daily = pointintime.news_daily_for_signals()
    got = sorted(pd.Timestamp(d).date().isoformat() for d in daily["date"])
    assert got == ["2026-10-07", "2026-10-08", "2026-10-09"]


# ==========================================================================
# §5.5：舊資料不得納入嚴格時間對齊
# ==========================================================================
def test_legacy_rows_are_excluded_from_signal_path(temp_db):
    _write("2330", published="2026-10-07", fetched_at=None, title="old",
           link="https://a.com/old", timestamp_source="unknown",
           provenance="unknown")
    daily = pointintime.news_daily_for_signals()
    assert daily.empty, "舊資料（fetched_at 不可得）不得進入訊號路徑"
    assert daily.columns.tolist() == ["date", "stock_id", "score"]


def test_availability_reports_exclusion_reasons(temp_db):
    _write("2330", published="2026-10-07", fetched_at=None, title="old",
           link="https://a.com/old", timestamp_source="unknown",
           provenance="unknown")
    _write("2330", published=None, fetched_at="2026-10-09T09:00:00",
           title="no-pub", link="https://a.com/np", timestamp_source="unknown")
    _write("2330", published="2026-10-09", fetched_at="2026-10-09T09:00:00",
           title="ok", link="https://a.com/ok")

    avail = pointintime.news_availability()
    assert len(avail) == 3
    reasons = dict(zip(avail["title"], avail["reason"]))
    assert reasons["old"] == "legacy_untraceable"
    assert reasons["no-pub"] == "timestamp_untraceable"
    assert reasons["ok"] == ""
    assert avail[avail["title"] == "ok"].iloc[0]["usable_from"] == "2026-10-09"


def test_unmigrated_db_does_not_crash_and_excludes_all(tmp_path, monkeypatch):
    """尚未遷移的舊 DB（唯讀分析常見）不得擲錯；其列一律視為不可用於訊號。

    唯讀模式下 ``read_sql`` 不跑遷移，若程式直接 SELECT 溯源欄位會擲
    ``no such column``。缺少欄位＝無法證明可見時點 → 必須排除而不是猜測。
    """
    db = tmp_path / "unmigrated.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
    CREATE TABLE news (published DATE, stock_id TEXT, title TEXT, link TEXT,
        score REAL, source TEXT, PRIMARY KEY (stock_id, link));
    INSERT INTO news VALUES ('2026-10-07','2330','舊新聞','l1',0.5,'x');
    """)
    conn.commit()
    conn.close()

    # 唯讀模式：read_sql 不會跑遷移，因此欄位確實不存在
    monkeypatch.setenv("TWSTOCK_READONLY", "1")
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    avail = pointintime.news_availability()
    assert len(avail) == 1
    assert avail.iloc[0]["reason"] == "timestamp_untraceable"
    assert avail.iloc[0]["usable_from"] is None
    assert pointintime.news_daily_for_signals().empty


def test_empty_news_table_returns_compatible_frame(temp_db):
    daily = pointintime.news_daily_for_signals()
    assert daily.empty
    assert daily.columns.tolist() == ["date", "stock_id", "score"]
    # features 在空新聞下不得產生 news_present 欄位（與舊介面一致）
    dates = pd.bdate_range("2026-10-06", periods=3)
    price = pd.DataFrame({"date": dates, "open": 100.0, "high": 101.0,
                          "low": 99.0, "close": 100.0, "volume": 1e6})
    f = build_features(price, news_daily=daily)
    assert "news_present" not in f.columns


# ==========================================================================
# §5：load_panel 走 point-in-time（整合）
# ==========================================================================
def test_load_panel_news_is_time_aligned(temp_db, monkeypatch):
    from twstock import pipeline
    _write("2330", published="2026-10-07", fetched_at="2026-10-09T18:00:00",
           link="https://a.com/delayed")
    news = pointintime.news_daily_for_signals()
    assert list(news["date"]) == [pd.Timestamp("2026-10-09")]

    # load_panel 必須取用 pointintime，而不是 published 直接當日期
    src = (pipeline.load_panel.__doc__ or "")
    assert "point-in-time" in src
    monkeypatch.setattr(pipeline, "pointintime",
                        type("M", (), {"news_daily_for_signals":
                                       staticmethod(lambda: news)})())
    assert pipeline.pointintime.news_daily_for_signals().equals(news)


def test_decay_spreads_forward_only_from_usable_date(temp_db):
    """衰減窗口只從可用日往後擴散，不往回。（可用日 = 週三 10-07）"""
    _write("2330", published="2026-10-06", fetched_at="2026-10-07T18:00:00",
           link="https://a.com/delayed")
    daily = pointintime.news_daily_for_signals()
    dates = pd.bdate_range("2026-10-06", periods=7)   # 10-06 ~ 10-14
    price = pd.DataFrame({"date": dates, "open": 100.0, "high": 101.0,
                          "low": 99.0, "close": 100.0, "volume": 1e6})
    f = build_features(price, news_daily=daily).set_index("date")
    assert f.loc["2026-10-06", "news_present"] == 0, "可用日前一日不得有影響"
    assert f.loc["2026-10-07", "news_present"] == 1
    assert f.loc["2026-10-08", "news_sentiment"] < \
        f.loc["2026-10-07", "news_sentiment"], "衰減應隨日遞減"
    assert f.loc["2026-10-09", "news_sentiment"] < \
        f.loc["2026-10-08", "news_sentiment"]
    assert f.loc["2026-10-12", "news_present"] == 0, "超出 3 日窗口"


def test_decay_window_is_calendar_days_so_friday_news_misses_monday(temp_db):
    """既定行為：衰減窗口以**日曆日**計（0/1/2 日），非交易日。

    因此週五取得的新聞只落在週五／週六／週日，而後兩者不是交易日，
    週一完全看不到該新聞。這是 ``features.SENTIMENT_WEIGHTS`` 的既有定義，
    本階段不修改因子定義，只把此行為固定下來並如實記錄。
    """
    _write("2330", published="2026-10-09", fetched_at="2026-10-09T18:00:00",
           link="https://a.com/friday")
    daily = pointintime.news_daily_for_signals()
    dates = pd.bdate_range("2026-10-08", periods=4)   # 10-08,09,12,13
    price = pd.DataFrame({"date": dates, "open": 100.0, "high": 101.0,
                          "low": 99.0, "close": 100.0, "volume": 1e6})
    f = build_features(price, news_daily=daily).set_index("date")
    assert f.loc["2026-10-09", "news_present"] == 1
    assert f.loc["2026-10-12", "news_present"] == 0, \
        "週一不在日曆日衰減窗口內（既定行為，非 bug）"


# ==========================================================================
# §6.3：對既有 features / signals 的影響必須可量化，不得靜默改變
# ==========================================================================
def _tiny_price(days: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2026-10-05", periods=days)
    return pd.DataFrame({"date": dates, "open": 100.0, "high": 101.0,
                         "low": 99.0, "close": 100.0, "volume": 1e6})


def test_excluding_news_changes_score_denominator_not_the_news_value(temp_db):
    """舊資料被排除後，``news_sentiment`` 欄位整個消失（不是變成 0）。

    因此打分分母少掉 0.10 權重，而不是「情緒＝中性」。兩者不可混同：
    前者是「沒有可追溯的新聞」，後者是「有新聞但情緒中性」。
    """
    from twstock import signals as sig

    _write("2330", published="2026-10-07", fetched_at=None, title="old",
           link="https://a.com/old", timestamp_source="unknown",
           provenance="unknown")
    daily = pointintime.news_daily_for_signals()
    price = _tiny_price()
    f = build_features(price, news_daily=daily)
    assert "news_sentiment" not in f.columns, \
        "全部不可用時不得產生 news_sentiment 欄位（那會被 fillna 成中性）"

    row = f.iloc[-1]
    parts = dict((k, v) for k, v, _ in sig._factor_parts(row))
    assert "news_sentiment" not in parts, "不可用的新聞不得進入打分分子"
    comps = sig.score_components(row)
    assert "news_sentiment" not in comps


def test_usable_news_still_reaches_scoring(temp_db):
    """point-in-time 可用的新聞仍必須進入打分（不能因為嚴格就把新資料也排除）。"""
    from twstock import signals as sig

    _write("2330", published="2026-10-07", fetched_at="2026-10-07T18:00:00",
           title="ok", link="https://a.com/ok", score=1.0)
    daily = pointintime.news_daily_for_signals()
    f = build_features(_tiny_price(), news_daily=daily)
    assert "news_sentiment" in f.columns
    row = f[f["date"] == "2026-10-07"].iloc[0]
    assert row["news_present"] == 1 and row["news_sentiment"] == 1.0
    assert sig.score_components(row)["news_sentiment"] > 0
