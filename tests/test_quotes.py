"""P3：行情來源介面、資料品質與可靠性測試。

全部以**注入的假 getter／假來源**驅動，**不連任何外部網路**；
合成行情一律標示為假資料（``FakeQuoteSource.is_fabricated``），
因此測試通過只證明「介面與管線正確」，**不證明任何真實供應商穩定**。

對應任務要求（§7 測試與交付）
  1. 正常行情解析及寫入
  2. 重複行情去重
  3. 舊行情不得覆蓋新行情
  4. 來源時間與取得時間分離
  5. 延遲行情正確分類
  6. 斷線、限流、格式錯誤及重試
  7. 交易時段與非交易時段判斷
  8. 過期資料不得觸發有效交易訊號
  9. Worker 重啟及單實例保護
 10. 正式資料庫安全與既有功能回歸
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from twstock import worker
from twstock.data import quotes, store
from twstock.data.quotes import (Assessment, FakeQuoteSource, HealthReport,
                                 MisQuoteSource, Quote, QuoteFetch, QuoteKind,
                                 QuoteStatus, SessionState, SourceHealth)

# 2026-10-16 為週五，10:23:45 在 09:00–13:30 連續交易時段內（台股無午休）。
TRADING_DAY = dt.datetime(2026, 10, 16, 10, 23, 45)
FETCHED = "2026-10-16T10:23:50"


def in_session(hh: int, mm: int = 0, ss: int = 0) -> dt.datetime:
    return TRADING_DAY.replace(hour=hh, minute=mm, second=ss, microsecond=0)


# --------------------------------------------------------------------------
# 測試替身（**不連網**）
# --------------------------------------------------------------------------
class FakeResponse:
    """最小 HTTP 回應替身。"""

    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class Clock:
    """可控制的單調時鐘：sleep 會推進時間，使限流與退避完全確定。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self.slept: list[float] = []

    def mono(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def mis_payload(code: str = "2330", *, price: str = "1050.0000",
                prev: str = "1040.0000", date: str = "20261016",
                time_: str = "10:23:45") -> dict:
    return {
        "msgArray": [{
            "c": code, "d": date, "t": time_, "z": price, "y": prev,
            "o": "1045.0000", "h": "1055.0000", "l": "1042.0000",
            "v": "12345", "tv": "5",
            "b": "1049.0000_1048.0000_1047.0000_1046.0000_1045.0000_",
            "a": "1050.0000_1051.0000_1052.0000_1053.0000_1054.0000_",
            "tlong": "1761000000000",
        }],
        "rtcode": "0000", "rtmessage": "OK",
    }


def mk_quote(stock_id: str = "2330", *, market_time: str = "",
             fetched_at: str = FETCHED, kind: str = QuoteKind.DELAYED.value,
             price: float | None = 100.0, prev_close: float | None = 100.0,
             time_source: str = quotes.TIME_SOURCE, source: str = "fake",
             total_volume: float | None = 10000.0) -> Quote:
    return Quote(stock_id=stock_id, source=source, kind=kind, price=price,
                 prev_close=prev_close, total_volume=total_volume,
                 market_time=market_time or in_session(10, 0).isoformat(),
                 market_time_raw=(market_time or in_session(10, 0).isoformat()
                                  ).replace("T", " "),
                 market_time_source=time_source, fetched_at=fetched_at,
                 market_state="open")


def rows(table: str) -> pd.DataFrame:
    return store.read_sql(f"SELECT * FROM {table}")


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    """與 test_worker_hardening 相同的隔離執行狀態目錄。"""
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(worker, "STATE_DIR", d)
    return d


# ==========================================================================
# 1. 正常行情解析及寫入
# ==========================================================================
def test_mis_payload_parses_all_documented_fields():
    res = quotes.parse_mis_payload(mis_payload(), ["2330"], fetched_at=FETCHED)
    assert res.ok, res.error
    assert res.resolved == 1 and res.malformed == 0
    q = res.quotes[0]
    assert q.stock_id == "2330"
    assert q.price == 1050.0 and q.prev_close == 1040.0
    assert q.open == 1045.0 and q.high == 1055.0 and q.low == 1042.0
    assert q.total_volume == 12345.0 and q.last_volume == 5.0
    assert q.bid == 1049.0 and q.ask == 1050.0
    assert q.market_time == "2026-10-16T10:23:45"


def test_mis_ignores_rtcode_and_detects_unmatched_symbol():
    """無效代碼仍回 HTTP 200／rtcode=0000，但 ``c`` 為空 → 不得判為成功。"""
    bad = {"msgArray": [{"c": "", "d": "20261016", "t": "10:23:45",
                         "z": "-", "y": "-"}], "rtcode": "0000"}
    res = quotes.parse_mis_payload(bad, ["2330"], fetched_at=FETCHED)
    assert res.ok is False
    assert res.resolved == 0
    assert res.error_class == "parse" and res.error == "no_symbol_matched_request"
    # 關鍵：這不是「來源正常但沒有行情」
    assert res.status == "failed", "不得降級成 success_empty"


def test_normal_quote_is_validated_and_written(temp_db):
    res = quotes.parse_mis_payload(mis_payload(), ["2330"], fetched_at=FETCHED)
    asmts = quotes.validate_batch(res.quotes, now=in_session(10, 24))
    assert all(a.accepted for a in asmts)
    summ = quotes.write_quotes(asmts, run_id="r1", source="mis",
                               kind=QuoteKind.REALTIME.value)
    assert summ.written == 1 and summ.rejected == 0
    df = rows("quote")
    assert len(df) == 1
    r = df.iloc[0]
    assert (r["source"], r["kind"], r["stock_id"]) == ("mis", "realtime", "2330")
    assert r["price"] == 1050.0 and r["status"] == QuoteStatus.OK.value
    assert r["run_id"] == "r1"


def test_finmind_minute_parses_last_row_and_tolerates_column_renames():
    rows_in = [{"date": "2026-10-16 10:00:00", "close": 100.0, "volume": 10},
               {"date": "2026-10-16 10:01:00", "deal_price": 101.5,
                "deal_volume": 20}]
    q = quotes.parse_finmind_minute(rows_in, "2330", fetched_at=FETCHED)
    assert q.price == 101.5 and q.last_volume == 20.0
    assert q.market_time.startswith("2026-10-16T10:01:00")
    assert q.kind == QuoteKind.DELAYED.value


def test_finmind_unknown_columns_is_parse_failure_not_guessed():
    q = quotes.parse_finmind_minute([{"foo": 1, "bar": 2}], "2330",
                                    fetched_at=FETCHED)
    assert q.error_class == "parse" and "unknown_columns" in q.error


def test_finmind_empty_list_is_empty_not_failure():
    q = quotes.parse_finmind_minute([], "2330", fetched_at=FETCHED)
    assert q.error_class == "empty"


# ==========================================================================
# 2. 重複行情去重
# ==========================================================================
def test_same_quote_written_only_once(temp_db):
    a = quotes.validate_batch([mk_quote()], now=in_session(10, 1))
    quotes.write_quotes(a, run_id="r1")
    b = quotes.validate_batch([mk_quote()], now=in_session(10, 1))
    summ = quotes.write_quotes(b, run_id="r2")
    assert summ.written == 0 and summ.rejected == 1
    assert summ.reasons == {"not_newer_than_stored": 1}
    assert len(rows("quote")) == 1, "重複行情不得重複寫入"


def test_duplicate_inside_one_batch_is_collapsed():
    q = mk_quote()
    asmts = quotes.validate_batch([q, q], now=in_session(10, 1))
    assert [a.accepted for a in asmts] == [True, False]
    assert asmts[1].reason == "duplicate_in_batch"


def test_write_is_idempotent_at_store_level(temp_db):
    """同一主鍵（來源／種類／標的／市場時間）以 if_conflict=ignore 寫入不新增列。"""
    df = pd.DataFrame([mk_quote().as_row()])
    assert store.write_df(df, "quote", if_conflict="ignore") == 1
    assert store.write_df(df, "quote", if_conflict="ignore") == 0
    assert len(rows("quote")) == 1


# ==========================================================================
# 3. 舊行情不得覆蓋新行情
# ==========================================================================
def test_older_market_time_is_rejected_against_stored(temp_db):
    # 市場時間不得晚於取得時間（時鐘一致性）；取得時間取 10:31
    fetched = in_session(10, 31).isoformat()
    newer = quotes.validate_batch([mk_quote(
        market_time=in_session(10, 30).isoformat(), fetched_at=fetched)],
        now=in_session(10, 31))
    assert quotes.write_quotes(newer, run_id="r1").written == 1
    older = quotes.validate_batch([mk_quote(
        market_time=in_session(10, 0).isoformat(), fetched_at=fetched)],
        now=in_session(10, 31))
    summ = quotes.write_quotes(older, run_id="r2")
    assert summ.written == 0 and summ.reasons == {"not_newer_than_stored": 1}
    stored = rows("quote")
    assert len(stored) == 1
    assert stored.iloc[0]["market_time"] == "2026-10-16T10:30:00"


def test_older_quote_is_rejected_even_within_same_batch_vs_previous():
    fetched = in_session(10, 31).isoformat()
    newer = mk_quote(market_time=in_session(10, 30).isoformat(),
                     fetched_at=fetched)
    older = mk_quote(market_time=in_session(10, 5).isoformat(),
                     fetched_at=fetched)
    asmts = quotes.validate_batch([newer, older], now=in_session(10, 31))
    assert asmts[0].accepted and not asmts[1].accepted
    assert asmts[1].reason == "duplicate_in_batch" or \
        asmts[1].reason == "not_newer_than_stored"


def test_newer_quote_is_appended_and_latest_quotes_returns_newest(temp_db):
    for hh, mm, px in ((10, 0, 100.0), (10, 30, 101.0)):
        a = quotes.validate_batch([mk_quote(
            market_time=in_session(hh, mm).isoformat(), price=px,
            fetched_at=in_session(hh, mm + 1).isoformat())],
            now=in_session(hh, mm + 1))
        assert quotes.write_quotes(a, run_id=f"r{hh}{mm}").written == 1
    df = rows("quote")
    assert len(df) == 2, "不同市場時間是不同快照，不得互相覆蓋"
    latest = quotes.latest_quotes(source="fake", kind=QuoteKind.DELAYED.value)
    assert len(latest) == 1 and latest.iloc[0]["price"] == 101.0


# ==========================================================================
# 4. 來源時間與取得時間分離
# ==========================================================================
def test_market_time_and_fetched_at_are_separate_facts():
    q = mk_quote(market_time=in_session(10, 23, 45).isoformat())
    assert q.market_time == "2026-10-16T10:23:45"
    assert q.fetched_at == FETCHED
    assert q.market_time != q.fetched_at
    row = q.as_row()
    assert row["market_time"] == q.market_time
    assert row["fetched_at"] == q.fetched_at
    assert row["market_time_raw"] == "2026-10-16 10:23:45"   # 來源原樣字串


def test_unknown_time_source_must_not_pass_validation():
    """取不到來源時間 → 不得當成新行情（也不得以取得時間冒充市場時間）。"""
    q = mk_quote(time_source=quotes.TIME_UNKNOWN, market_time="")
    a = quotes.validate_quote(q, now=in_session(10, 30))
    assert not a.accepted and a.reason == "market_time_unknown"
    assert a.error_class == "validation"


def test_mis_time_derived_from_tlong_is_marked_derived():
    payload = mis_payload()
    row = payload["msgArray"][0]
    del row["d"], row["t"]
    res = quotes.parse_mis_payload(payload, ["2330"], fetched_at=FETCHED)
    q = res.quotes[0]
    assert q.market_time_source == quotes.TIME_DERIVED
    assert "time_from_tlong" in q.notes and q.market_time


def test_mis_time_missing_entirely_is_unknown_and_rejected():
    payload = mis_payload()
    row = payload["msgArray"][0]
    del row["d"], row["t"], row["tlong"]
    q = quotes.parse_mis_payload(payload, ["2330"], fetched_at=FETCHED).quotes[0]
    assert q.market_time_source == quotes.TIME_UNKNOWN
    assert quotes.validate_quote(q, now=in_session(10, 30)).reason == \
        "market_time_unknown"


def test_market_time_after_fetch_is_rejected():
    """市場時間晚於取得時間超過容許值 → 時鐘有問題，不得寫入。"""
    q = mk_quote(market_time=in_session(12, 0).isoformat(),
                 fetched_at=in_session(10, 0).isoformat())
    assert quotes.validate_quote(q, now=in_session(12, 1)).reason == \
        "market_time_after_fetch"


# ==========================================================================
# 5. 延遲行情正確分類
# ==========================================================================
def test_four_quote_kinds_are_distinct():
    assert QuoteKind.REALTIME.is_intraday and QuoteKind.DELAYED.is_intraday
    assert not QuoteKind.EOD.is_intraday and not QuoteKind.HISTORICAL.is_intraday
    assert QuoteKind.REALTIME is not QuoteKind.DELAYED
    for k in QuoteKind:
        assert k.label


def test_delayed_source_is_not_advertised_as_realtime():
    """唯一預設啟用的來源必須是「延遲」，不得自稱即時。"""
    enabled = [s for s in quotes.SOURCE_CATALOG if s.enabled_by_default]
    assert enabled, "至少要有一個預設啟用的來源"
    assert {s.kind for s in enabled} == {QuoteKind.DELAYED.value}


def test_finmind_minute_kind_is_delayed():
    q = quotes.parse_finmind_minute(
        [{"date": "2026-10-16 10:00:00", "close": 100.0}], "2330")
    assert q.kind == QuoteKind.DELAYED.value


# ==========================================================================
# 6. 斷線、限流、格式錯誤及重試
# ==========================================================================
def test_network_failure_is_classified_and_retried_then_succeeds():
    calls = {"n": 0}

    def getter(url, *, params=None, timeout=10):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("連線中斷")
        return FakeResponse(200, mis_payload())

    clk = Clock()
    src = MisQuoteSource(getter=getter, sleep=clk.sleep, clock=clk.mono,
                         now=lambda: in_session(10, 24))
    res = src.fetch(["2330"])
    assert res.ok and calls["n"] == 2
    assert res.attempts == 2, "應重試一次"
    assert res.notes.endswith("retries=1")
    assert clk.slept[0] == quotes.RETRY_BASE_DELAY_S, "重試必須先退避"


def test_network_failure_exhausts_retries_and_reports_failure():
    def getter(url, *, params=None, timeout=10):
        raise ConnectionError("連線中斷")

    clk = Clock()
    src = MisQuoteSource(getter=getter, sleep=clk.sleep, clock=clk.mono,
                         now=lambda: in_session(10, 24))
    res = src.fetch(["2330"])
    assert res.ok is False
    assert res.error_class == "network" and "ConnectionError" in res.error
    assert res.attempts == quotes.DEFAULT_FETCH_ATTEMPTS
    # 關鍵：來源失敗**不得**降級成「正常無行情」
    assert res.status == "failed"


def test_http_429_is_treated_as_rate_limit_and_retried():
    clk = Clock()
    src = MisQuoteSource(getter=lambda url, **kw: FakeResponse(429, {}),
                         sleep=clk.sleep, clock=clk.mono,
                         now=lambda: in_session(10, 24))
    res = src.fetch(["2330"])
    assert res.error_class == "http_status" and res.http_status == 429
    assert res.attempts == quotes.DEFAULT_FETCH_ATTEMPTS
    assert len(clk.slept) >= 1


def test_parse_error_is_not_retried():
    calls = {"n": 0}

    def getter(url, *, params=None, timeout=10):
        calls["n"] += 1
        return FakeResponse(200, {"unexpected": "shape"})

    clk = Clock()
    src = MisQuoteSource(getter=getter, sleep=clk.sleep, clock=clk.mono,
                         now=lambda: in_session(10, 24))
    res = src.fetch(["2330"])
    assert res.error_class == "parse" and res.error == "missing_msgArray"
    assert calls["n"] == 1, "格式改變重試無用，不得重試"
    assert clk.slept == []


def test_json_decode_failure_is_parse_error():
    class Boom(FakeResponse):
        def json(self):
            raise ValueError("not json")

    src = MisQuoteSource(getter=lambda url, **kw: Boom(200, None),
                         sleep=lambda s: None, now=lambda: in_session(10, 24))
    res = src.fetch(["2330"], attempts=1)
    assert res.error_class == "parse" and "json_decode" in res.error


def test_no_symbols_requested_is_empty_not_network_error():
    src = FakeQuoteSource()
    res = src.fetch([])
    assert res.error_class == "empty" and res.ok is False


def test_rate_limiter_never_polls_faster_than_source_snapshot():
    clk = Clock()
    rl = quotes.RateLimiter(5.0, sleep=clk.sleep, clock=clk.mono)
    assert rl.wait() == 0.0, "第一次不等待"
    assert rl.wait() == 5.0, "立即再抓必須等滿最小間隔"
    assert clk.slept == [5.0]
    assert quotes.MIS_MIN_INTERVAL_S >= 5.0, "不得比交易所 5 秒快照更密"


def test_unknown_source_name_is_rejected(monkeypatch):
    monkeypatch.setenv("TWSTOCK_QUOTE_SOURCE", "not-a-source")
    with pytest.raises(KeyError):
        quotes.resolve_source()


def test_unset_source_env_resolves_to_none(monkeypatch):
    monkeypatch.delenv("TWSTOCK_QUOTE_SOURCE", raising=False)
    assert quotes.resolve_source() is None, "不得自行臆測預設來源"


def test_mis_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("TWSTOCK_QUOTE_ALLOW_MIS", raising=False)
    ok, why = MisQuoteSource().available()
    assert ok is False and "TWSTOCK_QUOTE_ALLOW_MIS" in why
    monkeypatch.setenv("TWSTOCK_QUOTE_ALLOW_MIS", "1")
    assert MisQuoteSource().available()[0] is True


# ==========================================================================
# 7. 交易時段與非交易時段判斷
# ==========================================================================
@pytest.mark.parametrize("when,expected", [
    (dt.datetime(2026, 10, 16, 7, 0), SessionState.CLOSED),
    (dt.datetime(2026, 10, 16, 8, 45), SessionState.PRE_OPEN),
    (dt.datetime(2026, 10, 16, 9, 0), SessionState.OPEN),
    (dt.datetime(2026, 10, 16, 13, 30), SessionState.OPEN),
    (dt.datetime(2026, 10, 16, 13, 31), SessionState.CLOSED),
    (dt.datetime(2026, 10, 17, 11, 0), SessionState.WEEKEND),   # 週六
])
def test_session_state_boundaries(when, expected):
    assert quotes.session_state(when) is expected
    assert quotes.is_trading_session(when) is (expected in
                                               (SessionState.PRE_OPEN,
                                                SessionState.OPEN))


def test_quote_outside_session_is_rejected():
    q = mk_quote(market_time=dt.datetime(2026, 10, 16, 14, 0).isoformat())
    assert quotes.validate_quote(q, now=in_session(10, 30)).reason == \
        "market_time_outside_session"


def test_weekend_timestamp_is_rejected():
    q = mk_quote(market_time=dt.datetime(2026, 10, 17, 11, 0).isoformat())
    assert quotes.validate_quote(q, now=in_session(10, 30)).reason == \
        "market_time_on_weekend"


def test_symbol_must_belong_to_pool():
    assert quotes.validate_quote(mk_quote("9999"),
                                 now=in_session(10, 30)).reason == \
        "symbol_not_in_pool"
    assert quotes.validate_quote(mk_quote(""), now=in_session(10, 30)).reason == \
        "symbol_missing"


def test_poll_skips_outside_trading_session_when_required(monkeypatch, temp_db):
    src = FakeQuoteSource(now=lambda: dt.datetime(2026, 10, 17, 11, 0))
    res = quotes.poll_once(["2330"], source=src, require_session=True,
                           now=dt.datetime(2026, 10, 17, 11, 0))
    assert res.skipped_reason and "非交易時段" in res.skipped_reason
    assert src.calls == 0, "非交易時段不得抓取"
    assert res.write.written == 0


# ==========================================================================
# 8. 過期資料不得觸發有效交易訊號
# ==========================================================================
def _health(*, health: str, age: float | None = 0.0, fabricated: bool = False,
            session: str = "open", last: str = "2026-10-16T10:30:00"):
    return HealthReport(source="mis", kind="realtime", health=health,
                        session=session, age_s=age, fabricated=fabricated,
                        last_market_time=last)


def test_fresh_healthy_quote_allows_intraday_gate():
    ok, why = quotes.intraday_signal_gate(_health(health=SourceHealth.NORMAL.value))
    assert ok and why == ""


@pytest.mark.parametrize("health", [SourceHealth.FAILED.value,
                                    SourceHealth.DELAYED.value,
                                    SourceHealth.UNKNOWN.value,
                                    SourceHealth.NON_TRADING_SESSION.value])
def test_stale_or_broken_health_blocks_intraday_gate(health):
    ok, why = quotes.intraday_signal_gate(_health(health=health))
    assert ok is False and why, "資料異常時禁止產生看似有效的即時訊號"


def test_fabricated_data_never_passes_gate():
    ok, why = quotes.intraday_signal_gate(
        _health(health=SourceHealth.NORMAL.value, fabricated=True))
    assert ok is False and "假" in why


def test_gate_blocks_when_no_market_time_available():
    ok, why = quotes.intraday_signal_gate(
        _health(health=SourceHealth.NORMAL.value, last=""))
    assert ok is False and "行情時間" in why


def test_quote_health_marks_stale_data_delayed(temp_db):
    quotes.write_quotes(quotes.validate_batch(
        [mk_quote(market_time=in_session(10, 0).isoformat())],
        now=in_session(10, 1)), run_id="r1", source="fake",
        kind=QuoteKind.DELAYED.value)
    df = quotes.latest_quotes(source="fake", kind=QuoteKind.DELAYED.value)
    rep = quotes.quote_health(df, source="fake", kind=QuoteKind.DELAYED.value,
                              now=in_session(10, 30), max_age_s=300)
    assert rep.health == SourceHealth.DELAYED.value
    assert rep.age_s == 1800.0
    assert quotes.intraday_signal_gate(rep)[0] is False


def test_quote_health_reports_failed_on_source_error(temp_db):
    rep = quotes.quote_health(pd.DataFrame(), source="mis", kind="realtime",
                              now=in_session(10, 30),
                              fetch_error_class="timeout",
                              fetch_error="ReadTimeout: timed out")
    assert rep.health == SourceHealth.FAILED.value
    assert "逾時" in rep.label


def test_quote_health_reports_unknown_when_nothing_stored(temp_db):
    rep = quotes.quote_health(pd.DataFrame(), source="mis", kind="realtime",
                              now=in_session(10, 30))
    assert rep.health == SourceHealth.UNKNOWN.value
    assert "尚無任何行情紀錄" in rep.reason


def test_quote_health_reports_non_trading_session(temp_db):
    rep = quotes.quote_health(pd.DataFrame(), source="mis", kind="realtime",
                              now=dt.datetime(2026, 10, 17, 11, 0))
    assert rep.health == SourceHealth.NON_TRADING_SESSION.value


def test_quote_health_unknown_when_source_not_configured():
    rep = quotes.quote_health(pd.DataFrame(), now=in_session(10, 30),
                              configured=False)
    assert rep.health == SourceHealth.UNKNOWN.value
    assert "未設定來源" in rep.reason


def test_source_health_covers_five_required_states():
    assert {s.value for s in SourceHealth} == {
        "normal", "delayed", "failed", "unknown", "non_trading_session"}


# ==========================================================================
# 9. Worker 重啟及單實例保護
# ==========================================================================
def test_quote_poll_disabled_unless_configured(monkeypatch):
    monkeypatch.delenv(worker.QUOTE_SOURCE_ENV, raising=False)
    assert worker.quote_poll_enabled() is False
    monkeypatch.setenv(worker.QUOTE_SOURCE_ENV, "fake")
    assert worker.quote_poll_enabled() is True


def test_run_quote_cycle_does_nothing_when_not_configured(monkeypatch, temp_db):
    monkeypatch.delenv(worker.QUOTE_SOURCE_ENV, raising=False)
    res = worker.run_quote_cycle(now=in_session(10, 30))
    assert res["status"] == "not_configured"
    assert worker.read_heartbeat()["quote_health"] == "unknown"


def test_quote_poll_due_respects_interval():
    now = in_session(10, 30)
    assert worker.quote_poll_due(None, now=now) is True
    assert worker.quote_poll_due({}, now=now) is True
    hb = {"quote_polled_at": (now - dt.timedelta(seconds=10)
                              ).isoformat(timespec="seconds")}
    assert worker.quote_poll_due(hb, now=now, interval=60) is False
    hb = {"quote_polled_at": (now - dt.timedelta(seconds=61)
                              ).isoformat(timespec="seconds")}
    assert worker.quote_poll_due(hb, now=now, interval=60) is True
    assert worker.quote_poll_due({"quote_polled_at": "壞掉的值"},
                                 now=now) is True


def test_run_quote_cycle_writes_heartbeat_and_audit_record(monkeypatch, temp_db):
    monkeypatch.setenv(worker.QUOTE_SOURCE_ENV, "fake")
    src = FakeQuoteSource(now=lambda: in_session(10, 30))
    res = worker.run_quote_cycle(force=True, now=in_session(10, 30),
                                 source="fake",
                                 poller=lambda **kw: quotes.poll_once(
                                     ["2330", "2317"],
                                     **{**kw, "source": src,
                                        "now": in_session(10, 30)}))
    assert res["status"] == "polled"
    hb = worker.read_heartbeat()
    assert hb["quote_polled_at"] == "2026-10-16T10:30:00"
    assert hb["quote_session"] == "open"
    assert hb["quote_fabricated"] is True, "合成來源必須在心跳中標示"
    # 稽核軌跡：輪詢也要留下 ingest_run 紀錄
    log_df = store.read_sql("SELECT * FROM ingest_run WHERE dataset='quote'")
    assert len(log_df) == 1
    assert log_df.iloc[0]["provenance"] == "recorded"


def test_run_quote_cycle_failure_is_recorded_not_raised(monkeypatch, temp_db):
    monkeypatch.setenv(worker.QUOTE_SOURCE_ENV, "fake")

    def boom(**kw):
        raise ConnectionError("來源斷線")

    res = worker.run_quote_cycle(force=True, now=in_session(10, 30),
                                 source="fake", poller=boom)
    assert res["status"] == "failed"
    hb = worker.read_heartbeat()
    assert hb["quote_health"] == "failed"
    assert "ConnectionError" in hb["quote_reason"]


def test_quote_failure_does_not_change_eod_cycle_outcome(monkeypatch, temp_db):
    """行情來源失敗**不得**讓 EOD 流程失敗，也不得讓它產生新訊號。"""
    monkeypatch.setenv(worker.QUOTE_SOURCE_ENV, "fake")
    monkeypatch.setattr(worker, "run_cycle", lambda force=False: "ok")
    real = worker.run_quote_cycle

    def failing_quote_cycle():
        return real(force=True, now=in_session(10, 30), source="fake",
                    poller=lambda **kw: (_ for _ in ()).throw(
                        ConnectionError("來源斷線")))

    monkeypatch.setattr(worker, "run_quote_cycle", failing_quote_cycle)
    assert worker.run_forever(force_once=True) == 0
    hb = worker.read_heartbeat()
    assert hb["last_cycle_ok"] is True
    assert hb["quote_health"] == "failed"


def test_heartbeat_survives_restart_and_keeps_quote_state(monkeypatch, temp_db):
    """重啟後心跳合併：行情狀態與 EOD 狀態互不覆蓋。"""
    worker.write_heartbeat(status="running", quote_health="delayed",
                           quote_polled_at="2026-10-16T10:30:00")
    worker.write_heartbeat(last_result="ok")           # 模擬另一輪只更新 EOD
    hb = worker.read_heartbeat()
    assert hb["quote_health"] == "delayed"
    assert hb["quote_polled_at"] == "2026-10-16T10:30:00"
    assert hb["last_result"] == "ok"


def test_worker_lock_still_prevents_second_instance(state_dir, monkeypatch):
    monkeypatch.setattr("twstock.ops.pid_alive", lambda pid: True)
    (state_dir / "worker.lock").write_text("424242 2026-01-01T00:00:00\n",
                                           encoding="utf-8")
    monkeypatch.setenv(worker.QUOTE_SOURCE_ENV, "fake")
    called = {"quote": 0}

    def spy():
        called["quote"] += 1
        return {"status": "polled"}

    monkeypatch.setattr(worker, "run_quote_cycle", spy)
    assert worker.run_forever(force_once=True) == 2
    assert called["quote"] == 0, "拒絕啟動時不得進行任何行情輪詢"
    assert not (state_dir / "stop.worker").exists()


def test_loop_quote_poll_is_bounded_by_interval(monkeypatch, temp_db):
    """間隔未到就不得再輪詢（避免打爆來源額度）。"""
    last = in_session(10, 30).isoformat(timespec="seconds")
    worker.write_heartbeat(quote_polled_at=last)
    calls = {"n": 0}
    monkeypatch.setattr(worker, "run_quote_cycle",
                        lambda: calls.__setitem__("n", calls["n"] + 1))
    assert worker.quote_poll_due(worker.read_heartbeat(),
                                 now=in_session(10, 30, 30)) is False
    assert calls["n"] == 0


# ==========================================================================
# 10. 正式資料庫安全與既有功能回歸
# ==========================================================================
def test_tests_never_touch_the_real_database(temp_db, tmp_path):
    """autouse fixture 必須把儲存層指向暫存 DB（正式 DB 為唯讀參考）。"""
    assert str(store.DB_PATH) == str(temp_db), "測試不得指向正式資料庫"
    assert Path(store.DB_PATH).parent == tmp_path, "必須隔離在 pytest 暫存目錄"
    from twstock.config import DB_PATH as REAL_DB_PATH
    assert Path(store.DB_PATH).resolve() != Path(REAL_DB_PATH).resolve(), \
        "不得指向正式資料庫"


def test_readonly_mode_blocks_quote_write(temp_db, monkeypatch):
    monkeypatch.setenv("TWSTOCK_READONLY", "1")
    with pytest.raises(RuntimeError, match="唯讀"):
        quotes.write_quotes(quotes.validate_batch([mk_quote()],
                                                  now=in_session(10, 30)),
                            run_id="r1")


def test_quote_write_does_not_disturb_existing_tables(temp_db):
    """回歸：新增 quote 表不得影響既有 price/news 寫入與讀取。"""
    store.write_df(pd.DataFrame([{
        "date": "2026-10-15", "stock_id": "2330", "open": 1.0, "high": 1.0,
        "low": 1.0, "close": 1.0, "volume": 1, "amount": 1, "turnover": 0.0,
    }]), "price")
    store.write_df(pd.DataFrame([{
        "published": "2026-10-15", "stock_id": "2330", "title": "t",
        "link": "https://example.com/a", "score": 0.0, "source": "test",
    }]), "news")
    quotes.write_quotes(quotes.validate_batch([mk_quote()],
                                              now=in_session(10, 30)),
                        run_id="r1")
    assert len(rows("price")) == 1 and len(rows("news")) == 1
    assert len(rows("quote")) == 1


def test_rejected_quotes_are_not_written_but_are_audited(temp_db):
    bad = mk_quote(price=-1.0)
    asmts = quotes.validate_batch([bad], now=in_session(10, 30))
    summ = quotes.write_quotes(asmts, run_id="r1")
    assert summ.written == 0 and summ.reasons == {"price_not_positive": 1}
    assert len(rows("quote")) == 0, "無效行情不得寫入行情序列"
    poll = quotes.PollResult(polled_at=FETCHED, source="fake", kind="delayed",
                             fetch=QuoteFetch(source="fake", kind="delayed",
                                              resolved=1, attempts=1),
                             write=summ, run_id="q1")
    assert quotes.write_poll_records(poll) == 1
    log = store.read_sql("SELECT status, error_class FROM ingest_run")
    assert log.iloc[0]["status"] == "failed"
    assert log.iloc[0]["error_class"] == "validation"
    assert "price_not_positive" in store.read_sql(
        "SELECT notes FROM ingest_run").iloc[0]["notes"]


def test_anomalous_price_beyond_daily_limit_is_rejected():
    q = mk_quote(price=200.0, prev_close=100.0)      # +100% 不可能是成交價
    a = quotes.validate_quote(q, now=in_session(10, 30))
    assert not a.accepted and a.reason.startswith("price_move_exceeds_limit")


def test_missing_price_is_empty_not_zero():
    q = mk_quote(price=None)
    a = quotes.validate_quote(q, now=in_session(10, 30))
    assert a.reason == "price_missing" and a.error_class == "empty"
    assert quotes._num("-") is None and quotes._num("") is None
    assert quotes._num("1,050.5") == 1050.5


def test_quote_kind_labels_are_human_readable():
    assert QuoteKind.HISTORICAL.label == "歷史"
    assert SourceHealth.NORMAL.label == "正常"
    assert quotes.SourceInfo.__dataclass_fields__["terms"]


def test_catalog_and_status_text_build_without_network(monkeypatch, temp_db):
    monkeypatch.delenv("TWSTOCK_QUOTE_SOURCE", raising=False)
    text = quotes.catalog_text()
    assert "沒有已查證、可無條件使用的免費官方台股盤中即時來源" in text
    status = quotes.status_text(now=in_session(10, 30))
    assert "TWSTOCK_QUOTE_SOURCE" in status
    assert "盤中訊號閘門 禁止" in status


def test_selftest_now_uses_synthetic_timestamp_outside_session(monkeypatch):
    at, why = quotes.selftest_now(now=dt.datetime(2026, 10, 17, 20, 0))  # 週六
    assert at.weekday() == 4 and at.hour == 10            # 退回週五 10:00
    assert "合成時間戳" in why, "必須明示時間戳為合成而非真實市場時間"
    at2, why2 = quotes.selftest_now(now=in_session(10, 30))
    assert at2 == in_session(10, 30) and "實際時間" in why2


def test_main_entry_points_exit_codes(monkeypatch, temp_db, capsys):
    assert quotes.main(["catalog"]) == 0
    assert quotes.main(["status"]) == 0
    monkeypatch.delenv("TWSTOCK_QUOTE_SOURCE", raising=False)
    assert quotes.main(["poll"]) == 2, "未設定來源時必須明確回報而非自行選來源"
    out = capsys.readouterr().out
    assert "不會" in out and "TWSTOCK_QUOTE_SOURCE" in out
    assert quotes.main(["selftest", "--stocks", "2330",
                        "--at", in_session(10, 0).isoformat()]) == 0
    assert "合成（假）行情" in capsys.readouterr().out


def test_selftest_never_touches_the_configured_db(temp_db, capsys):
    """自檢一律寫入**沙盒暫存 DB**：任何設定的資料庫都不得被合成資料污染。

    （實測教訓：本測試群先前允許 selftest 寫入設定的 DB，導致正式資料庫
    被寫入 20 列假行情；自此 selftest 強制沙盒化。）
    """
    args = ["selftest", "--stocks", "2330", "--at", in_session(10, 0).isoformat()]
    assert quotes.main(args) == 0
    assert len(rows("quote")) == 0, "自檢不得寫入設定的資料庫"
    assert quotes.main(args) == 0, "沙盒每次全新 → 每次都會有寫入"
    assert len(rows("quote")) == 0
    assert "合成（假）行情" in capsys.readouterr().out


def test_quote_rows_keep_source_error_redaction(temp_db):
    from twstock.data import status as st
    q = Quote(stock_id="2330", source="mis", market_time=in_session(10, 0).isoformat(),
              fetched_at=FETCHED, price=100.0, prev_close=100.0,
              error="token=abcd1234secret failed")
    assert "abcd1234secret" not in q.as_row()["error"]
    assert "redacted" in q.as_row()["error"]
    a = Assessment(quote=q, status=QuoteStatus.OK.value,
                   error_class=st.ErrorClass.NONE.value, reason="")
    assert a.accepted is True
    assert json.dumps(q.as_row(), ensure_ascii=False)   # 可序列化
