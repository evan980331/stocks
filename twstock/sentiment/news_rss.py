"""新聞抓取：Google News RSS（zh-TW）。規則式情緒打分：-1 ~ +1。

失敗語意（重要，修正前是無聲失敗）
  修正前 ``fetch_google_news`` 直接呼叫 ``feedparser.parse(url)``。feedparser 在
  網路失敗時**不拋例外**，只把 ``bozo`` 設起來並回傳空 entries，於是
  「抓取失敗」與「來源正常但沒有新聞」在資料層完全無法區分。

本模組現在以 ``requests`` 取得 HTTP 回應後才交給 feedparser 解析，因此可明確區分：
  - ``success_with_data``：正常回應且有新聞。
  - ``success_empty``：正常回應、確實沒有符合條件的資料。
  - ``failed``：網路失敗／逾時／HTTP 狀態異常／解析失敗（附可重現的錯誤分類）。

時間戳語意
  發布時間一律取自來源的 ``published_parsed``。取不到時**不得**以「今天」代替
  （那會憑空生成資料，並讓無法驗證的時間進入情緒衰減窗口），而是記為
  ``published=None``、``timestamp_source='unknown'``，由下游明確排除。
  來源時間若晚於抓取時間（無法驗證的未來時間）亦不採用。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
from dataclasses import dataclass, field
from urllib.parse import parse_qs, quote, urlparse

import feedparser
import requests

log = logging.getLogger(__name__)

POSITIVE_TERMS = [
    "創新高", "大漲", "大賺", "調升目標價", "買超", "獲利成長", "法說利多",
    "擴廠", "訂單暢旺", "寫新高", "金額破", "殖利率增", 
    "盈餘", "龍頭地位", "補漲", "外資買", "突破", "轉盈",
    "翻紅", "激勵", "中標", "領漲", "營收創高", "上調", "利多",
]
NEGATIVE_TERMS = [
    "創新低", "大跌", "重挫", "遭砍", "賣超", "虧損", "下修", "砍目標價",
    "利空", "警報", "制裁", "崩", "爆雷", "下市", "走跌", "賣壓", "反轉向下",
    "外資賣", "解禁", "庫存壓力", "減產", "遭罰", "訴訟", "跌停",
    "下探", "跌破", "轉虧", "股災", "降評",
]
# 抵銷常見誤標的正面詞（避免 EPS mere mention、龍頭等被當利多）
_NEUTRALIZE = {"EPS", "龍頭"}

GOOGLE_NEWS_RSS = ("https://news.google.com/rss/search?q={query}"
                   "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")

USER_AGENT = "twstock-research/1.0 (+local research; contact: local)"

# 情緒規則版本：規則或詞表一旦變動，分數就不可跨版本直接比較。
# 以詞表指紋入庫，讓「不同版本的分数混用」可被稽核發現。
def ruleset_fingerprint() -> str:
    blob = repr((sorted(POSITIVE_TERMS), sorted(NEGATIVE_TERMS),
                 sorted(_NEUTRALIZE)))
    return "news_rss/v1+" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


NEWS_RULESET_VERSION = ruleset_fingerprint()

# 狀態與錯誤分類（與 twstock.data.status 的值一致；此處不 import 以避免循環）
SUCCESS_WITH_DATA = "success_with_data"
SUCCESS_EMPTY = "success_empty"
FAILED = "failed"
ERR_NONE = "none"
ERR_NETWORK = "network"
ERR_TIMEOUT = "timeout"
ERR_HTTP = "http_status"
ERR_PARSE = "parse"

# 發布時間晚於抓取時間超過此容許量 → 視為無法驗證的未來時間，不採用
FUTURE_TOLERANCE_S = 6 * 3600


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _score_text(text: str) -> float:
    score = 0.0
    for t in POSITIVE_TERMS:
        if t in text:
            score += 1.0
    for t in NEGATIVE_TERMS:
        if t in text:
            score -= 1.0
    for t in _NEUTRALIZE:
        if t in text and t in POSITIVE_TERMS:
            score -= 1.0  # 抵銷誤標的正面詞
    if score > 0:
        return min(1.0, score / 3)
    if score < 0:
        return max(-1.0, score / 3)
    return 0.0


def _decode_google_link(link: str) -> str | None:
    """Google News RSS 的 link 可能包在 URL 參數裡，嘗試解出真實連結。"""
    try:
        u = urlparse(link)
        qs = parse_qs(u.query)
        for k in ("url", "u"):
            if k in qs and isinstance(qs[k][0], str):
                return qs[k][0]
    except Exception:  # noqa: BLE001
        pass
    return None


# --------------------------------------------------------------------------
# RSS 取得（唯一的 HTTP 進入點，供 news 與 experts 共用）
# --------------------------------------------------------------------------
@dataclass
class FeedFetchResult:
    """一次 RSS 取得的結果與可重現的診斷資訊。"""

    url: str
    status: str = FAILED
    error_class: str = ERR_NONE
    error: str = ""
    http_status: int | None = None
    attempts: int = 0
    fetched_at: str = ""
    entries: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in (SUCCESS_WITH_DATA, SUCCESS_EMPTY)

    @property
    def n_entries(self) -> int:
        return len(self.entries)


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    return isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()


def fetch_feed(url: str, *, timeout: float = 10.0, retries: int = 2,
               getter=None) -> FeedFetchResult:
    """取得並解析 RSS。回傳顯式狀態；**絕不**把失敗轉成空清單。

    ``getter`` 可注入（測試用），簽名為 ``getter(url, timeout=...)``。
    """
    fetched_at = _now_iso()
    get = getter or requests.get
    attempts = 0
    last_exc: BaseException | None = None
    resp = None
    while attempts < max(1, retries):
        attempts += 1
        try:
            resp = get(url, timeout=timeout,
                       headers={"User-Agent": USER_AGENT})
            break
        except Exception as e:  # noqa: BLE001
            last_exc = e
            resp = None
            if attempts < max(1, retries):
                log.warning("RSS 取得失敗（第 %d 次）：%s", attempts, e)

    if resp is None:
        cls = ERR_TIMEOUT if _is_timeout(last_exc) else ERR_NETWORK
        return FeedFetchResult(
            url=url, status=FAILED, error_class=cls,
            error=f"{type(last_exc).__name__}: {last_exc}" if last_exc else "",
            attempts=attempts, fetched_at=fetched_at)

    status_code = getattr(resp, "status_code", None)
    if status_code != 200:
        return FeedFetchResult(
            url=url, status=FAILED, error_class=ERR_HTTP,
            error=f"HTTP {status_code}", http_status=status_code,
            attempts=attempts, fetched_at=fetched_at)

    try:
        content = resp.content
    except Exception as e:  # noqa: BLE001
        return FeedFetchResult(
            url=url, status=FAILED, error_class=ERR_PARSE,
            error=f"讀取回應內容失敗：{type(e).__name__}: {e}",
            http_status=status_code, attempts=attempts,
            fetched_at=fetched_at)

    feed = feedparser.parse(content)
    entries = list(getattr(feed, "entries", []) or [])
    # bozo 表示解析出問題（非 XML、截斷、編碼異常）；若同時沒有任何條目，
    # 就是解析失敗，而不是「沒有新聞」。
    if getattr(feed, "bozo", 0) and not entries:
        exc = getattr(feed, "bozo_exception", None)
        return FeedFetchResult(
            url=url, status=FAILED, error_class=ERR_PARSE,
            error=f"解析失敗：{type(exc).__name__}: {exc}" if exc
            else "解析失敗（bozo 且無條目）",
            http_status=status_code, attempts=attempts, fetched_at=fetched_at)

    return FeedFetchResult(
        url=url, status=SUCCESS_WITH_DATA if entries else SUCCESS_EMPTY,
        http_status=status_code, attempts=attempts, fetched_at=fetched_at,
        entries=entries)


# --------------------------------------------------------------------------
# 新聞
# --------------------------------------------------------------------------
@dataclass
class NewsFetchResult:
    """單一標的的新聞抓取結果：狀態、診斷與新聞列。"""

    stock_id: str
    status: str = FAILED
    error_class: str = ERR_NONE
    error: str = ""
    articles: list[dict] = field(default_factory=list)
    http_status: int | None = None
    attempts: int = 0
    fetched_at: str = ""
    days: int = 7
    n_entries_seen: int = 0
    n_future_skipped: int = 0
    n_old_skipped: int = 0
    source_version: str = NEWS_RULESET_VERSION
    query: str = ""

    @property
    def n_fetched(self) -> int:
        """來源回應中實際看到的條目數（解析後、過濾前）。"""
        return self.n_entries_seen

    @property
    def n_stored(self) -> int:
        return len(self.articles)

    @property
    def n_missing_timestamp(self) -> int:
        """發布時間不可得的筆數（以 NULL 入庫，不臆測）。"""
        return sum(1 for a in self.articles
                   if a.get("timestamp_source") != "source")

    @property
    def n_neutral(self) -> int:
        return sum(1 for a in self.articles
                   if a.get("score_kind") == "neutral")

    @property
    def n_scored(self) -> int:
        return sum(1 for a in self.articles
                   if a.get("score_kind") == "scored")

    @property
    def n_failed(self) -> int:
        return 1 if self.status == FAILED else 0

    def as_notes(self) -> str:
        return (f"entities={self.n_entries_seen} stored={self.n_stored} "
                f"neutral={self.n_neutral} scored={self.n_scored} "
                f"no_timestamp={self.n_missing_timestamp} "
                f"future_skipped={self.n_future_skipped} "
                f"old_skipped={self.n_old_skipped}")


def _news_item(entry, stock_id: str, fetched_at: str, fetched_ts: float,
               cutoff_ts: float) -> tuple[dict | None, str]:
    """把單一 RSS 條目轉為新聞列。回傳 (列或 None, 排除原因)。"""
    ts = None
    if getattr(entry, "published_parsed", None):
        ts = dt.datetime(*entry.published_parsed[:6],
                         tzinfo=dt.timezone.utc).timestamp()
    if ts is not None and ts > fetched_ts + FUTURE_TOLERANCE_S:
        return None, "future"
    if ts is not None and ts < cutoff_ts:
        return None, "old"

    title = getattr(entry, "title", "") or ""
    link = getattr(entry, "link", "") or ""
    real = _decode_google_link(link)
    score = _score_text(title)
    if ts is None:
        published = None            # 不可臆測；不得以今天代替
        published_at = None
        timestamp_source = "unknown"
    else:
        published = dt.datetime.fromtimestamp(
            ts, tz=dt.timezone.utc).date().isoformat()
        published_at = dt.datetime.fromtimestamp(
            ts, tz=dt.timezone.utc).isoformat(timespec="seconds")
        timestamp_source = "source"
    item = {
        "published": published,
        "published_at": published_at,
        "stock_id": stock_id,
        "title": title,
        "link": real or link,
        "score": score,
        "score_kind": "neutral" if score == 0 else "scored",
        # 「來源正常但情緒中性」與「沒有新聞」是不同狀態，必須可分
        "source": (entry.get("source", {}) or {}).get("title", "google_news")
        if hasattr(entry, "get") else "google_news",
        "fetched_at": fetched_at,
        "timestamp_source": timestamp_source,
        "provenance": "recorded",
    }
    return item, ""


def fetch_google_news(stock_id: str, name: str, days: int = 7, *,
                      timeout: float = 10.0, retries: int = 2,
                      getter=None, now: dt.datetime | None = None,
                      ) -> NewsFetchResult:
    """以「代碼 或 公司名」搜尋 Google News RSS 的最近新聞。

    回傳 ``NewsFetchResult``（含狀態）。呼叫端可據此區分
    「取得資料／正常但無資料／抓取失敗」，不再把失敗偽裝成空清單。
    """
    query = quote(f"{stock_id} OR {name} 台股".replace("台股 ", " "))
    url = GOOGLE_NEWS_RSS.format(query=query)
    base = NewsFetchResult(stock_id=stock_id, days=days, query=query,
                           source_version=NEWS_RULESET_VERSION)
    feed = fetch_feed(url, timeout=timeout, retries=retries, getter=getter)
    base.fetched_at = feed.fetched_at
    base.attempts = feed.attempts
    base.http_status = feed.http_status
    base.error_class = feed.error_class
    base.error = feed.error
    base.status = feed.status
    if not feed.ok:
        log.warning("%s 新聞抓取失敗（%s）：%s", stock_id,
                    feed.error_class, feed.error)
        return base

    now = now or dt.datetime.now(dt.timezone.utc)
    fetched_ts = now.timestamp()
    cutoff_ts = fetched_ts - days * 86400
    base.n_entries_seen = feed.n_entries
    items: list[dict] = []
    for e in feed.entries[:20]:
        item, why = _news_item(e, stock_id, feed.fetched_at, fetched_ts,
                              cutoff_ts)
        if item is None:
            if why == "future":
                base.n_future_skipped += 1
            else:
                base.n_old_skipped += 1
            continue
        items.append(item)
    base.articles = items
    base.status = SUCCESS_WITH_DATA if items else SUCCESS_EMPTY
    return base
