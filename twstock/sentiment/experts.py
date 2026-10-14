"""Phase 2：名嘴 / 專業人士觀點來源。

來源類型（全部免金鑰，僅供研究用途）：
  1. YouTube 頻道 RSS — https://www.youtube.com/feeds/videos.xml?channel_id=XXX
     最新 15 則標題，適合追蹤財經節目與分析師頻道。
  2. Google News RSS — 以「分析師/名嘴姓名」為關鍵字搜尋其發言報導。
  3. CMoney 投資網誌 / 鉅亨網 — 以 Google News site: 網域限定搜尋。

注意：標題僅含部分資訊；語意打分沿用 news_rss 的規則式情緒。
授權與頻率限制請見 SETUP.md「資料來源合規」一節。

資料治理（本階段修正）
  修正前每個來源各自呼叫 ``feedparser.parse(url)`` 並在 ``bozo`` 時回傳 ``[]``，
  與「來源正常但沒有內容」無法區分；且發布時間取不到時以**今天**代替，等於
  憑空生成日期。現在一律走 ``news_rss.fetch_feed`` 取得顯式狀態，時間戳不可得時
  記為 NULL，並把每次來源的成敗寫入 ``ingest_run``（dataset='experts'）。
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from urllib.parse import quote

from ..data import status as st
from .news_rss import (ERR_NONE, FAILED, SUCCESS_EMPTY, FeedFetchResult,
                       _score_text, fetch_feed, ruleset_fingerprint)

log = logging.getLogger(__name__)

YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"

DATASET_EXPERTS = "experts"

# 台股常見財經節目 / 分析師頻道（channel_id 可自行替換或擴充）
YOUTUBE_CHANNELS: dict[str, str] = {
    # key = 頻道名, value = channel_id（留空則跳過）
    "CMoney理財寶": "UCceGk_IAmSygU-RYUmC_EOw",
}

# 分析師 / 名嘴姓名（用於 Google News 搜尋發言）
ANALYST_NAMES: list[str] = [
    "杜金龍", "李忠恩", "陳威良", "林漢偉", "朱家泓", "蔡明彥",
    "楊仁文", "許豐祿", "鄭廳宜", "李冠宇", "丁兆鳴", "游庭皓",
]

# 專業文章網域（用 site: 限定搜尋）
EXPERT_SITES: list[str] = [
    "cmnews.com.tw",   # CMoney 投資網誌
    "cnyes.com",       # 鉅亨網
    "moneydj.com",     # MoneyDJ
]


def _expert_items(entries, *, source: str, kind: str, fetched_at: str,
                  max_entries: int, cutoff_ts: float,
                  fetched_ts: float) -> tuple[list[dict], int]:
    """把 feed 條目轉為專家觀點列。回傳 (列, 略過筆數)。

    時間戳不可得時 ``published`` 為 ``None``（**不**以今天代替），
    並以 ``timestamp_source='unknown'`` 標記來源不可追溯。
    """
    out: list[dict] = []
    skipped = 0
    for e in entries[:max_entries]:
        ts = None
        if getattr(e, "published_parsed", None):
            ts = dt.datetime(*e.published_parsed[:6],
                             tzinfo=dt.timezone.utc).timestamp()
        if ts is not None and (ts < cutoff_ts or ts > fetched_ts + 6 * 3600):
            skipped += 1
            continue
        title = e.get("title", "") or ""
        out.append({
            "published": (dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
                          .date().isoformat()) if ts else None,
            "title": title,
            "link": e.get("link", "") or "",
            "score": _score_text(title),
            "source": source,
            "kind": kind,
            "fetched_at": fetched_at,
            "timestamp_source": "source" if ts else "unknown",
        })
    return out, skipped


def _fetch_source(url: str, *, source: str, kind: str, max_entries: int,
                  days: int, timeout: float, retries: int,
                  getter=None) -> tuple[list[dict], FeedFetchResult, int]:
    feed = fetch_feed(url, timeout=timeout, retries=retries, getter=getter)
    if not feed.ok:
        log.warning("專家來源 %s 取得失敗（%s）：%s", source,
                    feed.error_class, feed.error)
        return [], feed, 0
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    items, skipped = _expert_items(
        feed.entries, source=source, kind=kind, fetched_at=feed.fetched_at,
        max_entries=max_entries, cutoff_ts=now - days * 86400,
        fetched_ts=now)
    return items, feed, skipped


def fetch_youtube_channel(name: str, channel_id: str, max_entries: int = 15,
                          **kw) -> tuple[list[dict], FeedFetchResult, int]:
    """抓單一 YouTube 頻道最新影片標題。回傳 (列, feed 狀態, 略過筆數)。"""
    if not channel_id:
        return [], FeedFetchResult(
            url="", status=SUCCESS_EMPTY, error_class=ERR_NONE,
            error="未設定 channel_id"), 0
    return _fetch_source(YOUTUBE_RSS.format(cid=channel_id),
                         source=f"youtube:{name}", kind="video",
                         max_entries=max_entries, days=int(kw.get("days", 7)),
                         timeout=kw.get("timeout", 10.0),
                         retries=kw.get("retries", 2),
                         getter=kw.get("getter"))


def fetch_analyst_news(name: str, days: int = 7, **kw) -> tuple[list[dict], FeedFetchResult, int]:
    """以分析師姓名搜尋 Google News。"""
    q = quote(f'"{name}" 台股')
    return _fetch_source(GOOGLE_NEWS_RSS.format(q=q),
                         source=f"analyst:{name}", kind="comment",
                         max_entries=10, days=days,
                         timeout=kw.get("timeout", 10.0),
                         retries=kw.get("retries", 2),
                         getter=kw.get("getter"))


def fetch_site_news(site: str, max_entries: int = 10, **kw) -> tuple[list[dict], FeedFetchResult, int]:
    """以網域限定搜尋專業文章。"""
    q = quote(f"site:{site} 台股")
    return _fetch_source(GOOGLE_NEWS_RSS.format(q=q),
                         source=f"site:{site}", kind="article",
                         max_entries=max_entries, days=int(kw.get("days", 7)),
                         timeout=kw.get("timeout", 10.0),
                         retries=kw.get("retries", 2),
                         getter=kw.get("getter"))


@dataclass
class ExpertCollectResult:
    """專家觀點彙整結果：去重後清單、狀態紀錄與統計。"""

    items: list[dict] = field(default_factory=list)
    records: list[st.IngestRecord] = field(default_factory=list)
    n_raw: int = 0
    n_sources: int = 0
    n_failed_sources: int = 0

    @property
    def status(self) -> str:
        """整體狀態：全失敗→failed；部分失敗→partial_success。"""
        if self.n_sources == 0:
            return st.IngestStatus.SKIPPED.value
        if self.n_failed_sources == 0:
            return (st.IngestStatus.SUCCESS_WITH_DATA.value if self.items
                    else st.IngestStatus.SUCCESS_EMPTY.value)
        if self.n_failed_sources == self.n_sources:
            return st.IngestStatus.FAILED.value
        return st.IngestStatus.PARTIAL_SUCCESS.value


def collect_experts(with_videos: bool = True,
                    with_analysts: bool = True,
                    with_sites: bool = True,
                    *, run_id: str | None = None, days: int = 7,
                    timeout: float = 10.0, retries: int = 2,
                    getter=None, config_id: str = st.DEFAULT_CONFIG_ID,
                    ) -> ExpertCollectResult:
    """彙整所有專家觀點來源，回傳去重清單與每個來源的狀態紀錄。"""
    run_id = run_id or st.new_run_id()
    run_at = st.now_iso()
    version = ruleset_fingerprint()
    out = ExpertCollectResult()
    raw: list[dict] = []

    sources: list[tuple[str, tuple]] = []
    if with_videos:
        sources += [(f"youtube:{n}", (fetch_youtube_channel, (n, cid)))
                    for n, cid in YOUTUBE_CHANNELS.items()]
    if with_analysts:
        sources += [(f"analyst:{n}", (fetch_analyst_news, (n,)))
                    for n in ANALYST_NAMES]
    if with_sites:
        sources += [(f"site:{s}", (fetch_site_news, (s,)))
                    for s in EXPERT_SITES]

    for source, (fn, argv) in sources:
        out.n_sources += 1
        started = st.now_iso()
        try:
            items, feed, skipped = fn(*argv, days=days, timeout=timeout,
                                      retries=retries, getter=getter)
        except Exception as e:  # noqa: BLE001
            items, skipped = [], 0
            feed = FeedFetchResult(url="", status=FAILED,
                                   error_class=st.ErrorClass.UNKNOWN.value,
                                   error=f"{type(e).__name__}: {e}")
        raw.extend(items)
        if feed.status == FAILED:
            out.n_failed_sources += 1
            rec_status, cls = st.IngestStatus.FAILED.value, feed.error_class
        elif items:
            rec_status, cls = st.IngestStatus.SUCCESS_WITH_DATA.value, ERR_NONE
        else:
            rec_status, cls = st.IngestStatus.SUCCESS_EMPTY.value, ERR_NONE
        out.records.append(st.IngestRecord(
            run_id=run_id, dataset=DATASET_EXPERTS, run_at=run_at,
            source=source, task="collect_experts", stock_id="",
            started_at=started, ended_at=st.now_iso(),
            status=rec_status, error_class=cls, error=feed.error,
            rows_fetched=feed.n_entries, rows_parsed=len(items),
            rows_validated=sum(1 for i in items
                               if i.get("timestamp_source") == "source"),
            attempts=feed.attempts, source_version=version,
            config_id=config_id, provenance=st.RECORDED,
            notes=f"skipped={skipped}"))

    out.n_raw = len(raw)
    seen: set[str] = set()
    dedup: list[dict] = []
    for it in raw:
        key = it["link"] or (it["source"] + it["title"])
        if key in seen or not it["title"]:
            continue
        seen.add(key)
        dedup.append(it)
    # 時間不明的列排到最後，不以偽造日期參與排序
    dedup.sort(key=lambda x: (x["published"] or "", x["title"]), reverse=True)
    out.items = dedup
    st.write_records(out.records)
    log.info("專家觀點來源共取得 %d 則（原始 %d，來源 %d 個，失敗 %d 個；狀態 %s）",
             len(dedup), out.n_raw, out.n_sources, out.n_failed_sources,
             st.IngestStatus(out.status).label)
    return out
