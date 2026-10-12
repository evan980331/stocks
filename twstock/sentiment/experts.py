"""Phase 2：名嘴 / 專業人士觀點來源。

來源類型（全部免金鑰，僅供研究用途）：
  1. YouTube 頻道 RSS — https://www.youtube.com/feeds/videos.xml?channel_id=XXX
     最新 15 則標題，適合追蹤財經節目與分析師頻道。
  2. Google News RSS — 以「分析師/名嘴姓名」為關鍵字搜尋其發言報導。
  3. CMoney 投資網誌 / 鉅亨網 — 以 Google News site: 網域限定搜尋。

注意：標題僅含部分資訊；語意打分沿用 news_rss 的規則式情緒。
授權與頻率限制請見 SETUP.md「資料來源合規」一節。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser

from ..utils import today_str
from .news_rss import _score_text

log = logging.getLogger(__name__)

YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"

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


def fetch_youtube_channel(name: str, channel_id: str, max_entries: int = 15) -> list[dict]:
    """抓單一 YouTube 頻道最新影片標題。"""
    if not channel_id:
        return []
    url = YOUTUBE_RSS.format(cid=channel_id)
    feed = feedparser.parse(url)
    if feed.bozo and not feed.entries:
        log.warning("YouTube RSS 失敗 %s: %s", name, getattr(feed, "bozo_exception", ""))
        return []
    out: list[dict] = []
    for e in feed.entries[:max_entries]:
        ts = None
        if getattr(e, "published_parsed", None):
            ts = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        out.append({
            "published": ts.date().isoformat() if ts else today_str(),
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "score": _score_text(e.get("title", "")),
            "source": f"youtube:{name}",
            "kind": "video",
        })
    return out


def fetch_analyst_news(name: str, days: int = 7) -> list[dict]:
    """以分析師姓名搜尋 Google News。"""
    q = quote(f'"{name}" 台股')
    url = GOOGLE_NEWS_RSS.format(q=q)
    feed = feedparser.parse(url)
    out: list[dict] = []
    for e in feed.entries[:10]:
        ts = None
        if getattr(e, "published_parsed", None):
            ts = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        out.append({
            "published": ts.date().isoformat() if ts else today_str(),
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "score": _score_text(e.get("title", "")),
            "source": f"analyst:{name}",
            "kind": "comment",
        })
    return out


def fetch_site_news(site: str, max_entries: int = 10) -> list[dict]:
    """以網域限定搜尋專業文章。"""
    q = quote(f"site:{site} 台股")
    url = GOOGLE_NEWS_RSS.format(q=q)
    feed = feedparser.parse(url)
    out: list[dict] = []
    for e in feed.entries[:max_entries]:
        ts = None
        if getattr(e, "published_parsed", None):
            ts = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        out.append({
            "published": ts.date().isoformat() if ts else today_str(),
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "score": _score_text(e.get("title", "")),
            "source": f"site:{site}",
            "kind": "article",
        })
    return out


def collect_experts(with_videos: bool = True,
                    with_analysts: bool = True,
                    with_sites: bool = True) -> list[dict]:
    """彙整所有專家觀點來源，回傳去重後的清單。"""
    items: list[dict] = []
    if with_videos:
        for name, cid in YOUTUBE_CHANNELS.items():
            items.extend(fetch_youtube_channel(name, cid))
    if with_analysts:
        for nm in ANALYST_NAMES:
            items.extend(fetch_analyst_news(nm))
    if with_sites:
        for site in EXPERT_SITES:
            items.extend(fetch_site_news(site))

    seen: set[str] = set()
    dedup: list[dict] = []
    for it in items:
        key = it["link"] or (it["source"] + it["title"])
        if key in seen or not it["title"]:
            continue
        seen.add(key)
        dedup.append(it)
    dedup.sort(key=lambda x: x["published"], reverse=True)
    log.info("專家觀點來源共取得 %d 則（原始 %d）", len(dedup), len(items))
    return dedup
