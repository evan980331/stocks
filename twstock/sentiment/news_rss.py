"""新聞抓取：Google News RSS（zh-TW）。規則式情緒打分：-1 ~ +1。"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import quote, urlparse, parse_qs

import feedparser

from ..utils import today_str

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


def fetch_google_news(stock_id: str, name: str, days: int = 7) -> list[dict]:
    """以「代碼 或 公司名」搜尋 Google News RSS 的最近新聞。"""
    query = quote(f"{stock_id} OR {name} 台股".replace("台股 ", " "))
    url = GOOGLE_NEWS_RSS.format(query=query)
    feed = feedparser.parse(url)
    articles: list[dict] = []
    cutoff_ts = datetime.now(timezone.utc).timestamp() - days * 86400
    for e in feed.entries[:20]:
        ts = None
        if hasattr(e, "published_parsed") and e.published_parsed:
            ts = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).timestamp()
        if ts and ts < cutoff_ts:
            continue
        title = getattr(e, "title", "")
        link = getattr(e, "link", "")
        real = _decode_google_link(link)
        articles.append({
            "published": datetime.utcfromtimestamp(ts).date().isoformat() if ts else today_str(),
            "stock_id": stock_id,
            "title": title,
            "link": real or link,
            "score": _score_text(title),
            "source": (e.get("source", {}) or {}).get("title", "google_news")
            if hasattr(e, "get") else "google_news",
        })
    return articles
