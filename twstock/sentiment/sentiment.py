"""新聞情緒聚合：加權平均（近 3 日權重 1.0 / 0.66 / 0.4），LLM 可選升級。"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from . import news_rss
from ..data import store

log = logging.getLogger(__name__)

WEIGHTS = {0: 1.0, 1: 0.66, 2: 0.4}


def llm_score_available() -> bool:
    import os
    return bool(os.environ.get("OPENAI_API_KEY"))


def update_news_for(stock_id: str, name: str, days: int = 7) -> int:
    """抓新聞入庫，回傳筆數（replace 同 stock_id 的舊新聞避免重複累積）。"""
    articles = news_rss.fetch_google_news(stock_id, name, days=days)
    if not articles:
        return 0
    df = pd.DataFrame(articles)
    store.exec_sql("DELETE FROM news WHERE stock_id=?", (stock_id,))
    conn = store._connect()
    store._init(conn)
    try:
        df.to_sql("news", conn, if_exists="replace", index=False, method="multi", chunksize=100)
    finally:
        conn.close()
    return len(articles)
