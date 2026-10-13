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
    """抓新聞入庫，回傳筆數。

    重要（W3）：只以 ``(stock_id, link)`` 主鍵覆寫**這一檔**的新聞，絕不可整表覆蓋，
    否則每抓一檔股票就會清空其他所有股票的新聞。
    """
    articles = news_rss.fetch_google_news(stock_id, name, days=days)
    if not articles:
        return 0
    df = pd.DataFrame(articles)
    if "link" not in df.columns:
        df["link"] = ""
    # 同一檔同一天可能有多則新聞；link 為空時用標題當鍵，避免互相覆蓋
    df["link"] = df["link"].fillna("").astype(str)
    blank = df["link"] == ""
    df.loc[blank, "link"] = "title:" + df.loc[blank, "title"].fillna("").astype(str)
    return store.write_df(df, "news", key_cols=("stock_id", "link"))
