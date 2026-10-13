"""FinMind 資料抓取：日K、法人買賣、融資融券、月營收。

失敗語意（重要）：
  ``fetch_finmind`` 在**呼叫失敗**（網路/timeout/額度/格式錯誤）時回傳 ``None``；
  成功但該區間確實沒有資料時回傳**空 DataFrame**。
  上層據此區分「抓取失敗」與「真的沒資料」，絕不可把失敗當成零值或無訊號。
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time
from typing import Any

import pandas as pd

from ..config import CONFIG, FINMIND_API
from ..utils import retry_requests

log = logging.getLogger(__name__)


def fetch_finmind(dataset: str, data_id: str, start: str, end: str) -> pd.DataFrame | None:
    """呼叫 FinMind API v4。回傳 DataFrame（成功）或 None（失敗）。"""
    token = os.environ.get("FINMIND_TOKEN") or None
    params: dict[str, Any] = {
        "dataset": dataset,
        "data_id": data_id,
        "start_date": start,
        "end_date": end,
        "token": token,
    }
    try:
        r = retry_requests(FINMIND_API, params=params)
        payload = r.json()
    except Exception as e:  # noqa: BLE001
        log.warning("FinMind 連線失敗 (%s %s): %s", dataset, data_id, e)
        return None
    status = payload.get("status")
    # FinMind v4: status=200 表示成功（部分版本回 'success' 字串）
    if not (status == 200 or status == "success"):
        log.warning("FinMind %s %s 回應失敗: %s", dataset, data_id, payload.get("msg"))
        return None
    try:
        return pd.DataFrame(payload.get("data", []))
    except Exception as e:  # noqa: BLE001
        log.warning("FinMind %s %s 資料解析失敗: %s", dataset, data_id, e)
        return None


def fetch_price(stock_id: str, start: str, end: str) -> pd.DataFrame | None:
    df = fetch_finmind("TaiwanStockPrice", stock_id, start, end)
    if df is None or df.empty:
        return df
    df = df.rename(columns={
        "date": "date",
        "Trading_Volume": "volume",
        "Trading_money": "amount",
        "open": "open",
        "max": "high",
        "min": "low",
        "close": "close",
        "Trading_turnover": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_institutional(stock_id: str, start: str, end: str) -> pd.DataFrame | None:
    return fetch_finmind("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start, end)


def fetch_margin(stock_id: str, start: str, end: str) -> pd.DataFrame | None:
    return fetch_finmind("TaiwanStockMarginPurchaseShortSale", stock_id, start, end)


def fetch_revenue(stock_id: str, start: str, end: str) -> pd.DataFrame | None:
    return fetch_finmind("TaiwanStockMonthRevenue", stock_id, start, end)


def fetch_benchmark(start: str, end: str) -> pd.DataFrame | None:
    """基準（0050）日K，用於 ML 超額報酬標籤與回測基準比較。"""
    return fetch_price(CONFIG.benchmark, start, end)


def polite_sleep(seconds: float = 0.6) -> None:
    time.sleep(seconds)


def start_date(years_back: float) -> str:
    return (dt.date.today() - dt.timedelta(days=int(365.25 * years_back))).isoformat()
