"""FinMind 資料抓取：日K、法人買賣、融資融券、月營收。"""
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


def fetch_finmind(dataset: str, data_id: str, start: str, end: str) -> pd.DataFrame:
    """呼叫 FinMind API v4，回傳 DataFrame（失敗回空）。"""
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
        status = payload.get("status")
        # FinMind v4: status=200 表示成功（部分版本回 'success' 字串）
        ok = status == 200 or status == "success"
        if not ok:
            log.warning("FinMind %s %s: %s", dataset, data_id, payload.get("msg"))
            return pd.DataFrame()
        return pd.DataFrame(payload.get("data", []))
    except Exception as e:  # noqa: BLE001
        log.warning("FinMind fetch failed (%s %s): %s", dataset, data_id, e)
        return pd.DataFrame()


def fetch_price(stock_id: str, start: str, end: str) -> pd.DataFrame:
    df = fetch_finmind("TaiwanStockPrice", stock_id, start, end)
    if df.empty:
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


def fetch_institutional(stock_id: str, start: str, end: str) -> pd.DataFrame:
    return fetch_finmind("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start, end)


def fetch_margin(stock_id: str, start: str, end: str) -> pd.DataFrame:
    return fetch_finmind("TaiwanStockMarginPurchaseShortSale", stock_id, start, end)


def fetch_revenue(stock_id: str, start: str, end: str) -> pd.DataFrame:
    return fetch_finmind("TaiwanStockMonthRevenue", stock_id, start, end)


def polite_sleep(seconds: float = 0.6) -> None:
    time.sleep(seconds)


def start_date(years_back: float) -> str:
    return (dt.date.today() - dt.timedelta(days=int(365.25 * years_back))).isoformat()
