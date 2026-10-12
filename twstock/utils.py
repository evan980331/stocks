"""共用類型與工具。"""
from __future__ import annotations

import datetime as dt
from typing import Any

import requests


def today_str(offset_days: int = 0) -> str:
    d = dt.date.today() + dt.timedelta(days=offset_days)
    return d.isoformat()


def retry_requests(url: str, *, params: dict[str, Any] | None = None,
                   retries: int = 3, timeout: int = 30) -> requests.Response:
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last_exc = e
            import time
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET {url} failed after {retries} retries") from last_exc
