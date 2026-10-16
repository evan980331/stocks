"""Point-in-time 新聞可用性：發布時間與取得時間共同決定最早可用日期。

問題（本階段盤點發現的前視風險）
  ``pipeline.load_panel`` 曾以 ``published as date`` 直接餵給特徵衰減窗口。
  這代表一則**發布於 D、但在 D 之後才被抓到**的新聞會被回填到 D 及其後的
  衰減窗口——等於把「事後才看得到的資訊」塞進當時的交易訊號。

  另外既有 237 則歷史新聞的 ``fetched_at`` 全部為 NULL（寫入於狀態治理之前），
  「何時可見」不可重建；此類列**不得**納入需要嚴格時間對齊的訊號路徑。

可用性規則（保守，不臆測）
  一則新聞最早只能用於**同時滿足「已發布」且「已被本系統取得」**的交易日：

      usable_from = max(published 日期, fetched 日期)

  - ``fetched_at`` 缺失（舊資料、不可追溯）→ **不可用於訊號**，只出現在報告。
  - ``published`` 缺失（來源未提供發布時間）→ 同樣不可用於訊號。
  - 來源聲稱的發布時間晚於取得時間（無法驗證的未來時間）→ 不可用。
  - 訊號在 T 日收盤後計算、T+1 開盤才進場；因此「T 日收盤後才取得」的新聞
    對 T 日訊號仍屬可用（取得發生在訊號計算之前）。本模組只保證
    **不早於取得日**，不自行猜測當日內的先後順序。

設計原則
  - 只做過濾，不補值、不推估、不改寫既有列。
  - 舊資料一律排除於訊號路徑，並在報告中以「不可追溯」計數呈現。
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3

import pandas as pd

from . import store

log = logging.getLogger(__name__)

NEWS_SIGNAL_COLUMNS = ("date", "stock_id", "score")


def _to_date(value) -> dt.date | None:
    """把 DB 欄位值轉成日期；無法解析或為空時回 None（不猜測）。"""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def usable_from(published, fetched_at: str | None,
                timestamp_source: str | None = "source",
                provenance: str | None = "recorded") -> dt.date | None:
    """回傳該列最早可用日期；不可用於訊號時回 None。

    ``timestamp_source`` / ``provenance`` 預設視為可信，是**方便兩參數呼叫**；
    從 DB 讀出的路徑（``news_availability``）一律傳入實際欄位值，
    因此硬性排除仍然生效，不受此預設影響。

    判定順序即語意保證（與 ``status.classify`` 同樣採硬性排除優先）：
      1. 舊資料（``provenance='unknown'``）→ None（抓取時點不可重建）。
      2. 時間戳不可追溯（``timestamp_source != 'source'``）→ None。
      3. 發布日期或取得日期缺失／無法解析 → None。
      4. 取得日期早於發布日期（無法驗證的來源時間）→ None。
      5. 其餘 → max(published, fetched)。
    """
    if provenance == "unknown" or timestamp_source != "source":
        return None
    pub = _to_date(published)
    fet = _to_date(fetched_at)
    if pub is None or fet is None:
        return None
    if fet < pub:
        return None
    return max(pub, fet)


def _news_columns(conn) -> set[str]:
    """回傳 news 表實際存在的欄位集合；表不存在時回空集合。

    唯讀模式下 ``read_sql`` **不跑遷移**（為了不動使用者的 DB），
    因此尚未遷移的舊 DB 缺少溯源欄位。缺欄位時一律回空集合，
    下游把該列視為不可追溯（保守且正確：無法證明可見時點）。
    """
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='news'"
        ).fetchone()
        if not exists:
            return set()
        return {r[1] for r in conn.execute("PRAGMA table_info(news)").fetchall()}
    except sqlite3.Error:
        return set()


def news_availability() -> pd.DataFrame:
    """逐列的可用性判斷（含排除原因），供報告與稽核使用。

    欄位：``stock_id``、``title``、``published``、``fetched_at``、
    ``usable_from``（``None`` 表示不可用於訊號）、``reason``、``score``、
    ``news_uid``、``link_basis``。

    對**尚未遷移**的 DB（唯讀分析時常見）：只取實際存在的欄位，
    缺失者補 ``None`` → 列一律視為不可追溯，不會因欄位缺失而擲錯。
    """
    wanted = ["stock_id", "title", "published", "fetched_at", "score",
              "timestamp_source", "provenance", "news_uid", "link_basis"]
    conn = store._connect()
    try:
        known = _news_columns(conn)
    finally:
        conn.close()
    present = [c for c in wanted if c in known]
    if not known or not present:
        return pd.DataFrame(columns=["stock_id", "title", "published",
                                     "fetched_at", "usable_from", "reason",
                                     "score", "news_uid", "link_basis"])
    df = store.read_sql("SELECT " + ", ".join(present) + " FROM news")
    for col in wanted:
        if col not in df.columns:
            df[col] = None
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "title", "published",
                                     "fetched_at", "usable_from", "reason",
                                     "score", "news_uid", "link_basis"])
    rows = []
    for r in df.itertuples(index=False):
        prov = getattr(r, "provenance", None)
        tsrc = getattr(r, "timestamp_source", None)
        pub = _to_date(getattr(r, "published", None))
        fet = _to_date(getattr(r, "fetched_at", None))
        usable = usable_from(getattr(r, "published", None),
                             getattr(r, "fetched_at", None), tsrc, prov)
        if usable is None:
            if prov == "unknown":
                reason = "legacy_untraceable"
            elif tsrc != "source":
                reason = "timestamp_untraceable"
            elif pub is None:
                reason = "published_missing"
            elif fet is None:
                reason = "fetched_missing"
            else:
                reason = "fetched_before_published"
        else:
            reason = ""
        rows.append({"stock_id": str(getattr(r, "stock_id", "")),
                     "title": getattr(r, "title", ""),
                     "published": pub.isoformat() if pub else None,
                     "fetched_at": getattr(r, "fetched_at", None),
                     "usable_from": usable.isoformat() if usable else None,
                     "reason": reason, "score": getattr(r, "score", None),
                     "news_uid": getattr(r, "news_uid", None),
                     "link_basis": getattr(r, "link_basis", None)})
    return pd.DataFrame(rows)


def news_daily_for_signals() -> pd.DataFrame:
    """回傳可用於訊號的新聞日分數：``date`` = 最早可用日期。

    與舊介面 ``SELECT published as date ...`` 的差異：
      - ``date`` 改為 ``max(published, fetched)``，因此事後才抓到的新聞
        **不會**被回填到較早的交易日。
      - 舊資料（``fetched_at`` 為 NULL 或 ``provenance='unknown'``）一律排除，
        不以發布日冒充可用日。

    回傳欄位與舊介面相容（``date``、``stock_id``、``score``），
    供 ``features.build_features(news_daily=...)`` 直接使用。
    """
    avail = news_availability()
    usable = avail[avail["usable_from"].notna()]
    if usable.empty:
        if not avail.empty:
            log.info("point-in-time 新聞：0/%d 則可用於訊號"
                     "（其餘為舊資料或時間戳不可追溯）", len(avail))
        return pd.DataFrame(columns=list(NEWS_SIGNAL_COLUMNS))
    out = pd.DataFrame({
        "date": pd.to_datetime(usable["usable_from"]),
        "stock_id": usable["stock_id"].astype(str),
        "score": pd.to_numeric(usable["score"], errors="coerce"),
    })
    return out.dropna(subset=["score"]).reset_index(drop=True)
