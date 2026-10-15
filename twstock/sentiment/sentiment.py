"""新聞情緒聚合：加權平均（近 3 日權重 1.0 / 0.66 / 0.4），LLM 可選升級。

新聞狀態四態（修正前只有兩態：有筆數／0）
  1. **沒有新聞**            → ``success_empty``（來源正常、確實沒有資料）
  2. **有新聞但情緒中性**    → ``success_with_data``，列上 ``score_kind='neutral'``
  3. **有有效情緒分數**      → ``success_with_data``，列上 ``score_kind='scored'``
  4. **抓取失敗**            → ``failed``（附 network/timeout/http_status/parse 分類）

修正前 ``update_news_for`` 在 1 與 4 都回傳 ``0``，使失敗在資料層完全不可見。
本模組一律回傳 ``NewsIngestResult``，並把每次執行寫入 ``ingest_run``。
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import pandas as pd

from ..data import status as st
from ..data import store
from . import news_rss

log = logging.getLogger(__name__)

WEIGHTS = {0: 1.0, 1: 0.66, 2: 0.4}


def llm_score_available() -> bool:
    import os
    return bool(os.environ.get("OPENAI_API_KEY"))


@dataclass
class NewsIngestResult:
    """單檔新聞擷取結果：狀態、計數與寫入紀錄。"""

    stock_id: str
    status: str = st.IngestStatus.UNKNOWN.value
    error_class: str = st.ErrorClass.NONE.value
    error: str = ""
    rows_fetched: int = 0        # 來源回應中看到的條目數
    rows_parsed: int = 0         # 通過時間戳檢核、準備寫入的列數
    rows_written: int = 0        # **真正新增**的列數（已存在者整列跳過）
    n_existing: int = 0          # 已存在而跳過的列數（重複抓取不重計）
    n_neutral: int = 0
    n_scored: int = 0
    n_missing_timestamp: int = 0
    n_future_skipped: int = 0
    data_start: str = ""
    data_end: str = ""
    attempts: int = 0
    source_version: str = news_rss.NEWS_RULESET_VERSION
    days: int = 7
    record: st.IngestRecord | None = field(default=None, repr=False)

    @property
    def is_failure(self) -> bool:
        return self.status == st.IngestStatus.FAILED.value

    def label(self) -> str:
        return (f"{self.stock_id} 新聞 {st.IngestStatus(self.status).label}"
                f"（取得 {self.rows_fetched}／新增 {self.rows_written}，"
                f"既有 {self.n_existing}，"
                f"中性 {self.n_neutral}、有分數 {self.n_scored}、"
                f"時間不明 {self.n_missing_timestamp}）"
                + (f"｜{self.error_class}: {self.error}" if self.error else ""))


def _blank_result(stock_id: str, days: int) -> NewsIngestResult:
    return NewsIngestResult(stock_id=stock_id, days=days,
                            source_version=news_rss.NEWS_RULESET_VERSION)


def update_news_for(stock_id: str, name: str, days: int = 7, *,
                    run_id: str | None = None, run_at: str | None = None,
                    config_id: str = st.DEFAULT_CONFIG_ID,
                    timeout: float = 10.0, retries: int = 2,
                    getter=None, write_record: bool = True,
                    now: dt.datetime | None = None,
                    ) -> NewsIngestResult:
    """抓新聞入庫，回傳結構化結果（**不再**以 0 代表失敗或無資料）。

    重要（W3）：只以 ``(stock_id, link)`` 主鍵覆寫**這一檔**的新聞，絕不可整表覆蓋，
    否則每抓一檔股票就會清空其他所有股票的新聞。

    Idempotency（P2.8 改為「首次寫入即定案」）：同一 (stock_id, link)
    重複執行為 ``INSERT OR IGNORE``——**已存在的列整列跳過**，不覆寫。
    理由：重複抓取若覆寫既有列，會把 ``fetched_at``（point-in-time 可用起點）
    改成新的時間，使「當初什麼時候看得到這則新聞」變得不可重建；
    也可能把舊資料的 ``provenance='unknown'` 自動升級成已確認（任務禁止）。
    因此 ``rows_written`` 代表**真正新增**，``n_existing`` 代表重複抓取命中。
    """
    run_id = run_id or st.new_run_id()
    run_at = run_at or st.now_iso()
    started = st.now_iso()
    res = _blank_result(stock_id, days)

    fetched = news_rss.fetch_google_news(
        stock_id, name, days=days, timeout=timeout, retries=retries,
        getter=getter, now=now)
    res.rows_fetched = fetched.n_fetched
    res.n_future_skipped = fetched.n_future_skipped
    res.attempts = fetched.attempts
    res.source_version = fetched.source_version

    if fetched.status == news_rss.FAILED:
        # 失敗不得降級成「沒有新聞」
        res.status = st.IngestStatus.FAILED.value
        res.error_class = fetched.error_class
        res.error = fetched.error
    else:
        res.rows_parsed = fetched.n_stored
        res.n_neutral = fetched.n_neutral
        res.n_scored = fetched.n_scored
        res.n_missing_timestamp = fetched.n_missing_timestamp
        dates = [a["published"] for a in fetched.articles if a.get("published")]
        if dates:
            res.data_start, res.data_end = min(dates), max(dates)

        if not fetched.articles:
            res.status = st.IngestStatus.SUCCESS_EMPTY.value
            res.error_class = st.ErrorClass.EMPTY.value
        else:
            df = pd.DataFrame(fetched.articles)
            # 同一檔同一天可能有多則新聞；link 為空時用標題當鍵，避免互相覆蓋
            df["link"] = df["link"].fillna("").astype(str)
            blank = df["link"] == ""
            df.loc[blank, "link"] = ("title:"
                                     + df.loc[blank, "title"].fillna("").astype(str))
            # 溯源：讓每一列都能對回本次擷取任務與情緒規則版本
            df["run_id"] = run_id
            if "source_version" not in df.columns:
                df["source_version"] = news_rss.NEWS_RULESET_VERSION
            if "news_uid" not in df.columns:
                # 防御路徑：來源未提供識別碼時，以可重現的內容雜湊產生
                df["news_uid"] = [
                    news_rss.news_uid(
                        link=str(r.link), title=str(getattr(r, "title", "")),
                        published_at=(None if pd.isna(getattr(r, "published_at", None))
                                      else str(r.published_at)))
                    for r in df.itertuples(index=False)]
            try:
                n = store.write_df(df, "news", key_cols=("stock_id", "link"),
                                   if_conflict="ignore")
            except Exception as e:  # noqa: BLE001
                res.status = st.IngestStatus.FAILED.value
                res.error_class = st.ErrorClass.WRITE.value
                res.error = f"{type(e).__name__}: {e}"
            else:
                res.rows_written = n
                res.n_existing = int(len(df)) - int(n)
                # 狀態看「來源是否有提供有效資料」，而不是「這次有沒有新增」：
                # 重複抓取全部命中既有列時仍是 success_with_data（來源正常有資料），
                # 不可誤記為 success_empty（那會讓「沒有新聞」與「重複新聞」混同）。
                res.status = (st.IngestStatus.SUCCESS_WITH_DATA.value if len(df)
                              else st.IngestStatus.SUCCESS_EMPTY.value)

    ended = st.now_iso()
    rec = st.IngestRecord(
        run_id=run_id, dataset=st.DATASET_NEWS, run_at=run_at,
        source="google_news_rss", task="fetch_google_news",
        stock_id=stock_id, started_at=started, ended_at=ended,
        status=res.status, error_class=res.error_class, error=res.error,
        rows_fetched=res.rows_fetched, rows_parsed=res.rows_parsed,
        rows_validated=res.rows_parsed - res.n_missing_timestamp,
        rows_written=res.rows_written,
        data_start=res.data_start, data_end=res.data_end,
        attempts=res.attempts, source_version=res.source_version,
        config_id=config_id, provenance=st.RECORDED,
        notes=(fetched.as_notes() + f" existing={res.n_existing}"))
    res.record = rec
    if write_record:
        try:
            st.write_records([rec])
        except Exception as e:  # noqa: BLE001
            log.warning("寫入新聞擷取紀錄失敗：%s", e)
    return res
