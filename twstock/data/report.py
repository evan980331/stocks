"""資料品質報告：新聞筆數、有效情緒覆蓋率、抓取成功率／失敗率與狀態分佈。

設計原則
  - **不把新增狀態紀錄當成預測能力改善**：本模組只描述「資料的可信度與可追溯性」，
    不產出任何績效或因子強度指標。
  - **舊資料如實標示**：``provenance='unknown'`` 的列單獨統計，計入「不可追溯」，
    不計入抓取成功；也不推測當時是否成功。
  - **分母寫清楚**：覆蓋率一律同時給出「全部新聞列」與「發布時間可追溯的列」
    兩種分母，避免用模糊的單一比率誤導。

用於：
  python -m twstock.cli datacheck
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from . import status as st
from . import store

log = logging.getLogger(__name__)

NEWS_DATASET = st.DATASET_NEWS

REPORT_TITLE = "資料擷取與新聞品質報告"


# --------------------------------------------------------------------------
# 抓取狀態（來自 ingest_run）
# --------------------------------------------------------------------------
def ingest_status_summary(run_id: str | None = None) -> pd.DataFrame:
    """依資料集與狀態統計擷取紀錄。``run_id`` 為 None 時用最新一輪。"""
    if run_id is None:
        row = store.read_sql("SELECT MAX(run_id) AS r FROM ingest_run")
        run_id = (str(row.iloc[0]["r"])
                  if not row.empty and row.iloc[0]["r"] else None)
    if run_id is None:
        return pd.DataFrame(columns=["dataset", "status", "status_label", "n"])
    sql = ("SELECT dataset, status, COUNT(*) AS n FROM ingest_run "
           "WHERE run_id=? GROUP BY dataset, status ORDER BY dataset, status")
    df = store.read_sql(sql, (run_id,))
    if df.empty:
        return pd.DataFrame(columns=["dataset", "status", "status_label", "n"])
    df["status_label"] = [st.IngestStatus(s).label for s in df["status"]]
    return df


def latest_run_id() -> str | None:
    row = store.read_sql("SELECT MAX(run_id) AS r FROM ingest_run")
    if row.empty or not row.iloc[0]["r"]:
        return None
    return str(row.iloc[0]["r"])


def fetch_rates(dataset: str = NEWS_DATASET,
                run_id: str | None = None) -> dict:
    """抓取成功率／失敗率／不可追溯率（依結構化狀態，非布林 ok）。

    分母只含**狀態可判定**的紀錄；``unknown``（舊資料或無法追溯）另計，
    絕不併入成功率。
    """
    sql = "SELECT status, COUNT(*) AS n FROM ingest_run WHERE dataset=?"
    params: list = [dataset]
    if run_id:
        sql += " AND run_id=?"
        params.append(run_id)
    sql += " GROUP BY status"
    df = store.read_sql(sql, tuple(params))
    counts = ({str(r.status): int(r.n) for r in df.itertuples()}
              if not df.empty else {})

    unknown = counts.get(st.IngestStatus.UNKNOWN.value, 0)
    failed = counts.get(st.IngestStatus.FAILED.value, 0)
    with_data = counts.get(st.IngestStatus.SUCCESS_WITH_DATA.value, 0)
    empty = counts.get(st.IngestStatus.SUCCESS_EMPTY.value, 0)
    partial = counts.get(st.IngestStatus.PARTIAL_SUCCESS.value, 0)
    skipped = counts.get(st.IngestStatus.SKIPPED.value, 0)
    decided = with_data + empty + partial + failed + skipped
    ok_n = with_data + empty + partial
    return {
        "dataset": dataset,
        "run_id": run_id or latest_run_id(),
        "n_records": sum(counts.values()),
        "n_decided": decided,
        "n_unknown": unknown,
        "n_success_with_data": with_data,
        "n_success_empty": empty,
        "n_partial": partial,
        "n_failed": failed,
        "n_skipped": skipped,
        "success_rate": (ok_n / decided) if decided else None,
        "failure_rate": (failed / decided) if decided else None,
        "untraceable_rate": (unknown / sum(counts.values())
                             if sum(counts.values()) else None),
        "status_counts": counts,
    }


# --------------------------------------------------------------------------
# 新聞覆蓋率
# --------------------------------------------------------------------------
@dataclass
class NewsCoverage:
    """新聞資料的可追溯性與情緒覆蓋率。"""

    n_rows: int = 0
    n_with_known_date: int = 0
    n_unknown_date: int = 0
    n_scored: int = 0
    n_neutral: int = 0
    n_missing_score: int = 0
    n_recorded: int = 0
    n_untraceable: int = 0
    date_min: str = ""
    date_max: str = ""
    per_day: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def scored_rate_all(self) -> float | None:
        """有效情緒覆蓋率（分母＝全部新聞列）。"""
        return (self.n_scored / self.n_rows) if self.n_rows else None

    @property
    def scored_rate_dated(self) -> float | None:
        """有效情緒覆蓋率（分母＝發布時間可追溯的列）。

        這才是能進入情緒衰減窗口、實際會影響訊號的列。
        """
        return (self.n_scored / self.n_with_known_date
                if self.n_with_known_date else None)

    @property
    def untraceable_rate(self) -> float | None:
        return (self.n_untraceable / self.n_rows) if self.n_rows else None

    def to_text(self) -> str:
        pct = lambda v: "無資料" if v is None else f"{v:.1%}"  # noqa: E731
        lines = [
            f"  新聞列數              {self.n_rows}"
            f"（可追溯 {self.n_recorded}、不可追溯 {self.n_untraceable}）",
            f"  發布時間可追溯        {self.n_with_known_date}"
            f"（時間不明 {self.n_unknown_date}）",
            f"  有效情緒分數（scored）{self.n_scored}",
            f"  情緒中性（neutral）   {self.n_neutral}",
            f"  無情緒分數（missing） {self.n_missing_score}",
            f"  有效情緒覆蓋率        全部列 {pct(self.scored_rate_all)}"
            f"／可追溯列 {pct(self.scored_rate_dated)}",
            f"  不可追溯率             {pct(self.untraceable_rate)}",
            f"  涵蓋日期              {self.date_min or '—'} ~ "
            f"{self.date_max or '—'}",
        ]
        return "\n".join(lines)


def news_coverage() -> NewsCoverage:
    """計算新聞覆蓋率。舊資料（provenance=unknown）如實另計，不補猜。"""
    df = store.read_sql(
        "SELECT published, score, score_kind, timestamp_source, provenance "
        "FROM news")
    cov = NewsCoverage()
    if df.empty:
        return cov
    cov.n_rows = len(df)
    kind = df["score_kind"].astype("string").fillna("missing")
    ts = df["timestamp_source"].astype("string")
    prov = df["provenance"].astype("string")
    pub = df["published"].astype("string")

    cov.n_scored = int((kind == "scored").sum())
    cov.n_neutral = int((kind == "neutral").sum())
    cov.n_missing_score = int((kind == "missing").sum())
    cov.n_with_known_date = int((ts == "source").sum())
    cov.n_unknown_date = cov.n_rows - cov.n_with_known_date
    cov.n_untraceable = int((prov == st.UNKNOWN_PROVENANCE).sum())
    cov.n_recorded = cov.n_rows - cov.n_untraceable
    dates = pub[(ts == "source") & pub.notna()]
    if not dates.empty:
        cov.date_min, cov.date_max = str(dates.min()), str(dates.max())

    d = df.copy()
    d["date"] = pub.fillna("(時間不明)")
    d["_scored"] = (kind == "scored").astype(int)
    d["_neutral"] = (kind == "neutral").astype(int)
    d["_missing"] = (kind == "missing").astype(int)
    d["_unknown_ts"] = (ts != "source").astype(int)
    d["_untraceable"] = (prov == st.UNKNOWN_PROVENANCE).astype(int)
    cov.per_day = (
        d.groupby("date", dropna=False)
        .agg(rows=("score", "size"), scored=("_scored", "sum"),
             neutral=("_neutral", "sum"), missing_score=("_missing", "sum"),
             unknown_timestamp=("_unknown_ts", "sum"),
             untraceable=("_untraceable", "sum"))
        .reset_index().sort_values("date").reset_index(drop=True))
    return cov


# --------------------------------------------------------------------------
# 完整報告
# --------------------------------------------------------------------------
def build_report_text(run_id: str | None = None) -> str:
    """組合人類可讀的資料品質報告（供 CLI 與文件引用）。"""
    cov = news_coverage()
    rates = fetch_rates(NEWS_DATASET, run_id)
    overall = fetch_rates(st.DATASET_PRICE, run_id)
    legacy_news = cov.n_untraceable
    lines = [f"=== {REPORT_TITLE} ===",
             f"最新執行 run_id：{latest_run_id() or '（無紀錄）'}",
             "",
             "── 新聞資料品質 ──",
             cov.to_text(),
             "",
             "── 新聞抓取狀態（依 ingest_run） ──",
             f"  紀錄數                {rates['n_records']}"
             f"（狀態可判定 {rates['n_decided']}、不可追溯 {rates['n_unknown']}）",
             f"  成功（有資料）        {rates['n_success_with_data']}",
             f"  成功（無資料）        {rates['n_success_empty']}",
             f"  部分成功              {rates['n_partial']}",
             f"  失敗                  {rates['n_failed']}",
             f"  抓取成功率            {_pct(rates['success_rate'])}",
             f"  抓取失敗率            {_pct(rates['failure_rate'])}",
             f"  不可追溯率            {_pct(rates['untraceable_rate'])}",
             "",
             "── 價量抓取狀態（依 ingest_run） ──",
             f"  紀錄數                {overall['n_records']}"
             f"（狀態可判定 {overall['n_decided']}、不可追溯 {overall['n_unknown']}）",
             f"  成功（有資料）        {overall['n_success_with_data']}",
             f"  成功（無資料）        {overall['n_success_empty']}",
             f"  部分成功              {overall['n_partial']}",
             f"  失敗                  {overall['n_failed']}",
             ]
    if legacy_news:
        lines += ["",
                  "── 舊資料提醒 ──",
                  f"  {legacy_news} 則新聞寫入於狀態治理之前，"
                  "當時是否抓取成功無從判定，已標為不可追溯（unknown）。",
                  "  其分數欄位可直接觀察，但抓取成敗與來源版本不可重建；"
                  "不得視為抓取成功。"]
    lines += ["",
              "說明：本報告只描述資料可信度，"
              "不代表新聞因子或任何策略具備可交易優勢。"]
    return "\n".join(lines)


def _pct(v: float | None) -> str:
    return "無可判定紀錄" if v is None else f"{v:.1%}"


def build_report() -> dict:
    """結構化報告（供程式使用）。"""
    return {
        "run_id": latest_run_id(),
        "news_coverage": news_coverage(),
        "news_fetch_rates": fetch_rates(NEWS_DATASET),
        "price_fetch_rates": fetch_rates(st.DATASET_PRICE),
        "status_summary": ingest_status_summary(),
    }
