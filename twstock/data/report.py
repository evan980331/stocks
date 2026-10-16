"""資料品質報告：新聞筆數、有效情緒覆蓋率、抓取成功率／失敗率與狀態分佈。

設計原則
  - **不把新增狀態紀錄當成預測能力改善**：本模組只描述「資料的可信度與可追溯性」，
    不產出任何績效或因子強度指標。
  - **舊資料如實標示**：``provenance='unknown'`` 的列單獨統計，計入「不可追溯」，
    不計入抓取成功；也不推測當時是否成功。
  - **分母寫清楚**：覆蓋率一律同時給出「全部新聞列」與「發布時間可追溯的列」
    兩種分母，避免用模糊的單一比率誤導。
  - **兩種日期不可混用**（P2.8 §4）：``published``（來源聲稱的發布日）與
    ``fetched_at``（本系統實際取得日）必須分別統計、分別給涵蓋區間；
    「當日新增」不等於「當日發布」。
  - **欄位不齊不得擲錯**（P2.8）：唯讀分析尚未遷移的舊 DB 時缺少溯源欄位，
    缺失欄位一律視為「不可追溯」，而不是猜測。

用於：
  python -m twstock.cli datacheck
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from . import status as st
from . import store

log = logging.getLogger(__name__)

NEWS_DATASET = st.DATASET_NEWS

REPORT_TITLE = "資料擷取與新聞品質報告"

# 每日明細最多列幾日（避免舊資料過長時淹沒真正重要的摘要）
MAX_DAILY_ROWS = 31

# 來源明細最多列幾個來源
MAX_SOURCE_ROWS = 20

UNDATED_LABEL = "(時間不明)"            # 發布時間無法追溯的列
ADDED_UNDATED_LABEL = "(取得時間不明)"   # 取得時間無法追溯的列（舊資料）

# 股票關聯依據（link_basis）標籤；未記錄（舊資料）不得冒充已驗證。
BASIS_LABELS = {
    "title_match": "標題可驗證",
    "rss_query": "僅查詢關聯",
    "unknown": "無法確認",
}
BASIS_MISSING = "(未記錄)"
# 無法確認股票關聯的依據值（§4 要求「無法確認關聯的筆數」）
UNVERIFIABLE_BASIS = {"rss_query", "unknown", BASIS_MISSING}

# news.source 空值標籤
SOURCE_MISSING = "(來源不明)"

# news 表的欄位順序（僅取實際存在的欄位；缺欄位者補 None）
_NEWS_WANTED = ["published", "score", "score_kind", "timestamp_source",
                "provenance", "fetched_at", "news_uid", "link_basis", "source"]


# --------------------------------------------------------------------------
# 新聞表欄位探測（唯讀模式下不會遷移，缺欄位時不得擲錯）
# --------------------------------------------------------------------------
def _news_columns(conn) -> set[str]:
    """回傳 news 表實際存在的欄位集合；表不存在時回空集合。"""
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='news'"
        ).fetchone()
        if not exists:
            return set()
        return {r[1] for r in conn.execute("PRAGMA table_info(news)").fetchall()}
    except sqlite3.Error:
        return set()


def _safe_read(sql: str, params=None) -> pd.DataFrame:
    """讀取；表不存在時回空表（唯讀分析舊 DB 常見），不讓報告整個失敗。

    仍走 ``store.read_sql``，因此在**可寫**模式下仍會觸發輕量遷移
    （維持既有行為：舊 ``ingest_log`` 會被回填到 ``ingest_run``）。
    """
    try:
        return store.read_sql(sql, params)
    except (pd.errors.DatabaseError, sqlite3.Error) as e:  # noqa: BLE001
        log.warning("報告查詢失敗（視為無資料）：%s", e)
        return pd.DataFrame()


def _read_news(wanted: list[str]) -> pd.DataFrame:
    """讀取 news 表指定欄位；表或欄位不存在時回傳空／補 None 的 DataFrame。

    補 None 的語意是「不可追溯」，不是「值為 0」——下游一律把缺失視為未知。
    """
    conn = store._connect()
    try:
        known = _news_columns(conn)
    finally:
        conn.close()
    if not known:
        return pd.DataFrame(columns=wanted)
    cols = [c for c in wanted if c in known]
    if not cols:
        return pd.DataFrame(columns=wanted)
    df = store.read_sql("SELECT " + ", ".join(cols) + " FROM news")
    for c in wanted:
        if c not in df.columns:
            df[c] = None
    return df


def _text_series(df: pd.DataFrame, col: str) -> pd.Series:
    """取字串欄位並把空字串正規化為 NA（空字串不是有效值）。"""
    s = df[col].astype("string")
    return s.where(s.str.strip().ne("") & s.notna(), other=pd.NA)


def _date_prefix(s: pd.Series) -> pd.Series:
    """取 ISO 時間的前 10 碼（日期）；不足 10 碼者視為不可追溯。"""
    p = s.str.slice(0, 10)
    return p.where(p.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False), other=pd.NA)


# --------------------------------------------------------------------------
# 抓取狀態（來自 ingest_run）
# --------------------------------------------------------------------------
def ingest_status_summary(run_id: str | None = None) -> pd.DataFrame:
    """依資料集與狀態統計擷取紀錄。``run_id`` 為 None 時用最新一輪。"""
    if run_id is None:
        row = _safe_read("SELECT MAX(run_id) AS r FROM ingest_run")
        run_id = (str(row.iloc[0]["r"])
                  if not row.empty and row.iloc[0]["r"] else None)
    if run_id is None:
        return pd.DataFrame(columns=["dataset", "status", "status_label", "n"])
    sql = ("SELECT dataset, status, COUNT(*) AS n FROM ingest_run "
           "WHERE run_id=? GROUP BY dataset, status ORDER BY dataset, status")
    df = _safe_read(sql, (run_id,))
    if df.empty:
        return pd.DataFrame(columns=["dataset", "status", "status_label", "n"])
    df["status_label"] = [st.IngestStatus(s).label for s in df["status"]]
    return df


def latest_run_id() -> str | None:
    row = _safe_read("SELECT MAX(run_id) AS r FROM ingest_run")
    if row.empty or not row.iloc[0]["r"]:
        return None
    return str(row.iloc[0]["r"])


def fetch_rates(dataset: str = NEWS_DATASET,
                run_id: str | None = None) -> dict:
    """抓取成功率／失敗率／不可追溯率（依結構化狀態，非布林 ok）。

    分母只含**狀態可判定**的紀錄；``unknown``（舊資料或無法追溯）另計，
    絕不併入成功率。尚未建立 ``ingest_run`` 的舊 DB 回全零，不擲錯。
    """
    sql = "SELECT status, COUNT(*) AS n FROM ingest_run WHERE dataset=?"
    params: list = [dataset]
    if run_id:
        sql += " AND run_id=?"
        params.append(run_id)
    sql += " GROUP BY status"
    df = _safe_read(sql, tuple(params))
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
    """新聞資料的可追溯性、情緒覆蓋率、日期涵蓋與股票關聯依據。"""

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
    # P2.8 §4：去重後總數、取得時間涵蓋、關聯依據覆蓋率
    n_distinct_uid: int = 0
    n_uid_missing: int = 0
    n_fetched_known: int = 0
    n_fetched_unknown: int = 0
    fetched_min: str = ""
    fetched_max: str = ""
    basis_counts: dict = field(default_factory=dict)
    n_link_verified: int = 0
    n_link_unverifiable: int = 0
    added_per_day: pd.DataFrame = field(default_factory=pd.DataFrame)

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

    @property
    def unknown_timestamp_rate(self) -> float | None:
        """發布時間未知（來源未提供或不可追溯）的比例。"""
        return (self.n_unknown_date / self.n_rows) if self.n_rows else None

    @property
    def fetched_known_rate(self) -> float | None:
        """取得時間可追溯的比例（point-in-time 可用性的必要條件）。"""
        return (self.n_fetched_known / self.n_rows) if self.n_rows else None

    @property
    def link_verified_rate(self) -> float | None:
        """股票關聯可驗證率（分母＝全部新聞列）。"""
        return (self.n_link_verified / self.n_rows) if self.n_rows else None

    def to_text(self) -> str:
        pct = lambda v: "無資料" if v is None else f"{v:.1%}"  # noqa: E731
        fet = self.fetched_known_rate
        fetched_unknown = None if fet is None else 1 - fet
        lines = [
            f"  新聞列數              {self.n_rows}"
            f"（可追溯 {self.n_recorded}、不可追溯 {self.n_untraceable}）",
            f"  去重後新聞總數        {self.n_distinct_uid}"
            f"（依 news_uid；無法比對 {self.n_uid_missing}）",
            f"  發布時間可追溯        {self.n_with_known_date}"
            f"（時間不明 {self.n_unknown_date}）",
            f"  取得時間可追溯        {self.n_fetched_known}"
            f"（時間不明 {self.n_fetched_unknown}）",
            f"  有效情緒分數（scored）{self.n_scored}",
            f"  情緒中性（neutral）   {self.n_neutral}",
            f"  無情緒分數（missing） {self.n_missing_score}",
            f"  有效情緒覆蓋率        全部列 {pct(self.scored_rate_all)}"
            f"／可追溯列 {pct(self.scored_rate_dated)}",
            f"  發布時間未知率         {pct(self.unknown_timestamp_rate)}",
            f"  取得時間未知率         {pct(fetched_unknown)}",
            f"  不可追溯率             {pct(self.untraceable_rate)}",
            f"  股票關聯可驗證率       {pct(self.link_verified_rate)}"
            f"（無法確認 {self.n_link_unverifiable}）",
            f"  涵蓋日期（發布）      {self.date_min or '—'} ~ "
            f"{self.date_max or '—'}",
            f"  涵蓋日期（取得）      {self.fetched_min or '—'} ~ "
            f"{self.fetched_max or '—'}",
        ]
        return "\n".join(lines)


def news_coverage() -> NewsCoverage:
    """計算新聞覆蓋率。舊資料（provenance=unknown）如實另計，不補猜。

    缺少溯源欄位（尚未遷移的舊 DB）時，該欄位一律視為不可追溯，
    而不是擲錯或猜測。
    """
    df = _read_news(_NEWS_WANTED)
    cov = NewsCoverage()
    if df.empty:
        return cov
    cov.n_rows = len(df)
    # 缺欄位／缺值一律視為不可追溯（不可當成有效值而誤算）
    kind = df["score_kind"].astype("string").fillna("missing")
    ts = df["timestamp_source"].astype("string").fillna("")
    prov = df["provenance"].astype("string").fillna(st.UNKNOWN_PROVENANCE)
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

    # --- P2.8 §4：去重後總數 ---
    uid = _text_series(df, "news_uid")
    cov.n_distinct_uid = int(uid.nunique())
    cov.n_uid_missing = int(uid.isna().sum())

    # --- P2.8 §4：取得時間涵蓋（與發布時間分開） ---
    fdate = _date_prefix(_text_series(df, "fetched_at"))
    cov.n_fetched_known = int(fdate.notna().sum())
    cov.n_fetched_unknown = cov.n_rows - cov.n_fetched_known
    fd = fdate.dropna()
    if not fd.empty:
        cov.fetched_min, cov.fetched_max = str(fd.min()), str(fd.max())

    # --- P2.8 §4：股票關聯依據覆蓋率 ---
    basis = _text_series(df, "link_basis").fillna(BASIS_MISSING)
    cov.basis_counts = {str(k): int(v) for k, v in basis.value_counts().items()}
    cov.n_link_verified = int((basis == "title_match").sum())
    cov.n_link_unverifiable = cov.n_rows - cov.n_link_verified

    # --- 每日（依發布時間） ---
    d = df.copy()
    d["date"] = pub.fillna(UNDATED_LABEL)
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

    # --- 每日（依取得時間；語意是「當日新增」，與上面不同） ---
    a = df.copy()
    a["date"] = fdate.fillna(ADDED_UNDATED_LABEL)
    a["_scored"] = (kind == "scored").astype(int)
    a["_neutral"] = (kind == "neutral").astype(int)
    a["_missing"] = (kind == "missing").astype(int)
    cov.added_per_day = (
        a.groupby("date", dropna=False)
        .agg(added=("score", "size"), scored=("_scored", "sum"),
             neutral=("_neutral", "sum"), missing_score=("_missing", "sum"))
        .reset_index().sort_values("date").reset_index(drop=True))
    return cov


# --------------------------------------------------------------------------
# 每日明細：發布 vs 新增（兩種日期不可混用）
# --------------------------------------------------------------------------
def daily_news_counts(cov: NewsCoverage | None = None) -> pd.DataFrame:
    """每日**發布**新聞筆數（依 published，最近優先）。

    任務要求報告需明確顯示每日新聞筆數；此函式是唯一來源，
    報表與程式取用同一份分母，避免兩處算法不一致。
    """
    df = (cov or news_coverage()).per_day
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "rows", "scored", "neutral",
                                     "missing_score", "unknown_timestamp"])
    dated = df[df["date"] != UNDATED_LABEL].sort_values("date", ascending=False)
    undated = df[df["date"] == UNDATED_LABEL]
    return pd.concat([dated, undated], ignore_index=True)


def daily_added_counts(cov: NewsCoverage | None = None) -> pd.DataFrame:
    """每日**新增**新聞筆數（依 fetched_at，即本系統實際取得日，最近優先）。

    與 :func:`daily_news_counts` 語意不同：這裡回答「哪一天多累積了新聞」，
    不是「新聞的發布日是哪一天」。舊資料（fetched_at 不可得）歸入
    ``(取得時間不明)``，不得當成任何一天的新增。
    """
    df = (cov or news_coverage()).added_per_day
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "added", "scored", "neutral",
                                     "missing_score"])
    dated = df[df["date"] != ADDED_UNDATED_LABEL].sort_values(
        "date", ascending=False)
    undated = df[df["date"] == ADDED_UNDATED_LABEL]
    return pd.concat([dated, undated], ignore_index=True)


def _daily_table_lines(cov: NewsCoverage) -> list[str]:
    """每日新聞筆數段落（以直線分隔，不假裝 CJK 能精確對齊）。"""
    rows = daily_news_counts(cov)
    if rows.empty:
        return ["  每日新聞筆數：無資料"]
    out = ["  每日新聞筆數（依**發布時間** published；最近優先；"
           "只列狀態可觀察的欄位）",
           "    日期 | 筆數 | 有分數 | 中性 | 無分數 | 時間不明"]
    dated = rows[rows["date"] != UNDATED_LABEL]
    undated = rows[rows["date"] == UNDATED_LABEL]
    for r in dated.head(MAX_DAILY_ROWS).itertuples(index=False):
        out.append(f"    {r.date} | {r.rows} | {r.scored} | {r.neutral} "
                   f"| {r.missing_score} | {r.unknown_timestamp}")
    if len(dated) > MAX_DAILY_ROWS:
        out.append(f"    …（另有 {len(dated) - MAX_DAILY_ROWS} 日未列出）")
    for r in undated.itertuples(index=False):
        out.append(f"    {UNDATED_LABEL} | {r.rows} | {r.scored} | {r.neutral} "
                   f"| {r.missing_score} | {r.unknown_timestamp}")
    return out


def _daily_added_lines(cov: NewsCoverage) -> list[str]:
    """每日**新增**新聞段落（依取得時間）；與發布日明細不可混用。"""
    rows = daily_added_counts(cov)
    if rows.empty:
        return ["  每日新增新聞筆數：無資料"]
    out = ["  每日新增新聞筆數（依**取得時間** fetched_at；最近優先；"
           "「新增」不等於「發布」）",
           "    日期 | 新增 | 有分數 | 中性 | 無分數"]
    dated = rows[rows["date"] != ADDED_UNDATED_LABEL]
    undated = rows[rows["date"] == ADDED_UNDATED_LABEL]
    for r in dated.head(MAX_DAILY_ROWS).itertuples(index=False):
        out.append(f"    {r.date} | {r.added} | {r.scored} | {r.neutral} "
                   f"| {r.missing_score}")
    if len(dated) > MAX_DAILY_ROWS:
        out.append(f"    …（另有 {len(dated) - MAX_DAILY_ROWS} 日未列出）")
    for r in undated.itertuples(index=False):
        out.append(f"    {ADDED_UNDATED_LABEL} | {r.added} | {r.scored} "
                   f"| {r.neutral} | {r.missing_score}")
    return out


def _coverage_range_lines(cov: NewsCoverage) -> list[str]:
    """發布／取得兩種涵蓋區間，明確不得互相代替。"""
    return [
        "  資料涵蓋區間（依欄位分開統計，不可互相代替）",
        f"    依發布時間 published   {cov.date_min or '—'} ~ "
        f"{cov.date_max or '—'}（可追溯 {cov.n_with_known_date}／{cov.n_rows}）",
        f"    依取得時間 fetched_at  {cov.fetched_min or '—'} ~ "
        f"{cov.fetched_max or '—'}（可追溯 {cov.n_fetched_known}／{cov.n_rows}）",
    ]


def _basis_lines(cov: NewsCoverage) -> list[str]:
    """股票關聯依據分佈：可驗證 vs 僅查詢關聯，不得混淆。"""
    if not cov.basis_counts:
        return ["  股票關聯依據：無資料"]
    out = ["  股票關聯依據（link_basis；「僅查詢關聯」不得視為已驗證）"]
    for k in ("title_match", "rss_query", "unknown", BASIS_MISSING):
        if k in cov.basis_counts:
            out.append(f"    {BASIS_LABELS.get(k, k):<8} {cov.basis_counts[k]}")
    for k, v in cov.basis_counts.items():
        if k not in ("title_match", "rss_query", "unknown", BASIS_MISSING):
            out.append(f"    {k:<8} {v}")
    out.append(f"    無法確認關聯筆數      {cov.n_link_unverifiable}")
    return out


# --------------------------------------------------------------------------
# 各新聞來源：資料量、失敗率、最近成功時間
# --------------------------------------------------------------------------
def news_source_volume() -> pd.DataFrame:
    """依 ``news.source``（新聞的發布媒體）統計資料量。

    這**不是**擷取任務的來源：``news.source`` 是媒體名稱（如 UDN、中央社），
    ``ingest_run.source`` 是抓取管道；兩者命名空間不同，
    不得合併成一張表互相對應，否則會把「媒體」誤讀成「抓取來源」。
    """
    news = _read_news(["source", "score_kind", "provenance", "fetched_at"])
    if not news.empty:
        src = _text_series(news, "source").fillna(SOURCE_MISSING)
        kind = news["score_kind"].astype("string").fillna("missing")
        prov = news["provenance"].astype("string").fillna(
            st.UNKNOWN_PROVENANCE)
        fdate = _date_prefix(_text_series(news, "fetched_at"))
        n = pd.DataFrame({
            "source": src,
            "_scored": (kind == "scored").astype(int),
            "_untraceable": (prov == st.UNKNOWN_PROVENANCE).astype(int),
            "fetched": fdate,
        })
        news_side = (n.groupby("source", dropna=False)
                     .agg(n_news=("_scored", "size"),
                          n_scored=("_scored", "sum"),
                          n_untraceable=("_untraceable", "sum"),
                          last_fetched=("fetched", "max"))
                     .reset_index())
    else:
        return pd.DataFrame(columns=["source", "n_news", "n_scored",
                                     "n_untraceable", "last_fetched"])
    # 空值正規化為空字串（pd.NA 的布林判斷會擲錯，且「無」不等於任何日期）
    news_side["last_fetched"] = news_side["last_fetched"].fillna("")
    return news_side.sort_values("source").reset_index(drop=True)


def news_fetch_source_stats(run_id: str | None = None) -> pd.DataFrame:
    """依 ``ingest_run.source``（擷取管道）統計成敗與最近成功時間。

    失敗率的分母只含**狀態可判定**的紀錄；``unknown`` 另計，不併入成功率。
    「最近成功」只取來源端有正常結論（有資料／空結果／部分成功）者，
    失敗時間不得被當成成功時間。
    """
    sql = ("SELECT source, status, ended_at, run_at FROM ingest_run "
           "WHERE dataset=?")
    params: list = [NEWS_DATASET]
    if run_id:
        sql += " AND run_id=?"
        params.append(run_id)
    ir = _safe_read(sql, tuple(params))
    if not ir.empty:
        ir = ir.copy()
        ir["source"] = _text_series(ir, "source").fillna(SOURCE_MISSING)
        ok = ir["status"].isin([st.IngestStatus.SUCCESS_WITH_DATA.value,
                                st.IngestStatus.SUCCESS_EMPTY.value,
                                st.IngestStatus.PARTIAL_SUCCESS.value])
        decided = ir["status"].isin([
            st.IngestStatus.SUCCESS_WITH_DATA.value,
            st.IngestStatus.SUCCESS_EMPTY.value,
            st.IngestStatus.PARTIAL_SUCCESS.value,
            st.IngestStatus.FAILED.value,
            st.IngestStatus.SKIPPED.value])
        ir["_ok"] = ok.astype(int)
        ir["_failed"] = (ir["status"] == st.IngestStatus.FAILED.value).astype(int)
        ir["_decided"] = decided.astype(int)
        ir["_success_at"] = _text_series(ir, "ended_at").fillna(
            _text_series(ir, "run_at")).where(ok, other=pd.NA)
        ir["_success_status"] = ir["status"].where(ok, other=pd.NA)
        run_side = (ir.groupby("source", dropna=False)
                    .agg(n_records=("status", "size"),
                         n_failed=("_failed", "sum"),
                         n_decided=("_decided", "sum"),
                         last_success=("_success_at", "max"))
                    .reset_index())
        # 最近一次成功所對應的狀態（取 last_success 對應列）
        last_status = {}
        for s, grp in ir.dropna(subset=["_success_at"]).groupby("source"):
            idx = grp["_success_at"].idxmax()
            last_status[s] = grp.loc[idx, "_success_status"]
        run_side["last_success_status"] = [
            last_status.get(s, "") for s in run_side["source"]]
    else:
        return pd.DataFrame(columns=["source", "n_records", "n_failed",
                                     "n_decided", "failure_rate",
                                     "last_success", "last_success_status"])
    for c in ("n_records", "n_failed", "n_decided"):
        run_side[c] = (pd.to_numeric(run_side[c], errors="coerce")
                       .fillna(0).astype(int))
    run_side["failure_rate"] = pd.Series(
        [(f / d) if d else None for f, d in zip(run_side["n_failed"],
                                                run_side["n_decided"])],
        index=run_side.index, dtype=object)
    for c in ("last_success", "last_success_status"):
        run_side[c] = run_side.get(c, pd.Series([""] * len(run_side))).fillna("")
    return run_side[["source", "n_records", "n_failed", "n_decided",
                     "failure_rate", "last_success", "last_success_status"]] \
        .sort_values("source").reset_index(drop=True)


def _source_lines(run_id: str | None = None) -> list[str]:
    """各來源段落：依 ``news.source`` 與 ``ingest_run.source`` **分開**列出。"""
    out = ["  各新聞來源（兩種來源識別不同，分開列出，不得互相對應）"]

    vol = news_source_volume()
    if vol.empty:
        out.append("    新聞來源（news.source，發布媒體）：無資料")
    else:
        out.append("    新聞來源資料量（依 news.source，發布媒體）")
        out.append("      來源 | 資料量 | 有分數 | 不可追溯 | 最近取得")
        for r in vol.head(MAX_SOURCE_ROWS).itertuples(index=False):
            out.append(f"      {r.source} | {r.n_news} | {r.n_scored} "
                       f"| {r.n_untraceable} | {r.last_fetched or '—'}")
        if len(vol) > MAX_SOURCE_ROWS:
            out.append(f"      …（另有 {len(vol) - MAX_SOURCE_ROWS} 個來源未列出）")

    fs = news_fetch_source_stats(run_id)
    if fs.empty:
        out.append("    擷取來源（ingest_run.source，抓取管道）：無資料")
    else:
        out.append("    擷取來源成敗（依 ingest_run.source，抓取管道）")
        out.append("      來源 | 紀錄 | 失敗 | 失敗率 | 最近成功")
        for r in fs.itertuples(index=False):
            rate = ("無可判定" if r.failure_rate is None or pd.isna(r.failure_rate)
                    else f"{r.failure_rate:.1%}")
            out.append(f"      {r.source} | {r.n_records} | {r.n_failed} "
                       f"| {rate} | {r.last_success or '—'}"
                       f"{'（' + r.last_success_status + '）' if r.last_success_status else ''}")
    return out


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
             *_coverage_range_lines(cov),
             "",
             *_basis_lines(cov),
             "",
             *_daily_table_lines(cov),
             "",
             *_daily_added_lines(cov),
             "",
             *_source_lines(run_id),
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
                  "不得視為抓取成功。",
                  "  這些列同樣沒有取得時間，依 point-in-time 規則不得進入訊號路徑。"]
    lines += ["",
              "說明：本報告只描述資料可信度，"
              "不代表新聞因子或任何策略具備可交易優勢。"]
    return "\n".join(lines)


def _pct(v: float | None) -> str:
    return "無可判定紀錄" if v is None else f"{v:.1%}"
