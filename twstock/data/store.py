"""SQLite 儲存層：price、institution、margin、revenue、news、signals、experts、ingest_log、ingest_run。

寫入策略（重要）：
  所有寫入一律使用 ``INSERT OR REPLACE``（依各表 PRIMARY KEY 去重）。
  禁止使用 ``pandas.to_sql(if_exists="replace")``——它會 DROP 整張表，造成其他標的資料遺失。

唯讀保護：
  ``TWSTOCK_READONLY=1`` 時完全不寫入，僅以 mode=ro 開啟，用於安全分析既有資料。
  （此機制與任何雲端平台無關，純粹是本機保護。）

遷移原則（向後相容，絕不覆寫舊資料）：
  - 新欄位一律以 ``ALTER TABLE ... ADD COLUMN`` 補上（既有列得到 NULL）。
  - 舊資料的**狀態不可追溯**：``news.provenance``／``timestamp_source`` 補為
    ``'unknown'``，不得推測當時抓取是否成功。
  - 舊 ``ingest_log`` 只有布林 ``ok``，無從得知是「成功有資料」還是「成功無資料」，
    故一次性回填到 ``ingest_run`` 時一律標為 ``unknown``（由 ``meta`` 鍵守門，只做一次）。
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..config import DATA_DIR, DB_PATH, is_readonly

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS price (
    date TEXT, stock_id TEXT,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL, turnover REAL,
    PRIMARY KEY (stock_id, date)
);
CREATE TABLE IF NOT EXISTS institution (
    date TEXT, stock_id TEXT, name TEXT,
    buy REAL, sell REAL,
    PRIMARY KEY (stock_id, date, name)
);
CREATE TABLE IF NOT EXISTS margin (
    date TEXT, stock_id TEXT,
    MarginPurchaseBuy REAL, MarginPurchaseCashRepayment REAL,
    MarginPurchaseLimit REAL, MarginPurchaseSell REAL,
    MarginPurchaseTodayBalance REAL, MarginPurchaseYesterdayBalance REAL,
    Note TEXT,
    ShortSaleBuy REAL, ShortSaleCashRepayment REAL, ShortSaleLimit REAL,
    ShortSaleSell REAL, ShortSaleTodayBalance REAL,
    ShortSaleYesterdayBalance REAL, ShortSaleNote TEXT,
    PRIMARY KEY (stock_id, date)
);
CREATE TABLE IF NOT EXISTS revenue (
    date TEXT, stock_id TEXT, country TEXT,
    revenue REAL, revenue_month INTEGER, revenue_year INTEGER,
    create_time TEXT,
    PRIMARY KEY (stock_id, revenue_year, revenue_month)
);
CREATE TABLE IF NOT EXISTS news (
    published DATE, stock_id TEXT, title TEXT, link TEXT,
    score REAL, source TEXT,
    published_at TEXT,     -- 來源聲稱的發布時間（未知時為 NULL，不臆測）
    fetched_at TEXT,       -- 實際抓取時間（與發布時間必須可分）
    timestamp_source TEXT, -- source：有來源時間；unknown：時間不可得
    score_kind TEXT,       -- scored：有非零情緒分數；neutral：存在但中性；
                           -- missing：無有效分數
    provenance TEXT,       -- recorded：本階段之後；unknown：不可追溯的舊資料
    news_uid TEXT,         -- 穩定的新聞識別碼（內容雜湊；跨股票可比對，舊列為 NULL）
    content_hash TEXT,     -- 可重現的內容雜湊（標題＋正規化連結＋發布時間）
    run_id TEXT,           -- 寫入該列的擷取任務（→ ingest_run；舊列不可追溯）
    link_basis TEXT,       -- 股票關聯依據：title_match（標題可驗證）/
                           -- rss_query（僅依查詢關聯，內容無法驗證）/ unknown
    source_version TEXT,   -- 寫入該列時的情緒規則版本（可重現當初的分數）
    PRIMARY KEY (stock_id, link)
);
CREATE TABLE IF NOT EXISTS signals (
    date TEXT, stock_id TEXT, score REAL, reason TEXT,
    close REAL, rank INTEGER, action TEXT, ml_score REAL,
    PRIMARY KEY (date, stock_id)
);
CREATE TABLE IF NOT EXISTS experts (
    published DATE, source TEXT, title TEXT, link TEXT,
    score REAL, kind TEXT,
    PRIMARY KEY (source, link)
);
CREATE TABLE IF NOT EXISTS corporate_actions (
    stock_id TEXT, date TEXT, kind TEXT,
    factor REAL, price_scale INTEGER, detail TEXT,
    PRIMARY KEY (stock_id, date, kind)
);
CREATE TABLE IF NOT EXISTS ingest_log (
    run_at TEXT, stock_id TEXT, dataset TEXT,
    ok INTEGER, rows INTEGER, error TEXT,
    PRIMARY KEY (run_at, stock_id, dataset)
);
-- 擷取狀態治理：每次執行留下去重、可重建、向後相容的紀錄。
-- ok/rows 的布林語意不足以區分「成功但無資料」與「失敗」，故另立此表。
CREATE TABLE IF NOT EXISTS ingest_run (
    run_id TEXT, dataset TEXT, stock_id TEXT,
    run_at TEXT, source TEXT, task TEXT,
    started_at TEXT, ended_at TEXT,
    status TEXT, error_class TEXT, error TEXT,
    rows_fetched INTEGER, rows_parsed INTEGER,
    rows_validated INTEGER, rows_written INTEGER,
    data_start TEXT, data_end TEXT,
    attempts INTEGER, source_version TEXT, config_id TEXT,
    provenance TEXT, notes TEXT,
    PRIMARY KEY (run_id, dataset, stock_id)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT
);
"""

# 各表用於去重的主鍵（與 SCHEMA 一致）
PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "price": ("stock_id", "date"),
    "institution": ("stock_id", "date", "name"),
    "margin": ("stock_id", "date"),
    "revenue": ("stock_id", "revenue_year", "revenue_month"),
    "news": ("stock_id", "link"),
    "signals": ("date", "stock_id"),
    "experts": ("source", "link"),
    "corporate_actions": ("stock_id", "date", "kind"),
    "ingest_log": ("run_at", "stock_id", "dataset"),
    "ingest_run": ("run_id", "dataset", "stock_id"),
    "meta": ("key",),
}

# 舊資料無法追溯的標記值（與 twstock.data.status 一致；此處不 import 以避免循環）
UNKNOWN_PROVENANCE = "unknown"


def _ensure_dir() -> None:
    if is_readonly():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    """開啟 DB。唯讀模式用 file: URI + mode=ro，且不建表。"""
    if is_readonly():
        uri = Path(DB_PATH).resolve().as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True)
    _ensure_dir()
    return sqlite3.connect(DB_PATH)


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """輕量向前相容遷移：補上舊 DB 缺少的欄位，並標記不可追溯的舊資料。"""
    wanted = {
        "signals": [("ml_score", "REAL")],
        "price": [("amount", "REAL"), ("turnover", "REAL")],
        "news": [("published_at", "TEXT"), ("fetched_at", "TEXT"),
                 ("timestamp_source", "TEXT"), ("score_kind", "TEXT"),
                 ("provenance", "TEXT"), ("news_uid", "TEXT"),
                 ("content_hash", "TEXT"), ("run_id", "TEXT"),
                 ("link_basis", "TEXT"), ("source_version", "TEXT")],
    }
    for table, cols in wanted.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if not existing:
            continue
        for name, decl in cols:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                log.info("migrate: %s 新增欄位 %s", table, name)
    _ensure_news_indexes(conn)
    _mark_legacy_news(conn)
    _backfill_ingest_run(conn)


def _ensure_news_indexes(conn: sqlite3.Connection) -> None:
    """新聞表查詢索引（必須在補欄位**之後**才建，舊 DB 才不會因欄位不存在而失敗）。

    每日累積的冪等查詢、point-in-time 過濾（依 fetched_at / published）
    與跨股票同新聞比對（news_uid）都需要索引；否則會隨資料累積線性變慢。
    """
    for stmt in (
        "CREATE INDEX IF NOT EXISTS idx_news_uid ON news(news_uid)",
        "CREATE INDEX IF NOT EXISTS idx_news_fetched ON news(fetched_at)",
        "CREATE INDEX IF NOT EXISTS idx_news_published ON news(published)",
        "CREATE INDEX IF NOT EXISTS idx_news_run ON news(run_id)",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.Error as e:  # pragma: no cover - 防止索引問題阻斷開檔
            log.warning("建索引失敗（不影響寫入）：%s (%s)", stmt, e)


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))


def _mark_legacy_news(conn: sqlite3.Connection) -> None:
    """把本階段之前寫入的新聞標為不可追溯。

    可以重建的只有**分數本身是否存在**：``score`` 欄位就是直接證據
    （NULL → missing、0 → neutral、其餘 → scored）。
    但當時**抓取是否成功**無從得知，故 ``provenance`` 與 ``timestamp_source``
    一律標為 ``unknown``，絕不補猜成功。只處理尚未標記的列，可重複執行且不覆寫。
    """
    if _meta_get(conn, "news_legacy_marked"):
        return
    cols = {r[1] for r in conn.execute("PRAGMA table_info(news)").fetchall()}
    if not cols:
        return
    conn.execute(
        "UPDATE news SET provenance=? WHERE provenance IS NULL",
        (UNKNOWN_PROVENANCE,))
    conn.execute(
        "UPDATE news SET timestamp_source=? WHERE timestamp_source IS NULL",
        (UNKNOWN_PROVENANCE,))
    if "score_kind" in cols:
        conn.execute(
            "UPDATE news SET score_kind='missing' WHERE score_kind IS NULL "
            "AND score IS NULL")
        conn.execute(
            "UPDATE news SET score_kind='neutral' WHERE score_kind IS NULL "
            "AND score = 0")
        conn.execute(
            "UPDATE news SET score_kind='scored' WHERE score_kind IS NULL")
    n = conn.execute("SELECT COUNT(*) FROM news WHERE provenance=?",
                     (UNKNOWN_PROVENANCE,)).fetchone()[0]
    _meta_set(conn, "news_legacy_marked", dt.datetime.now().isoformat(
        timespec="seconds"))
    _meta_set(conn, "news_legacy_unknown_rows", str(int(n)))
    log.info("migrate: %d 列既有新聞標記為不可追溯（unknown）", int(n))


def _backfill_ingest_run(conn: sqlite3.Connection) -> None:
    """把舊 ``ingest_log`` 一次性回填到 ``ingest_run``，狀態一律 unknown。

    ``ingest_log.ok`` 只區分「有回應」與「呼叫失敗」，無法分辨「成功有資料」與
    「成功但無資料」；因此回填時不得推論，一律記為 unknown。以 meta 鍵守門，
    只做一次，且不覆寫既有（新的、狀態明確的）紀錄。
    """
    if _meta_get(conn, "ingest_run_backfilled"):
        return
    has_log = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ingest_log'"
    ).fetchone()
    if not has_log:
        return
    rows = conn.execute(
        "SELECT run_at, stock_id, dataset, ok, rows, error FROM ingest_log"
    ).fetchall()
    for run_at, stock_id, dataset, ok, nrows, error in rows:
        conn.execute(
            "INSERT OR IGNORE INTO ingest_run ("
            "run_id, dataset, stock_id, run_at, source, task, "
            "status, error_class, error, rows_written, attempts, "
            "provenance, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"legacy:{run_at}", dataset or "", stock_id or "", run_at or "",
             "legacy", dataset or "", "unknown", "unknown", error or "",
             int(nrows or 0), 0, UNKNOWN_PROVENANCE,
             "由 ingest_log 回填；當時無法區分成功有資料與成功無資料"))
    _meta_set(conn, "ingest_run_backfilled",
              dt.datetime.now().isoformat(timespec="seconds"))
    if rows:
        log.info("migrate: 回填 %d 筆舊 ingest_log 為 unknown 狀態", len(rows))


def _adapt(v: Any) -> Any:
    """把 pandas/numpy 值轉成 sqlite3 可接受的型別（NaN → NULL）。"""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        return str(v)
    if hasattr(v, "item"):  # numpy scalar
        try:
            return v.item()
        except Exception:  # noqa: BLE001
            return v
    return v


class WriteRejected(RuntimeError):
    """寫入被拒絕（欄位無交集、表不存在）。此前只寫 log 並回 0，
    會讓呼叫端把「完全沒寫入」誤記為成功。"""


def write_df(df: pd.DataFrame, table: str,
             key_cols: Sequence[str] | None = None,
             *, if_conflict: str = "replace") -> int:
    """以 INSERT OR REPLACE 寫入 DataFrame。

    - 僅寫入該表已存在的欄位（容忍來源欄位變動）。
    - 依主鍵去重，重跑同一任務不會產生重複、也不會刪掉其他標的的資料。
    - 回傳實際寫入列數。
    - 有資料但**無交集欄位**時擲出 ``WriteRejected``，不再靜默回 0，
      以免呼叫端把完全沒寫入當成成功。

    ``if_conflict``
      - ``"replace"``（預設）：依主鍵覆寫，適合價量等可重建資料。
      - ``"ignore"``：**首次寫入即定案**——已存在的主鍵整列跳過。
        新聞採用此模式：重複抓取不覆寫既有列，因此既有的
        ``fetched_at``（point-in-time 可用起點）、``provenance``
        與分數都不會被後來的抓取改寫，舊資料的 ``unknown``
        也不會被自動升級為已確認。回傳值＝**真正新增**的列數。
    """
    if df is None or df.empty:
        return 0
    if if_conflict not in ("replace", "ignore"):
        raise ValueError(f"if_conflict 必須是 replace 或 ignore，得到 {if_conflict!r}")
    if is_readonly():
        raise RuntimeError("唯讀模式（TWSTOCK_READONLY）不允許寫入")
    conn = _connect()
    try:
        _init(conn)
        known = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if not known:
            log.warning("write_df: 表 %s 不存在，拒絕寫入", table)
            raise WriteRejected(f"表 {table} 不存在")
        cols = [c for c in df.columns if c in known]
        if not cols:
            log.warning("write_df: %s 無交集欄位，拒絕寫入", table)
            raise WriteRejected(
                f"表 {table} 與來源欄位無交集（{len(df)} 列未寫入）")
        keys = tuple(key_cols or PRIMARY_KEYS.get(table, ()))
        on_conflict = ""
        if keys and all(k in known for k in keys):
            on_conflict = (" OR IGNORE" if if_conflict == "ignore"
                           else " OR REPLACE")
        else:
            log.warning("write_df: %s 主鍵 %s 不完整，改用 INSERT", table, keys)
        placeholders = ",".join("?" * len(cols))
        sql = (f"INSERT{on_conflict} INTO {table} ({','.join(cols)}) "
               f"VALUES ({placeholders})")
        rows = [tuple(_adapt(v) for v in rec)
                for rec in df[cols].itertuples(index=False, name=None)]
        before = conn.total_changes
        conn.executemany(sql, rows)
        conn.commit()
        if if_conflict == "ignore":
            # OR IGNORE 會略過既有主鍵，total_changes 只算真正新增的列
            return int(conn.total_changes - before)
        return len(rows)
    finally:
        conn.close()


def read_sql(sql: str, params: tuple | dict | None = None) -> pd.DataFrame:
    if is_readonly() and not Path(DB_PATH).exists():
        log.warning("唯讀模式找不到 DB 快照：%s", DB_PATH)
        return pd.DataFrame()
    conn = _connect()
    try:
        if not is_readonly():
            _init(conn)
        return pd.read_sql_query(sql, conn, params=params or ())
    finally:
        conn.close()


def exec_sql(sql: str, params: tuple | dict | None = None) -> None:
    if is_readonly():
        raise RuntimeError("唯讀模式（TWSTOCK_READONLY）不允許寫入，請取消該環境變數")
    conn = _connect()
    try:
        _init(conn)
        conn.execute(sql, params or ())
        conn.commit()
    finally:
        conn.close()


def table_exists(table: str) -> bool:
    if is_readonly() and not Path(DB_PATH).exists():
        return False
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,)).fetchone()
        return row is not None
    finally:
        conn.close()


def get_meta(key: str) -> str | None:
    df = read_sql("SELECT value FROM meta WHERE key=?", (key,))
    return df.iloc[0]["value"] if not df.empty else None


def set_meta(key: str, value: str) -> None:
    exec_sql("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))


def log_ingest(run_at: str, stock_id: str, dataset: str, ok: bool,
               rows: int = 0, error: str = "") -> None:
    """記錄單一 (run, stock, dataset) 的抓取結果，供健康檢查與失敗重試。

    保留舊介面（既有測試與呼叫端相依）；新程式碼請改用
    ``twstock.data.status.IngestRecord`` / ``write_records``，才能區分
    「成功有資料」與「成功但無資料」。
    """
    write_df(pd.DataFrame([{
        "run_at": run_at, "stock_id": stock_id, "dataset": dataset,
        "ok": 1 if ok else 0, "rows": int(rows), "error": error,
    }]), "ingest_log")
