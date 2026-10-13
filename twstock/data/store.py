"""SQLite 儲存層：price、institution、margin、revenue、news、signals、experts、ingest_log。

寫入策略（重要）：
  所有寫入一律使用 ``INSERT OR REPLACE``（依各表 PRIMARY KEY 去重）。
  禁止使用 ``pandas.to_sql(if_exists="replace")``——它會 DROP 整張表，造成其他標的資料遺失。

唯讀保護：
  ``TWSTOCK_READONLY=1`` 時完全不寫入，僅以 mode=ro 開啟，用於安全分析既有資料。
  （此機制與任何雲端平台無關，純粹是本機保護。）
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
CREATE TABLE IF NOT EXISTS ingest_log (
    run_at TEXT, stock_id TEXT, dataset TEXT,
    ok INTEGER, rows INTEGER, error TEXT,
    PRIMARY KEY (run_at, stock_id, dataset)
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
    "ingest_log": ("run_at", "stock_id", "dataset"),
    "meta": ("key",),
}


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
    """輕量向前相容遷移：補上舊 DB 缺少的欄位。"""
    wanted = {
        "signals": [("ml_score", "REAL")],
        "price": [("amount", "REAL"), ("turnover", "REAL")],
    }
    for table, cols in wanted.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if not existing:
            continue
        for name, decl in cols:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                log.info("migrate: %s 新增欄位 %s", table, name)


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


def write_df(df: pd.DataFrame, table: str,
             key_cols: Sequence[str] | None = None) -> int:
    """以 INSERT OR REPLACE 寫入 DataFrame。

    - 僅寫入該表已存在的欄位（容忍來源欄位變動）。
    - 依主鍵去重，重跑同一任務不會產生重複、也不會刪掉其他標的的資料。
    - 回傳實際寫入列數。
    """
    if df is None or df.empty:
        return 0
    if is_readonly():
        raise RuntimeError("唯讀模式（TWSTOCK_READONLY）不允許寫入")
    conn = _connect()
    try:
        _init(conn)
        known = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        cols = [c for c in df.columns if c in known]
        if not cols:
            log.warning("write_df: %s 無交集欄位，略過", table)
            return 0
        keys = tuple(key_cols or PRIMARY_KEYS.get(table, ()))
        on_conflict = ""
        if keys and all(k in known for k in keys):
            on_conflict = " OR REPLACE"
        else:
            log.warning("write_df: %s 主鍵 %s 不完整，改用 INSERT", table, keys)
        placeholders = ",".join("?" * len(cols))
        sql = (f"INSERT{on_conflict} INTO {table} ({','.join(cols)}) "
               f"VALUES ({placeholders})")
        rows = [tuple(_adapt(v) for v in rec)
                for rec in df[cols].itertuples(index=False, name=None)]
        conn.executemany(sql, rows)
        conn.commit()
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
    """記錄單一 (run, stock, dataset) 的抓取結果，供健康檢查與失敗重試。"""
    write_df(pd.DataFrame([{
        "run_at": run_at, "stock_id": stock_id, "dataset": dataset,
        "ok": 1 if ok else 0, "rows": int(rows), "error": error,
    }]), "ingest_log")
