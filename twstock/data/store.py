"""SQLite 儲存層：price、institution、margin、revenue、news。"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR, DB_PATH, is_readonly

log = logging.getLogger(__name__)


def _ensure_dir() -> None:
    if is_readonly():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    """開啟 DB。唯讀模式（Vercel）用 file: URI + mode=ro，且不建表。"""
    if is_readonly():
        uri = Path(DB_PATH).resolve().as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True)
    _ensure_dir()
    return sqlite3.connect(DB_PATH)


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
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
            close REAL, rank INTEGER, action TEXT,
            PRIMARY KEY (date, stock_id)
        );
        CREATE TABLE IF NOT EXISTS experts (
            published DATE, source TEXT, title TEXT, link TEXT,
            score REAL, kind TEXT,
            PRIMARY KEY (source, link)
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY, value TEXT
        );
        """
    )
    conn.commit()


def upsert(df: pd.DataFrame, table: str) -> int:
    """寫入 DataFrame（replace into）。"""
    if df is None or df.empty:
        return 0
    conn = _connect()
    try:
        _init(conn)
        df.to_sql(table, conn, if_exists="replace", index=False,
                  method="multi", chunksize=500)
        return len(df)
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
        raise RuntimeError("唯讀模式（Vercel）不允許寫入，請在本機執行 ingest/run")
    conn = _connect()
    try:
        _init(conn)
        conn.execute(sql, params or ())
        conn.commit()
    finally:
        conn.close()


def get_meta(key: str) -> str | None:
    df = read_sql("SELECT value FROM meta WHERE key=?", (key,))
    return df.iloc[0]["value"] if not df.empty else None


def set_meta(key: str, value: str) -> None:
    exec_sql("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))
