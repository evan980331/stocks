"""每日流程：資料抓取 → 特徵 → 打分 → ML（可選）→ 訊號入庫 → 通知。"""
from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import pandas as pd

from . import features as feat_mod
from .config import CONFIG, REPORTS_DIR
from .data import finmind_client as fm
from .data import store
from .sentiment import news_rss, sentiment as sent

log = logging.getLogger(__name__)


def ingest(stocks: list[str] | None = None) -> dict[str, int]:
    """抓日K + 法人 + 融資券 + 月營收 + 新聞。"""
    stocks = stocks or list(CONFIG.pool.keys())
    start = fm.start_date(CONFIG.lookback_years)
    end = dt.date.today().isoformat()
    counts: dict[str, int] = {}

    def _save(df: pd.DataFrame, table: str, sid: str) -> int:
        """Tolerant save：只寫入 table 已有的欄位。"""
        if df is None or df.empty:
            return 0
        conn = store._connect()
        store._init(conn)
        try:
            known_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            cols = [c for c in df.columns if c in known_cols]
            if not cols:
                log.warning("  %s 無交集欄位，略過", table)
                return 0
            store.exec_sql(f"DELETE FROM {table} WHERE stock_id=?", (sid,))
            df[cols].to_sql(table, conn, if_exists="append", index=False,
                            method="multi", chunksize=500)
            conn.commit()
            return len(df)
        finally:
            conn.close()

    for sid in stocks:
        log.info("ingest %s (%s)...", sid, CONFIG.pool.get(sid, ""))
        price = fm.fetch_price(sid, start, end)
        n_price = _save(price.drop(columns=["spread", "turnover"], errors="ignore"), "price", sid)
        inst = fm.fetch_institutional(sid, start, end)
        n_inst = _save(inst, "institution", sid)
        margin = fm.fetch_margin(sid, start, end)
        n_mg = _save(margin, "margin", sid)
        rev = fm.fetch_revenue(sid, "2015-01-01", end)
        n_rev = _save(rev, "revenue", sid)
        log.info("  price=%s inst=%s margin=%s revenue=%s", n_price, n_inst, n_mg, n_rev)

        n_news = sent.update_news_for(sid, CONFIG.pool.get(sid, ""), days=7)
        counts[sid] = n_price
        fm.polite_sleep(0.8)

    store.set_meta("last_ingest", end)
    log.info("ingest 完成：%s", counts)
    return counts


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """組合價量 + 籌碼 + 新聞日分數 → features 面板。回傳 (panel, news)。"""
    price = store.read_sql("SELECT * FROM price")
    inst = store.read_sql("SELECT * FROM institution")
    margin = store.read_sql("SELECT * FROM margin")
    rev = store.read_sql("SELECT * FROM revenue")
    news = store.read_sql(
        "SELECT published as date, stock_id, score FROM news")

    feature_rows: list[pd.DataFrame] = []
    for sid in CONFIG.pool:
        p = price[price["stock_id"] == sid]
        if p.empty:
            continue
        i = inst[inst["stock_id"] == sid] if not inst.empty else inst
        m = margin[margin["stock_id"] == sid] if not margin.empty else margin
        r = rev[rev["stock_id"] == sid] if not rev.empty else rev
        n = news[news["stock_id"] == sid] if not news.empty else news
        f = feat_mod.build_features(
            p, inst=i, margin=m, revenue=r, news_daily=n,
            momentum_windows=CONFIG.momentum_windows,
            rsi_window=CONFIG.rsi_window,
            vol_ma_window=CONFIG.volume_ma_window,
        )
        f["stock_id"] = sid
        feature_rows.append(f)

    panel = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
    return panel, news


def run_scores(stocks: list[str] | None = None) -> pd.DataFrame:
    """執行特徵 + 規則打分；若 ML 可用則合併 ml_score。"""
    panel, _news_df = load_panel()
    if panel.empty:
        log.warning("無資料可計算，請先 ingest")
        return pd.DataFrame()

    from . import signals as sig_mod
    features_by_stock = {sid: g for sid, g in panel.groupby("stock_id")}
    scores = sig_mod.score_stocks(features_by_stock)

    # ML（可選）
    from . import ml_model as ml
    model, meta = ml.train_walk_forward(panel)
    if model is not None:
        ml_scores = ml.predict_scores(model, panel)
        scores = scores.merge(ml_scores, on=["date", "stock_id"], how="left")
        scores["score"] = scores["score"] * 0.6 + scores["ml_score"].fillna(0) * 0.4

    # 入庫（replace 最近 60 日）
    if not scores.empty:
        conn = store._connect()
        store._init(conn)
        recent = scores[scores["date"] >= (dt.date.today() - dt.timedelta(days=60)).isoformat()]
        for sid in recent["stock_id"].unique():
            conn.execute("DELETE FROM signals WHERE stock_id=?", (sid,))
        recent.to_sql("signals", conn, if_exists="append", index=False,
                      method="multi", chunksize=500)
        conn.commit()
        conn.close()
    return scores


def daily_notify(top_n: int | None = None) -> bool:
    from . import signals as sig_mod
    from .notify import format_daily_signals, send_telegram
    today = dt.date.today().isoformat()
    all_scores = store.read_sql(
        "SELECT * FROM signals WHERE date = (SELECT MAX(date) FROM signals)")
    if all_scores.empty:
        log.warning("今日尚無訊號")
        return False
    top = sig_mod.pick_top_n(all_scores, today, top_n or CONFIG.top_n)
    if top.empty:
        log.info("今日無 top 訊號")
        return False
    text = format_daily_signals(today, top)
    ok = send_telegram(text)
    log.info("訊號輸出：\n%s", text)
    return ok


def run_backtest_cli() -> str:
    scores = store.read_sql("SELECT * FROM signals")
    if scores.empty:
        return "無訊號可回測，請先 run"
    price = store.read_sql("SELECT * FROM price")
    panel = {sid: g for sid, g in price.groupby("stock_id")
             if sid in CONFIG.pool or sid == CONFIG.benchmark}

    from .backtest import run_backtest
    result = run_backtest(scores, panel, top_n=CONFIG.top_n,
                          hold_days=CONFIG.hold_days)
    report = result.summary()

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"backtest_{dt.datetime.now():%Y%m%d_%H%M%S}.txt"
    out.write_text(report + "\n\n" + str(result.trades.head(50)), encoding="utf-8")
    return report + f"\n（詳細報告：{out}）"

def ingest_experts() -> int:
    """Phase 2：抓名嘴/專家觀點入庫，回傳筆數。"""
    from .sentiment.experts import collect_experts
    items = collect_experts()
    if not items:
        log.warning("專家來源無資料")
        return 0
    conn = store._connect()
    store._init(conn)
    try:
        df = pd.DataFrame(items)
        # replace 同 source 的舊資料，避免累積重複
        for src in df["source"].unique():
            conn.execute("DELETE FROM experts WHERE source=?", (src,))
        df.to_sql("experts", conn, if_exists="append", index=False,
                  method="multi", chunksize=200)
        conn.commit()
    finally:
        conn.close()
    store.set_meta("last_experts", dt.date.today().isoformat())
    log.info("專家觀點入庫 %d 則", len(items))
    return len(items)
