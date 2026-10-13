"""每日流程：資料抓取 → 特徵 → 打分 →（可選）ML → 訊號入庫 → 通知。

資料完整性原則：
  - 抓取失敗（``None``）與「真的沒資料」（空表）必須分開對待，失敗不得被當成 0 或無訊號。
  - 只有當至少一檔的**價量資料**成功寫入時，才更新 ``last_ingest`` 時間戳。
  - 每次抓取結果寫入 ``ingest_log`` 供健康檢查與失敗重試。
  - 所有寫入使用 ``store.write_df``（INSERT OR REPLACE），重跑不破壞其他標的資料。
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from . import features as feat_mod
from .config import CONFIG, REPORTS_DIR
from .data import finmind_client as fm
from .data import loader, store
from .data.quality import build_corporate_actions, clean_prices
from .sentiment import sentiment as sent

log = logging.getLogger(__name__)

# 價量資料是後續所有分析的根；沒有它就沒有可信訊號
PRICE_DATASET = "price"


def _write_price(df: pd.DataFrame | None, sid: str) -> tuple[bool, int, str]:
    """寫入價量；無效價格列（<=0 或缺失）一律剔除，成為缺口而非 0。"""
    if df is None:
        return False, 0, "fetch_failed"
    if df.empty:
        return True, 0, "empty"
    out = df.drop(columns=["spread", "turnover"], errors="ignore")
    out, dropped = clean_prices(out)
    if not dropped.empty:
        log.warning("  %s 剔除 %d 列無效價格（%s）", sid, len(dropped),
                    ", ".join(str(pd.Timestamp(d).date())
                              for d in dropped["date"].head(3)))
    n = store.write_df(out, "price", key_cols=("stock_id", "date"))
    err = f"dropped_invalid:{len(dropped)}" if not dropped.empty else ""
    return True, n, err


def _ingest_corporate_actions(sid: str, start: str, end: str,
                              run_at: str) -> int:
    """抓取分割與股利事件，計算還原調整因子並入庫。"""
    splits = fm.fetch_splits(sid, start, end)
    divs = fm.fetch_dividends(sid, start, end)
    if splits is None and divs is None:
        store.log_ingest(run_at, sid, "corporate_actions", ok=False,
                         error="fetch_failed")
        return 0
    price = store.read_sql(
        "SELECT date, close FROM price WHERE stock_id=? ORDER BY date", (sid,))
    actions, warns = build_corporate_actions(price, sid, splits, divs)
    for w in warns:
        log.warning("  %s", w)
    # 以本輪結果整批取代該檔的公司行動，避免已被驗證為無效的舊事件殘留
    store.exec_sql("DELETE FROM corporate_actions WHERE stock_id=?", (sid,))
    if not actions:
        store.log_ingest(run_at, sid, "corporate_actions", ok=True, rows=0)
        return 0
    df = pd.DataFrame([{
        "stock_id": a.stock_id, "date": str(a.date), "kind": a.kind,
        "factor": a.factor, "price_scale": int(a.price_scale), "detail": a.detail,
    } for a in actions])
    n = store.write_df(df, "corporate_actions",
                       key_cols=("stock_id", "date", "kind"))
    store.log_ingest(run_at, sid, "corporate_actions", ok=True, rows=n)
    log.info("  %s 公司行動 %d 筆（還原因子已更新）", sid, n)
    return n


def ingest(stocks: list[str] | None = None) -> dict[str, int]:
    """抓日K + 法人 + 融資券 + 月營收 + 新聞（含基準 0050）。"""
    stocks = stocks or list(CONFIG.pool.keys())
    start = fm.start_date(CONFIG.lookback_years)
    end = dt.date.today().isoformat()
    run_at = dt.datetime.now().isoformat(timespec="seconds")
    counts: dict[str, int] = {}
    price_ok = 0
    price_failed: list[str] = []

    def _save(df: pd.DataFrame | None, table: str, sid: str, dataset: str) -> int:
        if df is None:
            store.log_ingest(run_at, sid, dataset, ok=False, error="fetch_failed")
            return 0
        if df.empty:
            store.log_ingest(run_at, sid, dataset, ok=True, rows=0, error="empty")
            return 0
        known = store.PRIMARY_KEYS.get(table, ())
        n = store.write_df(df, table, key_cols=known)
        store.log_ingest(run_at, sid, dataset, ok=True, rows=n)
        return n

    # 基準（0050）：ML 超額報酬標籤與回測比較都需要
    bench = fm.fetch_benchmark(start, end)
    if bench is not None and not bench.empty:
        bdf = bench.drop(columns=["spread", "turnover"], errors="ignore")
        store.write_df(bdf, "price", key_cols=("stock_id", "date"))
        store.log_ingest(run_at, CONFIG.benchmark, PRICE_DATASET, ok=True, rows=len(bdf))
        log.info("基準 %s 價量 %d 筆", CONFIG.benchmark, len(bdf))
    else:
        store.log_ingest(run_at, CONFIG.benchmark, PRICE_DATASET, ok=False,
                         error="fetch_failed" if bench is None else "empty")
        log.warning("基準 %s 抓取失敗，ML 標籤與基準比較將不可用", CONFIG.benchmark)

    for sid in stocks:
        log.info("ingest %s (%s)...", sid, CONFIG.pool.get(sid, ""))
        ok, n_price, err = _write_price(fm.fetch_price(sid, start, end), sid)
        store.log_ingest(run_at, sid, PRICE_DATASET, ok=ok, rows=n_price, error=err)
        if ok and n_price > 0:
            price_ok += 1
        elif not ok:
            price_failed.append(sid)
            log.warning("  %s 價量抓取失敗（不會被記為成功）", sid)

        n_inst = _save(fm.fetch_institutional(sid, start, end), "institution", sid, "institution")
        n_mg = _save(fm.fetch_margin(sid, start, end), "margin", sid, "margin")
        n_rev = _save(fm.fetch_revenue(sid, "2015-01-01", end), "revenue", sid, "revenue")
        # 公司行動必須在價量寫入後計算（需要除息前收盤價）
        n_ca = _ingest_corporate_actions(sid, start, end, run_at)
        n_news = sent.update_news_for(sid, CONFIG.pool.get(sid, ""), days=7)
        store.log_ingest(run_at, sid, "news", ok=True, rows=n_news)
        log.info("  price=%s inst=%s margin=%s revenue=%s corp=%s news=%s",
                 n_price, n_inst, n_mg, n_rev, n_ca, n_news)
        counts[sid] = n_price
        fm.polite_sleep(0.8)

    # 只有真的有價量成功才更新新鮮度；失敗要留下可追蹤紀錄
    if price_ok > 0:
        store.set_meta("last_ingest", end)
        store.set_meta("last_ingest_ok_count", str(price_ok))
        store.set_meta("last_ingest_failed_count", str(len(price_failed)))
        store.set_meta("last_ingest_failed_ids", ",".join(price_failed))
        store.set_meta("last_ingest_run_at", run_at)
    else:
        log.error("本輪 ingest 沒有任何價量資料成功；不更新 last_ingest（避免誤判資料新鮮）")
        store.set_meta("last_ingest_failed_ids", ",".join(price_failed))
        store.set_meta("last_ingest_run_at", run_at)

    log.info("ingest 完成：價量成功 %d/%d；失敗 %s", price_ok, len(stocks), price_failed or "無")
    return counts


def load_benchmark(adjusted: bool = True) -> pd.DataFrame:
    """讀取基準（0050）日K；不存在時回空表。

    預設回傳**已還原**序列——未還原的 0050 在 1 拆 4 當日會顯示 -75% 的假崩盤，
    使基準報酬嚴重錯誤（實測曾因此產生 +67% 的假超額報酬）。
    """
    price = loader.load_price_panel(adjusted=adjusted,
                                   columns="date, stock_id, close")
    return loader.benchmark_series(price, CONFIG.benchmark)


def load_panel(stats: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """組合價量 + 籌碼 + 新聞日分數 → features 面板。回傳 (panel, news)。

    價格使用**已還原**序列，避免除權息缺口被當成真實下跌。
    """
    price = loader.load_price_panel(adjusted=True, stats=stats)
    inst = store.read_sql("SELECT * FROM institution")
    margin = store.read_sql("SELECT * FROM margin")
    rev = store.read_sql("SELECT * FROM revenue")
    news = store.read_sql("SELECT published as date, stock_id, score FROM news")

    feature_rows: list[pd.DataFrame] = []
    for sid in CONFIG.pool:
        p = price[price["stock_id"] == sid]
        if p.empty:
            log.warning("load_panel: %s 無價量資料，略過", sid)
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
    """執行特徵 + 規則打分；ML 可用時以排名百分位融合（尺度一致）。"""
    panel, _news_df = load_panel()
    if panel.empty:
        log.warning("無資料可計算，請先 ingest")
        return pd.DataFrame()

    from . import signals as sig_mod
    features_by_stock = {sid: g for sid, g in panel.groupby("stock_id")}
    scores = sig_mod.score_stocks(features_by_stock)
    if scores.empty:
        return scores

    # ML（可選）：缺基準或缺 lightgbm 時明確停用，不以 0 冒充
    from . import ml_model as ml
    ml_scores = pd.DataFrame(columns=["date", "stock_id", "ml_score"])
    model, meta = ml.train_walk_forward(panel, load_benchmark())
    if model is not None:
        ml_scores = ml.predict_scores(model, panel, load_benchmark())
        scores = sig_mod.fuse_scores(scores, ml_scores)
    else:
        log.info("ML 未啟用（%s），僅使用規則分數", meta.get("reason", "unknown"))
        scores = sig_mod.fuse_scores(scores, None)
    store.set_meta("ml_enabled", "1" if model is not None else "0")
    store.set_meta("ml_status", str(meta.get("reason", "")))

    if not scores.empty:
        keep = [c for c in ("date", "stock_id", "score", "reason", "close",
                            "ml_score") if c in scores.columns]
        store.write_df(scores[keep], "signals", key_cols=("date", "stock_id"))
    return scores


def daily_notify(top_n: int | None = None) -> bool:
    from . import signals as sig_mod
    from .notify import format_daily_signals, send_telegram
    latest = store.read_sql("SELECT MAX(date) AS d FROM signals")
    if latest.empty or latest.iloc[0]["d"] is None:
        log.warning("尚無任何訊號")
        return False
    latest_date = str(latest.iloc[0]["d"])
    all_scores = store.read_sql("SELECT * FROM signals WHERE date = ?", (latest_date,))
    if all_scores.empty:
        log.warning("最新日期 %s 無訊號", latest_date)
        return False
    top = sig_mod.pick_top_n(all_scores, latest_date, top_n or CONFIG.top_n)
    if top.empty:
        log.info("無 top 訊號")
        return False
    text = format_daily_signals(latest_date, top)
    ok = send_telegram(text)
    log.info("訊號輸出：\n%s", text)
    return ok


def run_backtest_cli() -> str:
    scores = store.read_sql("SELECT * FROM signals")
    if scores.empty:
        return "無訊號可回測，請先 run"
    price = loader.load_price_panel(adjusted=True)
    panel = {sid: g for sid, g in price.groupby("stock_id")}
    benchmark = panel.get(CONFIG.benchmark)

    from .backtest import run_backtest
    result = run_backtest(scores, panel, top_n=CONFIG.top_n,
                          hold_days=CONFIG.hold_days, benchmark=benchmark)
    report = result.summary()

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"backtest_{dt.datetime.now():%Y%m%d_%H%M%S}.txt"
    header = (
        f"回測區間：{result.equity.index[0] if len(result.equity) else '-'}"
        f" ~ {result.equity.index[-1] if len(result.equity) else '-'}\n"
        f"假設：T+1 開盤進場 / 到期日開盤出場 / 手續費 {CONFIG.commission_effective:.4%}"
        f"（低消 {CONFIG.buy_fixed_fee:.0f}）/ 證交稅 {CONFIG.tax_rate:.2%} / 滑價 {CONFIG.slippage:.2%}\n"
        f"單檔部位上限 {CONFIG.max_position_pct:.0%} / 漲跌停 {CONFIG.limit_pct:.0%}\n"
        f"價格：已套用除權息／分割還原；策略與基準採一致處理\n\n")
    out.write_text(header + report + "\n\n" + str(result.trades.head(80)), encoding="utf-8")
    return report + f"\n（詳細報告：{out}）"


def ingest_experts() -> int:
    """抓名嘴/專家觀點入庫，回傳筆數。"""
    from .sentiment.experts import collect_experts
    items = collect_experts()
    if not items:
        log.warning("專家來源無資料")
        return 0
    store.write_df(pd.DataFrame(items), "experts", key_cols=("source", "link"))
    store.set_meta("last_experts", dt.date.today().isoformat())
    log.info("專家觀點入庫 %d 則", len(items))
    return len(items)
