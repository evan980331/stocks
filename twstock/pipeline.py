"""每日流程：資料抓取 → 特徵 → 打分 →（可選）ML → 訊號入庫 → 通知。

資料完整性原則：
  - 抓取失敗（``None``）與「真的沒資料」（空表）必須分開對待，失敗不得被當成 0 或無訊號。
  - 只有當至少一檔的**價量資料**成功寫入時，才更新 ``last_ingest`` 時間戳。
  - 每次抓取結果寫入 ``ingest_run``（結構化狀態；可區分成功有資料／成功無資料／
    部分成功／失敗／未執行／不可追溯），另保留 ``ingest_log`` 舊介面。
  - 所有寫入使用 ``store.write_df``（INSERT OR REPLACE），重跑不破壞其他標的資料。
  - **不得**再把失敗記成成功：新聞失敗過去一律寫 ``ok=True``，本階段改為依狀態記錄。
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd

from . import features as feat_mod
from .config import CONFIG, REPORTS_DIR
from .data import finmind_client as fm
from .data import loader, store
from .data import status as st
from .data.quality import build_corporate_actions, clean_prices
from .sentiment import sentiment as sent

log = logging.getLogger(__name__)

# 價量資料是後續所有分析的根；沒有它就沒有可信訊號
PRICE_DATASET = "price"

# 寫入重試上限（可注入 sleep 以利測試）
WRITE_ATTEMPTS = 2


def _write_with_retry(df: pd.DataFrame, table: str,
                      key_cols: tuple | None = None, *, attempts: int = WRITE_ATTEMPTS,
                      sleep=None) -> tuple[int, str, str, int, bool]:
    """寫入並在失敗時重試。回傳 (列數, 錯誤分類, 錯誤訊息, 嘗試次數, 是否成功)。

    ``store.write_df`` 在欄位無交集時會擲 ``WriteRejected``；重試也失敗就**不**
    冒充成功，讓呼叫端把該次記為 ``failed/write``。
    """
    sleeper = sleep or time.sleep
    last: BaseException | None = None
    n_try = max(1, attempts)
    for i in range(1, n_try + 1):
        try:
            n = store.write_df(df, table, key_cols=key_cols)
            return n, st.ErrorClass.NONE.value, "", i, True
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("寫入 %s 第 %d/%d 次失敗：%s", table, i, n_try, e)
            if i < n_try:
                sleeper(0.2 * i)
    return (0, st.ErrorClass.WRITE.value,
            f"{type(last).__name__}: {last}", n_try, False)


def _mk_record(run_id: str, run_at: str, cfg: str, *, dataset: str,
               stock_id: str, source: str, task: str, status: str,
               error_class: str = st.ErrorClass.NONE.value, error: str = "",
               rows_fetched: int = 0, rows_parsed: int = 0,
               rows_validated: int = 0, rows_written: int = 0,
               data_start: str = "", data_end: str = "", attempts: int = 0,
               source_version: str = "", notes: str = "",
               started_at: str | None = None,
               ended_at: str | None = None) -> st.IngestRecord:
    """建立一筆擷取紀錄（時間欄位預設為現在，可覆寫）。"""
    return st.IngestRecord(
        run_id=run_id, dataset=dataset, run_at=run_at, source=source,
        task=task, stock_id=stock_id,
        started_at=started_at or st.now_iso(), ended_at=ended_at or st.now_iso(),
        status=status, error_class=error_class, error=error,
        rows_fetched=rows_fetched, rows_parsed=rows_parsed,
        rows_validated=rows_validated, rows_written=rows_written,
        data_start=data_start, data_end=data_end, attempts=attempts,
        source_version=source_version, config_id=cfg, provenance=st.RECORDED,
        notes=notes)


def _date_bounds(df: pd.DataFrame | None) -> tuple[str, str]:
    """資料涵蓋的日期範圍（無資料時為空字串）。"""
    if df is None or df.empty or "date" not in df.columns:
        return "", ""
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    if d.empty:
        return "", ""
    return str(d.min().date()), str(d.max().date())


def _write_price(df: pd.DataFrame | None, sid: str) -> dict:
    """清洗並寫入價量；無效價格列（<=0 或缺失）一律剔除，成為缺口而非 0。

    回傳 dict：``ok``（供舊 ``ingest_log`` 語意）、``rows``、``error``、``status``、
    ``error_class``、``attempts``、``data_start``／``data_end``、``fetched``。
    """
    if df is None:
        return {"ok": False, "rows": 0, "error": "fetch_failed",
                "status": st.IngestStatus.FAILED.value,
                "error_class": st.ErrorClass.NETWORK.value, "attempts": 1,
                "data_start": "", "data_end": "", "fetched": 0}
    if df.empty:
        # 來源正常回應但真的沒資料：成功但無資料，**不是**失敗
        return {"ok": True, "rows": 0, "error": "empty",
                "status": st.IngestStatus.SUCCESS_EMPTY.value,
                "error_class": st.ErrorClass.EMPTY.value, "attempts": 1,
                "data_start": "", "data_end": "", "fetched": 0}
    fetched = len(df)
    d0, d1 = _date_bounds(df)
    out = df.drop(columns=["spread", "turnover"], errors="ignore")
    out, dropped = clean_prices(out)
    if not dropped.empty:
        log.warning("  %s 剔除 %d 列無效價格（%s）", sid, len(dropped),
                    ", ".join(str(pd.Timestamp(d).date())
                              for d in dropped["date"].head(3)))
    n, cls, err, attempts, wrote = _write_with_retry(
        out, "price", key_cols=("stock_id", "date"))
    if not wrote:
        return {"ok": False, "rows": 0, "error": err,
                "status": st.IngestStatus.FAILED.value,
                "error_class": st.ErrorClass.WRITE.value,
                "attempts": attempts, "data_start": d0, "data_end": d1,
                "fetched": fetched}
    if not dropped.empty:
        err = f"dropped_invalid:{len(dropped)}"
    if n > 0:
        status, ec = st.IngestStatus.SUCCESS_WITH_DATA.value, st.ErrorClass.NONE.value
    else:
        # 清洗後無任何有效列 → 有回應但沒有可用資料（不是失敗，也不是成功有資料）
        status = st.IngestStatus.SUCCESS_EMPTY.value
        ec = st.ErrorClass.EMPTY.value
    return {"ok": True, "rows": n, "error": err, "status": status,
            "error_class": ec, "attempts": attempts,
            "data_start": d0, "data_end": d1, "fetched": fetched,
            "dropped": int(len(dropped))}


def _ingest_corporate_actions(sid: str, start: str, end: str,
                              run_at: str) -> dict:
    """抓取分割與股利事件，計算還原調整因子並入庫。

    回傳 ``{"rows": n, "status": ..., "error_class": ..., "error": ...}``。
    兩者皆失敗 → failed；只有一方失敗但另一方成功 → partial_success。
    """
    splits = fm.fetch_splits(sid, start, end)
    divs = fm.fetch_dividends(sid, start, end)
    n_failed = int(splits is None) + int(divs is None)
    if n_failed == 2:
        store.log_ingest(run_at, sid, "corporate_actions", ok=False,
                         error="fetch_failed")
        return {"rows": 0, "status": st.IngestStatus.FAILED.value,
                "error_class": st.ErrorClass.NETWORK.value,
                "error": "splits 與 dividends 均抓取失敗"}
    price = store.read_sql(
        "SELECT date, close FROM price WHERE stock_id=? ORDER BY date", (sid,))
    actions, warns = build_corporate_actions(price, sid, splits, divs)
    for w in warns:
        log.warning("  %s", w)
    # 以本輪結果整批取代該檔的公司行動，避免已被驗證為無效的舊事件殘留
    store.exec_sql("DELETE FROM corporate_actions WHERE stock_id=?", (sid,))
    if not actions:
        store.log_ingest(run_at, sid, "corporate_actions", ok=True, rows=0)
        status = (st.IngestStatus.PARTIAL_SUCCESS.value if n_failed
                  else st.IngestStatus.SUCCESS_EMPTY.value)
        return {"rows": 0, "status": status,
                "error_class": st.ErrorClass.EMPTY.value if not n_failed
                else st.ErrorClass.NETWORK.value,
                "error": "" if not n_failed else "部分事件來源抓取失敗",
                "notes": "; ".join(warns[:3]), "n_failed": n_failed}
    df = pd.DataFrame([{
        "stock_id": a.stock_id, "date": str(a.date), "kind": a.kind,
        "factor": a.factor, "price_scale": int(a.price_scale), "detail": a.detail,
    } for a in actions])
    n, cls, err, attempts, wrote = _write_with_retry(
        df, "corporate_actions", key_cols=("stock_id", "date", "kind"))
    if not wrote:
        store.log_ingest(run_at, sid, "corporate_actions", ok=False,
                         rows=0, error=cls)
        return {"rows": 0, "status": st.IngestStatus.FAILED.value,
                "error_class": st.ErrorClass.WRITE.value, "error": err,
                "attempts": attempts}
    store.log_ingest(run_at, sid, "corporate_actions", ok=True, rows=n)
    log.info("  %s 公司行動 %d 筆（還原因子已更新）", sid, n)
    return {"rows": n,
            "status": (st.IngestStatus.PARTIAL_SUCCESS.value if n_failed
                       else st.IngestStatus.SUCCESS_WITH_DATA.value),
            "error_class": st.ErrorClass.NETWORK.value if n_failed
            else st.ErrorClass.NONE.value,
            "error": "部分事件來源抓取失敗" if n_failed else "",
            "attempts": attempts, "n_failed": n_failed}


def ingest(stocks: list[str] | None = None) -> st.IngestSummary:
    """抓日K + 法人 + 融資券 + 月營收 + 新聞（含基準 0050）。

    回傳 ``IngestSummary``：含每檔價量列數（mapping 相容）與**結構化狀態**。
    呼叫端（例如 Worker）必須依 ``statuses`` / ``overall_status`` 判定成敗，
    不可再只看列數——「來源正常但當日無資料」不是失敗。
    """
    stocks = stocks or list(CONFIG.pool.keys())
    start = fm.start_date(CONFIG.lookback_years)
    end = dt.date.today().isoformat()
    run_at = st.now_iso()
    run_started = st.now_iso()
    run_id = st.new_run_id()
    cfg = st.config_id(CONFIG.pool, CONFIG.lookback_years)
    counts: dict[str, int] = {}
    statuses: dict[str, str] = {}
    records: list[st.IngestRecord] = []
    price_ok = 0
    price_failed: list[str] = []

    def _save(df: pd.DataFrame | None, table: str, sid: str,
              dataset: str) -> int:
        """寫入非價量資料集，並記下結構化狀態。"""
        t0 = st.now_iso()
        if df is None:
            store.log_ingest(run_at, sid, dataset, ok=False,
                             error="fetch_failed")
            records.append(_mk_record(
                run_id, run_at, cfg, dataset=dataset, stock_id=sid,
                source="finmind", task=table, status=st.IngestStatus.FAILED.value,
                error_class=st.ErrorClass.NETWORK.value, error="fetch_failed",
                started_at=t0))
            return 0
        if df.empty:
            store.log_ingest(run_at, sid, dataset, ok=True, rows=0,
                             error="empty")
            records.append(_mk_record(
                run_id, run_at, cfg, dataset=dataset, stock_id=sid,
                source="finmind", task=table,
                status=st.IngestStatus.SUCCESS_EMPTY.value,
                error_class=st.ErrorClass.EMPTY.value, error="empty",
                started_at=t0))
            return 0
        d0, d1 = _date_bounds(df)
        known = store.PRIMARY_KEYS.get(table, ())
        n, cls, err, attempts, wrote = _write_with_retry(df, table,
                                                        key_cols=known)
        if not wrote:
            store.log_ingest(run_at, sid, dataset, ok=False, rows=0, error=cls)
            records.append(_mk_record(
                run_id, run_at, cfg, dataset=dataset, stock_id=sid,
                source="finmind", task=table,
                status=st.IngestStatus.FAILED.value,
                error_class=st.ErrorClass.WRITE.value, error=err,
                rows_fetched=len(df), rows_parsed=len(df),
                data_start=d0, data_end=d1, attempts=attempts,
                started_at=t0))
            return 0
        store.log_ingest(run_at, sid, dataset, ok=True, rows=n)
        records.append(_mk_record(
            run_id, run_at, cfg, dataset=dataset, stock_id=sid,
            source="finmind", task=table,
            status=st.IngestStatus.SUCCESS_WITH_DATA.value,
            rows_fetched=len(df), rows_parsed=len(df), rows_validated=len(df),
            rows_written=n, data_start=d0, data_end=d1, attempts=attempts,
            started_at=t0))
        return n

    # 基準（0050）：ML 超額報酬標籤與回測比較都需要
    bench = fm.fetch_benchmark(start, end)
    if bench is not None and not bench.empty:
        bdf = bench.drop(columns=["spread", "turnover"], errors="ignore")
        b0, b1 = _date_bounds(bdf)
        bn, bcls, berr, battempts, bwrote = _write_with_retry(
            bdf, "price", key_cols=("stock_id", "date"))
        if bwrote:
            store.log_ingest(run_at, CONFIG.benchmark, PRICE_DATASET, ok=True,
                             rows=len(bdf))
            records.append(_mk_record(
                run_id, run_at, cfg, dataset=PRICE_DATASET,
                stock_id=CONFIG.benchmark, source="finmind",
                task="fetch_benchmark",
                status=st.IngestStatus.SUCCESS_WITH_DATA.value,
                rows_fetched=len(bdf), rows_parsed=len(bdf),
                rows_validated=len(bdf), rows_written=bn,
                data_start=b0, data_end=b1, attempts=battempts))
            log.info("基準 %s 價量 %d 筆", CONFIG.benchmark, len(bdf))
        else:
            store.log_ingest(run_at, CONFIG.benchmark, PRICE_DATASET, ok=False,
                             error=bcls)
            records.append(_mk_record(
                run_id, run_at, cfg, dataset=PRICE_DATASET,
                stock_id=CONFIG.benchmark, source="finmind",
                task="fetch_benchmark", status=st.IngestStatus.FAILED.value,
                error_class=st.ErrorClass.WRITE.value, error=berr,
                rows_fetched=len(bdf), attempts=battempts))
            log.warning("基準 %s 寫入失敗：%s", CONFIG.benchmark, berr)
    else:
        cls = (st.ErrorClass.NETWORK.value if bench is None
               else st.ErrorClass.EMPTY.value)
        store.log_ingest(run_at, CONFIG.benchmark, PRICE_DATASET, ok=False,
                         error="fetch_failed" if bench is None else "empty")
        records.append(_mk_record(
            run_id, run_at, cfg, dataset=PRICE_DATASET,
            stock_id=CONFIG.benchmark, source="finmind",
            task="fetch_benchmark",
            status=(st.IngestStatus.FAILED.value if bench is None
                    else st.IngestStatus.SUCCESS_EMPTY.value),
            error_class=cls,
            error="fetch_failed" if bench is None else "empty"))
        log.warning("基準 %s 抓取失敗，ML 標籤與基準比較將不可用", CONFIG.benchmark)

    for sid in stocks:
        log.info("ingest %s (%s)...", sid, CONFIG.pool.get(sid, ""))
        t0 = st.now_iso()
        res = _write_price(fm.fetch_price(sid, start, end), sid)
        n_price = int(res["rows"])
        store.log_ingest(run_at, sid, PRICE_DATASET, ok=bool(res["ok"]),
                         rows=n_price, error=str(res["error"]))
        records.append(_mk_record(
            run_id, run_at, cfg, dataset=PRICE_DATASET, stock_id=sid,
            source="finmind", task="fetch_price", status=str(res["status"]),
            error_class=str(res["error_class"]), error=str(res["error"]),
            rows_fetched=int(res.get("fetched", 0)),
            rows_parsed=int(res.get("fetched", 0)), rows_validated=n_price,
            rows_written=n_price, data_start=str(res.get("data_start", "")),
            data_end=str(res.get("data_end", "")),
            attempts=int(res.get("attempts", 0)), started_at=t0,
            notes=(f"dropped_invalid={res['dropped']}"
                   if res.get("dropped") else "")))
        statuses[sid] = str(res["status"])
        if bool(res["ok"]) and n_price > 0:
            price_ok += 1
        elif not bool(res["ok"]):
            price_failed.append(sid)
            log.warning("  %s 價量抓取失敗（不會被記為成功）", sid)

        n_inst = _save(fm.fetch_institutional(sid, start, end), "institution", sid, "institution")
        n_mg = _save(fm.fetch_margin(sid, start, end), "margin", sid, "margin")
        n_rev = _save(fm.fetch_revenue(sid, "2015-01-01", end), "revenue", sid, "revenue")
        # 公司行動必須在價量寫入後計算（需要除息前收盤價）
        ca = _ingest_corporate_actions(sid, start, end, run_at)
        n_ca = int(ca.get("rows", 0))
        records.append(_mk_record(
            run_id, run_at, cfg, dataset="corporate_actions", stock_id=sid,
            source="finmind", task="build_corporate_actions",
            status=str(ca.get("status")),
            error_class=str(ca.get("error_class", st.ErrorClass.NONE.value)),
            error=str(ca.get("error", "")), rows_written=n_ca,
            rows_parsed=n_ca, attempts=int(ca.get("attempts", 0)),
            notes=str(ca.get("notes", ""))))

        # 新聞：依結構化狀態記錄，**不得**一律 ok=True（過去抓取失敗被記為成功）
        news = sent.update_news_for(sid, CONFIG.pool.get(sid, ""), days=7,
                                    run_id=run_id, run_at=run_at, config_id=cfg)
        if isinstance(news, sent.NewsIngestResult):
            n_news = news.rows_written
            store.log_ingest(run_at, sid, "news",
                             ok=news.status != st.IngestStatus.FAILED.value,
                             rows=n_news,
                             error="" if not news.is_failure else news.error_class)
            if news.record is not None:
                records.append(news.record)
            if news.is_failure:
                log.warning("  %s 新聞抓取失敗（%s）：%s", sid,
                            news.error_class, news.error)
        else:
            # 非結構化回傳（例如測試替身）：狀態不可判定 → unknown，不冒充成功
            n_news = int(news) if isinstance(news, int) else 0
            store.log_ingest(run_at, sid, "news", ok=False, rows=n_news,
                             error="non_structured_result")
            records.append(_mk_record(
                run_id, run_at, cfg, dataset="news", stock_id=sid,
                source="google_news_rss", task="update_news_for",
                status=st.IngestStatus.UNKNOWN.value,
                error_class=st.ErrorClass.UNKNOWN.value,
                error="呼叫端回傳非結構化結果，狀態不可判定",
                rows_written=n_news))
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

    summary = st.IngestSummary(
        run_id=run_id, run_at=run_at, started_at=run_started,
        ended_at=st.now_iso(), counts=counts, statuses=statuses,
        records=records, failed_ids=price_failed)
    try:
        st.write_records(records)
    except Exception as e:  # noqa: BLE001
        log.warning("寫入擷取紀錄失敗：%s", e)
    store.set_meta("last_ingest_run_id", run_id)
    store.set_meta("last_ingest_status", summary.overall_status)

    log.info("ingest 完成：%s；價量成功 %d/%d；失敗 %s",
             summary.label(), price_ok, len(stocks), price_failed or "無")
    return summary


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
    """抓名嘴/專家觀點入庫，回傳筆數。

    來源失敗不得被計入成功：只有確實取得列數才更新 ``last_experts``，
    每個來源的成敗另以 ``ingest_run``（dataset='experts'）記錄。
    """
    from .sentiment.experts import collect_experts
    result = collect_experts()
    if not result.items:
        # 無資料或全失敗：不更新新鮮度，不冒充成功
        status = st.IngestStatus(result.status)
        if status is st.IngestStatus.FAILED:
            log.error("專家來源全部失敗（不會被記為成功）")
        else:
            log.warning("專家來源無資料（狀態 %s）", status.label)
        return 0
    store.write_df(pd.DataFrame(result.items), "experts",
                   key_cols=("source", "link"))
    store.set_meta("last_experts", dt.date.today().isoformat())
    store.set_meta("last_experts_status", result.status)
    log.info("專家觀點入庫 %d 則（狀態 %s；來源 %d 個，失敗 %d 個）",
             len(result.items), st.IngestStatus(result.status).label,
             result.n_sources, result.n_failed_sources)
    return len(result.items)
