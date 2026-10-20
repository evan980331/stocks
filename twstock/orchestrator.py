"""P5：本地自動化執行協調器（Orchestrator）——行情更新 → 資料品質檢查 →
訊號產生 → 模擬交易 → 帳務核對 → 健康狀態與異常事件。

設計邊界（本階段是**工程可靠性**，不是策略最佳化）
  - Windows 單機 + SQLite + 既有模組；不引入任何雲端、佇列或分散式元件。
  - 每一階段都**真正呼叫既有元件**（``pipeline.ingest`` / ``run_scores`` /
    ``worker.run_quote_cycle`` / ``paper.PaperEngine`` / ``paper.reconcile`` /
    ``ops``），不是另寫一套平行邏輯。
  - **純模擬交易**：本模組不下真單、不連券商；paper 階段受 P5-B 閘門約束。

輪次與階段（``run_round``）
  - 每輪唯一識別碼 ``run_id``（``p5-YYYYMMDDTHHMMSS-xxxx``）寫入 ``run_round``；
    每階段寫一列 ``run_stage``（開始/結束/耗時/狀態/筆數/錯誤/嘗試次數/分類）。
    追查鏈：``run_id`` → ``run_stage``（含 ingest run_id、signal_date）→
    ``paper_order.reason``（含 run_id）→ ``paper_fill`` → ``paper_ledger``。
  - 階段狀態沿用既有 ``IngestStatus`` 六態；輪次狀態為
    ``success`` / ``partial_success`` / ``failed`` / ``skipped``（＋執行中 ``running``）。
  - **冪等／續跑**：已完成的階段在重跑時**不重新執行**（狀態保留、不重複下單）；
    崩潰留下的 ``running`` 階段會重新執行；``ended_at IS NULL`` 的輪次自動續跑。
    訂單層再由 ``client_order_id``／``fill_id``／帳本 ``ref`` 三層唯一鍵防重播。
  - **單一來源失敗不得誤報整輪成功**：任一階段 ``failed`` → 輪次最高
    ``partial_success``；``failed`` 且無任何成功階段 → ``failed``。
  - **fail-closed 上游前置**：ingest / quotes / quality / signals 任一 ``failed``
    → paper 階段直接 ``skipped(kind=blocked)``，不產生新模擬交易。

保護（P5-C）
  - 單一實例鎖 ``state/p5run.lock``（PID 已死自動回收）；重試有限次＋指數退避
    （``worker.run_task_with_retry``）；輪次逾時預算用盡 → 後續階段標記
    ``skipped``（原因 round_timeout），不讓單輪永久卡住。
  - 資料庫交易原子性由 paper 引擎保證（BEGIN IMMEDIATE，失敗整筆回滾）；
    協調器只記錄結果，**絕不修改歷史帳務**。
  - Kill Switch：paper 階段被阻擋並記錄觸發原因；**不得自動解除**。

事件與通知（不假裝已送達）
  - 異常寫入 ``run_alert``（含層級與輪次/階段），並嘗試既有 Telegram 通道；
    **只有設定 ``TWSTOCK_ALERT_TELEGRAM=1`` 才外送**，``delivered`` 欄位
    如實記錄 ``recorded`` / ``telegram_sent`` / ``telegram_failed`` /
    ``not_configured``。

唯讀監控（P5-D）
  - ``audit_report``：帳務一致性（帳本/持倉/成交/訂單/快照）、未完成與重複
    訂單、worker 結果與最後成功時間、行情閘門是否允許新交易。**預設唯讀**。
  - ``health_report``：不只看程序存活——同時看資料新鮮度、錯誤率、帳務一致、
    Kill Switch 與心跳，可辨識「程序活著但資料已過期」。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import gate as gate_mod
from . import risk as risk_mod
from .config import is_readonly
from .data import quality as quality_mod
from .data import status as st
from .data import store
from .data.status import IngestStatus

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 階段與狀態常數
# ---------------------------------------------------------------------------
STAGES = ("ingest", "quotes", "quality", "signals", "paper", "reconcile",
          "health")
STAGE_LABEL = {
    "ingest": "行情更新（日K/量價/法人/籌碼/新聞）",
    "quotes": "盤中行情輪詢",
    "quality": "資料品質檢查",
    "signals": "訊號產生",
    "paper": "模擬交易",
    "reconcile": "帳務核對",
    "health": "健康狀態與事件",
}

#: paper 階段的上游前置：任一 ``failed`` → 停止新模擬交易（fail-closed）
UPSTREAM_FOR_PAPER = ("ingest", "quotes", "quality", "signals")

ROUND_RUNNING = "running"
ROUND_SUCCESS = "success"
ROUND_PARTIAL = "partial_success"
ROUND_FAILED = "failed"
ROUND_SKIPPED = "skipped"

KIND_POLICY = "policy"
KIND_BLOCKED = "blocked"

OK_STATUSES = (IngestStatus.SUCCESS_WITH_DATA.value,
               IngestStatus.SUCCESS_EMPTY.value)
TERMINAL_STATUSES = OK_STATUSES + (IngestStatus.PARTIAL_SUCCESS.value,
                                   IngestStatus.FAILED.value,
                                   IngestStatus.SKIPPED.value)

#: 事件外送開關（預設關閉 = 只寫事件表與 CLI，不假裝已送達外部）
ALERT_TELEGRAM_ENV = "TWSTOCK_ALERT_TELEGRAM"
TIMEOUT_ENV = "TWSTOCK_P5_TIMEOUT_S"
ATTEMPTS_ENV = "TWSTOCK_P5_ATTEMPTS"
DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_ATTEMPTS = 3

#: 程序心跳被視為「可能卡死」的門檻（秒）
HEARTBEAT_STALE_S = 3600.0

EXIT_CODES = {ROUND_SUCCESS: 0, ROUND_PARTIAL: 1, ROUND_FAILED: 2,
              ROUND_SKIPPED: 3}


# ---------------------------------------------------------------------------
# 資料結構
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    """單一階段的執行結果（狀態沿用 ``IngestStatus`` 六態）。"""

    status: str
    rows: int = 0
    error: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    kind: str = ""                      # skipped 時的分類：policy / blocked

    def label(self) -> str:
        base = (IngestStatus(self.status).label
                if self.status in IngestStatus._value2member_map_
                else self.status)
        extra = f"（{self.kind}）" if self.kind else ""
        return f"{base}{extra}｜筆數 {self.rows}" + \
               (f"｜{self.error}" if self.error else "")


@dataclass
class RoundResult:
    """一輪自動化工作的結果。"""

    run_id: str
    status: str
    trigger: str = "manual"
    started_at: str = ""
    ended_at: str = ""
    stages: dict[str, dict] = field(default_factory=dict)
    error: str = ""
    resumed: bool = False
    busy: bool = False

    @property
    def ok(self) -> bool:
        return self.status == ROUND_SUCCESS

    def label(self) -> str:
        parts = [f"輪次 {self.run_id or '—'}：{self.status}"]
        if self.resumed:
            parts.append("（續跑：已完成階段不重複執行）")
        for name, row in self.stages.items():
            parts.append(f"  - {STAGE_LABEL.get(name, name)}："
                         f"{row.get('status', '?')}"
                         + (f"（{row.get('error')}）" if row.get("error") else ""))
        if self.error:
            parts.append(f"  錯誤：{self.error}")
        return "\n".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "status": self.status,
                "trigger": self.trigger, "started_at": self.started_at,
                "ended_at": self.ended_at, "stages": self.stages,
                "error": self.error, "resumed": self.resumed,
                "busy": self.busy}


@dataclass
class RoundCtx:
    """傳給各階段函式的上下文（測試可注入 ``fns`` 覆寫任何階段）。"""

    run_id: str
    now: dt.datetime
    trigger: str
    fns: dict[str, Callable[["RoundCtx"], StageResult]] = field(
        default_factory=dict)


# ---------------------------------------------------------------------------
# 輪次／階段紀錄（run_round / run_stage）
# ---------------------------------------------------------------------------
def _now_iso(now: dt.datetime | None = None) -> str:
    return (now or dt.datetime.now()).isoformat(timespec="seconds")


def new_run_id(now: dt.datetime | None = None) -> str:
    """每輪唯一且可排序的識別碼（前綴 p5 以便與 ingest/quote 區分）。"""
    now = now or dt.datetime.now()
    import uuid
    return "p5-" + now.strftime("%Y%m%dT%H%M%S") + "-" + \
        uuid.uuid4().hex[:4]


def open_round_id() -> str | None:
    """尚未結束的輪次（崩潰/中斷後自動續跑的入口）。"""
    try:
        df = store.read_sql(
            "SELECT run_id FROM run_round WHERE ended_at IS NULL "
            "ORDER BY started_at DESC LIMIT 1")
    except Exception:  # noqa: BLE001
        return None
    return None if df.empty else str(df.iloc[0]["run_id"])


def round_due(now: dt.datetime) -> tuple[bool, str]:
    """排程輪是否到期（輪次層的冪等：同一天最多一輪成功/部分成功）。"""
    if now.weekday() >= 5:
        return False, "weekend（非交易日不自動執行）"
    if now.time() < dt.time(_eod_hour(), 0):
        return False, f"before_eod（{_eod_hour():02d}:00 後才自動執行）"
    try:
        done = store.read_sql(
            "SELECT run_id FROM run_round WHERE substr(started_at,1,10)=? "
            "AND status IN (?,?) LIMIT 1",
            (now.date().isoformat(), ROUND_SUCCESS, ROUND_PARTIAL))
    except Exception as e:  # noqa: BLE001
        return False, f"無法確認今日輪次（fail-closed）：{type(e).__name__}: {e}"
    if not done.empty:
        return False, f"already_done_today（{done.iloc[0]['run_id']}）"
    return True, "due"


def _eod_hour() -> int:
    """自動輪次最早執行的小時（與 worker 的 EOD 時間一致）。"""
    from .worker import EOD_HOUR
    return int(EOD_HOUR)


def _insert_round(run_id: str, trigger: str, now: dt.datetime, *,
                  status: str = ROUND_RUNNING, error: str = "",
                  notes: str = "", ended: bool = False) -> None:
    store.exec_sql(
        "INSERT OR IGNORE INTO run_round (run_id, trigger, started_at,"
        " ended_at, status, error, notes) VALUES (?,?,?,?,?,?,?)",
        (run_id, trigger, _now_iso(now),
         _now_iso(now) if ended else None, status, st.redact(error),
         notes[:400]))


def _finish_round(run_id: str, *, status: str, now: dt.datetime,
                  error: str = "", notes: str = "") -> None:
    store.exec_sql(
        "UPDATE run_round SET status=?, ended_at=?, error=?, notes=? "
        "WHERE run_id=?",
        (status, _now_iso(now), st.redact(error), notes[:400], run_id))


def _norm_val(v: Any) -> Any:
    """numpy 標量 → Python 原生型別（JSON 安全；API/CLI 共用）。"""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:  # noqa: BLE001
            return v
    return v


def p5_schema_ready() -> bool:
    """P5 資料表是否存在。

    舊 DB（或唯讀開啟的正式庫）可能尚未建立 run_* 三表：可寫模式會由
    ``store._init`` 自動補表（`CREATE TABLE IF NOT EXISTS`，不碰既有資料）；
    唯讀模式**不會**建立，此時監控查詢必須回空結果＋明確提示，不得崩潰。
    """
    try:
        return bool(store.table_exists("run_round")
                    and store.table_exists("run_stage"))
    except Exception:  # noqa: BLE001
        return False


def _norm_row(d: dict) -> dict:
    return {k: _norm_val(v) for k, v in d.items()}


def _stage_row(run_id: str, stage: str) -> dict | None:
    try:
        df = store.read_sql(
            "SELECT * FROM run_stage WHERE run_id=? AND stage=?",
            (run_id, stage))
    except Exception:  # noqa: BLE001
        return None
    return None if df.empty else _norm_row(df.iloc[0].to_dict())


def _begin_stage(run_id: str, stage: str, now: dt.datetime,
                 attempts: int) -> None:
    """寫入「執行中」列；崩潰時可由此辨識未完成階段（續跑會重做）。"""
    store.exec_sql(
        "INSERT OR REPLACE INTO run_stage (run_id, stage, started_at,"
        " ended_at, duration_s, status, rows, error, attempts, kind, detail)"
        " VALUES (?,?,?,NULL,NULL,?,0,?,?,?,?)",
        (run_id, stage, _now_iso(now), ROUND_RUNNING,
         st.redact(""), int(attempts), "", ""))


def _end_stage(run_id: str, stage: str, now: dt.datetime,
               result: StageResult, *, duration_s: float,
               attempts: int) -> None:
    store.exec_sql(
        "UPDATE run_stage SET ended_at=?, duration_s=?, status=?, rows=?,"
        " error=?, attempts=?, kind=?, detail=? WHERE run_id=? AND stage=?",
        (_now_iso(now), round(float(duration_s), 3), result.status,
         int(result.rows), st.redact(result.error), int(attempts),
         result.kind or "", json.dumps(result.detail or {}, ensure_ascii=False,
                                       default=str)[:2000],
         run_id, stage))


def load_stages(run_id: str) -> dict[str, dict]:
    """一輪所有階段列（含執行中，用於狀態彙總與續跑判斷）。"""
    try:
        df = store.read_sql(
            "SELECT * FROM run_stage WHERE run_id=?", (run_id,))
    except Exception:  # noqa: BLE001
        return {}
    if df.empty:
        return {}
    rows = [_norm_row(x) for x in df.to_dict(orient="records")]
    order = {s: i for i, s in enumerate(STAGES)}
    rows.sort(key=lambda r: order.get(str(r.get("stage")), 99))
    return {str(r["stage"]): r for r in rows}


def aggregate_status(stages: dict[str, dict]) -> str:
    """由階段狀態推導輪次狀態（單一失敗不得被誤報成整輪成功）。"""
    if not stages:
        return ROUND_SKIPPED
    statuses = [str(r.get("status", "")) for r in stages.values()]
    n_fail = sum(1 for s in statuses if s == IngestStatus.FAILED.value)
    n_partial = sum(1 for s in statuses
                    if s == IngestStatus.PARTIAL_SUCCESS.value)
    n_ok = sum(1 for s in statuses if s in OK_STATUSES)
    n_skip = sum(1 for s in statuses if s == IngestStatus.SKIPPED.value)
    blocked_skip = any(
        str(r.get("kind", "")) == KIND_BLOCKED
        and str(r.get("status", "")) == IngestStatus.SKIPPED.value
        for r in stages.values())
    if n_fail:
        return (ROUND_PARTIAL if (n_ok or n_partial or n_skip)
                else ROUND_FAILED)
    if n_partial or blocked_skip:
        return ROUND_PARTIAL
    if n_ok:
        return ROUND_SUCCESS
    return ROUND_SKIPPED


# ---------------------------------------------------------------------------
# 事件（run_alert）與通知通道
# ---------------------------------------------------------------------------
def _deliver(message: str) -> str:
    """嘗試外送通知；**未顯式啟用就絕不外送**，並如實回報送達狀態。"""
    if (os.environ.get(ALERT_TELEGRAM_ENV) or "").strip().lower() not in \
            ("1", "true", "yes"):
        return "not_configured"
    from .notify import send_telegram
    try:
        return "telegram_sent" if send_telegram(message) else \
            "telegram_failed"
    except Exception:  # noqa: BLE001
        return "telegram_failed"


def record_alert(level: str, message: str, *, run_id: str = "",
                 stage: str = "", kind: str = "") -> str:
    """記錄一則異常事件（同一輪同類同訊息去重；不假裝已送達）。"""
    msg = st.redact(message)
    try:
        dup = store.read_sql(
            "SELECT 1 FROM run_alert WHERE run_id=? AND kind=? AND message=? "
            "LIMIT 1", (run_id, kind, msg))
        if not dup.empty:
            return "dedup"
    except Exception:  # noqa: BLE001
        pass
    delivered = _deliver(f"[{level}] {msg}")
    store.write_df(pd.DataFrame([{
        "ts": _now_iso(), "level": level, "run_id": run_id, "stage": stage,
        "kind": kind, "message": msg, "delivered": delivered,
    }]), "run_alert")
    return delivered


def recent_alerts(limit: int = 50) -> list[dict]:
    if not p5_schema_ready():
        return []
    try:
        df = store.read_sql(
            "SELECT ts, level, run_id, stage, kind, message, delivered "
            "FROM run_alert ORDER BY alert_id DESC LIMIT ?", (int(limit),))
    except Exception as e:  # noqa: BLE001
        log.warning("讀取事件紀錄失敗（視為無事件）：%s", e)
        return []
    if df.empty:
        return []
    return [_norm_row(x) for x in df.to_dict(orient="records")]


# ---------------------------------------------------------------------------
# 階段實作（全部呼叫既有元件）
# ---------------------------------------------------------------------------
def _stage_ingest(ctx: RoundCtx) -> StageResult:
    """行情更新：抓日K/量價/法人/籌碼/月營收/新聞（含 0050 基準）。"""
    from .pipeline import ingest

    summary = ingest()
    status = summary.overall_status
    rows = int(sum(summary.counts.values()))
    detail = {"ingest_run_id": summary.run_id,
              "ingest_status": status,
              "failed_ids": list(summary.failed_ids),
              "status_counts": summary.status_counts()}
    error = "" if status != IngestStatus.PARTIAL_SUCCESS.value and \
        status != IngestStatus.FAILED.value else summary.label()
    return StageResult(status=status, rows=rows, error=error, detail=detail)


def _stage_quotes(ctx: RoundCtx) -> StageResult:
    """盤中行情輪詢（未設定來源 → 依規則略過；失敗不得假冒成功）。"""
    from . import worker

    if not worker.quote_poll_enabled():
        return StageResult(
            status=IngestStatus.SKIPPED.value, kind=KIND_POLICY,
            error=f"未設定 {worker.QUOTE_SOURCE_ENV}（不進行行情輪詢）")
    res = worker.run_quote_cycle(force=True, now=ctx.now)
    status = str(res.get("status", ""))
    if status in ("not_configured",):
        return StageResult(status=IngestStatus.SKIPPED.value, kind=KIND_POLICY,
                           error=str(res.get("reason", status)))
    if status == "skipped":
        return StageResult(status=IngestStatus.SKIPPED.value, kind=KIND_POLICY,
                           error=str(res.get("reason", "skipped")))
    if status == "failed":
        return StageResult(status=IngestStatus.FAILED.value,
                           error=str(res.get("reason", "quote poll failed")))
    poll = res.get("result")
    if poll is None:
        return StageResult(status=IngestStatus.FAILED.value,
                           error="行情輪詢無結果（無法確認來源狀態）")
    if getattr(poll, "skipped_reason", ""):
        return StageResult(status=IngestStatus.SKIPPED.value, kind=KIND_POLICY,
                           error=str(poll.skipped_reason))
    hard = (st.ErrorClass.NETWORK.value, st.ErrorClass.TIMEOUT.value,
            st.ErrorClass.HTTP_STATUS.value, st.ErrorClass.PARSE.value,
            st.ErrorClass.WRITE.value)
    err_class = getattr(getattr(poll, "fetch", None), "error_class", "") or ""
    written = int(getattr(poll.write, "written", 0))
    rejected = int(getattr(poll.write, "rejected", 0))
    detail = {"source": poll.source, "kind": poll.kind,
              "written": written, "rejected": rejected,
              "health": (poll.health.health if poll.health else ""),
              "poll_run_id": poll.run_id}
    if err_class in hard:
        return StageResult(status=IngestStatus.FAILED.value, rows=written,
                           error=getattr(poll.fetch, "error", "") or err_class,
                           detail=detail)
    if written and rejected:
        return StageResult(status=IngestStatus.PARTIAL_SUCCESS.value,
                           rows=written, error=poll.write.label(), detail=detail)
    if written == 0 and rejected > 0:
        return StageResult(status=IngestStatus.FAILED.value, rows=0,
                           error=poll.write.label(), detail=detail)
    if written:
        return StageResult(status=IngestStatus.SUCCESS_WITH_DATA.value,
                           rows=written, detail=detail)
    return StageResult(status=IngestStatus.SUCCESS_EMPTY.value, rows=0,
                       detail=detail)


def _stage_quality(ctx: RoundCtx) -> StageResult:
    """資料品質檢查：新鮮度、未來日期、無效價格、訊號與價量一致性。"""
    hard: list[str] = []
    soft: list[str] = []
    details: dict[str, Any] = {}

    daily_problems, daily_details = gate_mod.daily_basis(
        now=ctx.now, limits=risk_mod.DEFAULT_LIMITS)
    hard += daily_problems
    details["price"] = daily_details

    today = ctx.now.date().isoformat()
    try:
        fut = store.read_sql(
            "SELECT COUNT(*) AS n, MAX(date) AS m FROM price WHERE date>?",
            (today,))
        n_future = int(fut.iloc[0]["n"] or 0) if not fut.empty else 0
        details["n_future_rows"] = n_future
        if n_future:
            hard.append(
                f"價量表存在 {n_future} 列晚於系統時間的行情"
                f"（最晚 {fut.iloc[0]['m']}），時間不一致")
    except Exception as e:  # noqa: BLE001
        hard.append(f"價量時間檢查失敗（fail-closed）：{type(e).__name__}: {e}")

    since = (ctx.now.date() - dt.timedelta(days=10)).isoformat()
    n_checked = 0
    n_invalid = 0
    try:
        recent = store.read_sql(
            "SELECT stock_id, date, open, high, low, close, volume FROM price "
            "WHERE date>=?", (since,))
        n_checked = int(len(recent))
        if not recent.empty:
                bad = quality_mod.find_invalid_price_rows(recent)
                n_invalid = int(len(bad))
                if n_invalid:
                    dates = sorted({str(d)[:10] for d in bad["date"]})[:3] \
                        if "date" in bad.columns else []
                    # 最新一列無效 → hard（gate 也會擋）；其餘 → soft
                    latest_date = daily_details.get("latest_date")
                    latest_bad = bool(latest_date) and bool(
                        len(bad) and (bad["date"].astype(str).str[:10]
                                      == str(latest_date)).any())
                    (hard if latest_bad else soft).append(
                        f"近 10 日無效價格列 {n_invalid} 列"
                        f"（{'含最新交易日' if latest_bad else '非最新交易日'}，"
                        f"日期如 {', '.join(dates)}…）")
    except Exception as e:  # noqa: BLE001
        hard.append(f"價格品質檢查失敗（fail-closed）：{type(e).__name__}: {e}")

    try:
        sig = store.read_sql("SELECT MAX(date) AS d FROM signals")
        sig_date = None if sig.empty else sig.iloc[0]["d"]
        details["latest_signal_date"] = str(sig_date)[:10] if sig_date else None
        price_latest = daily_details.get("latest_date")
        if sig_date and price_latest and str(sig_date)[:10] > price_latest:
            soft.append(f"最新訊號日期（{str(sig_date)[:10]}）晚於價量資料"
                        f"（{price_latest}），須確認資料時點")
    except Exception as e:  # noqa: BLE001
        soft.append(f"訊號/價量一致性檢查失敗：{type(e).__name__}: {e}")

    details["n_recent_rows"] = n_checked
    details["n_invalid_recent"] = n_invalid
    problems = hard + soft
    if hard:
        return StageResult(status=IngestStatus.FAILED.value, rows=n_checked,
                           error="；".join(hard[:3]), detail=details)
    if soft:
        return StageResult(status=IngestStatus.PARTIAL_SUCCESS.value,
                           rows=n_checked, error="；".join(soft[:3]),
                           detail=details)
    return StageResult(status=IngestStatus.SUCCESS_WITH_DATA.value,
                       rows=n_checked, detail=details)


def _stage_signals(ctx: RoundCtx) -> StageResult:
    """訊號產生：特徵 + 規則打分（+ 可選 ML 融合），與 ``cli run`` 同一條路徑。"""
    from .pipeline import run_scores

    df = run_scores()
    if df is None or df.empty:
        return StageResult(status=IngestStatus.SUCCESS_EMPTY.value, rows=0,
                           detail={"note": "無訊號產出（無資料或分數為空）"})
    latest = str(df["date"].max())[:10] if "date" in df.columns else ""
    return StageResult(status=IngestStatus.SUCCESS_WITH_DATA.value,
                       rows=int(len(df)),
                       detail={"signal_date": latest, "rows": int(len(df))})


def _stage_paper(ctx: RoundCtx) -> StageResult:
    """模擬交易：受 P5-B 閘門約束（blocked → 不建立任何新單，可稽核）。"""
    from .paper import PaperEngine

    eng = PaperEngine()
    if eng.account() is None:
        return StageResult(status=IngestStatus.SKIPPED.value,
                           kind=KIND_BLOCKED,
                           error="未建立 paper 帳戶（安全停止：請先 paper init）")
    res = eng.run_paper_cycle(now=ctx.now, run_id=ctx.run_id)
    g = dict(res.get("gate") or {})
    orders = list(res.get("orders") or [])
    fills = list(res.get("fills") or [])
    n_new = sum(1 for o in orders if o.get("accepted") and not o.get("duplicate"))
    n_filled = sum(int(f.get("filled") or 0) > 0 for f in fills)
    detail = {"gate_allow_new_orders": bool(g.get("allow_new_orders", True)),
              "gate_kind": g.get("kind", ""), "gate_reasons": g.get("reasons", []),
              "orders_new": n_new, "orders_total": len(orders),
              "fills": n_filled, "signal_date": (g.get("details", {})
                                                 .get("price", {})
                                                 .get("latest_date"))}
    if not g.get("allow_new_orders", True):
        kind = str(g.get("kind") or KIND_BLOCKED)
        return StageResult(
            status=IngestStatus.SKIPPED.value, rows=n_filled, kind=kind,
            error="新單被行情品質閘門阻止：" + "；".join(
                list(g.get("reasons") or [])[:3]),
            detail={**detail, "blocked": list(res.get("blocked") or [])})
    if n_new or n_filled:
        return StageResult(status=IngestStatus.SUCCESS_WITH_DATA.value,
                           rows=n_new + n_filled, detail=detail)
    return StageResult(status=IngestStatus.SUCCESS_EMPTY.value, rows=0,
                       detail=detail)


def _stage_reconcile(ctx: RoundCtx) -> StageResult:
    """帳務核對（唯讀）：不一致只回報與阻擋，**絕不修帳**。"""
    from .paper import PaperEngine

    eng = PaperEngine()
    if eng.account() is None:
        return StageResult(status=IngestStatus.SKIPPED.value,
                           kind=KIND_BLOCKED,
                           error="未建立 paper 帳戶（無帳務可核對）")
    rec = eng.reconcile()
    detail = {"fills": int(rec.get("fills", 0)),
              "ledger": int(rec.get("ledger", 0)),
              "snapshots": int(rec.get("snapshots", 0)),
              "positions": rec.get("positions", {})}
    if rec.get("ok"):
        return StageResult(status=IngestStatus.SUCCESS_WITH_DATA.value,
                           rows=int(rec.get("fills", 0)), detail=detail)
    problems = list(rec.get("problems") or [])
    return StageResult(status=IngestStatus.FAILED.value, rows=0,
                       error=f"帳務不一致（{len(problems)} 項）："
                             + "；".join(problems[:3]),
                       detail={**detail, "problems": problems[:10]})


def _stage_health(ctx: RoundCtx) -> StageResult:
    """健康狀態：彙整程序/資料/帳務/閘門，並在異常時記錄事件。"""
    h = health_report(now=ctx.now)
    state = str(h.get("state", "unknown"))
    detail = {"state": state, "alive": h.get("process", {}).get("alive"),
              "data_fresh": h.get("data", {}).get("fresh"),
              "reconcile_ok": h.get("account", {}).get("reconcile_ok"),
              "gate_allow": h.get("gate", {}).get("allow_new_orders"),
              "error_rate": h.get("errors", {}).get("failure_rate")}
    if state != "healthy":
        level = "warning"
        if state in ("kill_switch", "inconsistent", "task_failed"):
            level = "error"
        # 手動輪次未執行 worker 屬正常；排程輪未執行 worker 才是異常
        if state not in ("worker_stopped", "not_started") or \
                ctx.trigger != "manual":
            record_alert(level, f"健康狀態 {state}：{h.get('summary', '')}",
                         run_id=ctx.run_id, stage="health", kind=f"health:{state}")
    return StageResult(status=IngestStatus.SUCCESS_WITH_DATA.value, rows=1,
                       detail=detail)


DEFAULT_FNS: dict[str, Callable[[RoundCtx], StageResult]] = {
    "ingest": _stage_ingest,
    "quotes": _stage_quotes,
    "quality": _stage_quality,
    "signals": _stage_signals,
    "paper": _stage_paper,
    "reconcile": _stage_reconcile,
    "health": _stage_health,
}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_round(*, now: dt.datetime | None = None, trigger: str = "manual",
              force: bool | None = None, run_id: str | None = None,
              deadline_s: float | None = None, attempts: int | None = None,
              sleep: Callable[[float], None] = time.sleep,
              fns: dict[str, Callable[[RoundCtx], StageResult]] | None = None,
              use_lock: bool = True) -> RoundResult:
    """執行一輪自動化工作（行情更新 → … → 健康狀態）。

    - ``force=False``（排程輪）依 ``round_due`` 決定是否執行；手動輪預設 force。
    - 已完成的階段**不重複執行**（冪等/續跑）；``run_id`` 可指定續跑特定輪次。
    - 單一實例鎖防止兩個協調器同時執行；重試有限次＋退避；輪次逾時後續階段
      標記跳過（不讓單輪永久卡住）。
    """
    from . import worker

    now = now or dt.datetime.now()
    if is_readonly():
        return RoundResult(run_id=run_id or "", status=ROUND_SKIPPED,
                           trigger=trigger, error="唯讀模式（TWSTOCK_READONLY）"
                                                  "不執行自動化輪次")
    fns = dict(fns or {})
    attempts = int(attempts if attempts is not None
                   else os.environ.get(ATTEMPTS_ENV, DEFAULT_ATTEMPTS))
    attempts = max(1, attempts)
    deadline_s = float(deadline_s if deadline_s is not None
                       else os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_S))
    if force is None:
        force = (trigger == "manual")

    lock = worker.WorkerLock(name="p5run") if use_lock else None
    if lock is not None and not lock.acquire():
        holder = lock.holder_pid()
        return RoundResult(run_id="", status=ROUND_SKIPPED, trigger=trigger,
                           busy=True,
                           error=f"已有 P5 輪次在執行（PID {holder}），"
                                 "拒絕同時啟動")
    try:
        # ---- 輪次識別：續跑未完成輪 / 指定輪 / 新輪 ----
        open_id = open_round_id()
        resumed = False
        if run_id:
            rid = run_id
            existing = _stage_row(rid, STAGES[0]) is not None or _round_exists(rid)
            resumed = bool(existing)
        elif open_id:
            rid, resumed = open_id, True
        else:
            rid = new_run_id(now)

        if not resumed:
            if not force:
                due, why = round_due(now)
                if not due:
                    _insert_round(rid, trigger, now, status=ROUND_SKIPPED,
                                  error=why, ended=True)
                    return RoundResult(run_id=rid, status=ROUND_SKIPPED,
                                       trigger=trigger,
                                       started_at=_now_iso(now),
                                       ended_at=_now_iso(now),
                                       error=why)
            _insert_round(rid, trigger, now, notes=f"trigger={trigger}")
        else:
            store.exec_sql(
                "UPDATE run_round SET status=?, notes=? WHERE run_id=?",
                (ROUND_RUNNING, f"resume of trigger={trigger} "
                                f"@{_now_iso(now)}", rid))
            log.warning("續跑未完成輪次 %s（已完成階段不重複執行）", rid)

        ctx = RoundCtx(run_id=rid, now=now, trigger=trigger, fns=fns)
        t_round = time.monotonic()
        deadline = (t_round + deadline_s) if deadline_s and deadline_s > 0 \
            else None

        for stage in STAGES:
            prev = _stage_row(rid, stage)
            prev_status = str(prev.get("status", "")) if prev else ""
            if prev and prev_status in TERMINAL_STATUSES:
                log.info("階段 %s 已完成（%s），跳過不重複執行", stage, prev_status)
                continue
            # 輪次逾時預算：不讓單一輪卡住後續排程
            if deadline is not None and time.monotonic() > deadline:
                sr = StageResult(
                    status=IngestStatus.SKIPPED.value, kind=KIND_BLOCKED,
                    error=f"輪次逾時（>{deadline_s:.0f}s），停止後續階段")
                _record_stage(rid, stage, now, sr, duration_s=0.0, attempts=0)
                continue
            # paper 的上游前置：任一上游失敗 → 停止新模擬交易（fail-closed）
            if stage == "paper":
                failed_up = [s for s in UPSTREAM_FOR_PAPER
                             if str((load_stages(rid).get(s) or {})
                                    .get("status", "")) == IngestStatus.FAILED.value]
                if failed_up:
                    sr = StageResult(
                        status=IngestStatus.SKIPPED.value, kind=KIND_BLOCKED,
                        error=f"上游階段失敗（{'、'.join(failed_up)}），"
                              "停止新模擬交易（fail-closed）")
                    _record_stage(rid, stage, now, sr, duration_s=0.0,
                                  attempts=0)
                    record_alert("warning", sr.error, run_id=rid, stage=stage,
                                 kind="upstream_failed")
                    continue
            fn = fns.get(stage) or DEFAULT_FNS[stage]
            attempts_used = int((prev or {}).get("attempts") or 0) + 1
            _begin_stage(rid, stage, now, attempts=attempts_used)
            t0 = time.monotonic()
            outcome = worker.run_task_with_retry(
                lambda: fn(ctx), name=stage, attempts=attempts, sleep=sleep)
            duration = time.monotonic() - t0
            if outcome.ok and isinstance(outcome.value, StageResult):
                sr = outcome.value
            elif outcome.ok:
                sr = StageResult(status=IngestStatus.SUCCESS_EMPTY.value)
            else:
                sr = StageResult(status=IngestStatus.FAILED.value,
                                 error=outcome.error)
            _record_stage(rid, stage, now, sr, duration_s=duration,
                          attempts=outcome.attempts)
            if sr.status == IngestStatus.FAILED.value:
                record_alert("error", f"{STAGE_LABEL.get(stage, stage)} 失敗："
                                      f"{sr.error}", run_id=rid, stage=stage,
                             kind="stage_failed")
            elif sr.status == IngestStatus.SKIPPED.value and \
                    sr.kind == KIND_BLOCKED:
                record_alert("warning", f"{STAGE_LABEL.get(stage, stage)} "
                                        f"被阻擋：{sr.error}",
                             run_id=rid, stage=stage, kind="stage_blocked")

        stages = load_stages(rid)
        overall = aggregate_status(stages)
        # 錯誤摘要只收「失敗」與「被阻擋（blocked）」的階段；
        # 依規則略過（policy，如未設定行情來源）不算錯誤，不得污染輪次結果。
        errors = [f"{s}: {r.get('error')}" for s, r in stages.items()
                  if r.get("error") and (
                      r.get("status") == IngestStatus.FAILED.value or
                      (r.get("status") == IngestStatus.SKIPPED.value
                       and str(r.get("kind", "")) == KIND_BLOCKED))]
        err = "；".join(errors[:3])
        _finish_round(rid, status=overall, now=now, error=err)
        try:
            store.set_meta("last_p5_run_id", rid)
            store.set_meta("last_p5_status", overall)
            store.set_meta("last_p5_ended_at", _now_iso(now))
            store.set_meta("last_p5_error", st.redact(err))
        except Exception as e:  # noqa: BLE001
            log.warning("寫入 P5 meta 失敗（不影響輪次結果）：%s", e)
        if overall in (ROUND_FAILED, ROUND_PARTIAL):
            record_alert("error" if overall == ROUND_FAILED else "warning",
                         f"輪次 {rid} {overall}：{err}", run_id=rid,
                         kind="round")
        return RoundResult(run_id=rid, status=overall, trigger=trigger,
                           started_at=(stages.get(STAGES[0], {})
                                       .get("started_at") or _now_iso(now)),
                           ended_at=_now_iso(now),
                           stages={k: {kk: v for kk, v in row.items()
                                       if kk in ("status", "rows", "error",
                                                 "kind", "attempts",
                                                 "duration_s")}
                                   for k, row in stages.items()},
                           error=err, resumed=resumed)
    finally:
        if lock is not None:
            lock.release()


def _round_exists(run_id: str) -> bool:
    try:
        df = store.read_sql("SELECT 1 FROM run_round WHERE run_id=? LIMIT 1",
                            (run_id,))
        return not df.empty
    except Exception:  # noqa: BLE001
        return False


def _record_stage(run_id: str, stage: str, now: dt.datetime,
                  result: StageResult, *, duration_s: float,
                  attempts: int) -> None:
    if _stage_row(run_id, stage) is None:
        _begin_stage(run_id, stage, now, attempts=attempts)
    _end_stage(run_id, stage, now, result, duration_s=duration_s,
               attempts=attempts)


# ---------------------------------------------------------------------------
# 健康狀態（P5-C；不只看程序存活）
# ---------------------------------------------------------------------------
def health_report(*, now: dt.datetime | None = None,
                  engine: Any = None) -> dict[str, Any]:
    """彙整程序存活、資料新鮮度、錯誤率、帳務一致與閘門狀態。

    可辨識「程序活著但資料已過期」：``process.alive=True`` 且
    ``data.fresh=False`` → 整體 ``state=data_stale``。
    """
    from . import ops, worker

    now = now or dt.datetime.now()
    comp = {r["name"]: r for r in ops.status()}
    wrow = comp.get("worker", {"pid": None, "alive": False})
    hb = worker.read_heartbeat()
    hb_at = str(hb.get("at") or "")
    hb_age_s: float | None = None
    if hb_at:
        try:
            hb_age_s = (now - dt.datetime.fromisoformat(hb_at)).total_seconds()
        except (TypeError, ValueError):
            hb_age_s = None
    try:
        stale_days = worker._compute_data_staleness()
    except Exception:  # noqa: BLE001
        stale_days = None
    limits = risk_mod.DEFAULT_LIMITS
    wstate = worker.derive_state(
        hb, stop_flag=worker.stop_requested("worker"),
        pid_alive=bool(wrow.get("alive")), data_staleness_days=stale_days,
        now=now)

    # ---- 資料新鮮度 ----
    price_problems, price_details = gate_mod.daily_basis(now=now, limits=limits)
    data_fresh = (price_details.get("staleness_days") is not None
                  and int(price_details["staleness_days"])
                  <= limits.max_data_staleness_days)
    latest_signal = None
    try:
        sig = store.read_sql("SELECT MAX(date) AS d FROM signals")
        if not sig.empty and sig.iloc[0]["d"]:
            latest_signal = str(sig.iloc[0]["d"])[:10]
    except Exception:  # noqa: BLE001
        latest_signal = None
    quote_hb = {k: hb[k] for k in hb if str(k).startswith("quote_")}

    # ---- 錯誤率（最近階段紀錄）----
    n_recent = n_failed = 0
    last_error = last_error_at = ""
    try:
        recent = store.read_sql(
            "SELECT status, error, ended_at FROM run_stage "
            "WHERE status IS NOT NULL ORDER BY ended_at DESC LIMIT 50")
        if not recent.empty:
            vals = [str(s) for s in recent["status"]]
            n_recent = len(vals)
            n_failed = sum(1 for s in vals if s == IngestStatus.FAILED.value)
            bad = recent[recent["status"] == IngestStatus.FAILED.value]
            if not bad.empty:
                last_error = str(bad.iloc[0]["error"] or "")
                last_error_at = str(bad.iloc[0]["ended_at"] or "")
    except Exception:  # noqa: BLE001
        pass
    failure_rate = (n_failed / n_recent) if n_recent else 0.0

    # ---- 帳務與閘門 ----
    from .paper import PaperEngine
    eng = engine or PaperEngine()
    acc = None
    reconcile_ok: bool | None = None
    problems: list[str] = []
    try:
        acc = eng.account()
    except Exception:  # noqa: BLE001
        acc = None
    if acc is not None:
        try:
            rec = eng.reconcile()
            reconcile_ok = bool(rec.get("ok", False))
            problems = list(rec.get("problems") or [])[:5]
        except Exception as e:  # noqa: BLE001
            reconcile_ok = False
            problems = [f"核對無法執行：{type(e).__name__}: {e}"]
    dec = gate_mod.evaluate(now=now, engine=eng if acc is not None else None,
                            limits=limits)
    ks = risk_mod.KillSwitch()
    kill = {"engaged": ks.engaged(), "reason": ks.reason()}

    # ---- 整體狀態（優先序：停止開關 > 帳務 > 資料 > 任務 > 程序）----
    if kill["engaged"]:
        state = "kill_switch"
    elif reconcile_ok is False:
        state = "inconsistent"
    elif not data_fresh:
        state = "data_stale"
    elif str(wstate.value) == "task_failed":
        state = "task_failed"
    elif not hb:
        state = "not_started"
    elif not wrow.get("alive"):
        state = "worker_stopped"
    elif str(wstate.value) in ("degraded", "data_stale"):
        state = "degraded"
    else:
        state = "healthy"

    summary = {
        "kill_switch": f"Kill Switch 已啟用（{kill['reason']}）"
        if kill["engaged"] else "",
        "inconsistent": "帳務不一致：" + "；".join(problems[:2]),
        "data_stale": f"行情延遲 {stale_days} 天（> {limits.max_data_staleness_days}）"
        if stale_days is not None else "行情延遲無法判定",
        "task_failed": "Worker 連續任務失敗",
        "not_started": "Worker 未啟動過",
        "worker_stopped": "Worker 未執行",
        "degraded": f"Worker 降級（{wstate.label}）",
        "healthy": "正常",
    }.get(state, state)

    return {
        "time": _now_iso(now),
        "state": state,
        "summary": summary,
        "ok": state == "healthy",
        "process": {
            "alive": bool(wrow.get("alive")), "pid": wrow.get("pid"),
            "worker_state": wstate.value, "worker_state_label": wstate.label,
            "heartbeat_at": hb_at, "heartbeat_age_s": hb_age_s,
            "last_success": hb.get("last_success"),
            "last_result": hb.get("last_result"),
            "last_cycle_status": hb.get("last_cycle_status"),
            "failure_reason": hb.get("failure_reason", ""),
            "consecutive_failures": int(hb.get("consecutive_failures") or 0),
            "stop_requested": worker.stop_requested("worker"),
        },
        "data": {
            "fresh": data_fresh,
            "latest_price_date": price_details.get("latest_date"),
            "staleness_days": price_details.get("staleness_days"),
            "max_staleness_days": limits.max_data_staleness_days,
            "latest_signal_date": latest_signal,
            "price_problems": price_problems,
            "quote": quote_hb,
        },
        "errors": {
            "recent_stages": n_recent, "failed_stages": n_failed,
            "failure_rate": round(float(failure_rate), 3),
            "last_error": st.redact(last_error),
            "last_error_at": last_error_at,
        },
        "account": {
            "exists": acc is not None, "reconcile_ok": reconcile_ok,
            "problems": problems,
        },
        "gate": dec.as_dict(),
        "kill_switch": kill,
    }


# ---------------------------------------------------------------------------
# 帳務稽核（P5-D；**預設唯讀**，不一致只回報）
# ---------------------------------------------------------------------------
def audit_report(*, now: dt.datetime | None = None, engine: Any = None
                 ) -> dict[str, Any]:
    """帳務與運行狀態稽核（唯讀）。

    涵蓋：帳本/持倉/成交/訂單/快照一致性、未完成與重複訂單、
    現金與市值與已實現/未實現/總資產計算一致性、worker 最近結果、
    行情閘門是否允許新模擬交易。發現不一致**只回報**，不修改任何資料。
    """
    from .paper import PaperEngine

    now = now or dt.datetime.now()
    eng = engine or PaperEngine()
    checks: list[dict[str, Any]] = []

    def _check(name: str, ok: bool, **kw: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), **kw})

    acc = None
    try:
        acc = eng.account()
    except Exception as e:  # noqa: BLE001
        _check("account", False, detail=f"帳戶無法讀取：{type(e).__name__}: {e}")
        return {"time": _now_iso(now), "ok": False, "readonly": True,
                "checks": checks}
    _check("account", acc is not None,
           detail=(f"mode={acc.get('mode')} status={acc.get('status')}"
                   if acc else "未建立 paper 帳戶（paper init）"))
    if acc is None:
        return {"time": _now_iso(now), "ok": False, "readonly": True,
                "checks": checks, "problems": ["未建立 paper 帳戶"]}
    aid = int(acc["account_id"])

    # 1) 帳務不變條件（帳本鏈/成交金額/持倉重放/訂單成交量/快照恆等式）
    rec = eng.reconcile()
    _check("reconcile", bool(rec.get("ok")),
           problems=list(rec.get("problems") or []),
           detail=f"fills={rec.get('fills')} ledger={rec.get('ledger')} "
                  f"snapshots={rec.get('snapshots')}")

    # 2) 訂單與成交關聯、未完成與重複訂單
    orders = store.read_sql(
        "SELECT order_id, client_order_id, stock_id, side, qty, filled_qty,"
        " status, signal_date, created_at FROM paper_order WHERE account_id=?",
        (aid,))
    fills = store.read_sql(
        "SELECT fill_id, order_id, stock_id, side, qty, fill_date FROM "
        "paper_fill WHERE account_id=?", (aid,))
    open_orders = orders[orders["status"].isin(("pending_new", "accepted",
                                                "partially_filled"))] \
        if not orders.empty else orders
    dup = []
    if not open_orders.empty:
        grp = open_orders.groupby(["stock_id", "side", "signal_date"]).size()
        dup = [f"{k[0]}/{k[1]}/{k[2]}×{int(v)}"
               for k, v in grp[grp > 1].items()]
    orphan = []
    if not fills.empty and not orders.empty:
        orphan = sorted(set(fills["order_id"]) - set(orders["order_id"]))
    _check("orders", not dup and not orphan,
           open_orders=int(len(open_orders)), fills=int(len(fills)),
           duplicates=dup, orphan_fills=orphan,
           detail=f"未完成 {len(open_orders)} 張；重複 {len(dup)}；"
                  f"孤兒成交 {len(orphan)}")

    # 3) 現金/持倉市值/已實現/未實現/總資產計算一致性（同一時點重算）
    try:
        snap = eng.snapshot(write=False, now=now)
        cash, positions = eng.cash_and_positions(aid)
        recomputed_equity = cash + sum(
            int((0 if st_.qty <= 0 else st_.avg_cost_share) * st_.qty)
            for st_ in positions.values() if st_.qty > 0)
        realized = sum(st_.realized for st_ in positions.values())
        ok_calc = (int(paper_micro(snap["equity"])) ==
                   int(paper_micro(snap["cash"]) + paper_micro(snap["market_value"]))
                   and int(paper_micro(snap["realized"])) == int(realized))
        _check("valuation", ok_calc,
               detail=f"equity={snap['equity']:,.2f} cash={snap['cash']:,.2f} "
                      f"mv={snap['market_value']:,.2f} "
                      f"realized={snap['realized']:,.2f} "
                      f"unrealized={snap['unrealized']:,.2f}",
               rebuilt_cash=paper_from_micro(cash),
               cost_basis_equity=paper_from_micro(recomputed_equity))
    except Exception as e:  # noqa: BLE001
        _check("valuation", False,
               detail=f"估值計算失敗：{type(e).__name__}: {e}")

    # 4) worker 最近執行結果、最後成功時間與失敗原因
    from . import ops, worker
    wrow = {r["name"]: r for r in ops.status()}.get("worker", {})
    hb = worker.read_heartbeat()
    hb_at = str(hb.get("at") or "")
    age_s: float | None = None
    if hb_at:
        try:
            age_s = (now - dt.datetime.fromisoformat(hb_at)).total_seconds()
        except (TypeError, ValueError):
            age_s = None
    alive = bool(wrow.get("alive"))
    hb_ok = (not alive) or (age_s is not None and age_s <= HEARTBEAT_STALE_S)
    _check("worker_heartbeat", hb_ok,
           detail=(f"alive={alive} pid={wrow.get('pid')} 心跳 {hb_at}"
                   + (f"（{age_s:.0f}s 前）" if age_s is not None else "")
                   + (f"；最後成功 {hb.get('last_success')}；"
                      f"失敗原因 {hb.get('failure_reason')}"
                      if hb.get("failure_reason") else "")),
           alive=alive, last_success=hb.get("last_success"),
           last_result=hb.get("last_result"),
           failure_reason=hb.get("failure_reason", ""),
           heartbeat_at=hb_at, heartbeat_age_s=age_s)

    # 5) 行情品質閘門目前是否允許新模擬交易
    dec = gate_mod.evaluate(now=now, engine=eng)
    _check("gate", dec.allow_new_orders, detail=dec.label,
           reasons=list(dec.reasons),
           allow_new_orders=dec.allow_new_orders,
           allow_fills=dec.allow_fills)

    # 6) 資料新鮮度
    _, price_details = gate_mod.daily_basis(now=now, limits=risk_mod.DEFAULT_LIMITS)
    _check("freshness",
           price_details.get("staleness_days") is not None
           and int(price_details["staleness_days"])
           <= risk_mod.DEFAULT_LIMITS.max_data_staleness_days,
           detail=f"最新價量 {price_details.get('latest_date')}"
                  f"（延遲 {price_details.get('staleness_days')} 天）")

    problems = [f"{c['name']}: {c.get('detail', '')}" for c in checks
                if not c["ok"]]
    return {"time": _now_iso(now), "ok": not problems, "readonly": True,
            "account_id": aid, "checks": checks, "problems": problems}


def paper_micro(x: float) -> int:
    from .paper import to_micro
    return int(to_micro(x))


def paper_from_micro(x: int) -> float:
    from .paper import from_micro
    return float(from_micro(x))


# ---------------------------------------------------------------------------
# 查詢（CLI / API）
# ---------------------------------------------------------------------------
def recent_rounds(limit: int = 5) -> list[dict]:
    if not p5_schema_ready():
        return []
    try:
        df = store.read_sql(
            "SELECT run_id, trigger, started_at, ended_at, status, error FROM "
            "run_round ORDER BY started_at DESC, run_id DESC LIMIT ?",
            (int(limit),))
    except Exception as e:  # noqa: BLE001
        log.warning("讀取輪次紀錄失敗（視為無紀錄）：%s", e)
        return []
    out: list[dict] = []
    for r in (_norm_row(x) for x in
              (df.to_dict(orient="records") if not df.empty else [])):
        rid = str(r["run_id"])
        stages = load_stages(rid)
        out.append({**r, "stages": {s: {"status": v.get("status"),
                                        "rows": v.get("rows"),
                                        "error": v.get("error"),
                                        "kind": v.get("kind"),
                                        "duration_s": v.get("duration_s")}
                                    for s, v in stages.items()}})
    return out


def recent_errors(limit: int = 20) -> dict[str, Any]:
    """最近的失敗階段與事件（CLI ``p5 errors`` 用）。"""
    if not p5_schema_ready():
        return {"stages": [], "alerts": [], "schema_ready": False}
    try:
        stages = store.read_sql(
            "SELECT run_id, stage, status, error, ended_at FROM run_stage "
            "WHERE status IN (?,?) ORDER BY ended_at DESC LIMIT ?",
            (IngestStatus.FAILED.value, IngestStatus.SKIPPED.value,
             int(limit)))
    except Exception as e:  # noqa: BLE001
        log.warning("讀取階段紀錄失敗（視為無錯誤）：%s", e)
        stages = pd.DataFrame()
    return {
        "schema_ready": True,
        "stages": ([_norm_row(x) for x in stages.to_dict(orient="records")]
                   if not stages.empty else []),
        "alerts": recent_alerts(limit),
    }


# ---------------------------------------------------------------------------
# CLI（python -m twstock.cli p5 <cmd>）
# ---------------------------------------------------------------------------
SUBCMDS = ("run", "status", "errors", "audit", "gate", "alerts")


def _gate_now(engine: Any = None) -> gate_mod.GateDecision:
    return gate_mod.evaluate(engine=engine)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="P5 本地自動化協調器（純模擬交易，不下真單）。"
                   f"子命令：{', '.join(SUBCMDS)}")
    ap.add_argument("cmd", nargs="?", default="status")
    ap.add_argument("--limit", type=int, default=None, help="顯示筆數")
    ap.add_argument("--due", action="store_true",
                    help="run：依排程判定執行（同日已有成功輪則跳過）")
    args = ap.parse_args(argv)
    limit = args.limit if args.limit is not None else 20

    if args.cmd == "run":
        res = run_round(trigger="scheduled" if args.due else "manual",
                        force=not args.due)
        print(res.label())
        return EXIT_CODES.get(res.status, 2) + (1 if res.busy else 0)
    if args.cmd == "status":
        rounds = recent_rounds(3)
        print("=== P5 自動化狀態 ===")
        if not p5_schema_ready():
            print("  P5 資料表尚未建立（唯讀模式不會建立）；"
                  "以可寫模式執行一次 p5 run 即自動補表（不碰既有資料）")
        for r in rounds:
            stages = "、".join(
                f"{s}={v.get('status')}" for s, v in r["stages"].items()) \
                or "（無階段）"
            print(f"  {r['run_id']} [{r['status']}] {r['started_at']}"
                  f" trigger={r['trigger']}")
            print(f"    {stages}")
            if r.get("error"):
                print(f"    錯誤：{r['error']}")
        if not rounds:
            print("  （尚無輪次紀錄；執行：python -m twstock.cli p5 run）")
        try:
            print(_gate_now().text())
        except Exception as e:  # noqa: BLE001
            print(f"  閘門無法判定：{e}")
        return 0
    if args.cmd == "errors":
        data = recent_errors(limit)
        print("=== P5 最近錯誤 ===")
        if not data.get("schema_ready", True):
            print("  P5 資料表尚未建立（唯讀模式不會建立）")
        for s in data["stages"]:
            print(f"  {s.get('ended_at')} {s.get('run_id')}/"
                  f"{s.get('stage')} [{s.get('status')}] {s.get('error')}")
        for a in data["alerts"]:
            print(f"  {a.get('ts')} [{a.get('level')}] {a.get('run_id')}"
                  f"/{a.get('stage')} {a.get('message')}"
                  f"（送達：{a.get('delivered')}）")
        if not data["stages"] and not data["alerts"]:
            print("  （無錯誤紀錄）")
        return 0
    if args.cmd == "alerts":
        if not p5_schema_ready():
            print("  P5 資料表尚未建立（唯讀模式不會建立）")
            return 0
        for a in recent_alerts(limit):
            print(f"  {a.get('ts')} [{a.get('level')}] {a.get('run_id')}"
                  f"/{a.get('stage')} {a.get('message')}"
                  f"（送達：{a.get('delivered')}）")
        return 0
    if args.cmd == "audit":
        rep = audit_report()
        print("=== P5 帳務稽核（唯讀；不一致只回報、不修帳）===")
        for c in rep["checks"]:
            mark = "✓" if c["ok"] else "✗"
            print(f"  {mark} {c['name']}: {c.get('detail', '')}")
            for p in (c.get("problems") or [])[:5]:
                print(f"      ⚠ {p}")
        if is_readonly():
            print("  （唯讀模式：本命令不建立/修改任何資料）")
        print(f"  總結：{'一致' if rep['ok'] else '發現不一致'}")
        return 0 if rep["ok"] else 1
    if args.cmd == "gate":
        from .paper import PaperEngine
        eng = PaperEngine()
        dec = _gate_now(eng if eng.account() is not None else None)
        print(dec.text())
        return 0 if dec.allow_new_orders else 1
    print(f"未知子命令：{args.cmd}（可用：{', '.join(SUBCMDS)}）")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
