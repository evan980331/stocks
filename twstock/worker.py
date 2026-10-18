"""本地監控 Worker：可持續執行、可觀測、可安全停止、可重啟恢復。

硬化重點（P3.1）
  1. **單一行實例鎖**（``state/<name>.lock``）：同名 Worker 只允許一個；PID 已死時
     自動回收殘留鎖。啟動競態時後到者會被拒絕，不會出現兩個 Worker 同時抓資料。
  2. **有上限的重試與退避**：任務失敗依 ``attempts`` 重試，延遲為指數退避
     （``base_delay * 2**n``，上限 ``max_delay``）。失敗**一律記錄**，不當成成功。
  3. **任務逾時偵測**：執行前寫入 ``task_started_at``；若行程在任務中被殺，
     重啟時會發現該欄位殘留且已逾時 → 標記 ``interrupted``，不會誤以為上一輪完成。
     （同步任務無法跨平台安全中斷，故此處做的是**偵測與恢復**，不是強制中止。）
  4. **健康狀態機**：``derive_state()`` 依心跳、停止旗標、失敗次數與資料新鮮度，
     區分 not_started / starting / running / task_running / data_stale / degraded /
     task_failed / stopping / stopped。
  5. **休市不輪詢**：非交易時段直接進入長睡眠，不對資料來源做無謂請求。

本 Worker **不做任何交易**：僅資料更新、特徵、打分與健康記錄。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

from .config import DATA_DIR, STATE_DIR

log = logging.getLogger(__name__)

EOD_HOUR = 16              # 收盤後執行時間（本地時間）
SLEEP_CHUNK = 2.0          # 睡眠分段，以便即時回應停止旗標
DEFAULT_TASK_TIMEOUT = 3600     # 任務逾時門檻（秒）
DEFAULT_MAX_FAILURES = 3        # 連續失敗達此次數 → task_failed
DATA_STALE_DAYS = 3             # 行情延遲超過此天數 → data_stale

# ---- P3：盤中行情輪詢（可選、非致命、預設完全不做）----
QUOTE_SOURCE_ENV = "TWSTOCK_QUOTE_SOURCE"
QUOTE_INTERVAL_S = 60           # 建議輪詢間隔（仍受來源限流）
QUOTE_MAX_AGE_S = 300           # 行情有效年齡門檻（秒）

T = TypeVar("T")


# --------------------------------------------------------------------------
# 狀態機
# --------------------------------------------------------------------------
class WorkerState(str, Enum):
    NOT_STARTED = "not_started"
    STARTING = "starting"
    RUNNING = "running"
    TASK_RUNNING = "task_running"
    DATA_STALE = "data_stale"
    DEGRADED = "degraded"
    TASK_FAILED = "task_failed"
    STOPPING = "stopping"
    STOPPED = "stopped"

    @property
    def label(self) -> str:
        return {
            "not_started": "服務未啟動",
            "starting": "正在啟動",
            "running": "正常運作",
            "task_running": "任務執行中",
            "data_stale": "資料過期",
            "degraded": "部分功能降級",
            "task_failed": "任務失敗",
            "stopping": "正在停止",
            "stopped": "已停止",
        }[self.value]


def derive_state(heartbeat: dict | None, *, stop_flag: bool = False,
                 pid_alive: bool | None = None,
                 data_staleness_days: int | None = None,
                 now: dt.datetime | None = None,
                 task_timeout: int = DEFAULT_TASK_TIMEOUT,
                 max_failures: int = DEFAULT_MAX_FAILURES,
                 stale_days: int = DATA_STALE_DAYS) -> WorkerState:
    """由可觀測狀態推導 Worker 健康狀態（純函式，可完整單元測試）。"""
    now = now or dt.datetime.now()
    hb = heartbeat or {}

    if stop_flag:
        return WorkerState.STOPPING
    if not hb:
        return WorkerState.NOT_STARTED

    status = hb.get("status")
    if status == "starting":
        return WorkerState.STARTING

    failures = int(hb.get("consecutive_failures", 0) or 0)
    # 失敗必須比「已停止」更醒目：停止後仍要看得出上一輪失敗
    if status == "error" or failures >= max_failures:
        return WorkerState.TASK_FAILED

    started = hb.get("task_started_at")
    if started and status in ("running", "task_running"):
        try:
            age = (now - dt.datetime.fromisoformat(started)).total_seconds()
        except (TypeError, ValueError):
            age = None
        if age is not None and age <= task_timeout:
            return WorkerState.TASK_RUNNING

    if failures > 0:
        return WorkerState.DEGRADED
    if status == "stopped":
        return WorkerState.STOPPED
    if pid_alive is False:
        return WorkerState.NOT_STARTED
    if data_staleness_days is not None and data_staleness_days > stale_days:
        return WorkerState.DATA_STALE
    return WorkerState.RUNNING


# --------------------------------------------------------------------------
# 單一實例鎖
# --------------------------------------------------------------------------
class WorkerLock:
    """以 PID 檔作為單一實例鎖；PID 已死時可回收。"""

    def __init__(self, name: str = "worker", path: Path | None = None) -> None:
        self.name = name
        self._explicit_path = Path(path) if path is not None else None
        self.held = False

    @property
    def path(self) -> Path:
        return self._explicit_path or (_state_dir() / f"{self.name}.lock")

    def holder_pid(self) -> int | None:
        if not self.path.exists():
            return None
        try:
            return int(self.path.read_text(encoding="utf-8").strip().split()[0])
        except (OSError, ValueError, IndexError):
            return None

    def acquire(self, pid: int | None = None) -> bool:
        """嘗試取得鎖。已有存活持有者時回 False（不覆蓋）。"""
        pid = pid if pid is not None else os.getpid()
        holder = self.holder_pid()
        if holder is not None and holder != pid:
            from .ops import pid_alive
            if pid_alive(holder):
                log.error("已有 Worker 在執行（PID %s），拒絕重複啟動", holder)
                return False
            log.warning("發現殘留鎖（PID %s 已不存在），回收", holder)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            f"{pid} {dt.datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8")
        self.held = True
        return True

    def release(self) -> None:
        if self.held and self.path.exists():
            self.path.unlink()
        self.held = False


# --------------------------------------------------------------------------
# 停止旗標 / 心跳
# --------------------------------------------------------------------------
def _state_dir() -> Path:
    """狀態目錄（於呼叫時讀取模組全域，方便測試替換）。"""
    p = Path(STATE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _paths(name: str) -> tuple[Path, Path]:
    d = _state_dir()
    return d / f"{name}.stop", d / "heartbeat.json"


def stop_requested(name: str) -> bool:
    stop_file, _ = _paths(name)
    return stop_file.exists()


def request_stop(name: str) -> None:
    stop_file, _ = _paths(name)
    stop_file.write_text(dt.datetime.now().isoformat(timespec="seconds"),
                         encoding="utf-8")


def clear_stop(name: str) -> None:
    stop_file, _ = _paths(name)
    if stop_file.exists():
        stop_file.unlink()


def heartbeat_path() -> Path:
    return _paths("worker")[1]


def _read_heartbeat_raw() -> dict:
    p = heartbeat_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_heartbeat() -> dict:
    return _read_heartbeat_raw()


def write_heartbeat(**fields: Any) -> None:
    """合併寫入心跳（保留既有欄位，避免遺失 consecutive_failures 等）。"""
    payload = _read_heartbeat_raw()
    payload.update(fields)
    payload.setdefault("name", "worker")
    payload["pid"] = os.getpid()
    payload["at"] = dt.datetime.now().isoformat(timespec="seconds")
    try:
        heartbeat_path().parent.mkdir(parents=True, exist_ok=True)
        heartbeat_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:  # noqa: BLE001
        log.warning("心跳寫入失敗：%s", e)


# --------------------------------------------------------------------------
# 有上限的重試
# --------------------------------------------------------------------------
@dataclass
class TaskOutcome:
    name: str
    ok: bool
    attempts: int = 0
    error: str = ""
    duration: float = 0.0
    value: Any = None

    def label(self) -> str:
        state = "成功" if self.ok else "失敗"
        return (f"{self.name} {state}（嘗試 {self.attempts} 次，"
                f"{self.duration:.1f}s）{self.error}")


def run_task_with_retry(task: Callable[[], T], *, name: str = "task",
                        attempts: int = 3, base_delay: float = 1.0,
                        max_delay: float = 30.0,
                        sleep: Callable[[float], None] = time.sleep,
                        on_retry: Callable[[int, Exception], None] | None = None,
                        ) -> TaskOutcome:
    """執行任務並在失敗時重試（指數退避，有上限）。

    ``sleep`` 可注入，測試時不需真的等待。最後一次失敗會完整記錄，不吞例外。
    """
    start = time.monotonic()
    last_err = ""
    for i in range(max(1, attempts)):
        try:
            value = task()
            return TaskOutcome(name=name, ok=True, attempts=i + 1,
                               duration=time.monotonic() - start, value=value)
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            log.warning("任務 %s 第 %d/%d 次失敗：%s", name, i + 1, attempts, last_err)
            if on_retry is not None:
                on_retry(i + 1, e)
            if i < attempts - 1:
                sleep(min(max_delay, base_delay * (2 ** i)))
    return TaskOutcome(name=name, ok=False, attempts=max(1, attempts),
                       error=last_err, duration=time.monotonic() - start)


# --------------------------------------------------------------------------
# 排程
# --------------------------------------------------------------------------
def _today() -> str:
    return dt.date.today().isoformat()


def _should_run(force: bool, *, now: dt.datetime | None = None,
                last_ingest: str | None = None) -> tuple[bool, str]:
    """是否應執行更新。休市或今日已完成則不輪詢。"""
    if force:
        return True, "forced"
    now = now or dt.datetime.now()
    if now.weekday() >= 5:
        return False, "weekend"
    if now.hour < EOD_HOUR:
        return False, f"before_eod({EOD_HOUR}:00)"
    if last_ingest is None:
        from .data import store
        last_ingest = store.get_meta("last_ingest")
    # 以注入的 now 為準，避免測試/回放時混用真實日期
    if last_ingest == now.date().isoformat():
        return False, "already_done_today"
    return True, "eod"


def _compute_data_staleness() -> int | None:
    from .data import store
    try:
        df = store.read_sql("SELECT MAX(date) AS d FROM price")
        latest = df.iloc[0]["d"] if not df.empty else None
        if not latest:
            return None
        d = dt.datetime.fromisoformat(str(latest)[:10]).date()
        return (dt.date.today() - d).days
    except Exception as e:  # noqa: BLE001
        # 無法判定延遲時不可謊報新鮮；記錄原因並回 None（狀態顯示為未知）
        log.warning("無法計算行情延遲：%s", e)
        return None


# --------------------------------------------------------------------------
# P3：盤中行情輪詢（完全獨立於 EOD 流程，失敗**不得**影響 ingest/打分）
# --------------------------------------------------------------------------
def quote_poll_enabled() -> bool:
    """是否啟用行情輪詢：未設定來源就**完全不動作**。

    刻意不自行選一個預設來源：那會在使用者不知情的情況下對外部服務
    發送請求，也可能觸發來源條款問題。
    """
    return bool((os.environ.get(QUOTE_SOURCE_ENV) or "").strip())


def quote_poll_due(heartbeat: dict | None, *, now: dt.datetime | None = None,
                   interval: int = QUOTE_INTERVAL_S) -> bool:
    """距離上次輪詢是否已達間隔（純函式，可完整測試）。"""
    now = now or dt.datetime.now()
    hb = heartbeat or {}
    last = hb.get("quote_polled_at")
    if not last:
        return True
    try:
        last_dt = dt.datetime.fromisoformat(str(last))
    except (TypeError, ValueError):
        return True
    return (now - last_dt).total_seconds() >= max(1, int(interval))


def run_quote_cycle(*, force: bool = False, now: dt.datetime | None = None,
                    source: str | None = None,
                    max_age_s: float = QUOTE_MAX_AGE_S,
                    poller: Callable[..., Any] | None = None) -> dict:
    """執行一次行情輪詢並更新心跳。**絕不呼叫 run_scores**。

    安全規則
      - 未設定來源 → 不抓取，只記下「未設定」（不假裝成功）。
      - 非交易時段且未 force → 不抓取。
      - 來源失敗 → 心跳標為 failed；**不碰價量、不產生新訊號**。
      - 任何例外都不往外拋到 EOD 流程（行情失敗不得讓整輪失敗）。
    """
    from .data import quotes

    now = now or dt.datetime.now()
    if source is None and not quote_poll_enabled():
        res = {"status": "not_configured",
               "reason": f"未設定 {QUOTE_SOURCE_ENV}"}
        write_heartbeat(quote_polled_at=now.isoformat(timespec="seconds"),
                        quote_health=quotes.SourceHealth.UNKNOWN.value,
                        quote_health_label=quotes.SourceHealth.UNKNOWN.label,
                        quote_reason=res["reason"])
        return res
    sess = quotes.session_state(now)
    if not force and sess in (quotes.SessionState.CLOSED,
                              quotes.SessionState.WEEKEND):
        res = {"status": "skipped", "reason": f"非交易時段（{sess.label}）",
               "session": sess.value}
        write_heartbeat(quote_polled_at=now.isoformat(timespec="seconds"),
                        quote_session=sess.value,
                        quote_reason=res["reason"])
        return res
    try:
        poll_fn = poller or quotes.poll_once
        res_poll = poll_fn(source=quotes.resolve_source(source), now=now,
                           run_id=st_run_id(now), max_age_s=max_age_s)
    except Exception as e:  # noqa: BLE001
        log.warning("行情輪詢失敗（不影響 EOD 流程）：%s", e)
        write_heartbeat(quote_polled_at=now.isoformat(timespec="seconds"),
                        quote_health=quotes.SourceHealth.FAILED.value,
                        quote_health_label=quotes.SourceHealth.FAILED.label,
                        quote_reason=f"{type(e).__name__}: {e}")
        return {"status": "failed", "reason": str(e)}
    hb = res_poll.health.as_heartbeat() if res_poll.health else {}
    hb["quote_polled_at"] = now.isoformat(timespec="seconds")
    hb["quote_result"] = res_poll.label()
    if res_poll.skipped_reason:
        hb["quote_reason"] = res_poll.skipped_reason
    write_heartbeat(**hb)
    try:
        quotes.write_poll_records(res_poll)
    except Exception as e:  # noqa: BLE001
        log.warning("行情輪詢紀錄寫入失敗：%s", e)
    return {"status": "polled", "result": res_poll,
            "health": hb.get("quote_health", "")}


def st_run_id(now: dt.datetime | None = None) -> str:
    """行情輪詢用的可排序 run_id（前綴 q 以便與 ingest 區分）。"""
    now = now or dt.datetime.now()
    return "q" + now.strftime("%Y%m%dT%H%M%S")


def run_cycle(force: bool = False) -> str:
    """執行一輪：判斷是否需要更新 → ingest → 打分。回傳狀態字串。

    成敗判定改依**結構化擷取狀態**（``IngestSummary``），不再只數價量列數：
      - 全部失敗（each record 皆 failed）→ 拋錯，Worker 記為 task_failed。
      - 有部分失敗 → 回 ``partial_success``，Worker 心跳會顯示而**不**謊報全數成功。
      - 來源正常但當日確實無資料 → 回 ``success_empty``，不是失敗。
    不可只依整體退出碼推論「所有資料均成功」（§4.4）。
    """
    from .data import status as st
    from .pipeline import ingest, run_scores

    should, why = _should_run(force)
    if not should:
        log.info("略過本輪（%s）", why)
        return why

    log.info("開始資料更新（%s）", why)
    summary = ingest()
    if summary.root_all_failed():
        # 價量（根資料集）全數失敗：即使新聞或籌碼有成功也不得繼續打分，
        # 否則會用陳舊價格產生看似正常的訊號。
        log.error("本輪價量擷取全數失敗（%s），保留原有資料與時間戳",
                  summary.label())
        raise RuntimeError(f"ingest 價量全數失敗：{summary.label()}")

    # 讓部分失敗可從可觀測狀態辨識，而不是只看退出碼
    write_heartbeat(
        ingest_run_id=summary.run_id,
        ingest_status=summary.overall_status,
        ingest_status_label=st.IngestStatus(summary.overall_status).label,
        ingest_usable_records=summary.usable_count,
        ingest_failed_records=summary.failed_count,
        ingest_unknown_records=summary.unknown_count,
        ingest_detail=summary.label())

    scores = run_scores()
    log.info("打分完成：%d 列", len(scores))
    overall = summary.overall_status
    if overall == st.IngestStatus.PARTIAL_SUCCESS.value:
        log.warning("本輪有部分擷取失敗（%s）；訊號以現有可用資料計算",
                    summary.label())
    elif overall == st.IngestStatus.SUCCESS_EMPTY.value:
        log.warning("本輪來源均正常回應但無新資料（%s）", summary.label())
    return "ok" if overall == st.IngestStatus.SUCCESS_WITH_DATA.value else overall


# --------------------------------------------------------------------------
# 主迴圈
# --------------------------------------------------------------------------
def run_forever(name: str = "worker", interval: int = 1800,
                force_once: bool = False, *, attempts: int = 3,
                task_timeout: int = DEFAULT_TASK_TIMEOUT,
                max_failures: int = DEFAULT_MAX_FAILURES,
                use_lock: bool = True) -> int:
    """主迴圈。回傳程序結束碼（0 正常、2 重複啟動、3 任務失敗）。"""
    lock = WorkerLock(name)
    if use_lock and not lock.acquire():
        return 2

    try:
        clear_stop(name)
        # 偵測上次是否在任務執行中被中斷（殘留 task_started_at）
        prev = _read_heartbeat_raw()
        interrupted = bool(prev.get("task_started_at")) and \
            prev.get("status") in ("running", "task_running")
        if interrupted:
            log.warning("偵測到上次執行於任務中被中斷，將以冪等寫入安全接續")

        log.info("Worker 啟動（interval=%ss, task_timeout=%ss）", interval, task_timeout)
        write_heartbeat(status="starting", consecutive_failures=0,
                        interrupted_last_run=interrupted,
                        task_started_at=None)

        if force_once:
            outcome = _run_cycle_task(attempts=attempts)
            write_heartbeat(
                status="running" if outcome.ok else "error",
                last_cycle_ok=outcome.ok,
                last_cycle_status=_cycle_status(outcome),
                consecutive_failures=0 if outcome.ok else 1,
                last_result=outcome.label(),
                failure_reason="" if outcome.ok else outcome.error,
                task_started_at=None, last_run=_today())
            if quote_poll_enabled():
                # 行情輪詢為獨立、非致命的一步：失敗不得改變 EOD 的成敗
                run_quote_cycle()
            return 0 if outcome.ok else 3

        failures = 0
        while not stop_requested(name):
            outcome = _run_cycle_task(attempts=attempts)
            failures = 0 if outcome.ok else failures + 1
            write_heartbeat(
                status="running" if outcome.ok else "error",
                last_cycle_ok=outcome.ok,
                last_cycle_status=_cycle_status(outcome),
                consecutive_failures=failures,
                last_result=outcome.label(),
                failure_reason="" if outcome.ok else outcome.error,
                task_started_at=None,
                last_success=_today() if outcome.ok else prev.get("last_success"),
                data_staleness_days=_compute_data_staleness())
            if failures >= max_failures:
                log.error("連續失敗 %d 次，進入降級狀態（持續重試但不中斷）", failures)

            slept = 0.0
            while slept < interval and not stop_requested(name):
                time.sleep(SLEEP_CHUNK)
                slept += SLEEP_CHUNK
                # P3：盤中行情輪詢（未設定來源／非交易時段／未到間隔則完全不動作）
                if quote_poll_enabled() and quote_poll_due(read_heartbeat()):
                    run_quote_cycle()
    finally:
        write_heartbeat(status="stopped", task_started_at=None)
        clear_stop(name)
        if use_lock:
            lock.release()
        log.info("Worker 已安全停止")
    return 0


def _cycle_status(outcome: TaskOutcome) -> str:
    """取得本輪 ingest 的結構化狀態（供心跳顯示部分失敗）。

    ``run_cycle`` 回傳字串；測試常以替身覆寫，故非字串時回空字串。
    """
    return outcome.value if isinstance(outcome.value, str) else ""


def _run_cycle_task(attempts: int = 3) -> TaskOutcome:
    """包裝 run_cycle：標記任務開始／結束，並在有上限的重試下執行。"""
    write_heartbeat(status="task_running",
                    task_started_at=dt.datetime.now().isoformat(timespec="seconds"))
    try:
        outcome = run_task_with_retry(run_cycle, name="run_cycle", attempts=attempts)
    finally:
        write_heartbeat(task_started_at=None, status="running")
    return outcome


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="台股本地監控 Worker")
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--once", action="store_true", help="立即執行一次後結束")
    ap.add_argument("--force", action="store_true", help="忽略時間窗，強制執行一次")
    ap.add_argument("--attempts", type=int, default=3, help="單輪任務重試次數上限")
    args = ap.parse_args(argv)
    if args.once or args.force:
        return run_forever(interval=args.interval, force_once=True,
                           attempts=args.attempts)
    return run_forever(interval=args.interval, attempts=args.attempts)


if __name__ == "__main__":
    raise SystemExit(main())
