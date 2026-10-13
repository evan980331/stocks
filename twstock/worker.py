"""本地監控 Worker：可持續執行、可觀測、可安全停止、可重啟恢復。

特性：
  - **安全停止**：每輪與睡眠期間檢查 `state/<name>.stop` 旗標，收到後完成當前工作才退出。
  - **心跳**：每輪寫入 `state/heartbeat.json`，供 `status`/`health` 判斷存活與最後成功時間。
  - **崩潰恢復**：重啟後先讀取 `last_ingest`/心跳，不重複抓取已完成的工作（冪等寫入）。
  - **不做交易**：本 Worker 僅做資料更新、特徵、打分與健康記錄；下單屬後續階段。

排程策略（延遲/EOD 資料）：
  - 預設僅在**台股收盤後**（本地時間 16:00 之後）且當日尚未成功 ingest 時執行完整流程。
  - ``--force`` 可忽略時間窗立即執行一次（供手動/測試）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

from .config import CONFIG, DATA_DIR
from .data import store

log = logging.getLogger(__name__)

STATE_DIR = Path(DATA_DIR) / "state"
EOD_HOUR = 16          # 收盤後執行時間（本地時間）
SLEEP_CHUNK = 2.0      # 睡眠分段，以便即時回應停止旗標


def _paths(name: str) -> tuple[Path, Path]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"{name}.stop", STATE_DIR / "heartbeat.json"


def stop_requested(name: str) -> bool:
    stop_file, _ = _paths(name)
    return stop_file.exists()


def request_stop(name: str) -> None:
    stop_file, _ = _paths(name)
    stop_file.write_text(dt.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def clear_stop(name: str) -> None:
    stop_file, _ = _paths(name)
    if stop_file.exists():
        stop_file.unlink()


def _write_heartbeat(name: str, **extra) -> None:
    _, hb = _paths(name)
    payload = {
        "name": name,
        "pid": _pid(),
        "at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(extra)
    try:
        hb.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:  # noqa: BLE001
        log.warning("心跳寫入失敗: %s", e)


def read_heartbeat() -> dict:
    _, hb = _paths("worker")
    if not hb.exists():
        return {}
    try:
        return json.loads(hb.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _pid() -> int:
    import os
    return os.getpid()


def _today() -> str:
    return dt.date.today().isoformat()


def _should_run(force: bool) -> tuple[bool, str]:
    if force:
        return True, "forced"
    now = dt.datetime.now()
    if now.weekday() >= 5:
        return False, "weekend"
    if now.hour < EOD_HOUR:
        return False, f"before_eod({EOD_HOUR}:00)"
    if store.get_meta("last_ingest") == _today():
        return False, "already_done_today"
    return True, "eod"


def run_cycle(force: bool = False) -> str:
    """執行一輪：判斷是否需要更新 → ingest → 打分。回傳狀態字串。"""
    from .pipeline import ingest, run_scores

    should, why = _should_run(force)
    if not should:
        log.info("略過本輪（%s）", why)
        return why

    log.info("開始資料更新（%s）", why)
    counts = ingest()
    ok = sum(1 for v in counts.values() if v and v > 0)
    if ok == 0:
        log.error("本輪無任何價量更新成功，保留原有資料與時間戳")
        return "ingest_failed"

    scores = run_scores()
    log.info("打分完成：%d 列", len(scores))
    return "ok"


def run_forever(name: str = "worker", interval: int = 1800,
                force_once: bool = False) -> int:
    """主迴圈。回傳程序結束碼。"""
    clear_stop(name)
    log.info("Worker 啟動（interval=%ss）", interval)
    _write_heartbeat(name, status="starting")
    try:
        if force_once:
            status = run_cycle(force=True)
            _write_heartbeat(name, status=status, last_run=_today())
            return 0

        while not stop_requested(name):
            try:
                status = run_cycle()
                _write_heartbeat(name, status=status, last_run=_today())
            except Exception as e:  # noqa: BLE001
                log.exception("本輪執行失敗，將於下一輪重試")
                _write_heartbeat(name, status="error", error=str(e))

            slept = 0.0
            while slept < interval and not stop_requested(name):
                time.sleep(SLEEP_CHUNK)
                slept += SLEEP_CHUNK
    finally:
        _write_heartbeat(name, status="stopped")
        clear_stop(name)
        log.info("Worker 已安全停止")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="台股本地監控 Worker")
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--once", action="store_true", help="立即執行一次後結束")
    ap.add_argument("--force", action="store_true", help="忽略時間窗，強制執行一次")
    args = ap.parse_args(argv)
    if args.once or args.force:
        return run_forever(interval=args.interval, force_once=True)
    return run_forever(interval=args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
