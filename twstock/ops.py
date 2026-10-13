"""本地服務生命週期管理：start / stop / status / health。

設計：
  - 每個元件一個 PID 檔（``data_store/state/<name>.pid``）。
  - **安全停止**：先寫入停止旗標讓行程於檢查點自行收尾，逾時才強制終止。
  - 不使用任何雲端/反向代理；所有服務僅綁定 127.0.0.1。
  - 不因強制終止而遺失已保存的交易狀態（DB 寫入為交易式且冪等）。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import DATA_DIR, DB_PATH

log = logging.getLogger(__name__)

STATE_DIR = Path(DATA_DIR) / "state"
COMPONENTS = ("dashboard", "worker")
LOOPBACK = "127.0.0.1"


def _ensure_state() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


def _pid_file(name: str) -> Path:
    return _ensure_state() / f"{name}.pid"


def _log_file(name: str) -> Path:
    return _ensure_state() / f"{name}.log"


def pid_alive(pid: int) -> bool:
    """跨平台存活檢查（Windows 不使用 os.kill，以免誤殺）。"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return False
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def _read_pid(name: str) -> int | None:
    p = _pid_file(name)
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _command(name: str, extra: list[str]) -> list[str]:
    if name == "dashboard":
        return [sys.executable, "-m", "twstock.cli", "dashboard",
                "--host", LOOPBACK, "--port", str(extra_port(extra))]
    if name == "worker":
        return [sys.executable, "-m", "twstock.worker", *extra]
    raise ValueError(f"未知元件：{name}")


def extra_port(extra: list[str]) -> int:
    if "--port" in extra:
        return int(extra[extra.index("--port") + 1])
    return 8000


def start(name: str, extra: list[str] | None = None) -> int:
    """啟動元件（detached）。回傳 PID；已在執行則回傳既有 PID。"""
    if name not in COMPONENTS:
        raise ValueError(f"未知元件：{name}（可選 {COMPONENTS}）")
    extra = extra or []
    existing = _read_pid(name)
    if existing and pid_alive(existing):
        print(f"[{name}] 已在執行（PID {existing}）")
        return existing

    # 清除殘留的停止旗標，讓新行程可正常運作
    from .worker import clear_stop
    clear_stop(name)

    logf = _log_file(name)
    cmd = _command(name, extra)
    creationflags = 0
    if os.name == "nt":
        creationflags = (subprocess.CREATE_NEW_PROCESS_GROUP
                         | getattr(subprocess, "DETACHED_PROCESS", 0))
    with open(logf, "ab") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf,
                                stdin=subprocess.DEVNULL,
                                creationflags=creationflags,
                                cwd=str(Path(__file__).resolve().parent.parent))
    _pid_file(name).write_text(str(proc.pid), encoding="utf-8")
    print(f"[{name}] 已啟動 PID {proc.pid}；日誌：{logf}")
    return proc.pid


def _signal_stop(pid: int) -> bool:
    """送出可攔截的終止訊號（非強制），讓行程有機會收尾。"""
    import signal
    try:
        if os.name == "nt":
            # 與 start() 的 CREATE_NEW_PROCESS_GROUP 搭配，觸發 KeyboardInterrupt/SIGBREAK
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            os.kill(pid, signal.SIGTERM)
        return True
    except Exception as e:  # noqa: BLE001
        # Windows 上 CTRL_BREAK_EVENT 可能回報錯誤但仍已送出；僅在逾時後才強制終止
        log.debug("送出終止訊號回報異常（%s）", e)
        return False


def stop(name: str, timeout: float = 20.0) -> bool:
    """安全停止：先送停止旗標與可攔截訊號，逾時才強制終止。"""
    from .worker import request_stop
    pid = _read_pid(name)
    if pid is None:
        print(f"[{name}] 未在執行")
        _pid_file(name).unlink(missing_ok=True)
        return True

    request_stop(name)      # worker 會於檢查點收尾
    _signal_stop(pid)       # dashboard(uvicorn) 會攔截此訊號優雅關閉
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            _pid_file(name).unlink(missing_ok=True)
            print(f"[{name}] 已安全停止（PID {pid}）")
            return True
        time.sleep(0.5)

    log.warning("[%s] 未於 %.0fs 內自行結束，強制終止 PID %s", name, timeout, pid)
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    time.sleep(1.0)
    _pid_file(name).unlink(missing_ok=True)
    print(f"[{name}] 已強制停止（PID {pid}）")
    return not pid_alive(pid)


def stop_all() -> None:
    for name in COMPONENTS:
        if _read_pid(name):
            stop(name)


def status() -> list[dict]:
    rows = []
    for name in COMPONENTS:
        pid = _read_pid(name)
        alive = bool(pid and pid_alive(pid))
        rows.append({"name": name, "pid": pid, "alive": alive})
    return rows


def print_status() -> None:
    for r in status():
        state = f"執行中 (PID {r['pid']})" if r["alive"] else "未執行"
        print(f"  {r['name']:<10} {state}")


def health() -> dict:
    """系統健康狀態（全部來自真實資料，不用假資料）。"""
    from .data import store

    now = dt.datetime.now()
    hb = {}
    hb_path = _ensure_state() / "heartbeat.json"
    if hb_path.exists():
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            hb = {}

    out: dict = {
        "time": now.isoformat(timespec="seconds"),
        "db_path": str(DB_PATH),
        "db_exists": Path(DB_PATH).exists(),
        "mode": _mode(),
    }

    try:
        price_max = store.read_sql("SELECT MAX(date) AS d FROM price")
        latest = price_max.iloc[0]["d"] if not price_max.empty else None
        out["latest_price_date"] = str(latest) if latest else None
        if latest:
            d = dt.datetime.fromisoformat(str(latest)[:10]).date()
            out["data_staleness_days"] = (dt.date.today() - d).days
        else:
            out["data_staleness_days"] = None
        sig = store.read_sql("SELECT MAX(date) AS d, COUNT(*) AS n FROM signals")
        if not sig.empty:
            out["latest_signal_date"] = str(sig.iloc[0]["d"]) if sig.iloc[0]["d"] else None
            out["signal_rows"] = int(sig.iloc[0]["n"])
        out["last_ingest"] = store.get_meta("last_ingest")
        out["last_ingest_failed_ids"] = store.get_meta("last_ingest_failed_ids") or ""
    except Exception as e:  # noqa: BLE001
        out["db_error"] = str(e)

    try:
        from . import ml_model
        out["ml"] = ml_model.ml_status()
    except Exception as e:  # noqa: BLE001
        out["ml"] = {"available": False, "reason": str(e)}

    try:
        from .risk import KillSwitch
        ks = KillSwitch()
        out["kill_switch"] = {"engaged": ks.engaged(), "reason": ks.reason()}
    except Exception as e:  # noqa: BLE001
        out["kill_switch"] = {"engaged": False, "reason": "", "error": str(e)}

    out["worker"] = {
        "pid": _read_pid("worker"),
        "alive": bool(_read_pid("worker") and pid_alive(_read_pid("worker") or 0)),
        "heartbeat": hb,
    }
    out["dashboard"] = {
        "pid": _read_pid("dashboard"),
        "alive": bool(_read_pid("dashboard") and pid_alive(_read_pid("dashboard") or 0)),
    }
    return out


def _mode() -> str:
    try:
        from .risk import current_mode
        return current_mode().value
    except Exception:  # noqa: BLE001
        return "paper"


def print_health() -> None:
    h = health()
    print("=== 系統健康狀態 ===")
    print(f"  時間          {h['time']}")
    print(f"  執行模式      {h['mode']}")
    print(f"  資料庫        {h['db_path']}（{'存在' if h['db_exists'] else '不存在'}）")
    print(f"  最新行情日期  {h.get('latest_price_date')}"
          f"（延遲 {h.get('data_staleness_days')} 天）")
    print(f"  最新訊號日期  {h.get('latest_signal_date')}  訊號列數 {h.get('signal_rows')}")
    print(f"  最後 ingest   {h.get('last_ingest')}"
          f"{'  失敗標的: ' + h['last_ingest_failed_ids'] if h.get('last_ingest_failed_ids') else ''}")
    ml = h.get("ml", {})
    print(f"  ML 模型       {'可用' if ml.get('available') else '停用'} {ml.get('reason','')}")
    ks = h.get("kill_switch", {})
    print(f"  Kill Switch   {'已啟用 ' + ks.get('reason','') if ks.get('engaged') else '未啟用'}")
    w = h.get("worker", {})
    wstate = "執行中" if w.get("alive") else "未執行"
    hbat = (w.get("heartbeat") or {}).get("at", "-")
    print(f"  Worker        {wstate}  最後心跳 {hbat}")
    d = h.get("dashboard", {})
    print(f"  儀表板        {'執行中' if d.get('alive') else '未執行'}"
          f"{'  → http://127.0.0.1:8000' if d.get('alive') else ''}")
    stale = h.get("data_staleness_days")
    if stale is not None and stale > 3:
        print(f"  ⚠️ 行情資料可能過期（延遲 {stale} 天），請執行 ingest")
