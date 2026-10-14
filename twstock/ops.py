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


def _cleanup_state(name: str) -> None:
    """行程確實結束後清除殘留狀態（PID 檔、停止旗標、Worker 鎖）。

    不清理會造成假狀態：例如停止旗標還在會讓健康狀態一直顯示「正在停止」。
    """
    _pid_file(name).unlink(missing_ok=True)
    (STATE_DIR / f"{name}.stop").unlink(missing_ok=True)
    if name == "worker":
        lock = STATE_DIR / f"{name}.lock"
        holder = None
        if lock.exists():
            try:
                holder = int(lock.read_text(encoding="utf-8").strip().split()[0])
            except (OSError, ValueError, IndexError):
                holder = None
        if holder is None or not pid_alive(holder):
            lock.unlink(missing_ok=True)


def stop(name: str, timeout: float = 20.0) -> bool:
    """安全停止。

    策略依元件而不同：
      - **worker**：先寫停止旗標，讓它在下一個檢查點自行收尾（真實的優雅停止）；
        只有在旗標未被理會時才送可攔截訊號，最後才強制終止。
      - **dashboard**（uvicorn）：無法輪詢旗標，直接送可攔截訊號。

    回傳值代表行程是否已結束，且**區分「自行收尾」與「被強制終止」**，
    不把強制終止回報成安全停止。
    """
    from .worker import request_stop
    pid = _read_pid(name)
    if pid is None:
        _cleanup_state(name)
        print(f"[{name}] 未在執行")
        return True

    cooperative = (name == "worker")
    request_stop(name)                       # worker 會於檢查點收尾
    if not cooperative:
        _signal_stop(pid)                    # dashboard 攔截訊號優雅關閉

    graceful = False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            graceful = True
            break
        time.sleep(0.5)

    if not graceful:
        _signal_stop(pid)                    # 再試可攔截訊號
        deadline = time.time() + min(5.0, timeout)
        while time.time() < deadline:
            if not pid_alive(pid):
                graceful = True
                break
            time.sleep(0.3)

    if not graceful:
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

    _cleanup_state(name)
    print(f"[{name}] {'已安全停止' if graceful else '已強制停止'}（PID {pid}）")
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
        # 擷取狀態治理：整體狀態與最近一輪的各資料集狀態（可辨識部分失敗）
        out["last_ingest_run_id"] = store.get_meta("last_ingest_run_id")
        out["last_ingest_status"] = store.get_meta("last_ingest_status")
        out["last_experts_status"] = store.get_meta("last_experts_status")
        try:
            from .data import report
            out["ingest_records"] = report.ingest_status_summary().to_dict(
                orient="records")
            out["news_fetch_rates"] = report.fetch_rates()
        except Exception as e:  # noqa: BLE001
            out["ingest_records"] = []
            out["ingest_records_error"] = str(e)
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

    wp = _read_pid("worker")
    walive = bool(wp and pid_alive(wp))
    try:
        from .worker import derive_state, stop_requested
        wstate = derive_state(
            hb, stop_flag=stop_requested("worker"), pid_alive=walive,
            data_staleness_days=out.get("data_staleness_days"))
        state_value, state_label = wstate.value, wstate.label
    except Exception as e:  # noqa: BLE001
        state_value, state_label = "unknown", f"無法判定（{e}）"
    out["worker"] = {
        "pid": wp,
        "alive": walive,
        "state": state_value,
        "state_label": state_label,
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
    if h.get("last_ingest_status"):
        from .data.status import IngestStatus
        try:
            st_label = IngestStatus(h["last_ingest_status"]).label
        except ValueError:
            st_label = str(h["last_ingest_status"])
        print(f"  擷取狀態      {st_label}"
              f"（run {h.get('last_ingest_run_id') or '-'}）")
    for row in h.get("ingest_records") or []:
        print(f"    - {row.get('dataset','?'):<18} {row.get('status_label','?')}"
              f" × {row.get('n','?')}")
    rates = h.get("news_fetch_rates") or {}
    if rates:
        def _p(v):
            return "—" if v is None else f"{v:.0%}"
        print(f"  新聞抓取      成功 {_p(rates.get('success_rate'))}"
              f"／失敗 {_p(rates.get('failure_rate'))}"
              f"／不可追溯 {_p(rates.get('untraceable_rate'))}"
              f"（紀錄 {rates.get('n_records', 0)}）")
    ml = h.get("ml", {})
    print(f"  ML 模型       {'可用' if ml.get('available') else '停用'} {ml.get('reason','')}")
    ks = h.get("kill_switch", {})
    print(f"  Kill Switch   {'已啟用 ' + ks.get('reason','') if ks.get('engaged') else '未啟用'}")
    w = h.get("worker", {})
    hbat = (w.get("heartbeat") or {}).get("at", "-")
    last_result = (w.get("heartbeat") or {}).get("last_result", "")
    print(f"  Worker        {w.get('state_label', '未執行')}  最後心跳 {hbat}")
    if last_result:
        print(f"                {last_result}")
    d = h.get("dashboard", {})
    print(f"  儀表板        {'執行中' if d.get('alive') else '未執行'}"
          f"{'  → http://127.0.0.1:8000' if d.get('alive') else ''}")
    stale = h.get("data_staleness_days")
    if stale is not None and stale > 3:
        print(f"  ⚠️ 行情資料可能過期（延遲 {stale} 天），請執行 ingest")
