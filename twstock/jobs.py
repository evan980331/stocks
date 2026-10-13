"""耗時任務的子程序隔離：架構評估用的**最小原型**（P3.2，未全面遷移）。

問題背景
  目前的 Worker 在**同一個行程**內執行耗時任務，因此任務逾時只能「事後偵測」，
  無法真正中止已在跑的工作（見 worker.py 的說明與 B15）。本模組把「單一耗時任務」
  移出父行程，讓逾時變成可強制中止，並讓崩潰狀態可被觀察與恢復。

責任劃分
  父程序（``JobRunner``，在 Worker 內使用）
    - 持有單一行實例鎖、排程節奏、心跳與健康狀態（由 worker.py 負責）。
    - 只負責：啟動子程序、等待、逾時處理、記錄結果。**不執行任務本身**。
    - 停止時**只操作自己啟動的 PID／handle**，絕不使用依名稱的批次終止，
      因此不會誤殺其他任務或使用者的程序。
  子程序（``python -m twstock.jobs --task ...``）
    - 只負責執行一個任務，並把結果以 JSON 原子寫入 ``--result``。
    - 支援合作式取消（檢查 ``--cancel-file``），讓「優雅停止」與「強制終止」可區分。

停止策略（三段式，順序固定；不把強制終止冒充成安全停止）
  1. **合作式取消**：建立取消檔，子程序在檢查點自行收尾、寫出 ``cancelled`` 結果。
     這是唯一跨平台且真正「優雅」的停止方式（唯一能讓子程序有機會收尾）。
  2. **可攔截訊號（僅 POSIX）**：SIGTERM，子程序可捕捉後收尾。
  3. **強制終止**：只針對本子程序的 PID 行程樹
     （Windows：``taskkill /PID <pid> /T /F``；POSIX：SIGKILL），並標示 ``forced``。

為何**不使用** Windows 的 ``CTRL_BREAK_EVENT``？（實測結論，非臆測）
  對以 ``CREATE_NEW_PROCESS_GROUP`` 建立的子程序送出 CTRL_BREAK_EVENT 時，
  子程序會以 ``0xC000013A``（STATUS_CONTROL_C_EXIT）**直接結束**，不會抛出一層
  可攔截的 ``KeyboardInterrupt``；連已明確 ``except KeyboardInterrupt`` 的任務也
  照樣被殺（實測：尚在 SQLite 交易中的任務被中斷訊號硬殺）。既然它做不到「讓
  子程序有機會收尾」，就不能拿來當優雅停止；把它當成優雅停止就是製造假的安全停止。
  ``CREATE_NEW_PROCESS_GROUP`` 仍保留：讓子程序不因父程序主控台的 Ctrl+C 而
  意外死亡，父程序才能先做完自己的清理再收掉它。

SQLite 寫入中途被殺
  ``sqlite-atomic`` 任務在交易中寫入後刻意停住；由測試在 commit 前強制終止，
  再驗證資料庫可讀且**沒有留下半筆資料**（依賴 SQLite 的交易回復日誌）。

指令（子程序入口）：
  python -m twstock.jobs --task ok --result /tmp/r.json --log /tmp/l.txt
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLL_INTERVAL = 0.05

# 狀態值（父程序回報用）
OK = "ok"
FAILED = "failed"
CRASHED = "crashed"
CANCELLED = "cancelled"
TIMEOUT = "timeout"

STATUS_LABEL = {
    OK: "任務成功",
    FAILED: "任務失敗（子程序回報）",
    CRASHED: "子程序崩潰（無結果檔）",
    CANCELLED: "任務已取消",
    TIMEOUT: "任務逾時",
}


# --------------------------------------------------------------------------
# 子程序端：任務註冊表
# --------------------------------------------------------------------------
TASKS: dict[str, Callable[[argparse.Namespace], dict]] = {}


def task(name: str):
    def deco(fn: Callable[[argparse.Namespace], dict]):
        TASKS[name] = fn
        return fn
    return deco


def _write_result(path: str | Path, payload: dict) -> None:
    """原子寫入結果檔（先寫暫存再 rename），避免父程序讀到半截 JSON。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _cancelled(args: argparse.Namespace) -> bool:
    cf = getattr(args, "cancel_file", "")
    return bool(cf) and Path(cf).exists()


@task("ok")
def _t_ok(args) -> dict:
    return {"status": OK, "note": "立即成功"}


@task("sleep")
def _t_sleep(args) -> dict:
    """單純睡 N 秒（可被中斷訊號打斷）。"""
    end = time.monotonic() + float(getattr(args, "seconds", 0.0))
    while time.monotonic() < end:
        time.sleep(0.02)
    return {"status": OK}


@task("stubborn")
def _t_stubborn(args) -> dict:
    """忽略中斷訊號的耗時工作（模擬無法安全打斷的長時間原生呼叫）。

    真實情境：卡住的網路請求、無法取消的第三方函式庫。這類任務只能靠
    強制終止，也正是子程序隔離存在的理由。
    """
    end = time.monotonic() + float(getattr(args, "seconds", 0.0))
    while time.monotonic() < end:
        try:
            time.sleep(0.02)
        except KeyboardInterrupt:      # 故意不理會
            continue
    return {"status": OK}


@task("cooperative")
def _t_cooperative(args) -> dict:
    """可被合作式取消的長任務：輪詢取消檔，收到即優雅收尾。"""
    end = time.monotonic() + float(getattr(args, "seconds", 0.0))
    while time.monotonic() < end:
        if _cancelled(args):
            return {"status": CANCELLED, "note": "收到取消要求，已自行收尾"}
        time.sleep(0.02)
    return {"status": OK}


@task("boom")
def _t_boom(args) -> dict:
    raise RuntimeError("模擬任務失敗（測試用）")


@task("crash")
def _t_crash(args) -> dict:
    """直接結束行程且不寫結果檔（模擬崩潰／被硬殺）。"""
    os._exit(9)


@task("sqlite-atomic")
def _t_sqlite_atomic(args) -> dict:
    """在交易中寫入後停住，用來驗證中途被殺不會留下半筆資料。"""
    import sqlite3

    db = str(getattr(args, "db", ""))
    rows = int(getattr(args, "rows", 0))
    hold = float(getattr(args, "hold", 0.0))
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS probe ("
                     "k TEXT PRIMARY KEY, v INTEGER)")
        conn.commit()
        conn.execute("BEGIN")
        for i in range(rows):
            conn.execute("INSERT OR REPLACE INTO probe VALUES (?, ?)",
                         (f"row{i}", i))
        if hold:
            # 仍在交易中，而且刻意**忽略中斷訊號**（模擬無法安全打斷的長時間
            # 寫入）。測試因此一定走到強制終止，真正驗證「被硬殺不留半筆資料」。
            end = time.monotonic() + hold
            while time.monotonic() < end:
                try:
                    time.sleep(0.02)
                except KeyboardInterrupt:
                    continue
        conn.commit()
    finally:
        conn.close()
    return {"status": OK, "rows": rows}


# --------------------------------------------------------------------------
# 子程序入口
# --------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="單一耗時任務的子程序入口")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--result", required=True, help="結果 JSON 輸出路徑")
    ap.add_argument("--cancel-file", default="", help="出現此檔即要求優雅取消")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--db", default="")
    ap.add_argument("--hold", type=float, default=0.0)
    return ap


def child_main(argv: list[str] | None = None) -> int:
    """子程序進入點。回傳 0 成功、4 未知任務、5 任務失敗。"""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_argparser().parse_args(argv)
    fn = TASKS.get(args.task)
    if fn is None:
        _write_result(args.result, {"status": FAILED,
                                    "error": f"未知任務 {args.task}"})
        return 4
    try:
        payload = fn(args) or {}
    except Exception as e:  # noqa: BLE001
        # 明確記錄失敗，不吞例外、也不冒充成功
        _write_result(args.result, {"status": FAILED,
                                    "error": f"{type(e).__name__}: {e}"})
        return 5
    payload.setdefault("status", OK)
    _write_result(args.result, payload)
    return 0


# --------------------------------------------------------------------------
# 父程序端
# --------------------------------------------------------------------------
@dataclass
class JobOutcome:
    name: str
    task: str
    status: str                              # ok / failed / crashed / cancelled / timeout
    forced: bool = False
    exit_code: int | None = None
    pid: int | None = None
    duration: float = 0.0
    result: dict = field(default_factory=dict)
    error: str = ""
    log_path: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK

    def label(self) -> str:
        base = STATUS_LABEL.get(self.status, self.status)
        tag = "（強制終止）" if self.forced else ""
        return (f"{self.name}/{self.task} {base}{tag}"
                f"｜exit={self.exit_code}｜{self.duration:.2f}s"
                + (f"｜{self.error}" if self.error else ""))


class JobRunner:
    """以子程序執行單一耗時任務，支援逾時與可區分的停止方式。

    只會操作自己啟動的 PID／Popen handle，不做任何依名稱的程序終止。
    """

    def __init__(self, *, state_dir: Path | None = None,
                 log_dir: Path | None = None,
                 python: str | None = None,
                 cwd: Path | None = None) -> None:
        self.state_dir = Path(state_dir or (PROJECT_ROOT / "data_store" / "state"))
        self.log_dir = Path(log_dir or self.state_dir)
        self.python = python or sys.executable
        self.cwd = Path(cwd or PROJECT_ROOT)
        self._proc: subprocess.Popen | None = None
        self._cancel_path: Path | None = None
        self._own_group: bool = False
        self.pid: int | None = None

    # ---- 內部工具 ----
    def _creation_flags(self) -> int:
        if os.name == "nt":
            # 新程序群組：必需的，才能只對這個子程序送 CTRL_BREAK_EVENT
            return subprocess.CREATE_NEW_PROCESS_GROUP
        return 0

    def _interrupt(self) -> None:
        """送出可攔截訊號（**僅 POSIX**），只影響本子程序。

        Windows 不送出任何訊號：CTRL_BREAK_EVENT 是立即終止而非可攔截中斷
        （見模組 docstring 的實測結論），送了就等於假的安全停止；
        此平台直接由後續的強制終止步驟處理。
        """
        proc = self._proc
        if proc is None or proc.pid is None or proc.poll() is not None:
            return
        if os.name == "nt":
            log.debug("Windows 不送中斷訊號（CTRL_BREAK_EVENT 等同立即終止）")
            return
        try:
            proc.terminate()
        except Exception as e:  # noqa: BLE001
            log.debug("送出 SIGTERM 失敗（%s）；將於 grace 後強制終止", e)

    def _spawn(self, name: str, task: str,
               task_args: dict[str, Any]) -> tuple[Path, Path, Path]:
        """啟動子程序（不等待），回傳 (log, result, cancel) 路徑。

        一定建立**獨立的程序群組**（Windows），否則中斷訊號會被送到整個
        主控台程序群組，可能誤殺父程序或不相關的行程。
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{name}.log"
        result_path = self.state_dir / f"{name}.result.json"
        cancel_path = self.state_dir / f"{name}.cancel"
        cancel_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
        self._cancel_path = cancel_path

        argv = [self.python, "-m", "twstock.jobs", "--task", task,
                "--result", str(result_path), "--cancel-file", str(cancel_path)]
        for k, v in task_args.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]

        flags = self._creation_flags()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as lf:
            proc = subprocess.Popen(
                argv, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL,
                creationflags=flags, cwd=str(self.cwd))
        self._proc = proc
        self.pid = proc.pid
        # 只有在確實建立了獨立程序群組時，才允許送 CTRL_BREAK_EVENT
        self._own_group = bool(
            flags & getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        log.info("啟動子程序 %s (task=%s pid=%s, 獨立群組=%s)",
                 name, task, proc.pid, self._own_group)
        return log_path, result_path, cancel_path

    def _force(self) -> bool:
        """強制終止**本子程序**（Windows 含其行程樹）。回傳是否已發出。"""
        proc = self._proc
        if proc is None or proc.pid is None or proc.poll() is not None:
            return False
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, check=False)
                return True
            except OSError as e:
                log.warning("taskkill 失敗（%s）", e)
        try:
            proc.kill()
            return True
        except OSError as e:
            log.warning("kill 失敗（%s）", e)
        return False

    def _wait(self, seconds: float) -> bool:
        """等待至多 seconds 秒；回傳是否已結束。"""
        proc = self._proc
        if proc is None:
            return True
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return True
            time.sleep(POLL_INTERVAL)
        return proc.poll() is not None

    @staticmethod
    def _contact_ok(pid: int) -> bool:
        """父程序可用來確認子程序是否仍存活（與 ops.pid_alive 一致的語意）。"""
        from .ops import pid_alive
        return pid_alive(pid)

    # ---- 對外 ----
    def run(self, name: str, *, task: str, timeout: float = 60.0,
            grace: float = 5.0, **task_args: Any) -> JobOutcome:
        """啟動子程序並等待；逾時則先優雅、再強制終止。

        ``task_args`` 會轉為命令列參數（``hold_seconds`` → ``--hold-seconds``）。
        """
        start = time.monotonic()
        log_path, result_path, cancel_path = self._spawn(name, task, task_args)
        proc = self._proc
        assert proc is not None

        finished = self._wait(timeout)
        forced = False
        if not finished:
            log.warning("任務 %s 逾時（> %.1fs），先要求優雅停止", name, timeout)
            # 1) 合作式取消：跨平台且行為確定（子程序在檢查點自行收尾）
            cancel_path.write_text(
                dt.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
            coop_wait = min(max(grace * 0.5, 0.2), 2.0)
            if not self._wait(coop_wait):
                # 2) 可攔截訊號（僅 POSIX；Windows 為 no-op）
                self._interrupt()
                if not self._wait(max(0.0, grace - coop_wait)):
                    # 3) 強制終止；明確標示為 forced，不冒充安全停止
                    log.error("任務 %s 未於 %.1fs 內收尾，強制終止 PID %s",
                              name, grace, proc.pid)
                    forced = self._force()
                    self._wait(max(3.0, grace))

        duration = time.monotonic() - start
        exit_code = proc.poll()
        result: dict = {}
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                log.warning("讀取結果檔失敗（%s）", e)

        status, error = self._classify(
            still_running=proc.poll() is None,
            interrupted=not finished, exit_code=exit_code, result=result)
        outcome = JobOutcome(name=name, task=task, status=status, forced=forced,
                             exit_code=exit_code, pid=proc.pid,
                             duration=duration, result=result,
                             error=error or str(result.get("error", "")),
                             log_path=str(log_path))
        self._proc = None
        self.pid = None
        return outcome

    @staticmethod
    def _classify(*, still_running: bool, interrupted: bool,
                  exit_code: int | None, result: dict) -> tuple[str, str]:
        """判定狀態。

        ``still_running``：介入後程序是否仍存活（不臆測成功）。
        ``interrupted``：是否已超過期限（即「這次停止是我們要求的」）。
        重點：只要有結果檔就要依它判定，不能因為超過期限就把
        「子程序已經自行收尾並回報取消」誤記成逾時；反之，逾時且沒有
        乾淨結果時也絕不冒充成功。
        """
        if still_running:
            return TIMEOUT, "介入後程序仍存活，無法確認已停止"
        r_status = str(result.get("status", "")).lower()
        code = exit_code
        if r_status == CANCELLED:
            return CANCELLED, ""
        if interrupted:
            if r_status == OK and code == 0:
                return OK, "完成但超過期限"
            return TIMEOUT, (str(result.get("error", ""))
                             or f"逾時後被終止（exit={code}）")
        if code == 0:
            if r_status == FAILED:
                return FAILED, str(result.get("error", ""))
            if r_status == OK:
                return OK, ""
            # 退出碼 0 但沒有結果檔 → 不可當成成功
            return CRASHED, "退出碼 0 但缺少結果檔（無法確認完成）"
        if not result:
            return CRASHED, f"子程序結束（exit={code}）且未留下結果檔"
        return FAILED, str(result.get("error", f"exit={code}"))

    def stop(self) -> bool:
        """中止**本 runner 啟動的**任務（先優雅、再強制）。回傳是否已結束。"""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return True
        if self._cancel_path is not None:
            try:
                self._cancel_path.write_text(
                    dt.datetime.now().isoformat(timespec="seconds"),
                    encoding="utf-8")
            except OSError:
                pass
        if self._wait(3.0):
            return True
        self._interrupt()
        if self._wait(2.0):
            return True
        self._force()
        return self._wait(5.0)


def main(argv: list[str] | None = None) -> int:
    """直接執行時等同子程序入口（供 `python -m twstock.jobs` 使用）。"""
    return child_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
