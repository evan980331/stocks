"""Worker 子程序隔離原型測試：逾時、停止、崩潰、錯誤處理與資料完整性。

這些測試會真的啟動子程序（`python -m twstock.jobs`），但每個任務都在 1~3 秒內
結束，不需要網路，也不會碰使用者的真實資料庫。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from twstock import jobs as J
from twstock.jobs import JobRunner, _write_result, child_main


@pytest.fixture()
def runner(tmp_path):
    """每個測試自己的 state/log 目錄，測試結束時確保沒有殘留子程序。"""
    r = JobRunner(state_dir=tmp_path / "state", log_dir=tmp_path / "logs")
    yield r
    try:
        r.stop()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
# 1. 正常成功
# --------------------------------------------------------------------------
def test_successful_task_reports_ok_with_result(runner):
    out = runner.run("j1", task="ok", timeout=30)
    assert out.status == J.OK
    assert out.ok is True
    assert out.exit_code == 0
    assert out.forced is False
    assert out.result.get("status") == J.OK
    assert out.pid is not None


def test_log_and_result_files_are_created(runner):
    out = runner.run("j2", task="ok", timeout=30)
    assert Path(out.log_path).exists()
    assert (runner.state_dir / "j2.result.json").exists()
    # 取消檔在成功路徑中不應殘留
    assert not (runner.state_dir / "j2.cancel").exists()


def test_outcome_label_includes_state_and_duration(runner):
    out = runner.run("j3", task="ok", timeout=30)
    label = out.label()
    assert "j3" in label and "任務成功" in label
    assert "exit=0" in label


# --------------------------------------------------------------------------
# 2. 失敗必須被記錄，不得默默當成功
# --------------------------------------------------------------------------
def test_task_exception_is_reported_as_failure(runner):
    out = runner.run("boom1", task="boom", timeout=30)
    assert out.status == J.FAILED
    assert out.ok is False
    assert "RuntimeError" in out.error
    assert "模擬任務失敗" in out.error
    assert out.exit_code == 5


def test_unknown_task_is_rejected_by_argparse():
    with pytest.raises(SystemExit) as e:
        J._build_argparser().parse_args(["--task", "no-such-task",
                                         "--result", "r.json"])
    assert e.value.code == 2


def test_child_exit_code_is_five_on_task_failure(tmp_path):
    r = tmp_path / "r.json"
    code = child_main(["--task", "boom", "--result", str(r)])
    assert code == 5
    payload = json.loads(r.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "RuntimeError" in payload["error"]


# --------------------------------------------------------------------------
# 3. 崩潰（無結果檔）不得被當成成功
# --------------------------------------------------------------------------
def test_child_crash_without_result_is_detected(runner):
    out = runner.run("crash1", task="crash", timeout=30)
    assert out.status == J.CRASHED
    assert out.ok is False
    assert out.result == {}
    assert "未留下結果檔" in out.error
    assert out.exit_code == 9


def test_classify_never_treats_clean_exit_without_result_as_success():
    status, err = JobRunner._classify(still_running=False, interrupted=False,
                                      exit_code=0, result={})
    assert status == J.CRASHED
    assert "無法確認完成" in err


def test_classify_table():
    c = JobRunner._classify
    assert c(still_running=False, interrupted=False, exit_code=0,
             result={"status": "ok"})[0] == J.OK
    assert c(still_running=False, interrupted=False, exit_code=0,
             result={"status": "cancelled"})[0] == J.CANCELLED
    assert c(still_running=False, interrupted=False, exit_code=0,
             result={"status": "failed", "error": "e"})[0] == J.FAILED
    assert c(still_running=False, interrupted=False, exit_code=5,
             result={"status": "failed", "error": "e"})[0] == J.FAILED
    assert c(still_running=False, interrupted=False, exit_code=9,
             result={})[0] == J.CRASHED
    assert c(still_running=True, interrupted=True, exit_code=None,
             result={})[0] == J.TIMEOUT


def test_classify_does_not_turn_graceful_cancel_into_timeout():
    """子程序已自行收尾並回報取消時，即使超過期限也必須記為 cancelled。"""
    status, _ = JobRunner._classify(still_running=False, interrupted=True,
                                    exit_code=0,
                                    result={"status": "cancelled"})
    assert status == J.CANCELLED


def test_classify_marks_late_completion_and_late_kill_distinctly():
    late_ok, note = JobRunner._classify(still_running=False, interrupted=True,
                                        exit_code=0, result={"status": "ok"})
    assert late_ok == J.OK and "超過期限" in note
    killed, _ = JobRunner._classify(still_running=False, interrupted=True,
                                    exit_code=1, result={})
    assert killed == J.TIMEOUT


# --------------------------------------------------------------------------
# 4. 逾時：合作式停止 vs 強制終止（可區分）
# --------------------------------------------------------------------------
def test_timeout_on_uncooperative_task_is_forced_and_really_dies(runner):
    """忽略中斷訊號的任務只能強制終止，且必須明確標示為 forced。"""
    out = runner.run("stub1", task="stubborn", timeout=1.0, grace=3.0,
                     seconds=60)
    assert out.status == J.TIMEOUT
    assert out.forced is True                 # 明確標示為強制終止
    assert out.pid is not None
    assert out.duration < 30
    assert not JobRunner._contact_ok(out.pid)  # 程序確實已不存在


def test_spawn_uses_its_own_process_group(runner, tmp_path):
    """必須建立獨立程序群組；否則 CTRL_BREAK 會誤殺父程序。"""
    runner._spawn("grp1", "sleep", {"seconds": 30})
    try:
        if __import__("os").name == "nt":
            assert runner._own_group is True
        assert JobRunner._contact_ok(runner.pid)
    finally:
        runner.stop()


def test_interrupt_is_a_noop_on_windows(runner):
    """Windows 上不得送出任何中斷訊號（CTRL_BREAK_EVENT 等同立即終止）。

    驗證方式：呼叫 _interrupt 後子程序必須仍存活 → 確實沒有送出會殺死它的訊號。
    """
    import os as _os
    if _os.name != "nt":
        pytest.skip("此防護僅針對 Windows 的 CTRL_BREAK_EVENT 語意")
    runner._spawn("grp2", "sleep", {"seconds": 30})
    try:
        runner._interrupt()
        time.sleep(0.3)
        assert JobRunner._contact_ok(runner.pid), "子程序被殺掉了 → 訊號確實被送出"
    finally:
        runner.stop()


def test_jobs_source_proves_no_signal_based_kill():
    """原始碼層級的可稽核性：模組不得呼叫 os.kill（也就無法誤殺其他行程）。"""
    src = Path(J.__file__).read_text(encoding="utf-8")
    assert "os.kill" not in src
    assert "CREATE_NEW_PROCESS_GROUP" in src   # 仍保留主控台事件隔離


def test_timeout_on_cooperative_task_stops_gracefully(runner):
    out = runner.run("coop1", task="cooperative", timeout=1.0, grace=6.0,
                     seconds=60)
    assert out.status == J.CANCELLED
    assert out.forced is False                # 自行收尾，不是被硬殺
    assert out.result.get("status") == J.CANCELLED
    assert not JobRunner._contact_ok(out.pid)


def test_cooperative_cancel_file_is_removed_after_success(runner):
    runner.run("coop2", task="cooperative", timeout=30, grace=6.0, seconds=0.1)
    assert not (runner.state_dir / "coop2.cancel").exists()


# --------------------------------------------------------------------------
# 5. 停止一個任務不得誤殺其他程序
# --------------------------------------------------------------------------
def test_stopping_one_job_leaves_other_jobs_running(tmp_path):
    """停止一個任務不得誤殺其他任務的程序。"""
    a = JobRunner(state_dir=tmp_path / "a", log_dir=tmp_path / "a")
    b = JobRunner(state_dir=tmp_path / "b", log_dir=tmp_path / "b")
    a._spawn("ja", "sleep", {"seconds": 30})
    b._spawn("jb", "sleep", {"seconds": 30})
    pa, pb = a.pid, b.pid
    try:
        assert a.stop() is True                # 只停 A
        time.sleep(0.3)
        assert not JobRunner._contact_ok(pa)
        assert JobRunner._contact_ok(pb)        # B 必須完全不受影響
    finally:
        b.stop()


def test_runner_never_uses_name_based_process_kill():
    """原始碼層級的防護：不得出現依名稱的批次終止。"""
    src = Path(J.__file__).read_text(encoding="utf-8")
    for bad in ("pkill", "killall", "taskkill /IM", "/IM "):
        assert bad not in src


# --------------------------------------------------------------------------
# 6. SQLite 寫入中途被殺：資料完整性
# --------------------------------------------------------------------------
def test_kill_during_sqlite_transaction_leaves_no_partial_rows(runner, tmp_path):
    """交易中途被硬殺（不給子程序任何收尾機會）仍不得留下半筆資料。"""
    db = tmp_path / "probe.db"
    out = runner.run("sq1", task="sqlite-atomic", timeout=1.0, grace=1.0,
                     db=str(db), rows=300, hold=60)
    assert out.status == J.TIMEOUT
    assert out.forced is True              # 確認真的走到強制終止那一步
    # 資料庫必須仍可讀，且交易中的 300 筆不得留下任何一筆
    conn = sqlite3.connect(str(db))
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        assert "probe" in tables               # 交易外的 CREATE TABLE 已提交
        n = conn.execute("SELECT COUNT(*) FROM probe").fetchone()[0]
        assert n == 0, f"交易未完整回復：留下 {n} 筆半截資料"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_sqlite_task_commits_when_it_completes(tmp_path):
    from twstock.jobs import PROJECT_ROOT
    import subprocess
    import sys

    db = tmp_path / "probe2.db"
    result = tmp_path / "r.json"
    p = subprocess.run(
        [sys.executable, "-m", "twstock.jobs", "--task", "sqlite-atomic",
         "--result", str(result), "--db", str(db), "--rows", "50",
         "--hold", "0"],
        cwd=str(PROJECT_ROOT), capture_output=True, timeout=60)
    assert p.returncode == 0
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "ok" and payload["rows"] == 50
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("SELECT COUNT(*) FROM probe").fetchone()[0] == 50
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 7. 結果檔原子寫入
# --------------------------------------------------------------------------
def test_write_result_is_atomic_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "r.json"
    _write_result(p, {"status": "ok", "n": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"status": "ok", "n": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_write_result_overwrites_previous_content(tmp_path):
    p = tmp_path / "r.json"
    _write_result(p, {"status": "ok"})
    _write_result(p, {"status": "failed", "error": "x"})
    assert json.loads(p.read_text(encoding="utf-8"))["status"] == "failed"
