"""P3.1 Worker 硬化：單一實例鎖、有上限重試、逾時偵測、健康狀態機、生命週期。"""
from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from twstock import worker
from twstock.ops import pid_alive
from twstock.worker import (TaskOutcome, WorkerLock, WorkerState, derive_state,
                            run_forever, run_task_with_retry)


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(worker, "STATE_DIR", d)
    return d


# --------------------------------------------------------------------------
# 健康狀態機
# --------------------------------------------------------------------------
def test_state_not_started_without_heartbeat():
    assert derive_state(None) is WorkerState.NOT_STARTED
    assert derive_state({}) is WorkerState.NOT_STARTED


def test_state_stopping_when_stop_flag_present():
    hb = {"status": "running"}
    assert derive_state(hb, stop_flag=True, pid_alive=True) is WorkerState.STOPPING


def test_state_starting_and_stopped():
    assert derive_state({"status": "starting"}, pid_alive=True) is WorkerState.STARTING
    assert derive_state({"status": "stopped"}, pid_alive=False) is WorkerState.STOPPED


def test_state_running_and_task_running():
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    assert derive_state({"status": "running"}, pid_alive=True,
                        now=now) is WorkerState.RUNNING
    hb = {"status": "running",
          "task_started_at": (now - dt.timedelta(seconds=30)).isoformat()}
    assert derive_state(hb, pid_alive=True, now=now) is WorkerState.TASK_RUNNING


def test_state_detects_task_timeout():
    """任務開始時間超過門檻 → 不得再宣稱「任務執行中」。"""
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    hb = {"status": "task_running",
          "task_started_at": (now - dt.timedelta(hours=3)).isoformat()}
    st = derive_state(hb, pid_alive=True, now=now, task_timeout=3600)
    assert st is not WorkerState.TASK_RUNNING
    assert st is WorkerState.RUNNING


def test_state_data_stale():
    st = derive_state({"status": "running"}, pid_alive=True,
                      data_staleness_days=10)
    assert st is WorkerState.DATA_STALE


def test_state_degraded_and_failed():
    assert derive_state({"status": "running", "consecutive_failures": 1},
                        pid_alive=True) is WorkerState.DEGRADED
    assert derive_state({"status": "running", "consecutive_failures": 3},
                        pid_alive=True, max_failures=3) is WorkerState.TASK_FAILED
    assert derive_state({"status": "error", "consecutive_failures": 1},
                        pid_alive=True) is WorkerState.TASK_FAILED


def test_all_states_have_labels():
    for st in WorkerState:
        assert st.label


# --------------------------------------------------------------------------
# 單一實例鎖
# --------------------------------------------------------------------------
def test_lock_acquire_and_release(state_dir):
    lock = WorkerLock("worker")
    assert lock.acquire() is True
    assert lock.path.exists()
    assert lock.holder_pid() == os.getpid()
    lock.release()
    assert not lock.path.exists()


def test_lock_refuses_when_other_live_process_holds_it(state_dir, monkeypatch):
    lock = WorkerLock("worker")
    lock.path.write_text("999999 2026-01-01T00:00:00\n", encoding="utf-8")
    monkeypatch.setattr("twstock.ops.pid_alive", lambda pid: True)
    assert lock.acquire(pid=os.getpid()) is False, "不得搶走存活中的鎖"
    # 原持有者資訊不得被覆蓋
    assert lock.holder_pid() == 999999


def test_lock_reclaims_stale_lock(state_dir, monkeypatch):
    lock = WorkerLock("worker")
    lock.path.write_text("999999 2026-01-01T00:00:00\n", encoding="utf-8")
    monkeypatch.setattr("twstock.ops.pid_alive", lambda pid: False)
    assert lock.acquire(pid=os.getpid()) is True
    assert lock.holder_pid() == os.getpid()


# --------------------------------------------------------------------------
# 有上限的重試
# --------------------------------------------------------------------------
def test_retry_succeeds_first_attempt():
    out = run_task_with_retry(lambda: 42, name="t", attempts=3)
    assert out.ok and out.attempts == 1 and out.value == 42


def test_retry_retries_with_recorded_backoff():
    slept: list[float] = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    out = run_task_with_retry(flaky, name="t", attempts=5, base_delay=1.0,
                              sleep=slept.append)
    assert out.ok and out.attempts == 3
    assert slept == [1.0, 2.0], f"應為指數退避：{slept}"


def test_retry_backoff_is_capped():
    slept: list[float] = []
    run_task_with_retry(lambda: (_ for _ in ()).throw(RuntimeError("x")),
                        name="t", attempts=6, base_delay=1.0, max_delay=4.0,
                        sleep=slept.append)
    assert max(slept) <= 4.0
    assert slept == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_retry_exhausts_and_reports_failure_without_swallowing():
    out = run_task_with_retry(lambda: (_ for _ in ()).throw(ValueError("bad")),
                              name="t", attempts=3, sleep=lambda s: None)
    assert out.ok is False
    assert out.attempts == 3
    assert "ValueError" in out.error and "bad" in out.error
    assert "失敗" in out.label()


def test_task_outcome_label():
    assert "成功" in TaskOutcome(name="x", ok=True, attempts=1).label()


# --------------------------------------------------------------------------
# 生命週期
# --------------------------------------------------------------------------
def test_force_once_success_writes_heartbeat_and_releases_lock(state_dir, monkeypatch):
    monkeypatch.setattr(worker, "run_cycle", lambda force=False: "ok")
    rc = run_forever(force_once=True)
    assert rc == 0
    hb = json.loads((state_dir / "heartbeat.json").read_text(encoding="utf-8"))
    # 結束時最後一次心跳標記為 stopped，但上一輪成功必須看得出來
    assert hb["status"] == "stopped"
    assert hb["last_cycle_ok"] is True
    assert hb["consecutive_failures"] == 0
    assert hb["task_started_at"] in (None, "")
    assert derive_state(hb, pid_alive=False) is WorkerState.STOPPED
    assert not (state_dir / "worker.lock").exists(), "結束後必須釋放鎖"


def test_force_once_failure_reports_error_state(state_dir, monkeypatch):
    def boom(force=False):
        raise RuntimeError("ingest 全部失敗")

    monkeypatch.setattr(worker, "run_cycle", boom)
    rc = run_forever(force_once=True, attempts=2)
    assert rc == 3
    hb = json.loads((state_dir / "heartbeat.json").read_text(encoding="utf-8"))
    # 失敗必須被記錄，且「停止後」仍看得出來（不得被 stopped 覆蓋掉）
    assert hb["last_cycle_ok"] is False
    assert hb["consecutive_failures"] >= 1
    assert "ingest" in hb["failure_reason"]
    assert derive_state(hb, pid_alive=False) is WorkerState.DEGRADED
    # 連續失敗達上限 → 任務失敗
    assert derive_state(hb, pid_alive=False,
                        max_failures=1) is WorkerState.TASK_FAILED


def test_run_forever_refuses_duplicate_instance(state_dir, monkeypatch):
    monkeypatch.setattr("twstock.ops.pid_alive", lambda pid: True)
    (state_dir / "worker.lock").write_text("424242 2026-01-01T00:00:00\n",
                                           encoding="utf-8")
    called = {"n": 0}

    def should_not_run(force=False):
        called["n"] += 1
        return "ok"

    monkeypatch.setattr(worker, "run_cycle", should_not_run)
    assert run_forever(force_once=True) == 2
    assert called["n"] == 0, "拒絕啟動時不得執行任何任務"


def test_interrupted_last_run_is_detected(state_dir, monkeypatch):
    """上次在任務中被中斷（殘留 task_started_at）→ 重啟需偵測並安全接續。"""
    (state_dir / "heartbeat.json").write_text(json.dumps({
        "status": "task_running",
        "task_started_at": "2026-01-01T00:00:00",
        "consecutive_failures": 0,
    }), encoding="utf-8")
    monkeypatch.setattr(worker, "run_cycle", lambda force=False: "ok")
    assert run_forever(force_once=True) == 0
    hb = json.loads((state_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert hb["interrupted_last_run"] is True


def test_stop_flag_is_cleared_after_run(state_dir, monkeypatch):
    monkeypatch.setattr(worker, "run_cycle", lambda force=False: "ok")
    worker.request_stop("worker")
    assert worker.stop_requested("worker") is True
    run_forever(force_once=True)
    assert worker.stop_requested("worker") is False


def test_heartbeat_merges_fields(state_dir):
    worker.write_heartbeat(status="running", consecutive_failures=1)
    worker.write_heartbeat(last_result="ok")
    hb = worker.read_heartbeat()
    assert hb["status"] == "running"          # 舊欄位保留
    assert hb["consecutive_failures"] == 1
    assert hb["last_result"] == "ok"


# --------------------------------------------------------------------------
# 安全界線
# --------------------------------------------------------------------------
def test_heartbeat_never_contains_secrets(state_dir, monkeypatch):
    monkeypatch.setenv("FINMIND_TOKEN", "super-secret-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "another-secret")
    worker.write_heartbeat(status="running")
    raw = (state_dir / "heartbeat.json").read_text(encoding="utf-8")
    assert "super-secret-token" not in raw
    assert "another-secret" not in raw


def test_pid_alive_helpers_are_sane():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(0) is False


def test_should_run_defaults(monkeypatch):
    # 休市（週末）不得輪詢
    sat = dt.datetime(2026, 1, 3, 18, 0, 0)          # 週六
    assert worker._should_run(False, now=sat) == (False, "weekend")
    # 收盤前不輪詢
    mon_morning = dt.datetime(2026, 1, 5, 10, 0, 0)
    assert worker._should_run(False, now=mon_morning)[0] is False
    # 收盤後且尚未更新 → 執行
    mon_evening = dt.datetime(2026, 1, 5, 18, 0, 0)
    should, why = worker._should_run(False, now=mon_evening, last_ingest="2026-01-01")
    assert should is True and why == "eod"
    # 今日已完成 → 不重複
    assert worker._should_run(False, now=mon_evening,
                              last_ingest=mon_evening.date().isoformat())[0] is False
