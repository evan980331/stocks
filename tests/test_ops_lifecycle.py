"""服務停止語意：狀態清理、殘留檔案、區分優雅停止與強制終止。"""
from __future__ import annotations

import os

import pytest

from twstock import ops


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ops, "STATE_DIR", d)
    return d


def test_cleanup_removes_pid_stop_and_stale_lock(state_dir, monkeypatch):
    monkeypatch.setattr(ops, "pid_alive", lambda pid: False)
    (state_dir / "worker.pid").write_text("999999", encoding="utf-8")
    (state_dir / "worker.stop").write_text("now", encoding="utf-8")
    (state_dir / "worker.lock").write_text("999999 2026-01-01T00:00:00\n",
                                           encoding="utf-8")

    ops._cleanup_state("worker")

    assert not (state_dir / "worker.pid").exists()
    assert not (state_dir / "worker.stop").exists()
    assert not (state_dir / "worker.lock").exists()


def test_cleanup_keeps_lock_of_live_holder(state_dir, monkeypatch):
    monkeypatch.setattr(ops, "pid_alive", lambda pid: True)
    (state_dir / "worker.lock").write_text("424242 2026-01-01T00:00:00\n",
                                           encoding="utf-8")
    ops._cleanup_state("worker")
    assert (state_dir / "worker.lock").exists(), "存活持有者的鎖不得被清掉"


def test_stop_when_not_running_is_clean(state_dir):
    assert ops.stop("worker") is True
    assert not (state_dir / "worker.pid").exists()
    assert not (state_dir / "worker.stop").exists()


def test_stop_reports_forced_when_process_never_exits(state_dir, monkeypatch):
    """行程不自行結束時必須回報『強制停止』，不可謊稱安全停止。"""
    monkeypatch.setattr(ops, "pid_alive", lambda pid: True)     # 永遠活著
    monkeypatch.setattr(ops, "_signal_stop", lambda pid: True)
    monkeypatch.setattr(ops.subprocess, "run", lambda *a, **k: None)
    (state_dir / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    ok = ops.stop("worker", timeout=0.1)
    assert ok is False, "行程仍在時不得回報成功"


def test_stop_reports_graceful_when_process_exits(state_dir, monkeypatch):
    calls = {"n": 0}

    def fake_alive(pid):
        calls["n"] += 1
        return calls["n"] < 2          # 第一次還活著，之後結束

    monkeypatch.setattr(ops, "pid_alive", fake_alive)
    monkeypatch.setattr(ops, "_signal_stop", lambda pid: True)
    (state_dir / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    assert ops.stop("worker", timeout=5.0) is True
    assert not (state_dir / "worker.pid").exists()


def test_mechanism_is_cooperative_flag_for_worker_but_signal_for_dashboard(
        state_dir, monkeypatch):
    """worker 必須先靠停止旗標自行收尾；dashboard 才直接送訊號。"""
    signals: list[str] = []
    monkeypatch.setattr(ops, "_signal_stop",
                        lambda pid: signals.append("signal") or True)

    def alive_then_dead(pid):
        return False                    # 立即結束

    monkeypatch.setattr(ops, "pid_alive", alive_then_dead)
    (state_dir / "worker.pid").write_text("1", encoding="utf-8")
    ops.stop("worker", timeout=1.0)
    assert signals == [], "worker 應先嘗試合作式停止，不需送訊號"

    (state_dir / "dashboard.pid").write_text("1", encoding="utf-8")
    ops.stop("dashboard", timeout=1.0)
    assert signals == ["signal"], "dashboard 應直接送可攔截訊號"


def test_mechanism_falls_back_to_signal_if_flag_ignored(state_dir, monkeypatch):
    signals: list[str] = []
    monkeypatch.setattr(ops, "_signal_stop",
                        lambda pid: signals.append("signal") or True)
    monkeypatch.setattr(ops, "pid_alive", lambda pid: False)
    (state_dir / "worker.pid").write_text("1", encoding="utf-8")
    ops.stop("worker", timeout=0.1)
    assert signals == [], "已結束就不需要再送訊號"
