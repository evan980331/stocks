"""Worker / ops 生命週期：停止旗標、PID 存活判斷、健康狀態欄位完整性。"""
from __future__ import annotations

import os

from twstock import ops, worker


def test_stop_flag_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "STATE_DIR", tmp_path)
    worker.clear_stop("worker")
    assert worker.stop_requested("worker") is False
    worker.request_stop("worker")
    assert worker.stop_requested("worker") is True
    worker.clear_stop("worker")
    assert worker.stop_requested("worker") is False


def test_pid_alive_detects_self_and_rejects_zero():
    assert ops.pid_alive(os.getpid()) is True
    assert ops.pid_alive(0) is False
    assert ops.pid_alive(-1) is False


def test_pid_alive_rejects_bogus_pid():
    # 極大且不存在的 PID 應回 False，而不是拋出例外
    assert ops.pid_alive(999_999_999) is False


def test_health_reports_real_state(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "STATE_DIR", tmp_path)
    monkeypatch.setattr(worker, "STATE_DIR", tmp_path)
    from twstock import risk
    monkeypatch.setattr(risk, "DATA_DIR", str(tmp_path))

    h = ops.health()
    for key in ("time", "db_path", "mode", "latest_price_date", "ml",
                "kill_switch", "worker", "dashboard"):
        assert key in h, f"健康狀態缺少 {key}"
    assert h["mode"] == "paper"                 # 預設非正式交易
    assert h["kill_switch"]["engaged"] is False
    assert h["ml"]["available"] is False        # 測試環境未安裝 lightgbm


def test_should_run_outside_trading_window():
    # 週末或收盤前不應觸發自動更新（避免對資料來源無謂輪詢）
    should, why = worker._should_run(force=False)
    assert isinstance(should, bool)
    assert why in {"weekend", "already_done_today"} or why.startswith("before_eod") \
        or why == "eod"
    assert worker._should_run(force=True) == (True, "forced")
