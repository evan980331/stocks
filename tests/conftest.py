"""共用測試夾具：將儲存層指向暫存 DB，避免污染使用者的真實資料。

實測發現（本階段修正）
  修正前只有明確要求 ``temp_db`` 的測試才會被隔離。``tests/test_ops_worker.py``
  的 ``test_should_run_outside_trading_window`` 沒有使用該 fixture，卻會經由
  ``worker._should_run`` → ``store.get_meta`` 以**可寫模式開啟正式 DB**，
  因此在跑 ``pytest`` 時對使用者的真實資料觸發了 schema 遷移
  （meta 記錄 ``news_legacy_marked = 2026-10-15T00:11:04``）。

  這正是不受控的寫入路徑，故改為 **autouse**：任何測試都無法再對真實 DB
  或執行狀態目錄寫入，而不必逐一記得加上 fixture。
"""
from __future__ import annotations

import os

import pytest

from twstock.data import store


@pytest.fixture(autouse=True)
def _isolate_runtime_state(tmp_path, monkeypatch):
    """把所有測試的儲存層與執行狀態隔離到 tmp_path。

    以 ``TWSTOCK_ALLOW_REAL_DB=1`` 可明確退出隔離（僅供刻意對真實資料的檢查）。
    """
    if os.environ.get("TWSTOCK_ALLOW_REAL_DB"):
        yield
        return
    monkeypatch.delenv("TWSTOCK_READONLY", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "isolated.db")
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    from twstock import ops, worker
    state = tmp_path / "state"
    monkeypatch.setattr(worker, "STATE_DIR", state)
    monkeypatch.setattr(ops, "STATE_DIR", state)
    yield


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """把 store 指向暫存 DB（可寫）。"""
    monkeypatch.delenv("TWSTOCK_READONLY", raising=False)
    db = tmp_path / "test.db"
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    yield db
    if db.exists():
        os.remove(db)
