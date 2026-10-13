"""共用測試夾具：將儲存層指向暫存 DB，避免污染使用者的真實資料。"""
from __future__ import annotations

import os

import pytest

from twstock.data import store


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
