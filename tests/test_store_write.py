"""W3：寫入必須冪等且不得覆蓋其他標的的資料（原 bug：to_sql(if_exists="replace") 會 DROP 整表）。"""
from __future__ import annotations

import pandas as pd

from twstock.data import store


def _news(sid: str, title: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "published": "2026-10-12", "stock_id": sid, "title": title,
        "link": f"https://example.com/{sid}", "score": 0.5, "source": "test",
    }])


def test_news_write_does_not_overwrite_other_stocks(temp_db):
    store.write_df(_news("2330", "台積電新聞"), "news")
    store.write_df(_news("2317", "鴻海新聞"), "news")

    df = store.read_sql("SELECT stock_id, title FROM news ORDER BY stock_id")
    ids = set(df["stock_id"])
    assert ids == {"2330", "2317"}, f"其他股票新聞被覆蓋了：{ids}"
    assert len(df) == 2


def test_write_is_idempotent_on_rerun(temp_db):
    df = _news("2330", "台積電新聞")
    store.write_df(df, "news")
    store.write_df(df, "news")          # 重跑同一任務
    assert store.read_sql("SELECT COUNT(*) AS n FROM news").iloc[0]["n"] == 1


def test_write_updates_score_without_duplicating(temp_db):
    store.write_df(_news("2330", "舊標題"), "news")
    updated = _news("2330", "新標題")
    store.write_df(updated, "news")
    df = store.read_sql("SELECT title FROM news")
    assert len(df) == 1 and df.iloc[0]["title"] == "新標題"


def test_price_write_across_stocks(temp_db):
    for sid in ("2330", "2317"):
        store.write_df(pd.DataFrame([{
            "date": "2026-10-12 00:00:00", "stock_id": sid, "open": 100.0,
            "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1e6,
            "amount": 1e8, "turnover": 0.01,
        }]), "price")
    assert store.read_sql("SELECT COUNT(DISTINCT stock_id) AS n FROM price").iloc[0]["n"] == 2


def test_readonly_mode_blocks_write(temp_db, monkeypatch):
    monkeypatch.setenv("TWSTOCK_READONLY", "1")
    try:
        store.write_df(_news("2330", "x"), "news")
    except RuntimeError as e:
        assert "唯讀" in str(e)
    else:  # pragma: no cover
        raise AssertionError("唯讀模式應禁止寫入")
