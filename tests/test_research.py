"""P2.8 §6：前瞻研究紀錄（合成資料，不連外）。

驗證重點：
  1. 凍結規格真的包含任務要求的六項：程式 SHA、因子權重、股票池、
     成本假設、交易規則、評估指標；且摘要可重現。
  2. 凍結不會被靜默覆寫：同版本不動、改版本回 drift 且不覆寫。
  3. 時間切點「首次寫入即定案」，重跑不篡改「當初知道什麼」。
  4. 歷史訊號被事後改寫時可被稽核出來。
  5. 未達最低觀察期間前只報告資料品質，不宣稱策略有效。
"""
from __future__ import annotations

import pandas as pd
import pytest

from twstock import research
from twstock.config import CONFIG
from twstock.data import store
from twstock.signals import WEIGHTS


# ==========================================================================
# §6.1：凍結規格
# ==========================================================================
def test_spec_records_everything_required(temp_db):
    s = research.research_spec()
    assert s["pool"]["n"] == len(CONFIG.pool)
    assert s["pool"]["ids"] == sorted(CONFIG.pool)
    assert s["factor_weights"] == dict(sorted(WEIGHTS.items()))
    assert s["cost_model"]["name"] == "typical"
    assert s["trading_rules"]["top_n"] == CONFIG.top_n
    assert s["trading_rules"]["hold_days"] == CONFIG.hold_days
    assert len(s["eval_metrics"]) >= 8
    assert s["spec_hash"]
    assert "code_sha" in s


def test_spec_hash_is_stable_across_calls(temp_db):
    a = research.research_spec(frozen_at="2026-01-01T00:00:00", sha="deadbeef")
    b = research.research_spec(frozen_at="2026-06-01T00:00:00", sha="deadbeef")
    assert a["spec_hash"] == b["spec_hash"], "凍結時間不應影響版本摘要"
    c = research.research_spec(frozen_at="2026-01-01T00:00:00", sha="cafe")
    assert c["spec_hash"] != a["spec_hash"], "程式 SHA 改變必須反映在摘要"


def test_freeze_is_idempotent_and_records_defaults(temp_db):
    assert research.freeze()["status"] == "created"
    assert research.freeze()["status"] == "unchanged"
    assert research.load_freeze() is not None


# ==========================================================================
# §6：凍結不得靜默漂移
# ==========================================================================
def test_drift_is_detected_and_not_overwritten(temp_db, monkeypatch):
    research.freeze()
    before = research.load_freeze()

    # 模擬因子權重被改動
    monkeypatch.setitem(WEIGHTS, "news_sentiment", 0.99)
    res = research.freeze()
    assert res["status"] == "drift"
    assert "factor_weights" in res["drift"]
    assert research.load_freeze()["spec_hash"] == before["spec_hash"], \
        "偵測到漂移時不得覆寫已凍結版本"

    forced = research.freeze(force=True)
    assert forced["status"] == "forced"
    assert research.load_freeze()["spec_hash"] != before["spec_hash"]


# ==========================================================================
# §6.2：每日可稽核的時間切點
# ==========================================================================
def _price(date: str, sid: str = "2330") -> dict:
    return {"date": date, "stock_id": sid, "open": 100.0, "high": 101.0,
            "low": 99.0, "close": 100.0, "volume": 1e6}


def _news(date: str, fetched: str | None, link: str) -> dict:
    return {"published": date, "stock_id": "2330", "title": "t", "link": link,
            "score": 0.5, "source": "x", "fetched_at": fetched,
            "timestamp_source": "source", "score_kind": "scored",
            "provenance": "recorded", "link_basis": "title_match"}


def test_cutoff_captures_known_data_and_quality_flags(temp_db):
    store.write_df(pd.DataFrame([_price("2026-10-13"),
                                 _price("2026-10-14")]), "price")
    store.write_df(pd.DataFrame([
        _news("2026-10-13", "2026-10-13T18:00:00", "a"),
        _news("2026-10-14", "2026-10-14T18:00:00", "b"),
    ]), "news")
    res = research.register_cutoff("2026-10-13")
    assert res["registered"] is True
    snap = res["snapshot"]
    assert snap["n_price_rows"] == 1
    assert snap["price_max"] == "2026-10-13"
    assert snap["n_news_fetched_upto"] == 1, "切點後才取得的新聞不得算進來"
    assert "has_price" in snap["notes"] and "news_fetched_known" in snap["notes"]
    assert snap["spec_hash"] == research.load_freeze()["spec_hash"] \
        if research.load_freeze() else True


def test_cutoff_is_first_write_wins(temp_db):
    """重跑不得改寫「當初知道什麼」（否則切點失去稽核意義）。"""
    store.write_df(pd.DataFrame([_price("2026-10-13")]), "price")
    assert research.register_cutoff("2026-10-13")["registered"] is True
    first = research.cutoffs().iloc[0]

    # 之後資料增加，重跑同一日
    store.write_df(pd.DataFrame([_price("2026-10-13", "2317"),
                                 _price("2026-10-14")]), "price")
    res = research.register_cutoff("2026-10-13")
    assert res["registered"] is False
    again = research.cutoffs().iloc[0]
    assert int(again["n_price_rows"]) == int(first["n_price_rows"]) == 1, \
        "切點不得被後來的資料回填改寫"


def test_cutoff_records_untraceable_legacy_news(temp_db):
    store.write_df(pd.DataFrame([
        {"published": "2026-10-13", "stock_id": "2330", "title": "old",
         "link": "old", "score": 0.0, "source": "x", "fetched_at": None,
         "timestamp_source": "unknown", "score_kind": "neutral",
         "provenance": "unknown"},
    ]), "news")
    snap = research.snapshot("2026-10-13")
    assert snap["n_news_fetched_upto"] == 0
    assert snap["n_news_untraceable"] == 1
    assert "news_untraceable_absent" not in snap["notes"]


# ==========================================================================
# §6.3：歷史訊號被改寫可被稽核
# ==========================================================================
def _signal(date: str, score: float) -> dict:
    return {"date": date, "stock_id": "2330", "score": score, "reason": "r",
            "close": 100.0, "rank": 1, "action": "buy", "ml_score": 0.0}


def test_signal_rewrite_is_detected(temp_db):
    store.write_df(pd.DataFrame([_signal("2026-10-13", 0.5)]), "signals")
    research.register_cutoff("2026-10-13")
    assert research.verify_cutoffs() == []

    # 事後改寫同一日訊號（例如因為結果不好想掩蓋）
    store.write_df(pd.DataFrame([_signal("2026-10-13", 0.9)]), "signals")
    problems = research.verify_cutoffs()
    assert problems and "2026-10-13" in problems[0]
    assert "改寫" in problems[0]


def test_verify_cutoffs_empty_without_records(temp_db):
    assert research.verify_cutoffs() == []


# ==========================================================================
# §6.5：未達最低觀察期間不得宣稱有效
# ==========================================================================
def test_observation_status_blocks_conclusions_before_minimum(temp_db):
    research.freeze()
    research.register_cutoff("2026-10-13")
    obs = research.observation_status(today="2026-10-20")
    assert obs["min_required"] == research.MIN_OBSERVATION_TRADING_DAYS
    assert obs["sufficient"] is False
    text = research.to_text(today="2026-10-20")
    assert "尚未達最低觀察期間" in text
    assert "只報告資料品質與覆蓋率" in text
    assert "不宣稱新聞因子或策略具備可交易優勢" in text


def test_observation_status_sufficient_after_long_period(temp_db):
    research.freeze()
    research.register_cutoff("2026-10-13")
    assert research.observation_status(today="2027-06-01")["sufficient"] is True
    assert "已達最低觀察期間" in research.to_text(today="2027-06-01")


def test_minimum_observation_is_declared_upfront_with_rationale(temp_db):
    """最低觀察期間必須事前宣告且附理由（不得依未來績效反推）。"""
    s = research.research_spec()
    assert s["min_observation"]["trading_days"] == \
        research.MIN_OBSERVATION_TRADING_DAYS
    assert s["min_observation"]["preferred_trading_days"] >= \
        research.MIN_OBSERVATION_TRADING_DAYS
    why = s["min_observation"]["rationale"]
    assert "不是依績效挑選" in why
    assert "檢定力" in why


def test_status_text_without_freeze_says_not_frozen(temp_db):
    text = research.to_text()
    assert "尚未凍結研究版本" in text
    assert "尚未開始累積前瞻資料" in text


def test_canonical_hash_is_order_insensitive():
    assert research.canonical_hash({"a": 1, "b": 2}) == \
        research.canonical_hash({"b": 2, "a": 1})
