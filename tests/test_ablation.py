"""P2.6：B17 新聞因子消融驗證的合成資料測試。

測試焦點：
  1. 方案 A 與生產路徑 ``score_row`` 完全一致（消融對象 = 策略真正使用的訊號）。
  2. 分解恆等式：新聞分子項、分母佔位項、C = B×den_b。
  3. 方案 D 的三態區分（缺值不冒充中性）。
  4. 排名不變性：W 同日常數 → 排名相同；W 跨股變動 → 排名**真的改變**。
  5. 配對差異檢定、分組價差、持倉/交易差異、門檻翻轉。
  6. ``build_features`` 新增的 ``news_present`` 欄位正確且向後相容。

所有預期值皆以公式明寫，不只驗證「不崩潰」。
"""
import numpy as np
import pandas as pd
import pytest

from twstock.ablation import (
    _score_from_parts,
    check_invariants,
    daily_group_spread,
    holdings_diff,
    news_present,
    news_state,
    paired_diff,
    rank_comparison,
    scale_spread_by_day,
    trade_key_set,
    variant_frame,
    variant_score,
)
from twstock.features import build_features
from twstock.signals import WEIGHTS, score_components, score_row

NEWS_W = WEIGHTS["news_sentiment"]          # 0.10
# 最小可用因子集（mom_20 0.12 + mom_60 0.06 + ma_alignment 0.09）= 0.27
BASE_MIN = WEIGHTS["mom_20"] + WEIGHTS["mom_60"] + WEIGHTS["ma_alignment"]


# --------------------------------------------------------------------------
# 合成特徵列
# --------------------------------------------------------------------------
def _prow(date: str, sid: str, *, kline=np.nan, ma=1, rev=np.nan,
          news=0.0, present=None, mom20=0.0, mom60=0.0) -> dict:
    """建一列特徵；未指定的因子欄位不放入 → ``_factor_parts`` 以預設值略過。"""
    d = {
        "date": pd.Timestamp(date), "stock_id": sid,
        "mom_20": mom20, "mom_60": mom60, "ma_alignment": ma,
        "kline_score": kline, "rev_yoy": rev,
        "news_sentiment": news,
    }
    if present is not None:
        d["news_present"] = float(present)
    return d


def _frame(rows: list[dict]) -> pd.DataFrame:
    return variant_frame(pd.DataFrame(rows))


# ==========================================================================
# 1. 方案 A 必須等於生產路徑
# ==========================================================================
def test_scheme_a_equals_production_score_row():
    """消融對象必須就是策略真正使用的訊號，否則比較無意義。"""
    for kw in ({"rev": 25.0, "news": 0.4},
               {"rev": np.nan, "news": 0.0},
               {"kline": 0.5, "ma": 0, "news": -0.7},
               {"kline": np.nan, "ma": 0, "news": 0.0}):
        row = pd.Series(_prow("2025-01-06", "2330", **kw))
        prod, _reason = score_row(row)
        assert variant_score(row, "A") == pytest.approx(prod, abs=1e-15)


# ==========================================================================
# 2. 分解恆等式
# ==========================================================================
def test_news_numerator_term_is_zero_when_news_is_zero():
    """A − A* = 0.10·v_news / W_A；v_news = 0 時嚴格等於 0。"""
    zero = pd.Series(_prow("2025-01-06", "2330", news=0.0))
    a = variant_score(zero, "A")
    a_ctrl = variant_score(zero, "Actrl")
    assert a == a_ctrl                      # 純 0，不是「接近 0」

    nz = pd.Series(_prow("2025-01-06", "2330", news=0.4))
    w_a = BASE_MIN + NEWS_W                 # 0.27 + 0.10 = 0.37
    expected_term = (NEWS_W * np.clip(0.4 * 1.5, -1, 1)) / w_a
    assert variant_score(nz, "A") - variant_score(nz, "Actrl") == \
        pytest.approx(expected_term, abs=1e-15)


def test_c_is_b_times_denominator_b():
    """C = B × den_b（正比例常數關係的實作驗證）。"""
    row = pd.Series(_prow("2025-01-06", "2330", rev=15.0, news=0.0))
    parts = [(k, v) for k, v in score_components(row).items()
             if k != "news_sentiment"]
    den_b = sum(WEIGHTS[k] for k, _ in parts)   # 0.27 + 0.11 = 0.38
    b = variant_score(row, "B")
    c = variant_score(row, "C")
    assert c == pytest.approx(b * den_b, abs=1e-15)


def test_check_invariants_passes_on_synthetic_panel():
    panel = pd.DataFrame([
        _prow("2025-01-06", "2330", rev=20.0, news=0.3),
        _prow("2025-01-06", "2317", rev=np.nan, news=0.0),
        _prow("2025-01-07", "2330", kline=0.4, news=-0.2, present=0.0),
    ])
    iv = check_invariants(panel)
    assert iv.ok, (iv.max_abs_news_term, iv.max_abs_den_err, iv.max_abs_scale_err)
    assert iv.n_rows == 3
    assert iv.max_abs_news_term <= 1e-12
    assert iv.max_abs_den_err <= 1e-12
    assert iv.max_abs_scale_err <= 1e-12


# ==========================================================================
# 3. 方案 D 的三態區分
# ==========================================================================
def test_d_equals_a_when_news_present_and_b_when_absent():
    present = pd.Series(_prow("2025-01-06", "2330", news=0.5, present=1.0))
    assert variant_score(present, "D") == variant_score(present, "A")

    absent = pd.Series(_prow("2025-01-06", "2330", news=0.5, present=0.0))
    assert variant_score(absent, "D") == variant_score(absent, "B")
    # 缺值不得冒充中性：D 的分母不含 0.10
    parts = list(score_components(absent).items())
    den_d = sum(WEIGHTS[k] for k, _ in parts if k != "news_sentiment")
    assert den_d == pytest.approx(BASE_MIN)       # 0.27，不含 0.10


def test_news_state_distinguishes_three_states():
    assert news_state(pd.Series(_prow("2025-01-06", "1", news=0.0,
                                      present=0.0))) == "absent"
    assert news_state(pd.Series(_prow("2025-01-06", "1", news=0.0,
                                      present=1.0))) == "neutral"
    assert news_state(pd.Series(_prow("2025-01-06", "1", news=0.33,
                                      present=1.0))) == "scored"
    # 「有記錄但情緒為 0」是有效中性，不等於缺值
    neutral = pd.Series(_prow("2025-01-06", "1", news=0.0, present=1.0))
    absent = pd.Series(_prow("2025-01-06", "1", news=0.0, present=0.0))
    assert variant_score(neutral, "D") != variant_score(absent, "D")


def test_news_present_falls_back_when_column_missing():
    """舊面板沒有 presence 欄位時退回 A 行為（視為存在），不臆測為缺值。"""
    old = pd.Series(_prow("2025-01-06", "1", news=0.0))   # 無 news_present
    assert "news_present" not in old.index
    assert news_present(old) is True
    no_news_col = pd.Series({"date": pd.Timestamp("2025-01-06"),
                             "stock_id": "1", "mom_20": 0.0})
    assert news_present(no_news_col) is False


# ==========================================================================
# 4. 排名不變性（本階段最關鍵的兩條）
# ==========================================================================
def test_rank_identical_when_available_weight_constant_within_day():
    """W 同日跨股為常數 → 分數不同但排名完全相同（尺度效應不影響排名）。"""
    rows = [
        _prow("2025-01-06", "A", rev=10.0),   # W_A = 0.27+0.11+0.10 = 0.48
        _prow("2025-01-06", "B", rev=20.0),
        _prow("2025-01-06", "C", rev=30.0),
        _prow("2025-01-06", "D", rev=-5.0),
    ]
    f = _frame(rows)
    assert f["w_a"].nunique() == 1            # 同日 W 為常數（前提成立）
    r = rank_comparison(f, "A", "B")
    assert r.identical_ratio == 1.0
    assert r.max_rank_move == 0
    assert r.max_abs_diff > 0                 # 分數確實不同
    assert r.spearman_mean == pytest.approx(1.0, abs=1e-12)
    assert r.relationship == "分數不同但排名完全相同"


def test_rank_changes_when_available_weight_varies_within_day():
    """W 同日跨股**變動** → 分母改變不是單一常數，排名真的會改變。

    這是 B17 影響排名的機制：稀釋程度逐股不同。
    """
    rows = [
        # W_A = 0.55（+kline 0.18），base = 0.09 + 0.18×0.2778 = 0.140004
        _prow("2025-01-06", "highW", kline=0.2778, ma=1),
        # W_A = 0.37，base = 0.09
        _prow("2025-01-06", "lowW", ma=1),
        # W_A = 0.48（+rev_yoy 0.11），base = 0.09 + 0.11 = 0.20
        _prow("2025-01-06", "midW", ma=1, rev=30.0),
        # W_A = 0.37，base = 0.09 × (−0.3) = −0.027
        _prow("2025-01-06", "bear", ma=0),
    ]
    f = _frame(rows)
    assert f["w_a"].nunique() > 1             # 同日 W 跨股變動（前提成立）

    # 逐股驗算，避免只信排名函式
    d = f.set_index("stock_id")
    assert d.loc["highW", "score_A"] == pytest.approx(0.140004 / 0.55, abs=1e-9)
    assert d.loc["lowW", "score_A"] == pytest.approx(0.09 / 0.37, abs=1e-9)
    assert d.loc["highW", "score_B"] == pytest.approx(0.140004 / 0.45, abs=1e-9)
    assert d.loc["lowW", "score_B"] == pytest.approx(0.09 / 0.27, abs=1e-9)

    # A 下 highW 高於 lowW；B 下 lowW 反超 → 名次互換
    assert d.loc["highW", "score_A"] > d.loc["lowW", "score_A"]
    assert d.loc["highW", "score_B"] < d.loc["lowW", "score_B"]

    r = rank_comparison(f, "A", "B")
    assert r.identical_ratio == 0.0
    assert r.max_rank_move == 1
    assert r.relationship == "非單一正比例常數 → 排名會改變"


def test_scale_spread_detects_non_constant_ratio():
    """理論尺度比 den_b/den_a 若同日跨股不為常數 → 不可宣告為單一比例常數。"""
    f = _frame([
        _prow("2025-01-06", "highW", kline=0.3),
        _prow("2025-01-06", "lowW", ma=1),
    ])
    sp = scale_spread_by_day(f, "A", "B")
    assert len(sp) == 1
    # highW: den_b/den_a = 0.45/0.55；lowW: 0.27/0.37
    expected_spread = abs(0.45 / 0.55 - 0.27 / 0.37)
    assert sp["ratio_spread"].iloc[0] == pytest.approx(expected_spread, abs=1e-9)
    assert sp["ratio_spread"].iloc[0] > 1e-6   # 非常數


# ==========================================================================
# 5. 配對差異檢定與分組
# ==========================================================================
def test_paired_diff_is_zero_for_identical_series():
    idx = pd.date_range("2025-01-01", periods=60, freq="B")
    x = pd.Series(np.linspace(0.01, 0.05, 60), index=idx)
    d = paired_diff(x, x, label="t|A−B@h5")
    assert d.n_pairs == 60
    assert d.mean_diff == 0.0
    assert not d.rejected_at                   # 常數差異 → NW 無法給顯著性


def test_paired_diff_recovers_known_shift():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=400, freq="B")
    base = pd.Series(rng.normal(0, 0.02, 400), index=idx)
    shift = 0.01
    other = base + shift + rng.normal(0, 0.002, 400)
    d = paired_diff(other, base, label="train|A−B@h5")
    assert d.mean_diff == pytest.approx(shift, abs=5e-4)
    assert d.rejected_at                       # 明確位移應被偵測
    # 自助法信賴區間必須涵蓋觀察到的差異均值
    assert d.ci_low <= d.mean_diff <= d.ci_high


def test_daily_group_spread_uses_contemporaneous_signal_only():
    """分組只用當期訊號：高訊號組報酬減低訊號組報酬即為價差。"""
    dates = pd.date_range("2025-01-06", periods=10, freq="B")
    rows = []
    for i, d in enumerate(dates):
        for j in range(10):
            rows.append({"date": d, "stock_id": f"s{j}",
                         "score_A": float(j),          # 依序遞增
                         "fwd_5": float(j) * 0.01})    # 與訊號完全同向
    sp = daily_group_spread(pd.DataFrame(rows), "score_A", 5)
    assert len(sp) == 10
    # Q5 平均（stock 8,9 → 0.08,0.09）減 Q1（stock 0,1 → 0.00,0.01）
    assert sp.iloc[0] == pytest.approx(0.085 - 0.005, abs=1e-12)


# ==========================================================================
# 6. 持倉／交易差異與門檻翻轉
# ==========================================================================
def _trades():
    return pd.DataFrame({
        "entry_date": ["2025-01-06", "2025-01-07"],
        "exit_date": ["2025-01-10", "2025-01-14"],
        "stock_id": ["2330", "2317"],
    })


def test_holdings_diff_counts_symmetric_difference():
    cal = list(pd.date_range("2025-01-06", periods=7, freq="B"))
    t = _trades()
    h_a = {cal[0]: {"2330"}, cal[1]: {"2330"}}
    h_b = {cal[0]: {"2317"}, cal[1]: {"2330"}}
    n, same, sym, cells = holdings_diff(h_a, h_b)
    assert n == 2                       # 只比較兩者皆有的日期
    assert same == 1                    # 第 2 天持倉相同
    # 第1天 {2330}^{2317} = 2；第2天 = 0 → 平均 1.0
    assert sym == pytest.approx(1.0)
    # 持倉不同的欄位總數 = Σ|A Δ B| = 2 + 0（不是聯集總數 3）
    assert cells == 2
    assert len(trade_key_set(t)) == 2


def test_threshold_flip_counts_crossings():
    """buy_threshold 為絕對門檻 → 尺度變化可能改變成交（必須精確計數）。"""
    f = _frame([
        # base = 0.09 + 0.11×clip(0/30) = 0.09；W_A = 0.48
        # A = 0.09/0.48 = 0.1875 < 0.2 → 不進場
        # B = 0.09/0.38 = 0.2368 ≥ 0.2 → 進場  →  唯一的翻轉
        _prow("2025-01-06", "flip", rev=0.0, news=0.0),
        # base = 0.09 + 0.11 = 0.20 → A = 0.4167、B = 0.5263，皆 ≥ 0.2
        _prow("2025-01-06", "always_in", rev=30.0, news=0.0),
        # base = 0.09 − 0.11 = −0.02 → A = −0.0417、B = −0.0526，皆 < 0.2
        _prow("2025-01-06", "never_in", rev=-30.0, news=0.0),
    ])
    thr = 0.2
    flips = int(((f["score_A"] >= thr) != (f["score_B"] >= thr)).sum())
    assert flips == 1
    d = f.set_index("stock_id")
    assert d.loc["flip", "score_A"] == pytest.approx(0.09 / 0.48, abs=1e-12)
    assert d.loc["flip", "score_B"] == pytest.approx(0.09 / 0.38, abs=1e-12)
    assert (d.loc["flip", "score_A"] < thr) and (d.loc["flip", "score_B"] >= thr)
    for sid in ("always_in", "never_in"):
        assert (d.loc[sid, "score_A"] >= thr) == (d.loc[sid, "score_B"] >= thr)


# ==========================================================================
# 7. build_features 的 news_present 欄位
# ==========================================================================
def _synth_price(days: int = 120, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=days)
    close = base + np.cumsum(rng.normal(0, 1, days))
    return pd.DataFrame({
        "date": dates, "open": close * (1 + rng.normal(0, 0.002, days)),
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.integers(1e5, 1e6, days), "amount": 1e8,
    })


def test_build_features_exposes_news_present_and_stays_compatible():
    p = _synth_price()
    news = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
        "stock_id": ["2330", "2330"],
        "score": [0.5, -0.3],
    })
    f = build_features(p, news_daily=news)
    # 向後相容：news_sentiment 仍被填為 0.0，無缺值
    assert f["news_sentiment"].isna().sum() == 0
    # presence 涵蓋 lag 0/1/2 → 2025-01-01(三)、01-02(四)、01-03(五)
    covered = f.loc[f["news_present"] == 1, "date"]
    assert set(pd.to_datetime(covered).dt.date) == {
        pd.Timestamp("2025-01-01").date(),
        pd.Timestamp("2025-01-02").date(),
        pd.Timestamp("2025-01-03").date(),
    }
    assert int(f["news_present"].sum()) == 3
    # 未覆蓋日：presence = 0 且 sentiment = 0（兩者可分，不是同一回事）
    rest = f[f["news_present"] == 0]
    assert (rest["news_sentiment"] == 0.0).all()
    # 覆蓋日的值與舊公式一致（(0.5 + (−0.3)) / 2 = 0.1）
    assert f.loc[f["date"] == pd.Timestamp("2025-01-01"),
                 "news_sentiment"].iloc[0] == pytest.approx(0.1, abs=1e-12)


def test_build_features_without_news_has_no_presence_column():
    f = build_features(_synth_price())
    assert "news_present" not in f.columns
    assert "news_sentiment" not in f.columns


# ==========================================================================
# 8. variant_frame 結構
# ==========================================================================
def test_variant_frame_columns_counts_and_states():
    f = _frame([
        _prow("2025-01-06", "2330", news=0.4, present=1.0),
        _prow("2025-01-06", "2317", news=0.0, present=1.0),
        _prow("2025-01-07", "2330", news=0.0, present=0.0),
    ])
    for col in ("score_A", "score_Actrl", "score_B", "score_C", "score_D",
                "w_a", "den_a", "den_b", "news_present", "news_state",
                "date", "stock_id"):
        assert col in f.columns, col
    assert len(f) == 3
    assert (f["score_A"] != f["score_B"]).all()   # 三列分數都不同
    st = f["news_state"].value_counts().to_dict()
    assert st == {"scored": 1, "neutral": 1, "absent": 1}
    # 分母恆等：den_a − den_b = 0.10（news 始終在 parts 中）
    assert np.allclose((f["den_a"] - f["den_b"]).to_numpy(), NEWS_W, atol=1e-12)


def test_variant_frame_keys_are_unique_for_forward_return_merge():
    f = _frame([_prow("2025-01-06", "2330"), _prow("2025-01-06", "2317"),
                _prow("2025-01-07", "2330")])
    dup = f.duplicated(subset=["date", "stock_id"]).sum()
    assert dup == 0
    assert len(f) == 3


# ==========================================================================
# P2.8：point-in-time 新聞可用性生效後，凍結報告的可重現性必須明確告知
# ==========================================================================
def _report(news_rows: int, n_rows_record: int) -> "AblationReport":
    from twstock.ablation import AblationReport
    rep = AblationReport(generated_at="2026-10-15T00:00:00")
    rep.news_rows = news_rows
    rep.n_rows_record = n_rows_record
    return rep


def test_notes_flag_frozen_report_not_reproducible_under_pointintime():
    """可進入訊號的新聞為 0 但 news 表有列 → 必須告知 B17 報告不可重現。

    B17 是凍結研究產物，其數字算在 P2.8 之前的面板上；若不提醒，
    重跑得出的不同數字會被誤讀成新發現。
    """
    from twstock.ablation import _notes
    notes = "\n".join(_notes(_report(news_rows=237, n_rows_record=0),
                             split=None, horizons=()))
    assert "point-in-time" in notes
    assert "不會**重現" in notes or "不會" in notes
    assert "237" in notes


def test_notes_do_not_claim_irreproducibility_when_news_is_usable():
    """有可用新聞列時不得誤報不可重現（否則提醒會失去意義）。"""
    from twstock.ablation import _notes
    notes = "\n".join(_notes(_report(news_rows=237, n_rows_record=61),
                             split=None, horizons=()))
    assert "point-in-time" not in notes
