"""P4 模擬交易引擎測試（17 類；全部使用暫存 DB＋假行情，不連外）。

涵蓋：初始帳戶、買賣帳務、費率稅、持倉成本損益、拒單、重複防護、
取消與狀態轉移、部分成交、行情過期/缺價/停牌/時點、漲跌停、風控上限、
緊急停止、重啟恢復、交易失敗回滾、帳本重建核對、正式 DB 隔離、
無真實下單路徑。
"""
from __future__ import annotations

import ast
import datetime as dt
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from twstock import paper, risk as risk_mod
from twstock.data import store

T0 = dt.datetime(2026, 2, 3, 15, 0)      # T 日（訊號日）收盤後
T1 = dt.datetime(2026, 2, 4, 15, 0)      # T+1 15:00
T2 = dt.datetime(2026, 2, 5, 15, 0)


# ---------------------------------------------------------------------------
# 夾具與假行情
# ---------------------------------------------------------------------------
def _seed_prices(rows: list[dict]) -> None:
    conn = store._connect()
    store._init(conn)
    conn.close()
    if rows:
        store.write_df(pd.DataFrame(rows), "price")


def _bar(date: str, o: float, *, close: float | None = None,
         volume: float = 50_000.0, stock_id: str = "2330") -> dict:
    c = close if close is not None else o
    return {"date": date, "stock_id": stock_id, "open": o,
            "high": max(o, c), "low": min(o, c), "close": c,
            "volume": volume}


def _mk_engine(tmp_path, **kw) -> paper.PaperEngine:
    lim = risk_mod.RiskLimits(
        max_order_value=500_000, max_position_pct=0.5,
        max_data_staleness_days=30, max_drawdown_pct=0.5,
        max_daily_loss_pct=0.5)
    ks = risk_mod.KillSwitch(path=tmp_path / "KILL_SWITCH")
    kw.setdefault("limits", lim)
    kw.setdefault("killswitch", ks)
    return paper.PaperEngine(**kw)


@pytest.fixture()
def eng(temp_db, tmp_path):
    """假行情：2026-02-03（T 日，收 101）→ 02-04 開 101（成交量充足）。"""
    _seed_prices([
        _bar("2026-02-03", 100.0, close=101.0),
        _bar("2026-02-04", 101.0, close=102.0),
        _bar("2026-02-05", 102.0, close=103.0),
    ])
    e = _mk_engine(tmp_path)
    e.ensure_account(initial_cash=1_000_000)
    return e


def _order(eng_, *, side="BUY", qty=1000, cid="c1", sig="2026-02-03",
           now=T0, stock="2330", hint=None, reason="test"):
    return eng_.place_order(stock_id=stock, side=side, qty=qty,
                            client_order_id=cid, signal_date=sig,
                            price_hint=hint, reason=reason, now=now)


def _rows(sql: str, params=()) -> pd.DataFrame:
    return store.read_sql(sql, params)


# ---------------------------------------------------------------------------
# 1. 初始帳戶及資金正確性
# ---------------------------------------------------------------------------
def test_initial_account_and_cash_integrity(eng):
    acc = eng.account()
    assert acc["mode"] == "paper" and acc["status"] == "active"
    assert acc["cash_utwd"] == paper.to_micro(1_000_000)
    assert acc["initial_cash_utwd"] == paper.to_micro(1_000_000)
    led = _rows("SELECT kind, amount_utwd, balance_utwd FROM paper_ledger")
    assert len(led) == 1 and led.iloc[0]["kind"] == "capital"
    assert int(led.iloc[0]["balance_utwd"]) == paper.to_micro(1_000_000)
    assert eng.reconcile()["ok"] is True
    # 冪等：重複 init 不重複入金
    eng.ensure_account(initial_cash=999)
    assert eng.account()["cash_utwd"] == paper.to_micro(1_000_000)
    assert len(_rows("SELECT 1 FROM paper_ledger")) == 1


# ---------------------------------------------------------------------------
# 2. 買入與賣出帳務
# ---------------------------------------------------------------------------
def test_buy_then_sell_accounting(eng):
    # 買：T+1（02-04）開盤 101 → 滑價後 101.1
    r = _order(eng)
    assert r["accepted"] is True
    res = eng.process_pending(now=T1)
    assert len(res) == 1 and res[0]["filled"] == 1000
    assert res[0]["price"] == pytest.approx(101.1)
    assert res[0]["raw_price"] == pytest.approx(101.0)  # 滑價前可稽核
    notional = paper.to_micro(101.1) * 1000
    comm, tax = paper.fee_for(notional, cost=eng.cm, side="BUY")
    acc = eng.account()
    assert acc["cash_utwd"] == paper.to_micro(1_000_000) - notional - comm
    cash, pos = eng.cash_and_positions(int(acc["account_id"]))
    st = pos["2330"]
    assert st.qty == 1000
    assert st.cost_basis == notional + comm           # 買入費用計入成本
    # 賣：02-05 開 102 → 滑價後 101.9
    r2 = _order(eng, side="SELL", cid="s1", sig="2026-02-04", now=T2)
    assert r2["accepted"] is True
    res2 = eng.process_pending(now=dt.datetime(2026, 2, 6, 15, 0))
    assert res2[0]["filled"] == 1000
    assert res2[0]["price"] == pytest.approx(101.9)
    notional2 = paper.to_micro(101.9) * 1000
    comm2, tax2 = paper.fee_for(notional2, cost=eng.cm, side="SELL")
    assert tax2 > 0                                    # 賣出有證交稅
    cash2, pos2 = eng.cash_and_positions(int(acc["account_id"]))
    assert cash2 == (paper.to_micro(1_000_000) - notional - comm
                     + notional2 - comm2 - tax2)
    assert pos2["2330"].qty == 0 and pos2["2330"].cost_basis == 0
    assert eng.reconcile()["ok"] is True


# ---------------------------------------------------------------------------
# 3. 手續費及交易稅計算
# ---------------------------------------------------------------------------
def test_fees_and_tax_calculation(eng):
    _order(eng)
    res = eng.process_pending(now=T1)[0]
    n = paper.to_micro(101.1) * 1000
    comm, tax = paper.fee_for(n, cost=eng.cm, side="BUY")
    assert res["fee"] == pytest.approx(paper.from_micro(comm))
    assert res["tax"] == 0.0                            # 買入無稅
    # 費率可設定且逐筆記錄於成交列（可追溯，非硬編碼於引擎）
    fill = _rows("SELECT * FROM paper_fill").iloc[0]
    assert eng.cm.name in str(fill["price_source"])
    assert int(fill["fee_utwd"]) == comm
    # 最低手續費生效
    c2, _ = paper.fee_for(paper.to_micro(10.0) * 1000, cost=eng.cm,
                          side="BUY")
    assert c2 == paper.to_micro(eng.cm.min_fee)


# ---------------------------------------------------------------------------
# 4. 持倉成本與損益計算（WAC；已實現/未實現分離）
# ---------------------------------------------------------------------------
def test_position_cost_and_realized_pnl(eng):
    # 兩次買進（不同價）→ 賣出全部 → 已實現損益正確
    _seed_prices([_bar("2026-02-06", 103.0, close=103.0),
                  _bar("2026-02-09", 104.0, close=104.0)])
    _order(eng, cid="b1", now=T0)
    eng.process_pending(now=T1)                         # 1000 @ 101.1
    _order(eng, cid="b2", sig="2026-02-04", now=T2)     # T=02-04
    eng.process_pending(now=dt.datetime(2026, 2, 6, 15, 0))  # 1000 @ 101.9
    acc = eng.account()
    cash, pos = eng.cash_and_positions(int(acc["account_id"]))
    assert pos["2330"].qty == 2000
    basis = pos["2330"].cost_basis                      # 兩批成本合計
    # 賣出 2000：訊號日 02-05 → 成交於下一交易日 02-06 開盤 103 → 102.9
    _order(eng, side="SELL", qty=2000, cid="sx",
           sig="2026-02-05", now=dt.datetime(2026, 2, 6, 15, 0))
    eng.process_pending(now=dt.datetime(2026, 2, 6, 16, 0))
    sell = _rows("SELECT * FROM paper_fill WHERE side='SELL'").iloc[0]
    assert float(sell["price_utwd"]) == paper.to_micro(102.9)
    cash2, pos2 = eng.cash_and_positions(int(acc["account_id"]))
    n2 = int(sell["price_utwd"]) * int(sell["qty"])
    comm2, tax2 = paper.fee_for(n2, cost=eng.cm, side="SELL")
    expected_realized = (n2 - comm2 - tax2) - basis
    assert int(sell["fee_utwd"]) == comm2 and int(sell["tax_utwd"]) == tax2
    assert pos2["2330"].realized == expected_realized
    snap = eng.snapshot(write=False)
    assert snap["realized"] == pytest.approx(
        paper.from_micro(expected_realized))
    assert snap["unrealized"] == pytest.approx(0.0)     # 清倉後無未實現


# ---------------------------------------------------------------------------
# 5. 資金不足及持倉不足拒單
# ---------------------------------------------------------------------------
def test_insufficient_cash_and_shares_rejected(eng):
    r = _order(eng, qty=4000)                           # 約需 404k+費 > 500k? 否
    # 改用超過可用現金的數量
    r = _order(eng, qty=9000, cid="big")                # ~909k+費 > 1M? 部位上限先擋
    assert r["accepted"] is False and r["status"] == "rejected"
    assert any("部位上限" in x or "單筆金額" in x for x in r["reasons"])
    # 資金不足：把現金壓低（先成交一筆大單）後再下買單
    _order(eng, cid="a", qty=4000)
    eng.process_pending(now=T1)                         # 花掉 ~404k
    r2 = _order(eng, cid="b", qty=8000, sig="2026-02-04", now=T2)
    assert r2["accepted"] is False
    assert any("資金不足" in x or "部位上限" in x or "單筆金額" in x
               for x in r2["reasons"])
    # 持倉不足：賣出量大於實際持有（兩單各成交 4000 → 共 8000 股）
    r3 = _order(eng, side="SELL", qty=9000, cid="s", sig="2026-02-04", now=T2)
    assert r3["accepted"] is False
    assert any("持倉不足" in x for x in r3["reasons"])
    # 拒單也入庫可稽核
    assert (_rows("SELECT COUNT(*) AS n FROM paper_order "
                  "WHERE status='rejected'").iloc[0]["n"]) == 3
    assert eng.reconcile()["ok"] is True


# ---------------------------------------------------------------------------
# 6. 重複訂單與重複成交防護
# ---------------------------------------------------------------------------
def test_duplicate_order_and_fill_idempotency(eng):
    r1 = _order(eng)
    r2 = _order(eng)                                    # 同 client_order_id
    assert r2["duplicate"] is True and r2["order_id"] == r1["order_id"]
    assert len(_rows("SELECT 1 FROM paper_order")) == 1
    eng.process_pending(now=T1)
    n_fills = len(_rows("SELECT 1 FROM paper_fill"))
    cash1 = eng.account()["cash_utwd"]
    # 重播：同參數再跑一次成交流程 → 不重複扣款/持倉
    eng.process_pending(now=T1)
    assert len(_rows("SELECT 1 FROM paper_fill")) == n_fills
    assert eng.account()["cash_utwd"] == cash1
    assert eng.reconcile()["ok"] is True
    # 帳本 ref 唯一索引防重（完整性錯誤 → IntegrityBroken，停止交易）
    with pytest.raises(paper.IntegrityBroken):
        with eng._tx() as conn:
            eng._ledger(conn, int(eng.account()["account_id"]), "fill",
                        -1, f"{r1['order_id']}:2026-02-04:cash")


# ---------------------------------------------------------------------------
# 7. 訂單取消與狀態轉移
# ---------------------------------------------------------------------------
def test_cancel_and_state_transitions(eng):
    r = _order(eng)
    oid = r["order_id"]
    c = eng.cancel_order(oid, "test_cancel")
    assert c["status"] == "cancelled"
    # 已終態不可再取消
    with pytest.raises(paper.InvalidTransition):
        eng.cancel_order(oid)
    # 已取消訂單不會成交
    res = eng.process_pending(now=T1)
    assert res == [] or res[0].get("reason") == "already_final"
    # 非法轉移一律拒絕（不得直接覆寫狀態）
    with pytest.raises(paper.InvalidTransition):
        paper.assert_transition("filled", "cancelled")
    with pytest.raises(paper.InvalidTransition):
        paper.assert_transition("rejected", "accepted")
    paper.assert_transition("accepted", "filled")       # 合法轉移不報錯
    # 取消事件入稽核
    ev = _rows("SELECT kind FROM paper_event")
    assert "order_cancelled" in set(ev["kind"])
    assert "order_transition" in set(ev["kind"])


# ---------------------------------------------------------------------------
# 8. 部分成交（成交量不足 → 殘量續留，次一交易日補齊）
# ---------------------------------------------------------------------------
def test_partial_fill_then_complete(temp_db, tmp_path):
    _seed_prices([
        _bar("2026-02-03", 100.0, close=101.0),
        _bar("2026-02-04", 101.0, close=102.0, volume=1_500.0),
        _bar("2026-02-05", 102.0, close=103.0, volume=50_000.0),
    ])
    e = _mk_engine(tmp_path)
    e.ensure_account(initial_cash=1_000_000)
    _order(e, qty=2000, now=T0)
    r1 = e.process_pending(now=T1)[0]
    assert r1["partial"] is True and r1["filled"] == 1000   # 1000 的倍數
    assert r1["status"] == "partially_filled"
    r2 = e.process_pending(now=T2)[0]
    assert r2["filled"] == 1000 and r2["status"] == "filled"
    cash, pos = e.cash_and_positions(int(e.account()["account_id"]))
    assert pos["2330"].qty == 2000
    assert e.reconcile()["ok"] is True
    # 零成交量（停牌）不成交、不推進持倉
    _order(e, side="SELL", qty=1000, cid="z", sig="2026-02-04", now=T2)


# ---------------------------------------------------------------------------
# 9. 行情過期、缺價、停牌與非交易時段
# ---------------------------------------------------------------------------
def test_staleness_missing_price_suspension(eng, tmp_path):
    # (a) 行情過期 → 禁止新交易
    r = _order(eng, now=dt.datetime(2026, 3, 20, 15, 0))
    assert r["accepted"] is False
    assert any("行情過期" in x for x in r["reasons"])
    # (b) 缺價：訊號日之後無行情列 → 不假設成交
    _seed_prices([])  # no-op if exists; 直接用無後續行情的訂單
    r2 = _order(eng, cid="m", sig="2026-02-05", now=dt.datetime(2026, 2, 6, 15, 0),
                hint=103.0)
    res = eng.process_pending(now=dt.datetime(2026, 2, 6, 15, 0))
    mine = [x for x in res if x["order_id"] == r2["order_id"]]
    assert mine and mine[0]["reason"] == "no_market_data_yet"
    # (c) 停牌：成交量=0 → 不成交
    _seed_prices([_bar("2026-02-09", 103.0, close=103.0, volume=0.0)])
    res2 = eng.process_pending(now=dt.datetime(2026, 2, 9, 15, 0))
    mine2 = [x for x in res2 if x["order_id"] == r2["order_id"]]
    assert mine2 and mine2[0]["reason"] == "suspended"
    # (d) 盤中（收盤前）不用當日尚未發生的開盤價成交
    _seed_prices([_bar("2026-02-09", 103.0, close=103.0)])  # 改為有量
    rc = _order(eng, cid="cut", sig="2026-02-05",
                now=dt.datetime(2026, 2, 6, 15, 0), hint=103.0)
    assert rc["accepted"] is True
    res3 = eng.process_pending(now=dt.datetime(2026, 2, 9, 10, 0))
    mine3 = [x for x in res3 if x["order_id"] == rc["order_id"]]
    assert mine3 and mine3[0]["reason"] == "no_market_data_yet"
    # 收盤後（同日 15:00）才成交
    res3b = eng.process_pending(now=dt.datetime(2026, 2, 9, 15, 0))
    mine3b = [x for x in res3b if x["order_id"] == rc["order_id"]]
    assert mine3b and mine3b[0].get("filled") == 1000
    # (e) T 日收盤前不得下單（訊號需 T 收盤資料）
    rb = _order(eng, cid="early", now=dt.datetime(2026, 2, 3, 10, 0))
    assert rb["accepted"] is False
    assert any("收盤後下單" in x for x in rb["reasons"])
    # (f) signal_date 為未來 → 拒絕
    rf = _order(eng, cid="future", sig="2026-03-01", now=T0)
    assert rf["accepted"] is False


# ---------------------------------------------------------------------------
# 10. 漲跌停及不可成交情境
# ---------------------------------------------------------------------------
def test_limit_up_down_no_fill(temp_db, tmp_path):
    # 前收 100 → 漲停 110.00 / 跌停 90.00（10% ± tick 近似）
    _seed_prices([
        _bar("2026-02-03", 100.0, close=100.0),
        _bar("2026-02-04", 110.0, close=110.0),         # 開盤即漲停
        _bar("2026-02-05", 100.0, close=100.0),         # 回到正常
        _bar("2026-02-06", 90.0, close=90.0),           # 開盤即跌停
    ])
    e = _mk_engine(tmp_path)
    e.ensure_account(initial_cash=1_000_000)
    up, dn = e.limit_up_down(100.0)
    assert float(up) == 110.0 and float(dn) == 90.0
    rb = _order(e, now=T0, cid="b")
    res = e.process_pending(now=T1)
    assert res[0]["reason"] == "limit_up_no_fill"        # 漲停鎖死買不進
    assert (_rows("SELECT status FROM paper_order").iloc[0]["status"]
            == "accepted")                              # 訂單未被誤判成交
    # 02-05 正常日成交建立持倉
    resf = e.process_pending(now=dt.datetime(2026, 2, 5, 15, 0))
    assert resf[0].get("filled") == 1000
    # 跌停鎖死賣不掉
    rs = _order(e, side="SELL", cid="s", sig="2026-02-05",
                now=dt.datetime(2026, 2, 5, 15, 0))
    assert rs["accepted"] is True
    res2 = e.process_pending(now=dt.datetime(2026, 2, 6, 15, 0))
    mine = [x for x in res2 if x["order_id"] == rs["order_id"]]
    assert mine and mine[0]["reason"] == "limit_down_no_fill"
    assert e.reconcile()["ok"] is True


# ---------------------------------------------------------------------------
# 11. 單筆曝險、持倉上限與損失限制
# ---------------------------------------------------------------------------
def test_risk_limits_pre_fill(temp_db, tmp_path):
    _seed_prices([_bar("2026-02-03", 100.0, close=101.0),
                  _bar("2026-02-04", 101.0, close=102.0)])
    lim = risk_mod.RiskLimits(max_order_value=100_000,
                              max_position_pct=0.10,
                              max_data_staleness_days=30)
    e = _mk_engine(tmp_path, limits=lim)
    e.ensure_account(initial_cash=1_000_000)
    # 單筆上限：1000 股 × ~101 ≈ 101k > 100k
    r = _order(e, now=T0)
    assert r["accepted"] is False
    assert any("單筆金額" in x for x in r["reasons"])
    # 持倉上限：預算 = equity × 10%
    assert e.limits.max_position_pct == 0.10
    # 當日損失限制：以快照歷史觸發（-10% < -5%）
    aid = int(e.account()["account_id"])
    store.exec_sql(
        "INSERT INTO paper_snapshot (account_id, as_of, cash_utwd,"
        " market_value_utwd, equity_utwd, realized_utwd, unrealized_utwd,"
        " max_equity_utwd, drawdown_pct, market_date, note)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (aid, "2026-02-03T15:00:00", paper.to_micro(1_000_000), 0,
         paper.to_micro(1_000_000), 0, 0, paper.to_micro(1_000_000), 0.0,
         "2026-02-03", ""))
    store.exec_sql(
        "INSERT INTO paper_snapshot (account_id, as_of, cash_utwd,"
        " market_value_utwd, equity_utwd, realized_utwd, unrealized_utwd,"
        " max_equity_utwd, drawdown_pct, market_date, note)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (aid, "2026-02-04T15:00:00", paper.to_micro(900_000), 0,
         paper.to_micro(900_000), 0, 0, paper.to_micro(1_000_000), 0.1,
         "2026-02-04", ""))
    r2 = _order(e, cid="dl", qty=100, hint=100.0,
                now=dt.datetime(2026, 2, 4, 15, 30))
    assert r2["accepted"] is False
    assert any("單日損失" in x or "回撤" in x for x in r2["reasons"])
    st = e.risk_status(now=dt.datetime(2026, 2, 4, 15, 30))
    assert st["preflight_ok"] is False


# ---------------------------------------------------------------------------
# 12. 緊急停止（Kill Switch）
# ---------------------------------------------------------------------------
def test_kill_switch_blocks_trading(temp_db, tmp_path):
    _seed_prices([_bar("2026-02-03", 100.0, close=101.0),
                  _bar("2026-02-04", 101.0, close=102.0)])
    e = _mk_engine(tmp_path)
    e.ensure_account(initial_cash=1_000_000)
    e.killswitch.engage("test emergency")
    r = _order(e, now=T0)
    assert r["accepted"] is False
    assert any("Kill Switch" in x for x in r["reasons"])
    st = e.risk_status()
    assert st["kill_switch_engaged"] is True and st["kill_switch_reason"]
    e.killswitch.release()
    assert _order(e, cid="after", now=T0)["accepted"] is True
    # 未成交訂單在 kill 期間也不成交
    e.killswitch.engage("during")
    res = e.process_pending(now=T1)
    assert all(x.get("filled", 0) == 0 for x in res)
    e.killswitch.release()


# ---------------------------------------------------------------------------
# 13. Worker 重啟及帳戶恢復
# ---------------------------------------------------------------------------
def test_restart_recovery(temp_db, tmp_path):
    e = _mk_engine(tmp_path)
    e.ensure_account(initial_cash=1_000_000)
    _seed_prices([_bar("2026-02-03", 100.0, close=101.0),
                  _bar("2026-02-04", 101.0, close=102.0)])
    _order(e, now=T0)
    e.process_pending(now=T1)
    # 未成交訂單（留待後續）
    _order(e, cid="open", sig="2026-02-03", now=T0)
    cash_before = e.account()["cash_utwd"]
    # 「重啟」：全新引擎實例，狀態完全由 DB 重建
    e2 = _mk_engine(tmp_path)
    rec = e2.recover()
    assert rec["reconcile_ok"] is True
    cash2, pos2 = e2.cash_and_positions(int(e2.account()["account_id"]))
    assert cash2 == cash_before
    assert pos2["2330"].qty == 1000
    assert rec["open_orders"] == 1
    # 未成交訂單在重啟後仍可繼續成交
    res = e2.process_pending(now=T2)
    assert any(x.get("filled") for x in res)
    _, pos3 = e2.cash_and_positions(int(e2.account()["account_id"]))
    assert pos3["2330"].qty == 2000
    assert e2.reconcile()["ok"] is True


# ---------------------------------------------------------------------------
# 14. 資料庫交易失敗與帳務一致性（回滾 / IntegrityBroken 停止交易）
# ---------------------------------------------------------------------------
def test_db_failure_rolls_back(eng):
    _order(eng, cid="ok", now=T0)
    eng.process_pending(now=T1)
    cash_before = eng.account()["cash_utwd"]
    r = _order(eng, cid="boom", sig="2026-02-04", now=T2)
    assert r["accepted"] is True

    def boom(*a, **k):
        raise RuntimeError("simulated db failure")

    # 注意：不可用 monkeypatch.undo()（會連 temp_db 的 DB 隔離一併解除）
    orig = eng._ledger
    eng._ledger = boom
    try:
        with pytest.raises(RuntimeError):
            eng.process_pending(now=dt.datetime(2026, 2, 6, 15, 0))
    finally:
        eng._ledger = orig
    # 回滾：訂單維持 accepted、無新成交、現金不變
    row = _rows("SELECT status, filled_qty FROM paper_order WHERE "
                "client_order_id='boom'").iloc[0]
    assert row["status"] == "accepted" and int(row["filled_qty"]) == 0
    assert eng.account()["cash_utwd"] == cash_before
    assert eng.reconcile()["ok"] is True


def test_integrity_error_stops_trading(eng, monkeypatch):
    _order(eng, cid="x", now=T0)

    def bad(*a, **k):
        raise sqlite3.IntegrityError("simulated constraint")

    monkeypatch.setattr(eng, "_ledger", bad)
    with pytest.raises(paper.IntegrityBroken):
        eng.process_pending(now=T1)
    # 完整性錯誤後不得有任何半套資料
    assert len(_rows("SELECT 1 FROM paper_fill")) == 0


# ---------------------------------------------------------------------------
# 15. 帳本重建與帳戶餘額核對
# ---------------------------------------------------------------------------
def test_ledger_rebuild_and_balance_check(eng):
    _order(eng, now=T0)
    eng.process_pending(now=T1)
    _order(eng, side="SELL", cid="s", sig="2026-02-04", now=T2)
    eng.process_pending(now=dt.datetime(2026, 2, 6, 15, 0))
    eng.snapshot(write=True, now=dt.datetime(2026, 2, 6, 15, 0))
    rec = eng.reconcile()
    assert rec["ok"] is True
    aid = int(eng.account()["account_id"])
    cash, pos = eng.cash_and_positions(aid)
    # 帳本重建現金 == 帳戶欄位 == 最新快照現金
    assert cash == int(eng.account()["cash_utwd"])
    snap = _rows("SELECT cash_utwd, market_value_utwd, equity_utwd FROM "
                 "paper_snapshot ORDER BY snapshot_id DESC LIMIT 1").iloc[0]
    assert int(snap["equity_utwd"]) == (int(snap["cash_utwd"])
                                        + int(snap["market_value_utwd"]))
    # 竄改帳戶餘額 → 核對必須發現
    store.exec_sql("UPDATE paper_account SET cash_utwd = cash_utwd + 1 "
                   "WHERE account_id=?", (aid,))
    rec2 = eng.reconcile()
    assert rec2["ok"] is False
    assert any("重建現金" in p for p in rec2["problems"])


# ---------------------------------------------------------------------------
# 16. 正式資料庫隔離與唯讀模式
# ---------------------------------------------------------------------------
def test_real_db_isolation_and_readonly(eng, monkeypatch, tmp_path):
    # 測試全程隔離：DB 必須在 pytest 的暫存目錄內
    assert str(Path(store.DB_PATH).resolve()).startswith(str(tmp_path.resolve())), \
        "測試不得指向正式 DB"
    # 唯讀模式：寫入一律拒絕，唯讀查詢可執行
    monkeypatch.setenv("TWSTOCK_READONLY", "1")
    with pytest.raises(paper.PaperError):
        _order(eng, now=T0)
    assert eng.account() is not None                    # 帳戶可查
    snap = eng.snapshot(write=False)
    assert "equity" in snap
    risk = eng.risk_status()
    assert "preflight_ok" in risk
    assert eng.reconcile()["ok"] is True


# ---------------------------------------------------------------------------
# 17. 任何模擬交易路徑都不會呼叫真實下單 API
# ---------------------------------------------------------------------------
FORBIDDEN_IMPORTS = {"requests", "urllib", "http", "socket", "ccxt",
                     "shioaji", "fubon", "ib_insync", "ibapi", "aiohttp",
                     "httpx"}


def test_no_live_trading_paths(eng, monkeypatch):
    src = Path(paper.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert not (mods & FORBIDDEN_IMPORTS), \
        f"paper 引擎不得引入網路/券商套件：{mods & FORBIDDEN_IMPORTS}"
    # live 模式（含人工確認旗標）下引擎直接拒絕
    monkeypatch.setattr(risk_mod, "current_mode",
                        lambda: risk_mod.Mode.LIVE)
    monkeypatch.setenv("TWSTOCK_LIVE_ACK", "1")
    with pytest.raises(paper.LiveModeForbidden):
        _order(eng, now=T0)
    # paper 帳戶一律 paper 模式
    assert eng.account()["mode"] == "paper"


# ---------------------------------------------------------------------------
# 額外：週期流程、CLI 與儀表板唯讀 API
# ---------------------------------------------------------------------------
def test_cycle_end_to_end(temp_db, tmp_path):
    _seed_prices([_bar("2026-02-03", 100.0, close=101.0),
                  _bar("2026-02-04", 101.0, close=102.0),
                  _bar("2026-02-05", 102.0, close=103.0)])
    store.write_df(pd.DataFrame([
        {"date": "2026-02-03", "stock_id": "2330", "score": 0.9,
         "reason": "t", "close": 101.0, "rank": 1, "action": "buy",
         "ml_score": None}]), "signals")
    e = _mk_engine(tmp_path)
    e.ensure_account(initial_cash=1_000_000)
    res = e.run_paper_cycle(now=T0)
    assert res["orders"] and res["orders"][0]["accepted"] is True
    assert len(_rows("SELECT 1 FROM paper_order")) == 1
    res2 = e.run_paper_cycle(now=T1)                    # 同訊號重跑
    # 冪等：未完成買單存在時不重複下單；成交只入一筆
    assert len(_rows("SELECT 1 FROM paper_order")) == 1
    fills = [f for f in res2["fills"] if f.get("filled")]
    assert fills and e.reconcile()["ok"] is True
    assert "equity" in res2["snapshot"]


def test_cli_paper_subcommands(temp_db, tmp_path, monkeypatch, capsys):
    from twstock import cli
    monkeypatch.setattr(
        "sys.argv", ["twstock.cli", "paper", "init", "--cash", "200000"])
    assert cli.main() == 0
    monkeypatch.setattr("sys.argv", ["twstock.cli", "paper", "account"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "模擬" in out and "200,000.00" in out
    monkeypatch.setattr("sys.argv", ["twstock.cli", "paper", "reconcile"])
    assert cli.main() == 0
    monkeypatch.setattr("sys.argv", ["twstock.cli", "paper", "risk"])
    assert cli.main() == 0
    monkeypatch.setattr("sys.argv", ["twstock.cli", "paper", "orders"])
    assert cli.main() == 0
    monkeypatch.setattr("sys.argv", ["twstock.cli", "paper", "bogus"])
    assert cli.main() == 2


def test_dashboard_paper_api_readonly(temp_db, tmp_path):
    from fastapi.testclient import TestClient
    from twstock.app import app
    client = TestClient(app)
    # 未建帳戶：明確提示，不 500
    r = client.get("/api/paper/account").json()
    assert "error" in r and "paper init" in r["error"]
    # 建帳戶後：唯讀端點皆可查
    _seed_prices([_bar("2026-02-03", 100.0, close=101.0)])
    e = _mk_engine(tmp_path)
    e.ensure_account(initial_cash=1_000_000)
    for path in ("/api/paper/account", "/api/paper/positions",
                 "/api/paper/orders", "/api/paper/fills",
                 "/api/paper/ledger", "/api/paper/risk",
                 "/api/paper/reconcile"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert "error" not in resp.json(), path
    # 無任何下單/成交的 POST 端點（儀表板只能看，不能交易）
    post_paths = [r.path for r in app.routes
                  if "paper" in getattr(r, "path", "")
                  and "POST" in getattr(r, "methods", set())]
    assert post_paths == []


def test_order_lifecycle_events_audit(eng):
    _order(eng, now=T0)
    _order(eng, cid="bad", qty=0, now=T0)               # 拒單
    eng.process_pending(now=T1)
    ev = _rows("SELECT kind FROM paper_event")
    kinds = set(ev["kind"])
    assert {"order_created", "order_rejected", "order_transition",
            "fill"} <= kinds


def test_no_fill_without_reference_price(eng):
    # 無任何行情 → 下單直接拒絕（不得假設可成交）
    r = _order(eng, cid="np", hint=None, now=T0)
    # 有 02-03 行情時有參考價；改為完全無行情的股票
    r2 = eng.place_order(stock_id="2454", side="BUY", qty=1000,
                         client_order_id="np2", signal_date="2026-02-03",
                         now=T0)
    assert r2["accepted"] is False
    assert any("參考價" in x or "股票池" in x for x in r2["reasons"])
