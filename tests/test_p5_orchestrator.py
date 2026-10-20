"""P5 本地自動化協調器測試（全隔離暫存 DB；不連外、不碰正式庫、不下真單）。

涵蓋任務 P5-F 的 14 類情境：
  1. 正常行情下完成一輪模擬工作
  2. 無行情資料時停止新交易
  3. 行情過期／時間戳未知時停止新交易
  4. 資料來源部分失敗時正確標記狀態
  5. 同一工作重複執行不產生重複訂單
  6. 同時啟動兩個 worker/協調器時避免衝突
  7. 執行中途崩潰後重新啟動（真實子程序崩潰）
  8. 資料庫交易失敗後帳務保持一致
  9. kill switch 啟用後拒絕新交易且不自動解除
 10. 帳務核對發現不一致時停止新交易且不自動改帳
 11. worker 逾時與重試上限
 12. 非交易日與盤中、盤後時間邊界
 13. 健康狀態能辨認「程序活著但資料已過期」
 14. 測試過程不改動正式資料庫內容
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from twstock import orchestrator as orc
from twstock import risk as risk_mod
from twstock import worker
from twstock.data import quotes, store
from twstock.data.status import IngestStatus
from twstock.paper import PaperEngine

NOW = dt.datetime(2026, 2, 5, 15, 0)      # 週四 15:00（收盤後）
SIG_DAY = "2026-02-04"                     # 訊號日（T）；成交用 T+1 開盤


# ---------------------------------------------------------------------------
# 夾具與假元件
# ---------------------------------------------------------------------------
def _bar(date: str, o: float, *, close: float | None = None,
         volume: float = 50_000.0, stock_id: str = "2330") -> dict:
    c = close if close is not None else o
    return {"date": date, "stock_id": stock_id, "open": o,
            "high": max(o, c), "low": min(o, c), "close": c,
            "volume": volume}


BASE_BARS = [
    _bar("2026-02-03", 40.0, close=41.0),
    _bar("2026-02-04", 41.0, close=41.5),
    _bar("2026-02-05", 42.0, close=43.0),
]


def _seed(bars: list[dict], signals: bool = True) -> None:
    conn = store._connect()
    store._init(conn)
    conn.close()
    if bars:
        store.write_df(pd.DataFrame(bars), "price")
    if signals:
        store.write_df(pd.DataFrame([
            {"date": SIG_DAY, "stock_id": "2330", "score": 0.9, "reason": "t",
             "close": 41.5, "rank": 1, "action": "buy", "ml_score": None}]),
            "signals")


def _ok_ingest(rows: int = 3):
    return lambda ctx: orc.StageResult(IngestStatus.SUCCESS_WITH_DATA.value,
                                       rows=rows,
                                       detail={"ingest_run_id": "fake-ingest"})


def _ok_signals():
    return lambda ctx: orc.StageResult(IngestStatus.SUCCESS_WITH_DATA.value,
                                       rows=1,
                                       detail={"signal_date": SIG_DAY})


def _run(**kw) -> orc.RoundResult:
    kw.setdefault("now", NOW)
    kw.setdefault("sleep", lambda s: None)
    fns = dict(kw.pop("fns", {}))
    fns.setdefault("ingest", _ok_ingest())
    fns.setdefault("signals", _ok_signals())
    kw["fns"] = fns
    return orc.run_round(**kw)


@pytest.fixture()
def p5(temp_db, tmp_path, monkeypatch):
    """隔離 DB＋狀態目錄；種行情與訊號，並建立 paper 帳戶。"""
    monkeypatch.setattr(risk_mod, "DATA_DIR", tmp_path)   # KillSwitch 隔離
    monkeypatch.delenv("TWSTOCK_QUOTE_SOURCE", raising=False)
    monkeypatch.delenv("TWSTOCK_ALERT_TELEGRAM", raising=False)
    _seed(BASE_BARS)
    eng = PaperEngine()
    eng.ensure_account(initial_cash=1_000_000)
    return eng


def _rows(sql: str, params=()) -> pd.DataFrame:
    return store.read_sql(sql, params)


# ---------------------------------------------------------------------------
# 1. 正常行情下完成一輪模擬工作
# ---------------------------------------------------------------------------
def test_round_success_with_normal_market(p5):
    res = _run()
    assert res.status == orc.ROUND_SUCCESS and res.ok
    assert res.run_id.startswith("p5-") and res.ended_at
    # 六＋1 個階段都有紀錄：開始/結束/耗時/狀態/筆數
    assert set(res.stages) == set(orc.STAGES)
    db = _rows("SELECT * FROM run_stage WHERE run_id=?", (res.run_id,))
    assert len(db) == len(orc.STAGES)
    assert db["ended_at"].notna().all() and db["started_at"].notna().all()
    done = db[db["status"] != IngestStatus.SKIPPED.value]
    assert (done["duration_s"].notna()).all()
    assert int(db["rows"].sum()) >= 1
    # 真的串接既有元件：ingest run_id 可追查
    ing = json.loads(db[db["stage"] == "ingest"].iloc[0]["detail"])
    assert ing["ingest_run_id"] == "fake-ingest"
    # 模擬交易真的執行：一張 2000 股買單 + T+1 開盤成交
    orders = _rows("SELECT * FROM paper_order")
    assert len(orders) == 1 and orders.iloc[0]["status"] == "filled"
    assert int(orders.iloc[0]["qty"]) == 2000
    fills = _rows("SELECT * FROM paper_fill")
    assert len(fills) == 1 and str(fills.iloc[0]["fill_date"]) == "2026-02-05"
    assert int(fills.iloc[0]["qty"]) == 2000
    # 帳本：capital + fill
    led = _rows("SELECT kind FROM paper_ledger ORDER BY ledger_id")
    assert list(led["kind"]) == ["capital", "fill"]
    # 追查鏈：run_id 進訂單 reason 與快照備註
    assert res.run_id in str(orders.iloc[0]["reason"])
    snap = _rows("SELECT note FROM paper_snapshot")
    assert res.run_id in str(snap.iloc[0]["note"])
    # meta 可查最後一輪
    assert store.get_meta("last_p5_run_id") == res.run_id
    assert store.get_meta("last_p5_status") == orc.ROUND_SUCCESS
    # 帳務一致
    assert p5.reconcile()["ok"] is True
    # 未設定行情來源 → quotes 階段依規則略過（policy），不污染輪次狀態
    assert res.stages["quotes"]["status"] == IngestStatus.SKIPPED.value
    assert res.stages["quotes"]["kind"] == orc.KIND_POLICY


# ---------------------------------------------------------------------------
# 2. 無行情資料時停止新交易
# ---------------------------------------------------------------------------
def test_no_market_data_blocks_trading(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(risk_mod, "DATA_DIR", tmp_path)
    monkeypatch.delenv("TWSTOCK_QUOTE_SOURCE", raising=False)
    _seed([], signals=False)                      # price 表零列
    eng = PaperEngine()
    eng.ensure_account(initial_cash=1_000_000)
    res = _run()
    # 品質檢查失敗、paper 階段被阻擋（上游/閘門 fail-closed）
    assert res.stages["quality"]["status"] == IngestStatus.FAILED.value
    assert res.stages["paper"]["status"] == IngestStatus.SKIPPED.value
    assert res.stages["paper"]["kind"] == orc.KIND_BLOCKED
    assert len(_rows("SELECT 1 FROM paper_order")) == 0, "無行情不得建立新單"
    assert res.status != orc.ROUND_SUCCESS
    # 事件有記錄，可從 CLI 查到
    assert any("停止新模擬交易" in str(s.get("error"))
               for s in orc.recent_errors()["stages"])


# ---------------------------------------------------------------------------
# 3. 行情過期 / 時間戳未知 / 來源失敗 / 延遲資料不得假裝即時
# ---------------------------------------------------------------------------
def test_stale_price_blocks_new_orders(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(risk_mod, "DATA_DIR", tmp_path)
    _seed([_bar("2026-01-28", 40.0, close=41.0)])   # 延遲 8 天 > 預設 3 天
    eng = PaperEngine()
    eng.ensure_account(initial_cash=1_000_000)
    r = eng.place_order(stock_id="2330", side="BUY", qty=1000,
                        client_order_id="stale", signal_date="2026-01-28",
                        now=NOW)
    assert r["accepted"] is False
    assert any("行情過期" in x for x in r["reasons"])
    g = orc.gate_mod.evaluate(now=NOW, engine=eng)
    assert not g.allow_new_orders and g.allow_fills   # 只擋新單，不擋既定成交


def _quote_row(source: str, kind: str, market_time: str, fetched_at: str,
               *, stock_id: str = "2330", price: float = 100.0,
               ts_source: str = "source") -> dict:
    return {"source": source, "kind": kind, "stock_id": stock_id,
            "market_time": market_time, "market_time_raw": market_time,
            "market_time_source": ts_source, "fetched_at": fetched_at,
            "price": price, "prev_close": price, "status": "ok"}


def _set_quote(rows: list[dict]) -> None:
    """重設 quote 表內容（每個子案例互不干扰）。"""
    store.exec_sql("DELETE FROM quote")
    if rows:
        store.write_df(pd.DataFrame(rows), "quote")


def test_intraday_gate_fail_closed(temp_db, tmp_path, monkeypatch):
    """盤中新單一律要求「可驗證的即時行情」；任一不明即 fail-closed。"""
    now = dt.datetime(2026, 2, 5, 10, 0)          # 盤中
    # (a) 未設定來源 → 拒絕（不臆測預設來源）
    monkeypatch.delenv("TWSTOCK_QUOTE_SOURCE", raising=False)
    probs, _ = orc.gate_mod.intraday_basis(now=now)
    assert any("未設定" in p for p in probs)
    # (b) 延遲來源 → 拒絕（不得把延遲假裝成可交易的盤中行情）
    monkeypatch.setenv("TWSTOCK_QUOTE_SOURCE", "finmind_minute")
    probs, _ = orc.gate_mod.intraday_basis(now=now)
    assert any("不是可交易的盤中即時行情" in p for p in probs)
    # (c) 即時來源但資料筆數為零 → 拒絕
    monkeypatch.setenv("TWSTOCK_QUOTE_SOURCE", "mis")
    probs, det = orc.gate_mod.intraday_basis(now=now)
    assert any("筆數為零" in p for p in probs)
    mt = (now - dt.timedelta(seconds=30)).isoformat(timespec="seconds")
    ft = (now - dt.timedelta(seconds=20)).isoformat(timespec="seconds")
    # (d) 時間戳未知（不可解析）→ 拒絕
    _set_quote([_quote_row("mis", "realtime", "not-a-time", ft)])
    probs, _ = orc.gate_mod.intraday_basis(now=now)
    assert any("行情時間未知" in p for p in probs)
    # (e) 過期行情（10 分鐘前 > 300 秒）→ 拒絕
    old = (now - dt.timedelta(seconds=600)).isoformat(timespec="seconds")
    _set_quote([_quote_row("mis", "realtime", old, old)])
    probs, _ = orc.gate_mod.intraday_basis(now=now)
    assert any("行情過期" in p for p in probs)
    # (f) 行情時間晚於系統時間（時鐘不一致）→ 拒絕
    fut = (now + dt.timedelta(seconds=300)).isoformat(timespec="seconds")
    _set_quote([_quote_row("mis", "realtime", fut, ft)])
    probs, _ = orc.gate_mod.intraday_basis(now=now)
    assert any("時鐘不一致" in p for p in probs)
    # (g) 新鮮、即時、通過既有 intraday_signal_gate → 允許
    _set_quote([_quote_row("mis", "realtime", mt, ft)])
    probs, det = orc.gate_mod.intraday_basis(now=now)
    assert probs == [], probs
    # 「已成功抓取」與「足以支援交易」分開記錄
    q = det["quote"]
    assert q["fetched_at"] and q["market_time"] and q["gate_ok"] is True
    # (h) 抓取時間缺失 → 拒絕（無法確認來源有運作）
    _set_quote([_quote_row("mis", "realtime", mt, "")])
    probs, _ = orc.gate_mod.intraday_basis(now=now)
    assert any("抓取時間" in p for p in probs)


# ---------------------------------------------------------------------------
# 4. 資料來源部分失敗時正確標記狀態（不得誤報整輪成功）
# ---------------------------------------------------------------------------
def test_partial_ingest_marks_round_partial(p5):
    def partial(ctx):
        return orc.StageResult(IngestStatus.PARTIAL_SUCCESS.value, rows=2,
                               error="2317 failed")
    res = _run(fns={"ingest": partial})
    assert res.stages["ingest"]["status"] == IngestStatus.PARTIAL_SUCCESS.value
    assert res.status == orc.ROUND_PARTIAL
    assert res.status != orc.ROUND_SUCCESS, "單一來源失敗不得誤報整輪成功"
    assert res.stages["ingest"]["error"], "部分失敗必須留下錯誤摘要"


def test_ingest_failure_blocks_paper_stage(p5):
    """根資料集全數失敗 → paper 階段直接阻擋（fail-closed）。"""
    def dead_ingest(ctx):
        raise RuntimeError("price source down")
    res = _run(fns={"ingest": dead_ingest}, attempts=2)
    assert res.stages["ingest"]["status"] == IngestStatus.FAILED.value
    assert res.stages["paper"]["status"] == IngestStatus.SKIPPED.value
    assert res.stages["paper"]["kind"] == orc.KIND_BLOCKED
    assert "上游階段失敗" in res.stages["paper"]["error"]
    assert len(_rows("SELECT 1 FROM paper_order")) == 0


# ---------------------------------------------------------------------------
# 5. 同一工作重複執行不產生重複訂單/成交/帳本
# ---------------------------------------------------------------------------
def test_rerun_same_day_no_duplicate_orders(p5):
    res1 = _run()
    assert res1.status == orc.ROUND_SUCCESS
    n_o, n_f, n_l = (len(_rows("SELECT 1 FROM paper_order")),
                     len(_rows("SELECT 1 FROM paper_fill")),
                     len(_rows("SELECT 1 FROM paper_ledger")))
    assert (n_o, n_f, n_l) == (1, 1, 2)
    res2 = _run()                                    # 同日再跑一輪
    assert res2.run_id != res1.run_id
    assert len(_rows("SELECT 1 FROM paper_order")) == n_o
    assert len(_rows("SELECT 1 FROM paper_fill")) == n_f
    assert len(_rows("SELECT 1 FROM paper_ledger")) == n_l
    assert p5.reconcile()["ok"] is True


def test_resume_skips_completed_stages(p5):
    """同一輪續跑：已完成階段不重複執行、不重複下單。"""
    calls = {"n": 0}

    def counting(ctx):
        calls["n"] += 1
        return orc.StageResult(IngestStatus.SUCCESS_WITH_DATA.value, rows=1)

    res1 = _run(fns={"ingest": counting})
    assert calls["n"] == 1
    # 以同一 run_id 續跑（模擬重試/重啟後再執行同一輪）
    res2 = orc.run_round(now=NOW, run_id=res1.run_id, force=True,
                         sleep=lambda s: None,
                         fns={"ingest": counting, "signals": _ok_signals()})
    assert res2.resumed is True and res2.run_id == res1.run_id
    assert calls["n"] == 1, "已完成的 ingest 階段不得重複執行"
    assert len(_rows("SELECT 1 FROM paper_order")) == 1


def test_round_due_scheduling_idempotence(p5):
    """排程層冪等：週末/未到點不執行；當日已有成功輪則不再跑。"""
    ok, why = orc.round_due(dt.datetime(2026, 2, 7, 16, 0))   # 週六
    assert ok is False and "weekend" in why
    ok, why = orc.round_due(dt.datetime(2026, 2, 5, 10, 0))   # 盤中（未到 EOD）
    assert ok is False and "before_eod" in why
    res1 = _run()                                             # 手動輪（15:00）
    assert res1.status == orc.ROUND_SUCCESS
    ok, why = orc.round_due(dt.datetime(2026, 2, 5, 16, 0))   # 同日 16:00
    assert ok is False and "already_done_today" in why
    # scheduled 輪（force=False）→ 依排程判定為 skipped，不重複下單
    n_orders = len(_rows("SELECT 1 FROM paper_order"))
    res2 = orc.run_round(now=dt.datetime(2026, 2, 5, 16, 0),
                         trigger="scheduled", force=False,
                         sleep=lambda s: None)
    assert res2.status == orc.ROUND_SKIPPED
    assert "already_done_today" in res2.error
    assert len(_rows("SELECT 1 FROM paper_order")) == n_orders


# ---------------------------------------------------------------------------
# 6. 同時啟動兩個協調器 → 後到者被拒絕
# ---------------------------------------------------------------------------
def test_second_coordinator_rejected(p5):
    # 以真實子程序持有鎖（跨行程語意；同行程 PID 相同會被視為自身）
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    lock_path = Path(store.DATA_DIR) / "state" / "p5run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(f"{holder.pid} 2026-02-05T15:00:00\n", encoding="utf-8")
    try:
        res = _run()
        assert res.status == orc.ROUND_SKIPPED and res.busy is True
        assert "已有 P5 輪次在執行" in res.error
        assert len(_rows("SELECT 1 FROM run_stage")) == 0
    finally:
        holder.terminate()
        holder.wait(timeout=30)
        lock_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 7. 執行中途崩潰後重新啟動（真實子程序崩潰 + 殘留鎖回收）
# ---------------------------------------------------------------------------
def test_crash_mid_round_then_restart_resumes(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(risk_mod, "DATA_DIR", tmp_path)
    monkeypatch.delenv("TWSTOCK_QUOTE_SOURCE", raising=False)
    _seed(BASE_BARS)
    PaperEngine().ensure_account(initial_cash=1_000_000)
    db = str(Path(store.DB_PATH).resolve())
    data_dir = str(Path(store.DATA_DIR).resolve())
    rid = "p5-20260205T120000-cra5"
    child = f"""
import os, sys
os.environ['TWSTOCK_DB_PATH'] = {db!r}
os.environ['TWSTOCK_DATA_DIR'] = {data_dir!r}
import datetime as dt
from twstock import orchestrator as orc
from twstock.data.status import IngestStatus
now = dt.datetime(2026, 2, 5, 12, 0)
rid = {rid!r}
orc._insert_round(rid, 'manual', now)
orc._begin_stage(rid, 'ingest', now, attempts=1)
orc._end_stage(rid, 'ingest', now,
               orc.StageResult(IngestStatus.SUCCESS_WITH_DATA.value, rows=3),
               duration_s=1.0, attempts=1)
orc._begin_stage(rid, 'signals', now, attempts=1)   # 執行中…
os._exit(9)                                          # 硬崩潰（無收尾）
"""
    proc = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, timeout=120,
        text=True, cwd=str(Path(orc.__file__).resolve().parent.parent))
    assert proc.returncode == 9, f"子程序應以 os._exit(9) 硬崩潰：{proc.stderr}"
    # 崩潰現場：輪次未結束、signals 階段停在執行中
    assert orc.open_round_id() == rid
    assert _rows("SELECT ended_at FROM run_stage WHERE run_id=? AND stage='signals'",
                 (rid,)).iloc[0]["ended_at"] is None
    # 殘留鎖（PID 已死）必須可回收
    (Path(store.DATA_DIR) / "state").mkdir(parents=True, exist_ok=True)
    (Path(store.DATA_DIR) / "state" / "p5run.lock").write_text(
        "4194304 2026-02-05T12:00:00\n", encoding="utf-8")
    calls = {"ingest": 0, "signals": 0}

    def cing(ctx):
        calls["ingest"] += 1
        return orc.StageResult(IngestStatus.SUCCESS_WITH_DATA.value, rows=3)

    def csig(ctx):
        calls["signals"] += 1
        return orc.StageResult(IngestStatus.SUCCESS_WITH_DATA.value, rows=1)

    res = _run(fns={"ingest": cing, "signals": csig})
    assert res.resumed is True and res.run_id == rid
    assert calls["ingest"] == 0, "崩潰前已完成的階段不得重複執行"
    assert calls["signals"] == 1, "崩潰時執行中的階段必須重做"
    assert res.ended_at, "重啟後輪次必須正常收尾"
    assert res.status in (orc.ROUND_SUCCESS, orc.ROUND_PARTIAL)
    assert orc.open_round_id() is None


# ---------------------------------------------------------------------------
# 8. 資料庫交易失敗後帳務保持一致（無半套成交/帳本）
# ---------------------------------------------------------------------------
def test_paper_failure_keeps_accounting_consistent(p5):
    def failing_paper(ctx):
        eng = PaperEngine()
        r = eng.place_order(stock_id="2330", side="BUY", qty=2000,
                            client_order_id="boom", signal_date=SIG_DAY,
                            now=NOW)
        assert r["accepted"] is True

        def boom(*a, **k):
            raise RuntimeError("simulated db failure")

        orig = eng._ledger
        eng._ledger = boom
        try:
            with pytest.raises(RuntimeError):
                eng.process_pending(now=NOW)
        finally:
            eng._ledger = orig
        raise RuntimeError("paper stage crashed after partial work")

    res = _run(fns={"paper": failing_paper})
    assert res.stages["paper"]["status"] == IngestStatus.FAILED.value
    # 回滾：無成交、無帳本事件，訂單維持未成交
    assert len(_rows("SELECT 1 FROM paper_fill")) == 0
    assert list(_rows("SELECT kind FROM paper_ledger")["kind"]) == ["capital"]
    order = _rows("SELECT status, filled_qty FROM paper_order").iloc[0]
    assert order["status"] == "accepted" and int(order["filled_qty"]) == 0
    assert p5.reconcile()["ok"] is True
    assert res.status == orc.ROUND_PARTIAL


# ---------------------------------------------------------------------------
# 9. kill switch：阻止新交易、記錄原因、不自動解除
# ---------------------------------------------------------------------------
def test_kill_switch_blocks_and_records(p5, tmp_path):
    ks = risk_mod.KillSwitch()
    ks.engage("P5 測試緊急停止")
    try:
        res = _run()
        assert res.stages["paper"]["status"] == IngestStatus.SKIPPED.value
        assert "Kill Switch" in str(res.stages["paper"]["error"])
        assert res.status == orc.ROUND_PARTIAL, "停止交易不得被報成整輪成功"
        alerts = orc.recent_alerts()
        assert any("Kill Switch" in str(a["message"]) for a in alerts)
        assert ks.engaged(), "kill switch 不得自動解除"
        res2 = _run()
        assert res2.stages["paper"]["status"] == IngestStatus.SKIPPED.value
        assert len(_rows("SELECT 1 FROM paper_order")) == 0
        assert ks.engaged()
    finally:
        ks.release()


# ---------------------------------------------------------------------------
# 10. 帳務核對不一致：停止新交易、只回報、不改帳
# ---------------------------------------------------------------------------
def test_inconsistent_account_blocks_and_reports(p5, monkeypatch):
    _run()                                     # 先建立一筆成交
    assert len(_rows("SELECT 1 FROM paper_fill")) == 1
    store.exec_sql("UPDATE paper_account SET cash_utwd = cash_utwd + 12345")
    before = {t: _rows(f"SELECT * FROM {t}").to_json()
              for t in ("paper_order", "paper_fill", "paper_ledger",
                        "paper_position")}
    # 稽核發現不一致，且**只回報**
    aud = orc.audit_report(now=NOW)
    assert aud["ok"] is False and aud["readonly"] is True
    rec = [c for c in aud["checks"] if c["name"] == "reconcile"][0]
    assert rec["ok"] is False and any("重建現金" in p for p in rec["problems"])
    # 唯讀證明：封死寫入函式後稽核仍可執行
    orig_wf, orig_es = store.write_df, store.exec_sql

    def _boom(*a, **k):
        raise AssertionError("稽核不得寫入資料庫")

    store.write_df = _boom
    store.exec_sql = _boom
    try:
        aud2 = orc.audit_report(now=NOW)
    finally:
        store.write_df, store.exec_sql = orig_wf, orig_es
    assert aud2["ok"] is False
    # 閘門阻擋新單與成交（fail-closed）
    g = orc.gate_mod.evaluate(now=NOW, engine=p5)
    assert not g.allow_new_orders and not g.allow_fills
    assert any("帳務核對不一致" in r for r in g.reasons)
    r = p5.place_order(stock_id="2330", side="BUY", qty=1000,
                       client_order_id="blocked", signal_date=SIG_DAY, now=NOW)
    assert r["accepted"] is False
    assert any("帳務" in x for x in r["reasons"])
    # 未自動修帳：稽核/健康/閘門再跑一輪後，帳務表與現金完全不變
    before = {t: _rows(f"SELECT * FROM {t}").to_json()
              for t in ("paper_order", "paper_fill", "paper_ledger",
                        "paper_position")}
    cash_before = int(p5.account()["cash_utwd"])
    orc.audit_report(now=NOW)
    orc.health_report(now=NOW)
    after = {t: _rows(f"SELECT * FROM {t}").to_json() for t in before}
    assert before == after
    assert int(p5.account()["cash_utwd"]) == cash_before, "不得静默修帳"
    # 再跑一輪：paper 階段被阻擋
    res = _run()
    assert res.stages["paper"]["status"] == IngestStatus.SKIPPED.value


# ---------------------------------------------------------------------------
# 11. 逾時與重試上限
# ---------------------------------------------------------------------------
def test_round_timeout_marks_remaining_stages_skipped(p5):
    def slow(ctx):
        time.sleep(0.5)                    # 單一階段超過整輪預算
        return orc.StageResult(IngestStatus.SUCCESS_WITH_DATA.value, rows=1)

    fns = {s: slow for s in orc.STAGES}
    res = orc.run_round(now=NOW, fns=fns, deadline_s=0.3,
                        sleep=lambda s: None)
    assert res.stages["ingest"]["status"] == IngestStatus.SUCCESS_WITH_DATA.value
    for s in orc.STAGES[1:]:
        assert res.stages[s]["status"] == IngestStatus.SKIPPED.value
        assert "逾時" in str(res.stages[s]["error"])
    assert res.status != orc.ROUND_SUCCESS, "逾時輪不得被報成整輪成功"


def test_stage_retry_limit_with_backoff(p5):
    sleeps: list[float] = []

    def always_fail(ctx):
        raise RuntimeError("source down")

    res = orc.run_round(now=NOW, fns={"ingest": always_fail}, attempts=3,
                        sleep=sleeps.append)
    assert res.stages["ingest"]["status"] == IngestStatus.FAILED.value
    assert sleeps == [1.0, 2.0], "必須指數退避且有次數上限"
    row = _rows("SELECT attempts, duration_s FROM run_stage WHERE run_id=?"
                " AND stage='ingest'", (res.run_id,)).iloc[0]
    assert int(row["attempts"]) == 3            # 含首次共 3 次，不無限重試
    assert float(row["duration_s"]) >= 0.0
    assert "RuntimeError" in res.stages["ingest"]["error"]


# ---------------------------------------------------------------------------
# 12. 非交易日與盤中、盤後時間邊界
# ---------------------------------------------------------------------------
def test_session_boundaries(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(risk_mod, "DATA_DIR", tmp_path)
    monkeypatch.delenv("TWSTOCK_QUOTE_SOURCE", raising=False)
    _seed(BASE_BARS)
    eng = PaperEngine()
    eng.ensure_account(initial_cash=1_000_000)
    # 週末（policy）：新單禁止，但不影響帳務與既有成交
    g = orc.gate_mod.evaluate(now=dt.datetime(2026, 2, 7, 15, 0), engine=eng)
    assert not g.allow_new_orders and g.kind == orc.KIND_POLICY
    assert g.allow_fills
    # 盤前（08:45）：允許 EOD 基準新單（成交仍走 T+1 開盤規則）
    g2 = orc.gate_mod.evaluate(now=dt.datetime(2026, 2, 5, 8, 45), engine=eng)
    assert g2.allow_new_orders and g2.basis == orc.gate_mod.Basis.EOD.value
    # 盤中（10:00）：未設定可驗證即時來源 → fail-closed 拒絕
    g3 = orc.gate_mod.evaluate(now=dt.datetime(2026, 2, 5, 10, 0), engine=eng)
    assert not g3.allow_new_orders and g3.kind == orc.KIND_BLOCKED
    assert g3.basis == orc.gate_mod.Basis.INTRADAY.value
    assert any("盤中" in r for r in g3.reasons)
    # 盤後（15:00）：允許（本專案設計路徑）
    g4 = orc.gate_mod.evaluate(now=NOW, engine=eng)
    assert g4.allow_new_orders and g4.basis == orc.gate_mod.Basis.EOD.value
    # 週末直接下單 → 拒單且理由可稽核
    r = eng.place_order(stock_id="2330", side="BUY", qty=1000,
                        client_order_id="weekend", signal_date=SIG_DAY,
                        now=dt.datetime(2026, 2, 7, 15, 0))
    assert r["accepted"] is False
    assert any("非交易日" in x for x in r["reasons"])


# ---------------------------------------------------------------------------
# 13. 健康狀態能辨認「程序活著但資料已過期」
# ---------------------------------------------------------------------------
def test_health_detects_alive_but_stale(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(risk_mod, "DATA_DIR", tmp_path)
    _seed([_bar("2026-01-28", 40.0, close=41.0)])     # 延遲 8 天
    now = NOW
    worker.write_heartbeat(status="running",
                           last_success="2026-01-28", consecutive_failures=0)
    pid_path = Path(store.DATA_DIR) / "state" / "worker.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")   # 本行程＝存活
    h = orc.health_report(now=now)
    assert h["process"]["alive"] is True, "程序活著"
    assert h["data"]["fresh"] is False, "但資料已過期"
    assert h["state"] == "data_stale" and h["ok"] is False
    assert h["data"]["staleness_days"] == 8
    # 健康階段會把此狀態寫進輪次與事件
    res = _run()
    detail = json.loads(
        _rows("SELECT detail FROM run_stage WHERE run_id=? AND stage='health'",
              (res.run_id,)).iloc[0]["detail"])
    assert detail["state"] == "data_stale" and detail["alive"] is True


# ---------------------------------------------------------------------------
# 14. 測試過程不改動正式資料庫內容
# ---------------------------------------------------------------------------
def test_isolation_from_real_db(p5, tmp_path):
    from twstock import config
    assert str(Path(store.DB_PATH).resolve()).startswith(str(tmp_path.resolve()))
    assert Path(store.DB_PATH).resolve() != Path(config.DB_PATH).resolve(), \
        "測試不得指向正式 DB"
    _run()                                        # 全輪寫入都落在暫存 DB
    assert str(Path(store.DB_PATH).resolve()).startswith(str(tmp_path.resolve()))


def test_readonly_mode_skips_round(p5, monkeypatch):
    monkeypatch.setenv("TWSTOCK_READONLY", "1")
    res = orc.run_round(now=NOW)
    assert res.status == orc.ROUND_SKIPPED
    assert "唯讀" in res.error


def test_readonly_legacy_db_without_p5_tables_is_tolerated(tmp_path, monkeypatch,
                                                           capsys):
    """舊 DB（尚無 P5 三表）以唯讀開啟 → 監控查詢回空結果＋提示，不得崩潰。"""
    import sqlite3
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setenv("TWSTOCK_READONLY", "1")
    assert orc.p5_schema_ready() is False
    assert orc.recent_rounds(3) == []
    assert orc.recent_alerts() == []
    data = orc.recent_errors()
    assert data["schema_ready"] is False and data["stages"] == []
    # CLI：status/errors/alerts 皆明確提示且退出碼 0（不崩潰）
    assert orc.main(["status"]) == 0 and "尚未建立" in capsys.readouterr().out
    assert orc.main(["errors"]) == 0 and "尚未建立" in capsys.readouterr().out
    assert orc.main(["alerts"]) == 0 and "尚未建立" in capsys.readouterr().out
    # 唯讀下 run_round 直接跳過；稽核回報問題而非崩溃
    assert orc.run_round(now=NOW).status == orc.ROUND_SKIPPED
    aud = orc.audit_report(now=NOW)
    assert aud["ok"] is False and aud["readonly"] is True


# ---------------------------------------------------------------------------
# 補充：run_paper_cycle 的閘門回報、唯讀 API、CLI、worker 整合
# ---------------------------------------------------------------------------
def test_run_paper_cycle_reports_gate(p5):
    res = p5.run_paper_cycle(now=NOW, run_id="p5-test")
    assert res["gate"]["allow_new_orders"] is True
    # 閘門關閉時：一張新單都不建立，理由入稽核欄位；既有訂單成交與快照照常
    store.exec_sql("UPDATE paper_account SET cash_utwd = cash_utwd + 1")
    res2 = p5.run_paper_cycle(now=NOW, run_id="p5-test2")
    assert res2["gate"]["allow_new_orders"] is False
    assert res2["orders"] == [] and "blocked" in res2
    assert any("帳務" in r for r in res2["blocked"])
    assert "snapshot" in res2


def test_p5_api_readonly(temp_db, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from twstock.app import app
    monkeypatch.setattr(risk_mod, "DATA_DIR", tmp_path)
    monkeypatch.delenv("TWSTOCK_QUOTE_SOURCE", raising=False)
    _seed(BASE_BARS)
    PaperEngine().ensure_account(initial_cash=1_000_000)
    client = TestClient(app)
    for path in ("/api/p5/health", "/api/p5/audit", "/api/p5/gate",
                 "/api/p5/runs", "/api/p5/alerts"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        body = resp.json()
        assert "error" not in body, f"{path}: {body}"
    assert client.get("/api/p5/audit").json()["readonly"] is True
    post = [r.path for r in app.routes
            if "/p5" in getattr(r, "path", "") and "POST" in
            getattr(r, "methods", set())]
    assert post == [], "監控 API 不得有下單/寫入端點"


def test_cli_p5_subcommands(p5, monkeypatch, capsys):
    from twstock import cli
    monkeypatch.setattr("sys.argv", ["twstock.cli", "p5", "status"])
    assert cli.main() == 0
    # audit / gate 以固定種子日無法對齊真實時鐘 → 以替身驗證轉交與輸出
    monkeypatch.setattr(
        orc, "audit_report",
        lambda **kw: {"ok": True, "readonly": True,
                      "checks": [{"name": "reconcile", "ok": True,
                                  "detail": "一致"}], "problems": []})
    monkeypatch.setattr("sys.argv", ["twstock.cli", "p5", "audit"])
    assert cli.main() == 0
    monkeypatch.setattr(
        orc.gate_mod, "evaluate",
        lambda **kw: orc.gate_mod.GateDecision(allow_new_orders=True,
                                               session="closed"))
    monkeypatch.setattr("sys.argv", ["twstock.cli", "p5", "gate"])
    assert cli.main() == 0
    monkeypatch.setattr("sys.argv", ["twstock.cli", "p5", "errors"])
    assert cli.main() == 0
    # run 轉交協調器（避免測試真的 ingest 連外）
    monkeypatch.setattr(
        orc, "run_round",
        lambda **kw: orc.RoundResult(run_id="p5-x", status=orc.ROUND_SUCCESS))
    monkeypatch.setattr("sys.argv", ["twstock.cli", "p5", "run"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "P5" in out or "p5-x" in out


def test_worker_p5_task_integration(monkeypatch, temp_db, tmp_path):
    monkeypatch.setattr(risk_mod, "DATA_DIR", tmp_path)
    called: dict = {}

    def fake_round(**kw):
        called.update(kw)
        return orc.RoundResult(run_id="p5-t", status=kw.pop("_status", "success"))

    monkeypatch.setattr(orc, "run_round",
                        lambda **kw: fake_round(**kw, _status="success"))
    out = worker._run_p5_task(attempts=1)
    assert out.ok and out.value == "success"
    assert called.get("trigger") == "worker"

    monkeypatch.setattr(orc, "run_round",
                        lambda **kw: fake_round(**kw, _status="partial_success"))
    out2 = worker._run_p5_task(attempts=1)
    assert out2.ok and out2.value == "partial_success", \
        "部分成功不計為任務失敗，但狀態必須如實顯示"

    def failed(**kw):
        return orc.RoundResult(run_id="p5-t", status=orc.ROUND_FAILED,
                               error="stage exploded")
    monkeypatch.setattr(orc, "run_round", failed)
    out3 = worker._run_p5_task(attempts=2)
    assert out3.ok is False and "stage exploded" in out3.error


def test_alerts_record_delivery_state(p5, monkeypatch):
    """事件必須如實記錄送達狀態：未設定通道 → not_configured（不假裝已送達）。"""
    monkeypatch.delenv("TWSTOCK_ALERT_TELEGRAM", raising=False)
    delivered = orc.record_alert("error", "測試事件：帳務不一致", run_id="p5-x",
                                 stage="reconcile", kind="test")
    assert delivered == "not_configured"
    rows = orc.recent_alerts()
    assert rows[0]["delivered"] == "not_configured"
    assert rows[0]["message"].startswith("測試事件")
    # 同輪同類同訊息去重（重跑不重複刷屏）
    assert orc.record_alert("error", "測試事件：帳務不一致", run_id="p5-x",
                            stage="reconcile", kind="test") == "dedup"
    assert len([a for a in orc.recent_alerts()
                if a["message"].startswith("測試事件")]) == 1
