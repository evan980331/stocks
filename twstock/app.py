"""FastAPI 儀表板：K線、訊號、新聞情緒。"""
from __future__ import annotations

import datetime as dt
import html as html_lib
import logging

import pandas as pd
from fastapi import FastAPI
from fastapi import Body
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse

from .config import CONFIG
from .data import store
from .signals import ACTION_CN

log = logging.getLogger(__name__)

app = FastAPI(title="台股預測儀表板")

# 欄位中文對照
COL_CN = {
    "rank": "排名",
    "stock_id": "股票代號",
    "stock_name": "股票名稱",
    "date": "日期",
    "published": "發布日",
    "score": "評分",
    "action": "建議",
    "reason": "理由",
    "close": "收盤價",
    "open": "開盤價",
    "high": "最高價",
    "low": "最低價",
    "volume": "成交量",
    "title": "新聞標題",
    "ml_score": "模型分數",
}

HTML = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>台股預測儀表板</title>
<style>
body {{ font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
       background:#0f1420; color:#e6e9f0; margin:0; padding:24px; }}
h1 {{ color:#7fd1ff; margin-bottom:8px; }}
h3 {{ margin-top:0; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(340px,1fr)); gap:16px; }}
.card {{ background:#171e2e; border-radius:12px; padding:16px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th,td {{ padding:6px 10px; text-align:left; border-bottom:1px solid #2a3448;
        white-space:nowrap; }}
td.reason-cell, td:last-child {{ white-space:normal; }}
th {{ color:#7fd1ff; font-weight:600; }}
.buy {{ color:#4caf50; font-weight:700; }} .avoid {{ color:#ef5350; font-weight:700; }}
.hold {{ color:#ffca28; font-weight:700; }}
.pos {{ color:#4caf50; }} .neg {{ color:#ef5350; }}
.num {{ text-align:right; font-variant-numeric: tabular-nums; }}
.empty {{ color:#8b95a7; padding:8px 0; }}
.reason {{ font-size:13px; color:#aeb8cc; }}
/* ===== AI 側邊小視窗 ===== */
#aiToggle {{ position:fixed; right:20px; bottom:20px; z-index:60;
  background:linear-gradient(135deg,#7b2ff7,#2f80ed); color:#fff; border:0;
  border-radius:24px; padding:12px 18px; font-size:15px; cursor:pointer;
  box-shadow:0 6px 20px rgba(47,128,237,.45); }}
#aiPanel {{ position:fixed; top:0; right:0; height:100%; width:380px; max-width:92vw;
  background:#141b2d; border-left:1px solid #2a3448; z-index:70;
  transform:translateX(105%); transition:transform .25s ease;
  display:flex; flex-direction:column; box-shadow:-8px 0 28px rgba(0,0,0,.45); }}
#aiPanel.open {{ transform:translateX(0); }}
#aiPanel header {{ padding:16px; border-bottom:1px solid #2a3448;
  display:flex; justify-content:space-between; align-items:center; }}
#aiPanel header b {{ color:#7fd1ff; font-size:16px; }}
#aiClose {{ background:none; border:0; color:#8b95a7; font-size:20px; cursor:pointer; }}
#aiBody {{ padding:16px; overflow-y:auto; flex:1; white-space:pre-wrap;
  font-size:14px; line-height:1.75; color:#e6e9f0; }}
#aiAsk {{ padding:12px 16px; border-top:1px solid #2a3448; display:flex; gap:8px; }}
#aiAsk input {{ flex:1; background:#0f1420; border:1px solid #2a3448;
  border-radius:8px; padding:9px 10px; color:#e6e9f0; font-size:14px; }}
#aiAsk button {{ background:#2f80ed; border:0; border-radius:8px; color:#fff;
  padding:0 14px; cursor:pointer; }}
#aiAsk button:disabled {{ opacity:.5; cursor:wait; }}
.ai-msg {{ margin-bottom:14px; }}
.ai-tag {{ font-size:12px; color:#7fd1ff; margin-bottom:4px; }}
.ai-loading {{ color:#8b95a7; }}
</style></head><body>
<h1>📊 台股預測儀表板</h1>
<p id="statusStrip" style="color:#aeb8cc;font-size:13px;margin:0 0 14px">{status_strip}</p>
<div class="grid">
  <div class="card"><h3>🎯 最新訊號 Top {topn}</h3>{signals_table}</div>
  <div class="card"><h3>📰 最新新聞情緒</h3>{news_table}</div>
</div>
<div class="card" style="margin-top:16px"><h3>📈 各股最新報價</h3>{price_table}</div>
<button id="aiToggle">🤖 AI 建議</button>
<aside id="aiPanel">
  <header><b>🤖 AI 投資建議</b><button id="aiClose">✕</button></header>
  <div id="aiBody"><div class="ai-loading">點「產生建議」開始。</div></div>
  <div id="aiAsk">
    <input id="aiQ" placeholder="問 AI 任何問題（例：台積電能追嗎？）">
    <button id="aiGo">產生</button>
  </div>
</aside>
<script>
const panel=document.getElementById('aiPanel'),
      body=document.getElementById('aiBody'),
      go=document.getElementById('aiGo'),
      q=document.getElementById('aiQ');
document.getElementById('aiToggle').onclick=()=>panel.classList.add('open');
document.getElementById('aiClose').onclick=()=>panel.classList.remove('open');
function addMsg(tag,text){{
  const d=document.createElement('div'); d.className='ai-msg';
  const t=document.createElement('div'); t.className='ai-tag'; t.textContent=tag;
  const b=document.createElement('div'); b.textContent=text;
  d.appendChild(t); d.appendChild(b); body.appendChild(d);
  body.scrollTop=body.scrollHeight; return d;
}}
async function askAI(){{
  const question=q.value.trim();
  go.disabled=true;
  const m=addMsg('思考中…','正在呼叫 Gemini，請稍候…');
  m.classList.add('ai-loading');
  try{{
    const r=await fetch('/api/ai',{{method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{question}})}});
    const j=await r.json();
    body.removeChild(m);
    addMsg(j.ok?'🤖 Gemini':'⚠️ 降級建議', j.text||j.error||'無回應');
  }}catch(e){{ body.removeChild(m); addMsg('⚠️ 錯誤',String(e)); }}
  go.disabled=false;
}}
go.onclick=askAI;
q.addEventListener('keydown',e=>{{ if(e.key==='Enter') askAI(); }});
</script>
</body></html>"""


LT = chr(60)   # HTML 標籤開頭
GT = chr(62)   # HTML 標籤結尾


def _cn_name(stock_id: str) -> str:
    return CONFIG.pool.get(stock_id, "")


def _df_to_html(df: pd.DataFrame, cols: list[str],
                action_col: str | None = None,
                score_col: str | None = None,
                num_cols: tuple = ()) -> str:
    """將 DataFrame 轉為中文表頭的 HTML 表格（空表回空態訊息）。"""
    if df.empty:
        return LT + 'p class="empty"' + GT + "尚無資料" + LT + "/p" + GT
    d = df[cols].copy()
    # 日期欄去掉時間部分（同時相容字串與 datetime64）
    for c in d.columns:
        if c in ("date", "published"):
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                d[c] = d[c].dt.strftime("%Y-%m-%d")
            else:
                d[c] = d[c].map(lambda v: str(v)[:10] if pd.notna(v) else "")
    # 先轉義原文，後續再包上 HTML 標籤（escape=False 需自行處理）
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].map(lambda v: html_lib.escape(str(v)) if pd.notna(v) else "")

    if action_col and action_col in d.columns:
        d[action_col] = d[action_col].map(lambda a: ACTION_CN.get(a, a))
        d[action_col] = [
            LT + 'span class="' + ("buy" if v == "買進" else
                                   "avoid" if v == "迴避" else "hold") + '"'
            + GT + str(v) + LT + "/span" + GT
            for v in d[action_col]
        ]

    if score_col and score_col in d.columns:
        d[score_col] = [
            (LT + 'span class="' + ("pos" if float(v) >= 0 else "neg") + '"'
             + GT + ("%+.3f" % float(v)) + LT + "/span" + GT)
            if pd.notna(v) else "" for v in d[score_col]
        ]

    for c in num_cols:
        if c in d.columns and c != score_col:
            d[c] = d[c].map(lambda v: "{:,.0f}".format(float(v)) if pd.notna(v) else "")

    if "reason" in d.columns:
        d["reason"] = d["reason"].map(
            lambda r: LT + 'span class="reason"' + GT + str(r) + LT + "/span" + GT)

    d.columns = [COL_CN.get(c, c) for c in d.columns]
    return d.to_html(index=False, border=0, classes="", justify="left", escape=False)


def _ensure_actions(df: pd.DataFrame) -> pd.DataFrame:
    """action 欄若未落庫則即時計算，避免儀表板一律顯示「持有」而誤導。"""
    from .signals import pick_top_n
    df = df.copy()
    if "action" not in df.columns:
        df["action"] = None
    if df["action"].isna().all() and len(df):
        date = df["date"].iloc[0]
        top = pick_top_n(df, date, top_n=len(df))
        df["action"] = df["stock_id"].map(
            top.set_index("stock_id")["action"]).fillna("HOLD")
    else:
        df["action"] = df["action"].fillna("HOLD")
    return df


def _status_strip() -> str:
    """資料新鮮度與系統狀態（全部來自真實執行狀態，不用假資料）。"""
    try:
        from .ops import health
        h = health()
    except Exception as e:  # noqa: BLE001
        return html_lib.escape(f"狀態無法取得：{e}")
    parts = [f"模式 {h.get('mode')}"]
    parts.append(f"最新行情 {h.get('latest_price_date') or '無'}")
    stale = h.get("data_staleness_days")
    if stale is not None:
        parts.append(f"延遲 {stale} 天" + (" ⚠️" if stale > 3 else ""))
    parts.append(f"最新訊號 {h.get('latest_signal_date') or '無'}")
    ml = h.get("ml", {})
    parts.append("ML " + ("可用" if ml.get("available") else f"停用({ml.get('reason','')})"))
    ks = h.get("kill_switch", {})
    if ks.get("engaged"):
        parts.append("🛑 Kill Switch 已啟用")
    w = h.get("worker", {})
    hb = (w.get("heartbeat") or {}).get("at")
    parts.append("Worker " + ("執行中" if w.get("alive") else "未執行") +
                 (f"（心跳 {hb}）" if hb else ""))
    if h.get("last_ingest_failed_ids"):
        parts.append("⚠️ ingest 失敗：" + str(h["last_ingest_failed_ids"]))
    return html_lib.escape(" ｜ ".join(str(p) for p in parts))


@app.get("/api/health")
def api_health() -> dict:
    """系統健康狀態（供本機監控與 script 檢查）。"""
    from .ops import health
    return jsonable_encoder(health())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    sig = store.read_sql(
        "SELECT * FROM signals WHERE date=(SELECT MAX(date) FROM signals) "
        "ORDER BY score DESC LIMIT 8")
    if not sig.empty:
        sig = _ensure_actions(sig)
        sig["stock_name"] = sig["stock_id"].map(_cn_name)
        sig = sig.sort_values("score", ascending=False).reset_index(drop=True)
        sig["rank"] = range(1, len(sig) + 1)

    news = store.read_sql(
        "SELECT published, stock_id, title, score FROM news "
        "ORDER BY published DESC LIMIT 12")
    if not news.empty:
        news["stock_name"] = news["stock_id"].map(_cn_name)

    price = store.read_sql(
        "SELECT p.stock_id, p.date, p.open, p.high, p.low, p.close, p.volume "
        "FROM price p WHERE p.date = (SELECT MAX(date) FROM price "
        "WHERE stock_id=p.stock_id) ORDER BY p.stock_id")
    if not price.empty:
        price["stock_name"] = price["stock_id"].map(_cn_name)

    sig_html = _df_to_html(
        sig, ["rank", "stock_id", "stock_name", "date", "score", "action", "reason"],
        action_col="action", score_col="score")
    news_html = _df_to_html(
        news, ["published", "stock_id", "stock_name", "score", "title"],
        score_col="score")
    price_html = _df_to_html(
        price, ["stock_id", "stock_name", "date", "open", "high", "low",
                "close", "volume"],
        num_cols=("open", "high", "low", "close", "volume"))
    return HTML.format(topn=CONFIG.top_n, signals_table=sig_html,
                       news_table=news_html, price_table=price_html,
                       status_strip=_status_strip())


@app.get("/api/signals")
def api_signals() -> list[dict]:
    """最新一期訊號；rank/action 若未落庫則即時計算。"""
    df = store.read_sql(
        "SELECT * FROM signals WHERE date=(SELECT MAX(date) FROM signals) "
        "ORDER BY score DESC"
    )
    if df.empty:
        return []
    df = _ensure_actions(df)
    df["rank"] = range(1, len(df) + 1)
    return jsonable_encoder(df.to_dict(orient="records"))


@app.get("/api/price/{stock_id}")
def api_price(stock_id: str, days: int = 120) -> list[dict]:
    df = store.read_sql(
        "SELECT date, open, high, low, close, volume FROM price WHERE stock_id=? "
        "ORDER BY date DESC LIMIT ?", (stock_id, days))
    return jsonable_encoder(df.to_dict(orient="records"))


@app.post("/api/ai")
def api_ai(body: dict = Body(default={})) -> dict:
    """AI 側邊視窗：有金鑰呼叫 Gemini，否則降級為規則式建議。"""
    from .ai import gemini
    question = (body or {}).get("question") or None
    if gemini.has_key():
        return gemini.ask(question)
    return {"ok": False, "text": gemini.rule_based_advice(),
            "error": "未設定 GEMINI_API_KEY"}


@app.get("/api/experts")
def api_experts(limit: int = 15) -> dict:
    """Phase 2：專家/名嘴觀點來源。"""
    df = store.read_sql(
        "SELECT published, source, title, score, kind FROM experts "
        "ORDER BY published DESC LIMIT ?", (limit,))
    if df.empty:
        return {"items": [], "hint": "尚無資料，本機執行: python -m twstock.cli experts"}
    return {"items": df.to_dict(orient="records"), "hint": ""}


# ---------------------------------------------------------------------------
# P4：模擬交易（paper）唯讀 API。所有端點僅查詢、不建立訂單、不成交；
# 下單與成交只在本機 CLI（python -m twstock.cli paper …）。
# ---------------------------------------------------------------------------
_PAPER_HINT = "尚未建立 paper 帳戶，本機執行：python -m twstock.cli paper init"


def _paper_records(df) -> list:
    """DataFrame → JSON 安全 records（NULL → None，避免 NaN 序列化 500）。"""
    if df is None or df.empty:
        return []
    clean = df.astype(object).where(df.notna(), None)
    return jsonable_encoder(clean.to_dict(orient="records"))


@app.get("/api/paper/account")
def api_paper_account() -> dict:
    """紙上帳戶總覽（唯讀）；含帳務一致性。"""
    from .paper import PaperEngine, PaperError
    eng = PaperEngine()
    try:
        acc = eng.account()
        if acc is None:
            return {"error": _PAPER_HINT}
        return {"summary": eng.account_text(),
                "reconcile": eng.reconcile()}
    except (PaperError, RuntimeError) as e:
        return {"error": str(e)}


@app.get("/api/paper/positions")
def api_paper_positions() -> dict:
    from .paper import PaperEngine, PaperError
    eng = PaperEngine()
    try:
        if eng.account() is None:
            return {"error": _PAPER_HINT}
        df = eng.positions_view()
        return {"items": _paper_records(df)}
    except (PaperError, RuntimeError) as e:
        return {"error": str(e)}


@app.get("/api/paper/orders")
def api_paper_orders(status: str | None = None, limit: int = 50) -> dict:
    from .paper import PaperEngine, PaperError
    eng = PaperEngine()
    try:
        if eng.account() is None:
            return {"error": _PAPER_HINT}
        df = eng.orders_view(status=status, limit=limit)
        return {"items": _paper_records(df)}
    except (PaperError, RuntimeError) as e:
        return {"error": str(e)}


@app.get("/api/paper/fills")
def api_paper_fills(limit: int = 50) -> dict:
    from .paper import PaperEngine, PaperError
    eng = PaperEngine()
    try:
        if eng.account() is None:
            return {"error": _PAPER_HINT}
        df = eng.fills_view(limit=limit)
        return {"items": _paper_records(df)}
    except (PaperError, RuntimeError) as e:
        return {"error": str(e)}


@app.get("/api/paper/ledger")
def api_paper_ledger(limit: int = 50) -> dict:
    from .paper import PaperEngine, PaperError
    eng = PaperEngine()
    try:
        if eng.account() is None:
            return {"error": _PAPER_HINT}
        df = eng.ledger_view(limit=limit)
        return {"items": _paper_records(df)}
    except (PaperError, RuntimeError) as e:
        return {"error": str(e)}


@app.get("/api/paper/risk")
def api_paper_risk() -> dict:
    """風控狀態（唯讀；含 kill switch 與拒絕原因）。"""
    from .paper import PaperEngine, PaperError
    eng = PaperEngine()
    try:
        return eng.risk_status()
    except (PaperError, RuntimeError) as e:
        return {"error": str(e)}


@app.get("/api/paper/reconcile")
def api_paper_reconcile() -> dict:
    """帳務一致性核對（唯讀；帳本重建 vs 帳戶/持倉/成交）。"""
    from .paper import PaperEngine, PaperError
    eng = PaperEngine()
    try:
        return eng.reconcile()
    except (PaperError, RuntimeError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# P5：自動化協調器唯讀監控 API（全部 GET；**無任何下單/成交端點**）。
# 回傳明確狀態、錯誤原因與最後更新時間；不一致時只回報、不修改資料。
# ---------------------------------------------------------------------------

def _safe(fn, *a, **kw) -> dict:
    """呼叫監控函式；失敗時回明確 error（不 500、不静默）。"""
    try:
        return jsonable_encoder(fn(*a, **kw))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "time": dt.datetime.now().isoformat(
            timespec="seconds")}


@app.get("/api/p5/health")
def api_p5_health() -> dict:
    """系統健康狀態（程序存活＋資料新鮮度＋錯誤率＋帳務＋閘門）。"""
    from .orchestrator import health_report, recent_rounds
    return {"health": _safe(health_report), "runs": _safe(recent_rounds, 3)}


@app.get("/api/p5/audit")
def api_p5_audit() -> dict:
    """帳務稽核（唯讀）；發現不一致只回報，不修帳。"""
    from .orchestrator import audit_report
    return _safe(audit_report)


@app.get("/api/p5/gate")
def api_p5_gate() -> dict:
    """行情品質閘門目前是否允許新增模擬交易（含原因）。"""
    from .gate import evaluate
    from .paper import PaperEngine
    eng = PaperEngine()
    try:
        engine = eng if eng.account() is not None else None
    except Exception:  # noqa: BLE001
        engine = None
    return _safe(lambda: evaluate(engine=engine).as_dict())


@app.get("/api/p5/runs")
def api_p5_runs(limit: int = 5) -> dict:
    """最近的自動化輪次與各階段結果（成功/部分成功/失敗/跳過）。"""
    from .orchestrator import recent_rounds
    return {"items": _safe(recent_rounds, limit)}


@app.get("/api/p5/alerts")
def api_p5_alerts(limit: int = 50) -> dict:
    """異常事件紀錄（含實際送達狀態；未設定通道時為 not_configured）。"""
    from .orchestrator import recent_alerts
    return {"items": _safe(recent_alerts, limit)}
