"""Gemini AI 建議引擎（Phase 2 側邊視窗）。

金鑰由 .env 的 GEMINI_API_KEY 提供；未設定時由 UI 降級為規則式建議。
使用官方 Interactions API（v1beta/interactions），舊 generateContent 已下線。
"""
from __future__ import annotations

import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # 確保 GEMINI_API_KEY 在模組載入時即可用

log = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

SYSTEM_PROMPT = (
    "你是一位嚴謹的台股研究助理。請用繁體中文回答，"
    "格式為：(1) 市場短評 (2) 個股觀察 (3) 風險提醒。"
    "每點一行，簡潔具體，避免空話。"
    "絕對不要保證獲利，並註明資料僅供研究參考。"
)


def has_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _safe(store, sql: str) -> pd.DataFrame:
    """表不存在 / 唯讀缺表時回空，不中斷。"""
    try:
        return store.read_sql(sql)
    except Exception as e:  # noqa: BLE001
        log.info("context query skipped: %s", e)
        return pd.DataFrame()


def build_context() -> str:
    """彙整目前訊號、新聞、專家觀點，組成給 LLM 的上下文。"""
    from ..data import store

    parts: list[str] = []

    sig = _safe(store,
                "SELECT stock_id, score, reason FROM signals "
                "WHERE date=(SELECT MAX(date) FROM signals) "
                "ORDER BY score DESC LIMIT 8")
    if not sig.empty:
        rows = [f"- {r.stock_id} 評分{r.score:+.2f}：{r.reason}"
                for r in sig.itertuples()]
        parts.append("【最新訊號 Top8】\n" + "\n".join(rows))

    news = _safe(store,
                 "SELECT stock_id, title, score FROM news "
                 "ORDER BY published DESC LIMIT 8")
    if not news.empty:
        rows = [f"- [{r.stock_id}] {r.title}（情緒{r.score:+.2f}）"
                for r in news.itertuples()]
        parts.append("【最新新聞】\n" + "\n".join(rows))

    exp = _safe(store,
                "SELECT source, title FROM experts "
                "ORDER BY published DESC LIMIT 8")
    if not exp.empty:
        rows = [f"- {r.source}：{r.title}" for r in exp.itertuples()]
        parts.append("【專家/名嘴觀點】\n" + "\n".join(rows))

    return "\n\n".join(parts) or "（尚無資料，請先於本機執行 ingest 與 run）"


def _extract_text(data: dict) -> str:
    """從 Interactions 回應抽文字；相容舊 generateContent 格式。"""
    out: list[str] = []
    for st in data.get("steps", []):
        for c in st.get("content", []):
            if isinstance(c, dict) and c.get("text"):
                out.append(c["text"])
    if out:
        return "".join(out).strip()
    if data.get("outputText"):
        return str(data["outputText"]).strip()
    for cand in data.get("candidates", []):
        for pt in cand.get("content", {}).get("parts", []):
            if pt.get("text"):
                out.append(pt["text"])
    return "".join(out).strip()


def ask(question: str | None = None, timeout: int = 30) -> dict:
    """呼叫 Gemini，回傳 {ok, text, error}。"""
    if not has_key():
        return {"ok": False, "text": "", "error": "未設定 GEMINI_API_KEY"}

    ctx = build_context()
    user = (question or "請依上述資料給出今日台股操作建議。").strip()
    payload = {
        "model": f"models/{DEFAULT_MODEL}",
        "input": f"{SYSTEM_PROMPT}\n\n{ctx}\n\n【問題】{user}",
    }
    key = os.environ["GEMINI_API_KEY"]
    import time
    last_err = ""
    # 503/429 為暫時性高需求，重試 3 次（指數退避）
    for attempt in range(3):
        try:
            r = requests.post(GEMINI_URL, headers={"x-goog-api-key": key},
                              json=payload, timeout=timeout)
            if r.status_code in (429, 503):
                last_err = f"HTTP {r.status_code}（服務繁忙）"
                log.warning("Gemini %s，%ds 後重試", r.status_code,
                            2 ** attempt)
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                log.warning("Gemini HTTP %s: %s", r.status_code, r.text[:300])
                return {"ok": False, "text": "", "error": f"HTTP {r.status_code}"}
            text = _extract_text(r.json())
            if not text:
                return {"ok": False, "text": "", "error": "空回應"}
            return {"ok": True, "text": text, "error": ""}
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            log.warning("Gemini 呼叫失敗: %s", e)
            time.sleep(2 ** attempt)
    return {"ok": False, "text": "", "error": last_err or "重試逾次"}


def rule_based_advice() -> str:
    """無金鑰時的降級建議（純規則，離線可用）。"""
    from ..data import store
    sig = _safe(store,
                "SELECT stock_id, score, action, reason FROM signals "
                "WHERE date=(SELECT MAX(date) FROM signals) "
                "ORDER BY score DESC LIMIT 5")
    if sig.empty:
        return ("尚無訊號資料。請先於本機執行：\n"
                "  python -m twstock.cli ingest\n"
                "  python -m twstock.cli run")
    lines = ["【規則式建議】（未設定 AI 金鑰）", ""]
    for r in sig.itertuples():
        act = {"BUY": "買進", "HOLD": "持有", "AVOID": "迴避"}.get(r.action, "持有")
        lines.append(f"・{r.stock_id}｜{act}｜評分 {r.score:+.2f}")
        lines.append(f"  依據：{r.reason}")
    lines += ["", "⚠️ 僅供研究參考，不構成投資建議。"]
    return "\n".join(lines)
