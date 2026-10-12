"""Telegram 推送：每日訊號摘要。"""
from __future__ import annotations

import logging
import os

import pandas as pd
import requests

log = logging.getLogger(__name__)


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.info("未設定 Telegram，略過推送。訊號內容：\n%s", text)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("Telegram 推送失敗: %s", e)
        return False


def format_daily_signals(date: str, top: pd.DataFrame) -> str:
    lines = [f"📈 台股波段訊號 {date}", ""]
    for _, r in top.iterrows():
        emoji = "🟢" if r["action"] == "BUY" else ("🔴" if r["action"] == "AVOID" else "⚪")
        lines.append(f"{emoji} {r['stock_id']}  評分 {r['score']:+.2f}  建議 {r['action']}")
        if r.get("reason"):
            lines.append(f"   理由：{r['reason']}")
    lines.append("")
    lines.append("⚠️ 僅供研究參考，非投資建議")
    return "\n".join(lines)
