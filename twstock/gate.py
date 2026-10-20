"""P5-B：行情品質閘門（Market / Trading Gate）——把 P3 的資料品質接到 P4 成交路徑。

核心語意（**fail-closed**）
  無法確認行情新鮮度、資料來源失敗、資料筆數為零、帳務不一致或 Kill Switch
  啟用時，一律「停止新增模擬交易」並記錄原因；**不猜、不放行、不靜默**。

兩種「可以交易」必須分開（任務明確要求）
  1. ``allow_new_orders``：是否允許**建立新的模擬訂單**（買與賣都不例外）。
  2. ``allow_fills``：是否允許**既有未完成訂單繼續成交**。
     P4 既有語意是「行情過期只禁止新單，既定訂單仍可依已存行情成交」，
     因此資料新鮮度問題只擋新單；只有 Kill Switch 與**帳務不一致**才連成交
     都停（``risk.py``：發現失誤不得繼續交易）。
     **停止交易 ≠ 清倉**：本閘門不刪持倉、不主動平倉、不取消既有訂單。

「行情已成功抓取」≠「行情足以支援這次交易」（兩者分開判斷）
  - ``fetched_at``（抓取時間）只能證明**來源有被呼叫**。
  - ``market_time``（市場時間）才決定**資料夠不夠新**。
  拿到抓取時間但市場時間過期／未知 → 仍然禁止交易（見 ``intraday_basis``）。

時間語意
  - **行情時間**（``market_time``）與**系統時間**（``now``）分開判斷：
    行情時間晚於系統時間超過容忍值 → 時鐘不一致 → fail-closed。
  - 日線（price 表）以「最新交易日 ↔ 系統日期」的延遲天數判斷新鮮度；
    出現在**未來**的日線列由 ``quality`` 階段與成交時點規則攔截
    （``paper._attempt_fill`` 拒用晚於 ``now`` 的 bar；``signal_date`` 不得為
    未來）——因為回測/測試資料本來就會帶 T+1 的 bar，閘門不誤殺既有語意。
  - 只有日線或**延遲/盤後**資料時，**不得**假裝具備可交易的盤中行情：
    盤中（session=OPEN）建立新單一律要求「即時（realtime）來源 + 新鮮 +
    通過既有 ``intraday_signal_gate``」，否則拒絕。

時段規則（新單）
  - 週末（非交易日）：不新增模擬訂單（``kind=policy``）。
    **限制**：沒有國定假日曆（與 ``quotes.session_state`` 相同的已知限制），
    假日的防護由「盤中無新鮮即時行情」與「日線過期」兩道 fail-closed 補。
  - 盤前（08:30–09:00）：允許（依前一交易日收盤訊號，成交仍走 T+1 開盤規則）。
  - 盤中（09:00–13:30）：必須有可驗證的即時行情；未設定來源 → 拒絕。
  - 盤後（13:30–24:00、00:00–08:30）：允許（EOD 基準，本專案的設計路徑）。

本模組**只讀**：不寫任何表格、不發網路請求（不引入任何網路套件）。
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from . import risk as risk_mod
from .data import quotes
from .data import store

log = logging.getLogger(__name__)

#: 行情時間允許超前系統時間的容忍值（秒）；超過即視為時鐘不一致
MARKET_TIME_TOLERANCE_S = 120.0
QUOTE_SOURCE_ENV = "TWSTOCK_QUOTE_SOURCE"


class GateKind(str, Enum):
    """拒絕原因的分類（決定整輪狀態：blocked → partial_success，policy → success）。"""

    NONE = ""
    POLICY = "policy"      # 排程/時段/使用者設定導致（正常運作的一部分）
    BLOCKED = "blocked"    # 資料或帳務無法確認 → fail-closed

    @property
    def label(self) -> str:
        return {"": "允許", "policy": "依規則略過", "blocked": "資料/風險阻斷"}[self.value]


class Basis(str, Enum):
    """這次判定所依賴的行情基礎。"""

    EOD = "eod"            # 日線（盤後訊號，本專案設計路徑）
    INTRADAY = "intraday"  # 盤中即時行情
    NONE = "none"          # 沒有可用基礎

    @property
    def label(self) -> str:
        return {"eod": "日線（盤後）", "intraday": "盤中即時",
                "none": "無可用資料"}[self.value]


# ---------------------------------------------------------------------------
# 判定結果
# ---------------------------------------------------------------------------
@dataclass
class GateDecision:
    allow_new_orders: bool = True
    allow_fills: bool = True
    reasons: list[str] = field(default_factory=list)      # 擋新單的理由
    fill_reasons: list[str] = field(default_factory=list) # 擋成交的理由
    session: str = ""
    kind: str = GateKind.NONE.value
    basis: str = Basis.EOD.value
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.allow_new_orders

    @property
    def label(self) -> str:
        if self.allow_new_orders:
            return "允許新增模擬交易"
        return f"禁止新增模擬交易（{GateKind(self.kind).label}）"

    def as_dict(self) -> dict[str, Any]:
        return {
            "allow_new_orders": self.allow_new_orders,
            "allow_fills": self.allow_fills,
            "reasons": list(self.reasons),
            "fill_reasons": list(self.fill_reasons),
            "session": self.session,
            "kind": self.kind,
            "basis": self.basis,
            "basis_label": Basis(self.basis).label,
            "details": self.details,
            "label": self.label,
        }

    def text(self) -> str:
        L = [f"=== 行情品質閘門（{Basis(self.basis).label}）===",
             f"  交易時段   {quotes.SessionState(self.session).label}"
             if self.session else "  交易時段   —",
             f"  新增模擬單 {'✓ 允許' if self.allow_new_orders else '✗ 禁止'}",
             f"  既有單成交 {'✓ 允許' if self.allow_fills else '✗ 禁止'}"]
        for r in self.reasons:
            L.append(f"    ⛔ {r}")
        for r in self.fill_reasons:
            L.append(f"    ⛔（成交）{r}")
        q = self.details.get("quote") or {}
        if q:
            L.append(f"  行情來源   {q.get('source')}（{q.get('kind')}）"
                     f"健康={q.get('health', '—')}")
            L.append(f"  行情時間   {q.get('market_time') or '—'}"
                     f"｜抓取時間 {q.get('fetched_at') or '—'}"
                     f"｜年齡 {q.get('age_s', '—')}")
        p = self.details.get("price") or {}
        if p:
            L.append(f"  日線       最新 {p.get('latest_date') or '—'}"
                     f"（延遲 {p.get('staleness_days', '—')} 天，"
                     f"門檻 {p.get('max_staleness_days', '—')} 天）")
        return "\n".join(L)


# ---------------------------------------------------------------------------
# 日線基礎（price 表）——決定「盤後 EOD 基準」能不能用
# ---------------------------------------------------------------------------
def daily_basis(*, now: dt.datetime,
                limits: risk_mod.RiskLimits | None = None
                ) -> tuple[list[str], dict[str, Any]]:
    """日線（price 表）是否足以支撐新一輪模擬交易。回傳 ``(問題清單, 詳情)``。

    檢查項（任一不過 → 禁止新單）：
      1. 讀取失敗 / 表為空（**資料筆數為零**）。
      2. 延遲超過 ``limits.max_data_staleness_days``（**行情過期**）。
      3. 每檔「最新一列」價格異常（缺值 / ≤0；**異常價格**）。
    未來日期的日線列不在這裡擋（成交與訊號時點規則已攔截；由 quality 階段
    顯式回報），避免把測試/回測資料的 T+1 bar 誤殺成不可交易。
    """
    lim = limits or risk_mod.DEFAULT_LIMITS
    problems: list[str] = []
    details: dict[str, Any] = {"max_staleness_days": int(lim.max_data_staleness_days)}
    try:
        df = store.read_sql("SELECT MAX(date) AS m FROM price")
    except Exception as e:  # noqa: BLE001
        return [f"價量資料無法讀取（fail-closed）：{type(e).__name__}: {e}"], {
            **details, "price_error": f"{type(e).__name__}: {e}"}
    latest = None if df.empty else df.iloc[0]["m"]
    if latest is None or pd.isna(latest):
        details["latest_date"] = None
        details["staleness_days"] = None
        return ["無價量資料（price 表為零列），無法確認行情新鮮度"], details
    try:
        d = dt.date.fromisoformat(str(latest)[:10])
    except ValueError:
        details["latest_date"] = str(latest)
        return ["最新價量日期不可解析（時間未知，fail-closed）"], details
    details["latest_date"] = d.isoformat()
    stale = (now.date() - d).days
    details["staleness_days"] = stale
    if stale > lim.max_data_staleness_days:
        problems.append(
            f"行情過期 {stale} 天（> {lim.max_data_staleness_days}），"
            "停止新增模擬交易")
    # 每檔最新一列：異常價格（缺值/0/負值）不得當成可交易的參考價
    try:
        latest_rows = store.read_sql(
            "SELECT p.stock_id, p.date, p.open, p.close, p.volume FROM price p "
            "JOIN (SELECT stock_id, MAX(date) AS d FROM price GROUP BY stock_id) m "
            "ON p.stock_id=m.stock_id AND p.date=m.d")
    except Exception as e:  # noqa: BLE001
        problems.append(f"最新行情列無法讀取（fail-closed）：{type(e).__name__}: {e}")
        return problems, details
    bad: list[str] = []
    for _, r in latest_rows.iterrows():
        o, c = r.get("open"), r.get("close")
        if o is None or c is None or pd.isna(o) or pd.isna(c) \
                or float(o) <= 0 or float(c) <= 0:
            bad.append(f"{r['stock_id']}@{r['date']}")
    details["n_latest_checked"] = int(len(latest_rows))
    details["n_invalid_latest"] = len(bad)
    if bad:
        problems.append(
            f"最新行情列價格異常 {len(bad)} 檔（{', '.join(bad[:3])}…），"
            "資料品質不合格")
    return problems, details


# ---------------------------------------------------------------------------
# 盤中即時基礎（quote 表）——決定「盤中新單」能不能用
# ---------------------------------------------------------------------------
def intraday_basis(*, now: dt.datetime,
                   max_age_s: float = quotes.DEFAULT_MAX_AGE_S
                   ) -> tuple[list[str], dict[str, Any]]:
    """盤中（session=OPEN）建立新單所需的即時行情基礎。

    回傳 ``(問題清單, 詳情)``；**沒有任何一項可被當成「應該沒問題」**：
      1. 未設定來源 → 拒絕（不臆測預設來源；P3 決定沿用）。
      2. 來源種類不是 ``realtime``（延遲/盤後/歷史）→ 拒絕：
         **不得把延遲或日線假裝成可交易的盤中行情**。
      3. ``quote`` 表零列（來源失敗或從未寫入）→ 拒絕。
      4. 市場時間（``market_time``）不可得／不可解析（**時間戳未知**）→ 拒絕。
      5. 市場時間晚於系統時間超過容忍值（**時鐘不一致**）→ 拒絕。
      6. 市場時間年齡 > ``max_age_s``（**行情過期**）→ 拒絕。
      7. 抓取時間（``fetched_at``）缺失 → 拒絕（無法確認來源有運作）。
      8. 既有 ``quote_health`` + ``intraday_signal_gate`` 未通過 → 拒絕。
    ``details['quote']`` 同時記錄 ``market_time`` 與 ``fetched_at``，
    讓「抓到了」與「夠新」可被分別查證。
    """
    problems: list[str] = []
    src_name = (os.environ.get(QUOTE_SOURCE_ENV) or "").strip()
    details: dict[str, Any] = {
        "max_age_s": float(max_age_s),
        "quote": {"source": src_name or "", "kind": "", "market_time": "",
                  "fetched_at": "", "age_s": None, "health": ""},
        "configured": bool(src_name),
    }
    if not src_name:
        problems.append(
            f"盤中未設定可驗證的即時行情來源（{QUOTE_SOURCE_ENV}），"
            "fail-closed 停止盤中新單")
        return problems, details
    info = quotes.source_info(src_name)
    if info is None:
        problems.append(f"未知行情來源 {src_name}（無法查證種類，fail-closed）")
        return problems, details
    details["quote"]["kind"] = info.kind
    if info.kind != quotes.QuoteKind.REALTIME.value:
        problems.append(
            f"來源 {src_name} 是{quotes.QuoteKind(info.kind).label}行情，"
            "不是可交易的盤中即時行情（不得假裝即時）")
        return problems, details
    try:
        df = quotes.latest_quotes(source=src_name, kind=info.kind)
    except Exception as e:  # noqa: BLE001
        problems.append(f"行情讀取失敗（fail-closed）：{type(e).__name__}: {e}")
        return problems, details
    if df is None or df.empty:
        problems.append("行情資料筆數為零（來源失敗或尚無任何已寫入行情）")
        return problems, details
    details["quote"]["n_symbols"] = int(df["stock_id"].nunique()) \
        if "stock_id" in df.columns else 0
    if not details["quote"]["n_symbols"]:
        problems.append("行情標的數為零")
    # ---- 市場時間（決定夠不夠新）----
    times = pd.to_datetime(df.get("market_time"), errors="coerce").dropna()
    if times.empty:
        problems.append("行情時間未知（market_time 不可得/不可解析），fail-closed")
    else:
        newest = times.max().to_pydatetime()
        details["quote"]["market_time"] = newest.isoformat(timespec="seconds")
        age = (now - newest).total_seconds()
        details["quote"]["age_s"] = round(age, 1)
        if newest > now + dt.timedelta(seconds=MARKET_TIME_TOLERANCE_S):
            problems.append(
                f"行情時間（{details['quote']['market_time']}）晚於系統時間，"
                "時鐘不一致，fail-closed")
        elif age > max_age_s:
            problems.append(
                f"行情過期 {age:.0f} 秒（> {max_age_s:.0f} 秒），不得用於盤中新單")
    # ---- 抓取時間（只能證明有抓取，不能證明夠新）----
    fetches = pd.to_datetime(df.get("fetched_at"), errors="coerce").dropna()
    if fetches.empty:
        problems.append("缺少抓取時間（fetched_at），無法確認來源有運作")
    else:
        fmax = fetches.max().to_pydatetime()
        details["quote"]["fetched_at"] = fmax.isoformat(timespec="seconds")
        details["quote"]["fetched_recently"] = \
            (now - fmax).total_seconds() <= max_age_s
    # ---- 既有健康狀態與盤中訊號閘門（不重造輪子）----
    try:
        health = quotes.quote_health(df, source=src_name, kind=info.kind,
                                     now=now, max_age_s=max_age_s)
    except Exception as e:  # noqa: BLE001
        problems.append(f"健康狀態無法判定（fail-closed）：{type(e).__name__}: {e}")
        return problems, details
    details["quote"]["health"] = health.health
    details["quote"]["health_reason"] = health.reason
    ok, why = quotes.intraday_signal_gate(health)
    details["quote"]["gate_ok"] = bool(ok)
    if not ok:
        problems.append(f"盤中訊號閘門未通過：{why}")
    return problems, details


# ---------------------------------------------------------------------------
# 既有訂單是否還能成交（只有 Kill Switch 與帳務不一致才停）
# ---------------------------------------------------------------------------
def fill_block_reasons(*, engine: Any = None,
                       killswitch: risk_mod.KillSwitch | None = None
                       ) -> list[str]:
    """阻擋**既有未完成訂單**繼續成交的理由（其餘問題只擋新單）。

    - Kill Switch：既定未成交訂單也必須停（P4 既有語意，可稽核）。
    - 帳務不一致：繼續成交只會讓不一致擴大 → 停止交易、**不修帳**。
    不包含「行情過期／非交易時段」——既定訂單仍可依已存行情成交（P4 語意）。
    """
    ks = killswitch or risk_mod.KillSwitch()
    out: list[str] = []
    if ks.engaged():
        out.append(f"Kill Switch 已啟用：{ks.reason() or '已啟動'}")
    if engine is not None:
        try:
            acc = engine.account()
            if acc is None:
                out.append("未建立 paper 帳戶（無法確認帳務，停止成交）")
                return out
            rec = engine.reconcile()
        except Exception as e:  # noqa: BLE001
            out.append(f"帳務核對無法執行（fail-closed）：{type(e).__name__}: {e}")
            return out
        if not rec.get("ok", False):
            probs = list(rec.get("problems") or [])
            out.append(
                f"帳務核對不一致（{len(probs)} 項，僅回報不修帳）："
                + "；".join(probs[:3]))
    return out


# ---------------------------------------------------------------------------
# 總判定
# ---------------------------------------------------------------------------
def evaluate(*, now: dt.datetime | None = None, engine: Any = None,
             limits: risk_mod.RiskLimits | None = None,
             killswitch: risk_mod.KillSwitch | None = None,
             max_age_s: float = quotes.DEFAULT_MAX_AGE_S) -> GateDecision:
    """判斷「現在能不能新增模擬交易、既有訂單還能不能成交」。

    檢查順序（前面的失敗不掩蓋後面的問題；所有問題都會被記錄）：
      Kill Switch → 日線基礎 → 帳戶/帳務核對 → 時段規則 → 盤中即時基礎。
    回傳的 ``reasons`` 全部通過才 ``allow_new_orders=True``（fail-closed）。
    """
    now = now or dt.datetime.now()
    lim = limits or risk_mod.DEFAULT_LIMITS
    ks = killswitch or risk_mod.KillSwitch()
    data_reasons: list[str] = []
    policy_reasons: list[str] = []
    fill_reasons: list[str] = []
    details: dict[str, Any] = {}

    sess = quotes.session_state(now)

    # 1) Kill Switch（新單與成交都停；不自動解除）
    if ks.engaged():
        r = f"Kill Switch 已啟用：{ks.reason() or '已啟動'}"
        data_reasons.append(r)
        fill_reasons.append(r)
    details["kill_switch"] = ks.engaged()

    # 2) 日線基礎（資料筆數、過期、異常價格）
    price_problems, price_details = daily_basis(now=now, limits=lim)
    details["price"] = price_details
    data_reasons += price_problems

    # 3) 帳戶與帳務核對（不一致 → 新單與成交都停；只回報、絕不修帳）
    if engine is not None:
        acc = None
        try:
            acc = engine.account()
        except Exception:  # noqa: BLE001
            acc = None
        details["account_exists"] = acc is not None
        if acc is None:
            msg = "未建立 paper 帳戶（paper init），無法確認帳務與模擬交易"
            data_reasons.append(msg)
            fill_reasons.append(msg)
        else:
            try:
                rec = engine.reconcile()
                problems = list(rec.get("problems") or [])
                details["reconcile_ok"] = bool(rec.get("ok", False))
                details["reconcile_problems"] = problems[:5]
                if not rec.get("ok", False):
                    msg = (f"帳務核對不一致（{len(problems)} 項，僅回報不修帳）："
                           + "；".join(problems[:3]))
                    data_reasons.append(msg)
                    fill_reasons.append(msg)
            except Exception as e:  # noqa: BLE001
                details["reconcile_ok"] = False
                msg = f"帳務核對無法執行（fail-closed）：{type(e).__name__}: {e}"
                data_reasons.append(msg)
                fill_reasons.append(msg)

    # 4) 時段規則（只影響新單）
    if sess is quotes.SessionState.WEEKEND:
        policy_reasons.append("非交易日（週末）不新增模擬訂單")
    elif sess is quotes.SessionState.OPEN:
        # 盤中：必須有可驗證的即時行情，否則不得假裝可交易
        intra_problems, intra_details = intraday_basis(now=now, max_age_s=max_age_s)
        details["quote"] = intra_details.get("quote", {})
        data_reasons += intra_problems
    # 盤前（PRE_OPEN）與盤後（CLOSED）：EOD 基準允許（成交仍走 T+1 規則）

    # ---- 彙總 ----
    reasons = data_reasons + policy_reasons
    if any(("無價量資料" in p) or ("無法讀取" in p) or ("不可解析" in p)
           for p in price_problems):
        basis = Basis.NONE.value          # 沒有任何可依賴的行情基礎
    elif sess is quotes.SessionState.OPEN:
        basis = Basis.INTRADAY.value      # 盤中判定（要求即時行情）
    else:
        basis = Basis.EOD.value           # 盤後 EOD 基準（本專案設計路徑）
    dec = GateDecision(
        allow_new_orders=not reasons,
        allow_fills=not fill_reasons,
        reasons=reasons,
        fill_reasons=fill_reasons,
        session=sess.value,
        kind=(GateKind.BLOCKED.value if data_reasons
              else (GateKind.POLICY.value if policy_reasons else GateKind.NONE.value)),
        basis=basis,
        details=details,
    )
    return dec
