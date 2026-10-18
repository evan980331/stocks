"""行情來源介面、資料品質、健康狀態與故障處理（P3）。

本階段只處理**行情來源、資料管線、監控與可靠性**：不接入券商 API、不下單、
不改變既有策略權重／曝險／持有期間。系統的訊號仍是**盤後（EOD）**產生，
本模組不會產生任何「即時訊號」。

四種行情必須分開，不得混稱「即時」
  - ``realtime``   來源標示為盤中即時（例如交易所 5 秒快照）
  - ``delayed``    來源明確延遲，或**盤中但延遲未載明** → 保守歸此類
  - ``eod``        盤後（當日收盤後才產生的日資料）
  - ``historical`` 歷史（非當日）資料，**不得**當成新行情

時間戳語義（混用即失去意義）
  - ``market_time``       來源表示的市場時間；``market_time_raw`` 保留來源原樣字串，
                          ``market_time`` 只是正規化 ISO（排序／去重），不是新資訊
  - ``fetched_at``        本系統實際取得時間
  - ``market_time_source`` ``source``（來源提供）／``derived``（可驗證推導，如毫秒 epoch）
                          ／``unknown``（不可得）→ **unknown 不得當成新行情**

可靠性規則（皆有對應測試）
  1. 同一筆行情重複取得不得重複寫入（主鍵＝冪等去重鍵）。
  2. **較舊的行情不得覆蓋較新的有效資料**：寫入前與 DB 既有最大市場時間比較。
  3. 來源失敗（網路／逾時／HTTP／解析）**不得**降級成「正常無行情」。
  4. 缺值、非正價格、超出當日漲跌停價位、代碼不符、日期非當日、
     非交易時段、時間來源不可得 → 一律不得當成有效新行情。
  5. 過期行情不得通過 :func:`intraday_signal_gate`（未來盤中訊號必須先過此閘）。

指令
  ``python -m twstock.cli quotes``            # 來源評估與健康狀態（不連網）
  ``python -m twstock.cli quotes poll``       # 以設定的來源實際抓取一次
  ``python -m twstock.cli quotes selftest``   # 假來源端到端驗證管線（明確標示為假資料）
  ``python -m twstock.quotes [...]``
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import CONFIG, REPORTS_DIR, is_readonly
from ..utils import today_str
from . import status as st
from . import store

log = logging.getLogger(__name__)

DATASET_QUOTE = "quote"
QUOTE_TABLE = "quote"
SOURCE_VERSION = "quotes/v1"

# 盤中行情可視為「有效」的最大年齡（秒）。超過即不得用於任何盤中判斷。
DEFAULT_MAX_AGE_S = 300
# 建議輪詢間隔（秒）；實際仍受來源限流，且不得比來源快照更密。
DEFAULT_POLL_INTERVAL_S = 60
# 抓取失敗的重試次數（含首次）與退避上限（秒）。
DEFAULT_FETCH_ATTEMPTS = 2
RETRY_BASE_DELAY_S = 1.0
RETRY_MAX_DELAY_S = 4.0

# 只有「重試可能變好」的失敗才重試：舊碼會對格式錯誤也重試，
# 那只是把同一個解析失敗做三次。
RETRYABLE_CLASSES = frozenset({st.ErrorClass.NETWORK.value,
                               st.ErrorClass.TIMEOUT.value,
                               st.ErrorClass.HTTP_STATUS.value})

# 台股交易時段（台北時間）：09:00–13:30 連續交易，**沒有午休**。
SESSION_OPEN = dt.time(9, 0)
SESSION_CLOSE = dt.time(13, 30)
PRE_OPEN = dt.time(8, 30)

TIME_SOURCE = "source"
TIME_DERIVED = "derived"
TIME_UNKNOWN = "unknown"


class QuoteKind(str, Enum):
    REALTIME = "realtime"
    DELAYED = "delayed"
    EOD = "eod"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {"realtime": "即時", "delayed": "延遲", "eod": "盤後",
                "historical": "歷史", "unknown": "未知"}[self.value]

    @property
    def is_intraday(self) -> bool:
        """盤中行情（即時或延遲）；盤後／歷史不算。"""
        return self in (QuoteKind.REALTIME, QuoteKind.DELAYED)


class QuoteStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    REJECTED = "rejected"

    @property
    def label(self) -> str:
        return {"ok": "有效", "degraded": "有效但異常", "rejected": "已拒絕"}[self.value]


class SourceHealth(str, Enum):
    """來源健康狀態（任務要求至少五態，互斥且可稽核）。"""

    NORMAL = "normal"
    DELAYED = "delayed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    NON_TRADING_SESSION = "non_trading_session"

    @property
    def label(self) -> str:
        return {"normal": "正常", "delayed": "延遲", "failed": "失敗",
                "unknown": "未知", "non_trading_session": "非交易時段"}[self.value]


class SessionState(str, Enum):
    WEEKEND = "weekend"
    PRE_OPEN = "pre_open"
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {"weekend": "週末休市", "pre_open": "開盤前", "open": "交易中",
                "closed": "收盤後", "unknown": "無法判定"}[self.value]


# --------------------------------------------------------------------------
# 來源評估目錄（程式與文件同源；「已查證」與「未查證」必須分開寫）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceInfo:
    """單一行情來源的可稽核描述（授權、限流、延遲、認證）。"""

    name: str
    label: str
    kind: str                       # QuoteKind value
    docs_url: str
    auth: str
    rate_limit: str
    delay: str
    terms: str
    enabled_by_default: bool
    implemented: bool               # 介面與解析器是否已實作
    verified: str                   # 本階段**實際查證**到什麼（不可寫未查證的保證）
    caveat: str = ""


SOURCE_CATALOG: tuple[SourceInfo, ...] = (
    SourceInfo(
        name="mis",
        label="TWSE MIS 行情端點（mis.twse.com.tw getStockInfo.jsp）",
        kind=QuoteKind.REALTIME.value,
        docs_url="https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_2330.tw",
        auth="無（未觀察到需要 API key／cookie；此為觀察事實，非官方保證）",
        rate_limit="未查到官方限流值；社群觀察約 10 秒更新、單次可帶多檔。"
                   "本專案自設 5 秒最快間隔（交易所快照為 5 秒）",
        delay="交易所稱 5 秒行情快照；**回應未標示即時／延遲旗標**",
        terms="⚠️ TWSE 網站條款禁止未經同意的自動化下載／擷取；"
              "正式行情資訊需另訂契約並付費。**商業轉散布屬授權問題**",
        enabled_by_default=False,
        implemented=True,
        verified="端點可被一般 HTTP GET 取得並回傳 JSON；無效代碼回 HTTP 200 且 "
                 "msgArray 內 c 為空 → **不可只看 rtcode 判定成功**",
        caveat="欄位 tv／pz／ps／mt 無官方定義；本模組只使用可驗證語意的欄位，"
               "無官方定義者一律不猜。因授權限制，需 TWSTOCK_QUOTE_ALLOW_MIS=1 才啟用。",
    ),
    SourceInfo(
        name="finmind_minute",
        label="FinMind TaiwanStockPriceMinute（分鐘／5 秒取樣）",
        kind=QuoteKind.DELAYED.value,
        docs_url="https://finmind.github.io/tutor/TaiwanMarket/DataList/",
        auth="可選 token（FINMIND_TOKEN）；本專案既有來源",
        rate_limit="官方文件：未帶 token 300 次/小時、帶 token 600 次/小時"
                   "（20 檔 × 每輪 → 每小時最多約 15~30 輪）",
        delay="文件說明為 5 秒間隔取樣的盤中資料；**實際交付延遲未載明**"
              "→ 本專案保守歸為「延遲」，不稱即時",
        terms="FinMind 專案聲明為教育／非商業用途；各資料集授權需個別確認",
        enabled_by_default=True,
        implemented=True,
        verified="資料集與欄位名稱見官方文件；**本階段未實際呼叫外部 API**"
                 "（測試一律以注入的假 getter 驅動）",
        caveat="免費層限流嚴格，且延遲未載明 → 不得作為交易決策依據。",
    ),
    SourceInfo(
        name="fake",
        label="假行情來源（僅供測試與管線自檢）",
        kind=QuoteKind.DELAYED.value,
        docs_url="",
        auth="無",
        rate_limit="無",
        delay="無（合成）",
        terms="不適用",
        enabled_by_default=False,
        implemented=True,
        verified="由固定種子產生確定性序列；**is_fabricated=True，不得當成真實行情**",
        caveat="輸出必須明確標示為假資料；不得用於任何績效或交易結論。",
    ),
    SourceInfo(
        name="fugle",
        label="Fugle（富果）MarketData API",
        kind=QuoteKind.REALTIME.value,
        docs_url="https://developer.fugle.tw/docs/data/http-api/intraday/quote/",
        auth="需註冊並取得 API key（Basic 免費）；不需券商帳戶",
        rate_limit="Basic 免費層：REST 60 次/分鐘、WebSocket 5 訂閱／1 連線",
        delay="文件標示為即時（無延遲 SLA）",
        terms="禁止未授權轉送、轉售或提供給第三方；商業使用需另確認",
        enabled_by_default=False,
        implemented=False,
        verified="官方文件已查證（欄位含最後成交價／時間／量、五檔）",
        caveat="**未實作**：需使用者自行註冊取得 API key；本階段不代為註冊。",
    ),
    SourceInfo(
        name="shioaji",
        label="永豐金 Shioaji（券商 API）",
        kind=QuoteKind.REALTIME.value,
        docs_url="https://sinotrade.github.io/tutor/market_data/streaming/stocks/",
        auth="需券商帳戶與 API 金鑰（需使用者決定與辦理）",
        rate_limit="200 檔訂閱、5 連線／人；行情查詢 10 秒 50 次",
        delay="盤中訂閱推送（文件無延遲 SLA）",
        terms="券商帳戶條款；轉散布需另取得授權",
        enabled_by_default=False,
        implemented=False,
        verified="官方文件已查證",
        caveat="**未實作且 BLOCKED**：需券商帳戶（同時是 P4/P6 的前置條件）。",
    ),
    SourceInfo(
        name="fubon_neo",
        label="富邦新一代 API（券商 API）",
        kind=QuoteKind.REALTIME.value,
        docs_url="https://www.fbs.com.tw/TradeAPI/en/docs/market-data/intraday/quote/",
        auth="需富邦證券帳戶、憑證與 API 同意書",
        rate_limit="盤中行情 300 次/分鐘；WebSocket 300 檔／連線、7 連線",
        delay="盤中即時（無延遲 SLA）",
        terms="禁止未授權截取、轉送、轉售",
        enabled_by_default=False,
        implemented=False,
        verified="官方文件已查證",
        caveat="**未實作且 BLOCKED**：需券商帳戶。",
    ),
    SourceInfo(
        name="yahoo",
        label="Yahoo Finance（yfinance，.TW／.TWO）",
        kind=QuoteKind.DELAYED.value,
        docs_url="https://help.yahoo.com/kb/SLN2310.html",
        auth="無官方 API key（非官方套件）",
        rate_limit="無公開保證配額",
        delay="官方延遲表列台股為 **20 分鐘延遲**",
        terms="Yahoo 條款禁止未經許可的自動化蒐集與再散布",
        enabled_by_default=False,
        implemented=False,
        verified="官方延遲表已查證；非交易所授權的即時來源",
        caveat="**未實作**：條款與延遲皆不利，且無 SLA，不列為預設選項。",
    ),
    SourceInfo(
        name="twse_openapi",
        label="TWSE／TPEx OpenAPI（政府資料開放）",
        kind=QuoteKind.EOD.value,
        docs_url="https://openapi.twse.com.tw/v1/swagger.json",
        auth="無",
        rate_limit="未查到逐端點限流值",
        delay="**盤後**：個股日收盤價資料每日約 13:50 後更新",
        terms="政府資料開放授權條款（可商用、需標示來源）；"
              "**不涵蓋** MIS 盤中端點",
        enabled_by_default=False,
        implemented=False,
        verified="官方 Swagger 與資料集頁已查證；端點為盤後報告型資料",
        caveat="**未實作**：可用於日K交叉驗證，但**不是**盤中行情來源。",
    ),
)


def source_info(name: str) -> SourceInfo | None:
    for s in SOURCE_CATALOG:
        if s.name == name:
            return s
    return None


# --------------------------------------------------------------------------
# 交易時段（只用本地時間；**沒有假日曆**，此為已知限制）
# --------------------------------------------------------------------------
def session_state(now: dt.datetime | None = None) -> SessionState:
    """由本地時間判定交易時段。

    **限制**：沒有台灣假日曆，因此國定假日會被判為 ``OPEN``。
    這正是「來源在交易時段內長時間沒有新行情 → 標為延遲／失敗」這個
    品質訊號存在的原因：寧可顯示延遲，也不假裝正常。
    """
    now = now or dt.datetime.now()
    if now.weekday() >= 5:
        return SessionState.WEEKEND
    t = now.time()
    if t < PRE_OPEN:
        return SessionState.CLOSED
    if t < SESSION_OPEN:
        return SessionState.PRE_OPEN
    if t <= SESSION_CLOSE:
        return SessionState.OPEN
    return SessionState.CLOSED


def is_trading_session(now: dt.datetime | None = None) -> bool:
    return session_state(now) in (SessionState.PRE_OPEN, SessionState.OPEN)


# --------------------------------------------------------------------------
# 行情資料結構
# --------------------------------------------------------------------------
def _num(value: Any) -> float | None:
    """來源數值轉換：``"-"``／空字串／非數字**一律回 None（缺值）**，不當成 0。"""
    if value is None:
        return None
    s = str(value).strip()
    if s in ("", "-", "--", "N/A", "null", "None"):
        return None
    try:
        f = float(s.replace(",", ""))
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


@dataclass(frozen=True)
class Quote:
    """一筆行情快照（來源無關）。``market_time`` 與 ``fetched_at`` 必須可分。"""

    stock_id: str
    source: str
    kind: str = QuoteKind.UNKNOWN.value
    price: float | None = None
    prev_close: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    last_volume: float | None = None
    total_volume: float | None = None
    bid: float | None = None
    ask: float | None = None
    market_time: str = ""
    market_time_raw: str = ""
    market_time_source: str = TIME_UNKNOWN
    fetched_at: str = ""
    market_state: str = ""
    status: str = QuoteStatus.OK.value
    error_class: str = st.ErrorClass.NONE.value
    error: str = ""
    notes: str = ""
    run_id: str = ""

    @property
    def day(self) -> str:
        return self.market_time[:10] if self.market_time else ""

    @property
    def kind_label(self) -> str:
        return QuoteKind(self.kind).label if self.kind in QuoteKind._value2member_map_ \
            else self.kind

    def as_row(self) -> dict:
        return {
            "source": self.source, "kind": self.kind, "stock_id": self.stock_id,
            "market_time": self.market_time, "market_time_raw": self.market_time_raw,
            "market_time_source": self.market_time_source,
            "fetched_at": self.fetched_at, "price": self.price,
            "prev_close": self.prev_close, "open": self.open, "high": self.high,
            "low": self.low, "last_volume": self.last_volume,
            "total_volume": self.total_volume, "bid": self.bid, "ask": self.ask,
            "market_state": self.market_state, "status": self.status,
            "error_class": self.error_class, "error": st.redact(self.error),
            "run_id": self.run_id, "notes": st.redact(self.notes),
        }


@dataclass
class QuoteFetch:
    """一次抓取的結果（含可重建的失敗資訊）。"""

    source: str
    kind: str
    quotes: list[Quote] = field(default_factory=list)
    requested: list[str] = field(default_factory=list)
    error_class: str = st.ErrorClass.NONE.value
    error: str = ""
    http_status: int | None = None
    attempts: int = 0
    resolved: int = 0          # 實際對應到 requested 標的的筆數
    malformed: int = 0         # 解析不出必要欄位的筆數
    notes: str = ""

    @property
    def ok(self) -> bool:
        return (self.error_class == st.ErrorClass.NONE.value
                and self.resolved > 0)

    @property
    def status(self) -> str:
        """對應 ``status.classify`` 的狀態語意（失敗**不得**降級成空結果）。"""
        return st.classify(error_class=self.error_class,
                           written=self.resolved if self.ok else 0)

    def label(self) -> str:
        return (f"{self.source}（{self.kind}）：要求 {len(self.requested)} 檔、"
                f"對應 {self.resolved} 檔、解析失敗 {self.malformed} 檔"
                + (f"｜{st.ErrorClass(self.error_class).label}" if self.error else ""))


# --------------------------------------------------------------------------
# 限流（保護來源額度；sleep 與 clock 可注入以利測試）
# --------------------------------------------------------------------------
class RateLimiter:
    """最小間隔限流器；**不得**比來源快照更密。"""

    def __init__(self, min_interval_s: float, *,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None
        self.waited = 0.0

    def wait(self) -> float:
        """必要時等待並回傳實際等待秒數（第一次不等待）。"""
        now = self._clock()
        waited = 0.0
        if self._last is not None:
            delta = now - self._last
            if delta < self.min_interval_s:
                waited = self.min_interval_s - delta
                self._sleep(waited)
        self._last = now + waited
        self.waited += waited
        return waited


# --------------------------------------------------------------------------
# 來源介面
# --------------------------------------------------------------------------
class QuoteSource:
    """行情來源介面：不同供應商可替換。

    實作只負責**取得與解析**；驗證、去重、寫入與健康狀態一律由本模組的
    純函式處理，因此換來源不會改變資料品質規則。
    """

    name = "base"
    is_fabricated = False             # True 表示合成資料，不得當成真實行情

    def __init__(self, *, getter: Callable[..., Any] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], dt.datetime] = dt.datetime.now) -> None:
        self._get = getter or _default_getter
        self._sleep = sleep
        self._now = now
        self.info = source_info(self.name) or SourceInfo(
            name=self.name, label=self.name, kind=QuoteKind.UNKNOWN.value,
            docs_url="", auth="", rate_limit="", delay="", terms="",
            enabled_by_default=False, implemented=True, verified="")

    # ---- 子類別覆寫 ----
    def _fetch_raw(self, stock_ids: list[str]) -> QuoteFetch:  # pragma: no cover
        raise NotImplementedError

    def fetch(self, stock_ids: list[str], *,
              attempts: int = DEFAULT_FETCH_ATTEMPTS,
              base_delay: float = RETRY_BASE_DELAY_S,
              max_delay: float = RETRY_MAX_DELAY_S) -> QuoteFetch:
        """取得行情，可重試；任何例外都轉成可重建的失敗分類，**不吞例外**。

        重試規則（可完全離線測試）
          - 只重試網路／逾時／HTTP 狀態（含 429 限流）→ 指數退避，上限 ``max_delay``。
          - **不重試**解析失敗：格式變了，重試只會得到同一個結果。
          - ``attempts`` 為實際執行的抓取回合數（>1 代表曾重試）。
        """
        ids = [str(s).strip() for s in (stock_ids or []) if str(s).strip()]
        if not ids:
            return QuoteFetch(source=self.name, kind=self.info.kind,
                              error_class=st.ErrorClass.EMPTY.value,
                              error="no_symbols_requested")
        rounds = max(1, int(attempts))
        res: QuoteFetch | None = None
        retries = 0
        executed = 0
        for i in range(rounds):
            executed += 1
            try:
                res = self._fetch_raw(ids)
            except Exception as e:  # noqa: BLE001
                cls = _classify_exception(e)
                log.warning("行情來源 %s 失敗（%s）：%s", self.name, cls, e)
                res = QuoteFetch(source=self.name, kind=self.info.kind,
                                 error_class=cls,
                                 error=f"{type(e).__name__}: {e}")
            if res.error_class == st.ErrorClass.NONE.value and res.resolved > 0:
                break
            if res.error_class not in RETRYABLE_CLASSES or i == rounds - 1:
                break
            delay = min(float(max_delay), float(base_delay) * (2 ** i))
            log.info("行情來源 %s 可重試失敗（%s），%.1fs 後重試（第 %d/%d 次）",
                     self.name, res.error_class, delay, i + 2, rounds)
            self._sleep(delay)
            retries += 1
        assert res is not None  # rounds >= 1 保證
        res.requested = ids
        if retries:
            res.notes = ";".join(x for x in (res.notes, f"retries={retries}") if x)
        res.attempts = executed
        return res

    def available(self) -> tuple[bool, str]:
        """來源是否可用（未設定認證／授權時明確回報原因，不假裝可用）。"""
        if self.is_fabricated:
            return True, "合成來源（僅供測試／自檢）"
        if self.name == "mis":
            if os.environ.get("TWSTOCK_QUOTE_ALLOW_MIS") != "1":
                return False, ("MIS 為未公開的網頁端點，TWSE 條款禁止未經同意的"
                               "自動化擷取；需設 TWSTOCK_QUOTE_ALLOW_MIS=1 明確啟用")
        return True, ""


def _default_getter(url: str, *, params: dict | None = None,
                    timeout: int = 10) -> Any:  # pragma: no cover - 需網路
    import requests
    return requests.get(url, params=params, timeout=timeout)


def _classify_exception(e: BaseException) -> str:
    """例外 → 可重建的錯誤分類（與 ``status.ErrorClass`` 同一詞彙）。"""
    name = type(e).__name__.lower()
    if "timeout" in name or isinstance(e, TimeoutError):
        return st.ErrorClass.TIMEOUT.value
    if "json" in name or isinstance(e, ValueError):
        return st.ErrorClass.PARSE.value
    if "http" in name or "status" in name:
        return st.ErrorClass.HTTP_STATUS.value
    mod = type(e).__module__
    if "requests" in mod or "urllib" in mod or "socket" in mod or "connection" in name:
        return st.ErrorClass.NETWORK.value
    return st.ErrorClass.UNKNOWN.value


# --------------------------------------------------------------------------
# 來源 1：TWSE MIS（需明確啟用）
# --------------------------------------------------------------------------
MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
MIS_MIN_INTERVAL_S = 5.0           # 交易所快照 5 秒，不得更密


def mis_channel(stock_id: str, ex: str = "tse") -> str:
    return f"{ex}_{stock_id}.tw"


def parse_mis_payload(payload: Any, requested: list[str] | None = None, *,
                      fetched_at: str = "", market_state: str = "",
                      run_id: str = "") -> QuoteFetch:
    """解析 MIS ``getStockInfo.jsp`` 回應（純函式，可完全離線測試）。

    **不可只看 ``rtcode``**：無效代碼同樣回 HTTP 200 與 ``rtcode="0000"``，
    但 ``msgArray`` 內的 ``c`` 為空。因此成功判定改為「有幾筆對應到 requested」。
    """
    req = [str(s).strip() for s in (requested or [])]
    out = QuoteFetch(source="mis", kind=QuoteKind.REALTIME.value)
    if not isinstance(payload, dict):
        return QuoteFetch(source="mis", kind=QuoteKind.REALTIME.value,
                          error_class=st.ErrorClass.PARSE.value,
                          error="payload_not_a_dict")
    rows = payload.get("msgArray")
    if rows is None:
        return QuoteFetch(source="mis", kind=QuoteKind.REALTIME.value,
                          error_class=st.ErrorClass.PARSE.value,
                          error="missing_msgArray")
    if not isinstance(rows, list):
        return QuoteFetch(source="mis", kind=QuoteKind.REALTIME.value,
                          error_class=st.ErrorClass.PARSE.value,
                          error="msgArray_not_a_list")
    quotes: list[Quote] = []
    malformed = 0
    for row in rows:
        if not isinstance(row, dict):
            malformed += 1
            continue
        code = str(row.get("c") or "").strip()
        raw_time = " ".join(x for x in (str(row.get("d") or "").strip(),
                                        str(row.get("t") or "").strip()) if x)
        mtime, tsrc, terr = _mis_market_time(row, raw_time)
        if not code:
            # 來源回了回應，但這筆不對應任何標的 → 記為解析失敗，不是「沒有行情」
            malformed += 1
        quotes.append(Quote(
            stock_id=code, source="mis", kind=QuoteKind.REALTIME.value,
            price=_num(row.get("z")), prev_close=_num(row.get("y")),
            open=_num(row.get("o")), high=_num(row.get("h")), low=_num(row.get("l")),
            last_volume=_num(row.get("tv")), total_volume=_num(row.get("v")),
            bid=_first_level(row.get("b")), ask=_first_level(row.get("a")),
            market_time=mtime, market_time_raw=raw_time,
            market_time_source=tsrc, fetched_at=fetched_at,
            market_state=market_state,
            notes=";".join(x for x in (terr, "c_missing" if not code else "") if x),
            run_id=run_id))
    if req:
        resolved = sum(1 for q in quotes if q.stock_id in req)
    else:
        resolved = sum(1 for q in quotes if q.stock_id)
    out.quotes = quotes
    out.malformed = malformed
    out.resolved = int(resolved)
    if resolved == 0:
        out.error_class = st.ErrorClass.PARSE.value
        out.error = "no_symbol_matched_request"
    return out


def _mis_market_time(row: dict, raw: str) -> tuple[str, str, str]:
    """由 ``d``＋``t`` 取得市場時間；取不到時退回 ``tlong``（標為 derived）。"""
    d = str(row.get("d") or "").strip()
    t = str(row.get("t") or "").strip()
    iso = ""
    if len(d) == 8 and d.isdigit() and t:
        try:
            iso = dt.datetime.strptime(f"{d} {t}", "%Y%m%d %H:%M:%S").isoformat()
        except ValueError:
            iso = ""
    if iso:
        return iso, TIME_SOURCE, ""
    tl = _num(row.get("tlong"))
    if tl is not None and tl > 1_000_000_000_000:      # 毫秒 epoch
        try:
            iso = dt.datetime.fromtimestamp(tl / 1000.0).isoformat(timespec="seconds")
            return iso, TIME_DERIVED, "time_from_tlong"
        except (OverflowError, OSError, ValueError):
            pass
    return "", TIME_UNKNOWN, "market_time_unavailable"


def _first_level(value: Any) -> float | None:
    """五檔字串（底線分隔）取第一檔；``"-"`` 一律回 None。"""
    if value is None:
        return None
    parts = [p for p in str(value).split("_") if p.strip() not in ("", "-")]
    return _num(parts[0]) if parts else None


class MisQuoteSource(QuoteSource):
    """TWSE MIS 來源。**預設停用**：TWSE 條款禁止未經同意的自動化擷取。"""

    name = "mis"

    def __init__(self, *, ex: str = "tse",
                 clock: Callable[[], float] = time.monotonic, **kw) -> None:
        kw.setdefault("sleep", time.sleep)
        super().__init__(**kw)
        self.ex = ex
        self.limiter = RateLimiter(MIS_MIN_INTERVAL_S, sleep=self._sleep,
                                   clock=clock)

    def available(self) -> tuple[bool, str]:
        return super().available()

    def _fetch_raw(self, stock_ids: list[str]) -> QuoteFetch:
        self.limiter.wait()
        params = {"ex_ch": "|".join(mis_channel(s, self.ex) for s in stock_ids),
                  "json": "1", "delay": "0"}
        r = self._get(MIS_URL, params=params)
        status_code = getattr(r, "status_code", None)
        # 先看 HTTP 狀態：4xx／5xx（含 429 限流）是連線層失敗，
        # 不應該被之後的 JSON 解析失敗蓋掉而誤判為「格式變了」。
        if status_code is not None and int(status_code) >= 400:
            return QuoteFetch(source=self.name, kind=self.info.kind,
                              error_class=st.ErrorClass.HTTP_STATUS.value,
                              error=f"http_{status_code}", http_status=status_code,
                              attempts=1)
        try:
            payload = r.json()
        except Exception as e:  # noqa: BLE001
            return QuoteFetch(source=self.name, kind=self.info.kind,
                              error_class=st.ErrorClass.PARSE.value,
                              error=f"json_decode: {e}", http_status=status_code,
                              attempts=1)
        res = parse_mis_payload(payload, stock_ids,
                                fetched_at=self._now().isoformat(timespec="seconds"),
                                market_state=session_state(self._now()).value)
        res.http_status = status_code
        res.attempts = 1
        return res


# --------------------------------------------------------------------------
# 來源 2：FinMind 分鐘資料（本專案既有來源；延遲未載明 → 歸為延遲）
# --------------------------------------------------------------------------
FINMIND_MINUTE_DATASET = "TaiwanStockPriceMinute"
FINMIND_MIN_INTERVAL_S = 6.0       # 保護 300~600 次/小時額度


def parse_finmind_minute(rows: Any, stock_id: str, *, fetched_at: str = "",
                         market_state: str = "", run_id: str = "") -> Quote:
    """解析 FinMind 分鐘資料的**最後一筆**為一則行情。

    欄位名稱容錯（格式變更時**回報**而非臆測）：價格接受
    ``deal_price``／``close``／``price``，量接受 ``volume``／``deal_volume``，
    時間接受 ``date``／``time``／``Time``。找不到就標為解析失敗。
    """
    q = dict(stock_id=stock_id, source="finmind_minute",
             kind=QuoteKind.DELAYED.value, fetched_at=fetched_at,
             market_state=market_state, run_id=run_id)
    if rows is None:
        return Quote(**q, error_class=st.ErrorClass.PARSE.value,
                     error="minute_payload_not_a_list")
    if not isinstance(rows, list):
        return Quote(**q, error_class=st.ErrorClass.PARSE.value,
                     error="minute_payload_not_a_list")
    if not rows:
        # 來源正常回應但沒有資料（例如開盤前）→ 空結果，**不是**失敗
        return Quote(**q, error_class=st.ErrorClass.EMPTY.value, error="empty")
    last = rows[-1]
    if not isinstance(last, dict):
        return Quote(**q, error_class=st.ErrorClass.PARSE.value,
                     error="minute_row_not_a_dict")
    price = _first_key(last, ("deal_price", "close", "price"))
    volume = _first_key(last, ("volume", "deal_volume", "Trading_Volume"))
    raw_time = _first_key(last, ("date", "time", "Time"))
    if price is None or raw_time is None:
        return Quote(**q, error_class=st.ErrorClass.PARSE.value,
                     error=("unknown_columns:" + ",".join(sorted(map(str, last)))[:200]))
    iso = _iso_datetime(raw_time)
    return Quote(**q, price=_num(price), last_volume=_num(volume),
                 market_time=iso, market_time_raw=str(raw_time),
                 market_time_source=TIME_SOURCE if iso else TIME_UNKNOWN,
                 notes="" if iso else "market_time_unparseable")


def _default_finmind_fetcher(dataset: str, data_id: str, start: str,
                             end: str):  # pragma: no cover - 需網路
    from . import finmind_client as fm
    return fm.fetch_finmind(dataset, data_id, start, end)


def _first_key(row: dict, names: tuple[str, ...]) -> Any:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def _iso_datetime(value: Any) -> str:
    """把來源時間轉為正規化 ISO；無法解析時回空字串（不得臆測）。"""
    s = str(value).strip()
    if not s:
        return ""
    try:
        return pd.Timestamp(s).to_pydatetime().isoformat()
    except (ValueError, TypeError, OverflowError):
        return ""


class FinMindMinuteQuoteSource(QuoteSource):
    """FinMind 分鐘資料來源（延遲未載明 → 一律標為 ``delayed``）。"""

    name = "finmind_minute"

    def __init__(self, *, fetcher: Callable[..., Any] | None = None, **kw) -> None:
        super().__init__(**kw)
        self._fetch_ds = fetcher or _default_finmind_fetcher
        self.limiter = RateLimiter(FINMIND_MIN_INTERVAL_S, sleep=self._sleep)

    def _fetch_raw(self, stock_ids: list[str]) -> QuoteFetch:
        epoch = self._now()
        fetched_at = epoch.isoformat(timespec="seconds")
        state = session_state(epoch).value
        quotes: list[Quote] = []
        malformed = 0
        errs: list[str] = []
        attempts = 0
        for sid in stock_ids:
            self.limiter.wait()
            attempts += 1
            try:
                df = self._fetch_ds(FINMIND_MINUTE_DATASET, sid,
                                    today_str(), today_str())
            except Exception as e:  # noqa: BLE001
                return QuoteFetch(source=self.name, kind=self.info.kind,
                                  error_class=_classify_exception(e),
                                  error=f"{type(e).__name__}: {e}",
                                  attempts=attempts)
            if df is None:
                # 抓取失敗（網路／額度／格式）→ 整批視為失敗，不混成「沒有行情」
                return QuoteFetch(source=self.name, kind=self.info.kind,
                                  error_class=st.ErrorClass.NETWORK.value,
                                  error=f"fetch_failed:{sid}", attempts=attempts)
            rows = df.to_dict("records")
            q = parse_finmind_minute(rows, sid, fetched_at=fetched_at,
                                     market_state=state)
            if q.error_class == st.ErrorClass.PARSE.value:
                malformed += 1
                errs.append(f"{sid}:{q.error}")
            quotes.append(q)
        resolved = sum(1 for q in quotes
                       if q.stock_id in stock_ids
                       and q.error_class not in (st.ErrorClass.PARSE.value,))
        out = QuoteFetch(source=self.name, kind=self.info.kind, quotes=quotes,
                         attempts=attempts, malformed=malformed,
                         resolved=int(resolved),
                         notes=";".join(errs)[:300])
        if resolved == 0:
            out.error_class = st.ErrorClass.PARSE.value
            out.error = "no_symbol_resolved"
        return out


# --------------------------------------------------------------------------
# 來源 3：假來源（測試與管線自檢；**明確標示為假資料**）
# --------------------------------------------------------------------------
class FakeQuoteSource(QuoteSource):
    """確定性合成行情。``is_fabricated=True``，**不得**當成真實行情。"""

    name = "fake"
    is_fabricated = True

    def __init__(self, *, kind: str = QuoteKind.DELAYED.value, seed: int = 0,
                 start_price: float = 100.0, error: BaseException | None = None,
                 error_every: int = 0, **kw) -> None:
        super().__init__(**kw)
        self.kind_override = kind
        self.seed = seed
        self.start_price = start_price
        self.error = error
        self.error_every = max(0, int(error_every))
        self.calls = 0
        self.info = SourceInfo(
            name="fake", label="假行情來源（測試／自檢用）", kind=kind,
            docs_url="", auth="無", rate_limit="無", delay="無（合成）",
            terms="不適用", enabled_by_default=False, implemented=True,
            verified="確定性合成資料；不得作為任何真實結論的依據",
            caveat="輸出必須標示為假資料。")

    def available(self) -> tuple[bool, str]:
        return True, "合成來源（僅供測試／自檢）"

    def _fetch_raw(self, stock_ids: list[str]) -> QuoteFetch:
        self.calls += 1
        if self.error is not None and (self.error_every <= 0
                                       or self.calls % self.error_every == 0):
            raise self.error
        now = self._now()
        step = len(str(self.calls))
        quotes = []
        for i, sid in enumerate(stock_ids):
            px = round(self.start_price + ((i + self.calls) % 7) * 0.5, 2)
            quotes.append(Quote(
                stock_id=sid, source="fake", kind=self.kind_override,
                price=px, prev_close=self.start_price,
                last_volume=1000.0 * step, total_volume=100_000.0,
                market_time=now.replace(microsecond=0).isoformat(),
                market_time_raw=now.strftime("%Y%m%d %H:%M:%S"),
                market_time_source=TIME_SOURCE,
                fetched_at=now.isoformat(timespec="seconds"),
                market_state=session_state(now).value,
                notes="FABRICATED_DATA"))
        return QuoteFetch(source="fake", kind=self.kind_override, quotes=quotes,
                          resolved=len(quotes), attempts=1)


SOURCE_TYPES: dict[str, type[QuoteSource]] = {
    "mis": MisQuoteSource,
    "finmind_minute": FinMindMinuteQuoteSource,
    "fake": FakeQuoteSource,
}


def resolve_source(name: str | None = None, **kw) -> QuoteSource | None:
    """由參數或 ``TWSTOCK_QUOTE_SOURCE`` 決定來源；**未設定回 None**（不臆測預設）。

    刻意不在未設定的情況下自行選一個來源來輪詢：那會在使用者不知情的
    情況下開始對外部服務發送請求。
    """
    key = (name or os.environ.get("TWSTOCK_QUOTE_SOURCE") or "").strip()
    if not key:
        return None
    cls = SOURCE_TYPES.get(key)
    if cls is None:
        raise KeyError(f"未知行情來源 {key}；可用：{sorted(SOURCE_TYPES)}")
    return cls(**kw)


# --------------------------------------------------------------------------
# 資料品質驗證（純函式）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Assessment:
    quote: Quote
    status: str
    error_class: str
    reason: str

    @property
    def accepted(self) -> bool:
        return self.status != QuoteStatus.REJECTED.value


def validate_quote(q: Quote, *, pool: dict | None = None,
                   prev_market_time: str | None = None,
                   now: dt.datetime | None = None,
                   limit_pct: float | None = None) -> Assessment:
    """單筆行情驗證（純函式，順序即語意保證）。

    拒絕理由一律回可重建的 ``reason`` 字串，且**不得**與來源失敗混淆。
    """
    now = now or dt.datetime.now()
    pool = pool if pool is not None else CONFIG.pool
    limit = CONFIG.limit_pct if limit_pct is None else float(limit_pct)

    def reject(reason: str, cls: str = st.ErrorClass.VALIDATION.value) -> Assessment:
        return Assessment(q, QuoteStatus.REJECTED.value, cls, reason)

    # 1) 代碼必須在股票池且與請求一致（不臆測關聯）
    if not q.stock_id:
        return reject("symbol_missing")
    if pool and q.stock_id not in pool:
        return reject("symbol_not_in_pool")
    # 2) 時間來源不可得 → 不得當成新行情
    if q.market_time_source == TIME_UNKNOWN or not q.market_time:
        return reject("market_time_unknown")
    # 3) 非正常交易時段的時間戳不得當成盤中行情
    try:
        mtime = dt.datetime.fromisoformat(q.market_time)
    except ValueError:
        return reject("market_time_unparseable")
    if session_state(mtime) is SessionState.WEEKEND:
        return reject("market_time_on_weekend")
    if not (SESSION_OPEN <= mtime.time() <= SESSION_CLOSE):
        return reject("market_time_outside_session")
    # 4) 價格必填且為正
    if q.price is None:
        return reject("price_missing", st.ErrorClass.EMPTY.value)
    if not math.isfinite(q.price) or q.price <= 0:
        return reject("price_not_positive")
    # 5) 相對前收的異常跳動（超過當日漲跌停幅度 → 不可能是成交價）
    if q.prev_close and q.prev_close > 0:
        move = q.price / q.prev_close - 1.0
        if abs(move) > limit + 1e-9:
            return reject(f"price_move_exceeds_limit:{move:+.4f}")
    # 6) 舊行情不得覆蓋新行情
    if prev_market_time and q.market_time <= prev_market_time:
        return reject("not_newer_than_stored",
                      st.ErrorClass.VALIDATION.value)
    # 7) 成交時間與取得時間的先後不得矛盾（取得早於市場時間 → 時鐘問題）
    if q.fetched_at:
        try:
            if dt.datetime.fromisoformat(q.market_time) > \
                    dt.datetime.fromisoformat(q.fetched_at) + dt.timedelta(seconds=120):
                return reject("market_time_after_fetch")
        except ValueError:
            pass
    status = (QuoteStatus.DEGRADED.value
              if (q.last_volume is None and q.total_volume is None)
              else QuoteStatus.OK.value)
    return Assessment(q, status, st.ErrorClass.NONE.value, "")


def validate_batch(quotes: list[Quote], *, pool: dict | None = None,
                   prev: dict[str, str] | None = None,
                   now: dt.datetime | None = None) -> list[Assessment]:
    """批次驗證。

    兩種重複都要擋：同一批內**完全相同**的鍵只留第一筆
    （``duplicate_in_batch``）；同一批內同一標的**較舊**的時間戳不得通過
    （``not_newer_than_stored``）──否則「舊不得覆蓋新」只在跨批次成立，
    批次內仍可插入舊資料。
    """
    best: dict[str, str] = dict(prev or {})   # 複製，不改動呼叫端的 dict
    seen: set[tuple] = set()
    out: list[Assessment] = []
    for q in quotes:
        key = (q.source, q.kind, q.stock_id, q.market_time)
        if key in seen:
            out.append(Assessment(q, QuoteStatus.REJECTED.value,
                                  st.ErrorClass.VALIDATION.value,
                                  "duplicate_in_batch"))
            continue
        seen.add(key)
        a = validate_quote(q, pool=pool, prev_market_time=best.get(q.stock_id),
                           now=now)
        if a.accepted and q.market_time > best.get(q.stock_id, ""):
            best[q.stock_id] = q.market_time
        out.append(a)
    return out


def stored_market_times(source: str, kind: str | None = None) -> dict[str, str]:
    """每個標的**已寫入**的最新市場時間（用於「舊不得覆蓋新」與延遲判斷）。"""
    try:
        if kind:
            df = store.read_sql(
                "SELECT stock_id, MAX(market_time) AS m FROM quote "
                "WHERE source=? AND kind=? GROUP BY stock_id", (source, kind))
        else:
            df = store.read_sql(
                "SELECT stock_id, MAX(market_time) AS m FROM quote "
                "WHERE source=? GROUP BY stock_id", (source,))
    except Exception as e:  # noqa: BLE001 - 舊 DB 可能還沒有 quote 表
        log.warning("讀取既有行情失敗（視為無既有行情）：%s", e)
        return {}
    if df.empty:
        return {}
    return {str(r.stock_id): str(r.m) for r in df.itertuples(index=False)
            if r.m}


@dataclass
class WriteSummary:
    """一次行情寫入的完整結果（含被拒絕的原因統計）。"""

    received: int = 0
    duplicates: int = 0
    rejected: int = 0
    written: int = 0
    dry_run: bool = False
    reasons: dict[str, int] = field(default_factory=dict)

    def add_reason(self, reason: str, n: int = 1) -> None:
        if reason:
            self.reasons[reason] = self.reasons.get(reason, 0) + n

    def label(self) -> str:
        parts = [f"收到 {self.received}",
                 (f"將寫入 {self.written}（不寫入）" if self.dry_run
                  else f"寫入 {self.written}"),
                 f"重複 {self.duplicates}", f"拒絕 {self.rejected}"]
        if self.reasons:
            top = sorted(self.reasons.items(), key=lambda kv: -kv[1])[:3]
            parts.append("拒絕原因 " + "、".join(f"{k}×{v}" for k, v in top))
        return "；".join(parts)


def write_quotes(assessments: list[Assessment], *, run_id: str = "",
                 source: str | None = None, kind: str | None = None
                 ) -> WriteSummary:
    """只寫入**通過驗證**的行情；重複不重寫、較舊不覆蓋較新。

    被拒絕的行情不會寫進 ``quote``，但呼叫端會把它們記進 ``ingest_run``
    （狀態＋錯誤分類＋原因），因此稽核軌跡完整而不污染行情序列。
    """
    summ = WriteSummary(received=len(assessments))
    accepted = [a for a in assessments if a.accepted]
    for a in assessments:
        if not a.accepted:
            summ.rejected += 1
            summ.add_reason(a.reason)
    if not accepted:
        return summ
    # 以 DB 既有最大市場時間為準（**跨批次**防護：舊不得覆蓋新）
    prev = stored_market_times(source or accepted[0].quote.source,
                               kind or accepted[0].quote.kind)
    fresh: list[Assessment] = []
    for a in accepted:
        stored = prev.get(a.quote.stock_id)
        if stored and a.quote.market_time <= stored:
            summ.rejected += 1
            summ.add_reason("not_newer_than_stored")
            continue
        fresh.append(a)
    if not fresh:
        return summ
    rows = []
    for a in fresh:
        row = a.quote.as_row()
        row["status"] = a.status
        # 本批次 run_id（來源列沒帶時以呼叫端為準），供稽核軌跡串起同一次輪詢
        if run_id and not row.get("run_id"):
            row["run_id"] = run_id
        rows.append(row)
    df = pd.DataFrame(rows)
    # if_conflict="ignore"：主鍵（source, kind, stock_id, market_time）已存在則跳過
    # → 重複取得同一筆行情不會重複寫入，也不會改寫既有列
    try:
        n = store.write_df(df, QUOTE_TABLE, if_conflict="ignore")
    except store.WriteRejected as e:
        log.error("行情寫入被拒（欄位不符）：%s", e)
        raise
    summ.written = int(n)
    summ.duplicates = int(len(fresh) - n)
    if summ.duplicates:
        summ.add_reason("already_stored", summ.duplicates)
    return summ


def latest_quotes(*, source: str | None = None, kind: str | None = None) -> pd.DataFrame:
    """每個標的**最新**的一筆已寫入行情（依 market_time 取最大）。"""
    sub = ("SELECT source, kind, stock_id, MAX(market_time) AS m FROM quote"
           " WHERE 1=1")
    params: list[Any] = []
    if source:
        sub += " AND source=?"
        params.append(source)
    if kind:
        sub += " AND kind=?"
        params.append(kind)
    sub += " GROUP BY source, kind, stock_id"
    sql = ("SELECT q.* FROM quote q JOIN (" + sub + ") t "
           "ON q.source=t.source AND q.kind=t.kind "
           "AND q.stock_id=t.stock_id AND q.market_time=t.m "
           "ORDER BY q.stock_id")
    try:
        return store.read_sql(sql, tuple(params) if params else None)
    except Exception as e:  # noqa: BLE001
        log.warning("讀取最新行情失敗（視為無資料）：%s", e)
        return pd.DataFrame()


# --------------------------------------------------------------------------
# 健康狀態與訊號閘門
# --------------------------------------------------------------------------
@dataclass
class HealthReport:
    source: str
    kind: str
    health: str
    session: str
    reason: str = ""
    n_symbols: int = 0
    last_market_time: str = ""
    age_s: float | None = None
    max_age_s: float = DEFAULT_MAX_AGE_S
    last_error_class: str = st.ErrorClass.NONE.value
    last_error: str = ""
    fabricated: bool = False

    @property
    def label(self) -> str:
        return (f"{SourceHealth(self.health).label}"
                + (f"（{self.reason}）" if self.reason else ""))

    def as_heartbeat(self) -> dict:
        return {"quote_source": self.source, "quote_kind": self.kind,
                "quote_health": self.health,
                "quote_health_label": SourceHealth(self.health).label,
                "quote_session": self.session,
                "quote_last_market_time": self.last_market_time,
                "quote_age_s": self.age_s,
                "quote_symbols": self.n_symbols,
                "quote_reason": self.reason,
                "quote_fabricated": self.fabricated}

    def to_text(self) -> str:
        L = [f"來源 {self.source}（{self.kind}）",
             f"  健康狀態   {self.label}",
             f"  交易時段   {SessionState(self.session).label}",
             f"  標的數     {self.n_symbols}",
             f"  最新市場時間 {self.last_market_time or '—'}"
             + (f"（{self.age_s:.0f} 秒前）" if self.age_s is not None else ""),
             f"  有效門檻   {self.max_age_s:.0f} 秒"]
        if self.last_error:
            L.append(f"  上次錯誤   {st.ErrorClass(self.last_error_class).label}"
                     f"：{self.last_error}")
        if self.fabricated:
            L.append("  ⚠️ 此為**合成（假）資料**，不得當成真實行情")
        return "\n".join(L)


def quote_health(df: pd.DataFrame | None = None, *, source: str | None = None,
                 kind: str | None = None, now: dt.datetime | None = None,
                 max_age_s: float = DEFAULT_MAX_AGE_S,
                 fetch_error_class: str = st.ErrorClass.NONE.value,
                 fetch_error: str = "",
                 configured: bool = True,
                 fabricated: bool = False) -> HealthReport:
    """來源健康狀態（五態，互斥）。

    判定順序（新的失敗必須蓋掉「沒資料」的表面平靜）：
      1. 硬性抓取失敗 → ``failed``
      2. 未設定來源 → ``unknown``
      3. 非交易時段 → ``non_trading_session``（收盤時「沒有新資料」是預期）
      4. 無任何行情 → ``unknown``（**不假裝正常**）
      5. 合成（假）來源 → ``unknown``（不是真實市場狀態；訊號閘門一律拒絕）
      6. 最新行情年齡 > 門檻 → ``delayed``
      7. 否則 → ``normal``
    """
    now = now or dt.datetime.now()
    sess = session_state(now).value
    rep = HealthReport(source=source or "（未設定）", kind=kind or "unknown",
                       health=SourceHealth.UNKNOWN.value, session=sess,
                       max_age_s=float(max_age_s),
                       last_error_class=fetch_error_class,
                       last_error=st.redact(fetch_error),
                       fabricated=bool(fabricated))
    if st.ErrorClass(fetch_error_class) in (
            st.ErrorClass.NETWORK, st.ErrorClass.TIMEOUT,
            st.ErrorClass.HTTP_STATUS, st.ErrorClass.PARSE,
            st.ErrorClass.WRITE):
        rep.health = SourceHealth.FAILED.value
        rep.reason = st.ErrorClass(fetch_error_class).label
        return rep
    df = df if df is not None else pd.DataFrame()
    if not configured:
        rep.reason = "未設定來源（TWSTOCK_QUOTE_SOURCE）"
        return rep
    if session_state(now) is not SessionState.OPEN:
        # 收盤時段「沒有新資料」是預期，不是未知；優先報非交易時段
        rep.health = SourceHealth.NON_TRADING_SESSION.value
        rep.reason = SessionState(sess).label
        return rep
    if df is None or len(df) == 0:
        rep.reason = "尚無任何行情紀錄"
        return rep
    rep.n_symbols = int(df["stock_id"].nunique()) if "stock_id" in df.columns else 0
    times = pd.to_datetime(df.get("market_time"), errors="coerce").dropna()
    if times.empty:
        rep.reason = "行情時間不可解析"
        return rep
    newest = times.max().to_pydatetime()
    rep.last_market_time = newest.isoformat(timespec="seconds")
    rep.age_s = max(0.0, (now - newest).total_seconds())
    if fabricated:
        # 合成資料一律不得被當成「正常」的可用行情；但仍保留可觀測事實
        rep.health = SourceHealth.UNKNOWN.value
        rep.reason = "合成（假）資料"
        return rep
    if rep.age_s > max_age_s:
        rep.health = SourceHealth.DELAYED.value
        rep.reason = f"最新行情已 {rep.age_s:.0f} 秒未更新（門檻 {max_age_s:.0f} 秒）"
        return rep
    rep.health = SourceHealth.NORMAL.value
    return rep


def intraday_signal_gate(health: HealthReport, *,
                         allow_non_session: bool = False) -> tuple[bool, str]:
    """盤中訊號閘門：**資料過期或來源異常時禁止產生看似有效的即時訊號**。

    這是未來盤中訊號路徑**必須**先通過的檢查；本階段訊號仍是盤後產生，
    因此此函式目前不影響任何既有流程。
    """
    if health.fabricated:
        return False, "合成（假）資料不得用於訊號"
    if health.health == SourceHealth.FAILED.value:
        return False, f"來源失敗：{health.reason}"
    if health.health == SourceHealth.UNKNOWN.value:
        return False, f"來源狀態未知：{health.reason}"
    if health.health == SourceHealth.DELAYED.value:
        return False, f"行情過期：{health.reason}"
    if health.health == SourceHealth.NON_TRADING_SESSION.value \
            and not allow_non_session:
        return False, f"非交易時段：{health.reason}"
    if not health.last_market_time:
        return False, "沒有可用的行情時間"
    return True, ""


# --------------------------------------------------------------------------
# 抓取 → 驗證 → 寫入（一次完整輪詢）
# --------------------------------------------------------------------------
@dataclass
class PollResult:
    polled_at: str
    source: str
    kind: str
    fetch: QuoteFetch | None = None
    write: WriteSummary = field(default_factory=WriteSummary)
    health: HealthReport | None = None
    run_id: str = ""
    skipped_reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.fetch and self.fetch.ok and self.write.written >= 0
                    and not self.skipped_reason)

    def label(self) -> str:
        if self.skipped_reason:
            return f"{self.source}：略過（{self.skipped_reason}）"
        parts = [self.source]
        if self.fetch is not None:
            parts.append(self.fetch.label())
        parts.append(self.write.label())
        if self.health is not None:
            parts.append(f"健康={self.health.health}")
        return "｜".join(parts)


def poll_once(stock_ids: list[str] | None = None, *,
              source: QuoteSource | None = None,
              now: dt.datetime | None = None,
              run_id: str = "",
              max_age_s: float = DEFAULT_MAX_AGE_S,
              require_session: bool = False,
              dry_run: bool = False) -> PollResult:
    """抓取一次行情並寫入（來源未設定時明確回報，不假裝成功）。

    ``require_session=True`` 時非交易時段直接略過（Worker 用）。
    ``dry_run=True`` 時只驗證不寫入。
    """
    now = now or dt.datetime.now()
    src = source or resolve_source()
    rid = run_id or st.new_run_id(now)
    if src is None:
        return PollResult(polled_at=now.isoformat(timespec="seconds"),
                          source="（未設定）", kind="unknown", run_id=rid,
                          skipped_reason="未設定 TWSTOCK_QUOTE_SOURCE",
                          health=quote_health(configured=False, now=now))
    ids = stock_ids or list(CONFIG.pool.keys())
    if require_session and session_state(now) in (SessionState.CLOSED,
                                                 SessionState.WEEKEND):
        return PollResult(
            polled_at=now.isoformat(timespec="seconds"), source=src.name,
            kind=src.info.kind, run_id=rid,
            skipped_reason=f"非交易時段（{session_state(now).label}）",
            health=quote_health(latest_quotes(source=src.name), source=src.name,
                                kind=src.info.kind, now=now, configured=True,
                                fabricated=src.is_fabricated))
    ok, why = src.available()
    if not ok:
        return PollResult(polled_at=now.isoformat(timespec="seconds"),
                          source=src.name, kind=src.info.kind, run_id=rid,
                          skipped_reason=why,
                          health=quote_health(latest_quotes(source=src.name),
                                              source=src.name, kind=src.info.kind,
                                              now=now, configured=False,
                                              fabricated=src.is_fabricated))
    res = src.fetch(ids)
    prev = stored_market_times(src.name, src.info.kind)
    assessments = validate_batch(res.quotes, prev=prev, now=now)
    summ = WriteSummary(received=len(assessments), dry_run=bool(dry_run))
    if dry_run:
        for a in assessments:
            if a.accepted:
                summ.written += 1
            else:
                summ.rejected += 1
                summ.add_reason(a.reason)
    else:
        summ = write_quotes(assessments, run_id=rid, source=src.name,
                            kind=src.info.kind)
    health = quote_health(latest_quotes(source=src.name, kind=src.info.kind),
                          source=src.name, kind=src.info.kind, now=now,
                          max_age_s=max_age_s,
                          fetch_error_class=res.error_class,
                          fetch_error=res.error,
                          fabricated=src.is_fabricated)
    return PollResult(polled_at=now.isoformat(timespec="seconds"),
                      source=src.name, kind=src.info.kind, fetch=res,
                      write=summ, health=health, run_id=rid)


def write_poll_records(poll: PollResult, fetch: QuoteFetch | None = None) -> int:
    """把一次輪詢寫進 ``ingest_run``（記憶體版；寫入由呼叫端決定）。

    回傳 ``IngestRecord`` 清單供 ``status.write_records`` 使用。
    """
    f = fetch or poll.fetch
    err_class = f.error_class if f is not None else st.ErrorClass.NONE.value
    err = f.error if f is not None else ""
    written = int(poll.write.written)
    rejected = int(poll.write.rejected)
    if err_class in (st.ErrorClass.NETWORK.value, st.ErrorClass.TIMEOUT.value,
                     st.ErrorClass.HTTP_STATUS.value, st.ErrorClass.PARSE.value,
                     st.ErrorClass.WRITE.value):
        status = st.IngestStatus.FAILED.value
    elif written and rejected:
        # 部分行情通過、部分被品質規則拒絕 → 不得只報「成功」
        status = st.IngestStatus.PARTIAL_SUCCESS.value
        if not err:
            err_class = st.ErrorClass.VALIDATION.value
            err = poll.write.label()
    elif written == 0 and rejected > 0:
        # 來源有回應但**沒有任何一筆**通過驗證 → 這是失敗，不是「正常無資料」
        status = st.IngestStatus.FAILED.value
        err_class = st.ErrorClass.VALIDATION.value
        err = poll.write.label()
    elif written:
        status = st.IngestStatus.SUCCESS_WITH_DATA.value
    else:
        status = st.IngestStatus.SUCCESS_EMPTY.value
    rec = st.IngestRecord(
        run_id=poll.run_id, dataset=DATASET_QUOTE, run_at=poll.polled_at,
        source=poll.source, task="poll_quotes", stock_id="",
        started_at=poll.polled_at, ended_at=st.now_iso(), status=status,
        error_class=err_class, error=err,
        rows_fetched=len(f.quotes) if f is not None else 0,
        rows_parsed=int(f.resolved) if f is not None else 0,
        rows_validated=written,
        rows_written=written,
        attempts=int(f.attempts) if f is not None else 0,
        source_version=SOURCE_VERSION, provenance=st.RECORDED,
        notes=poll.write.label()[:400])
    try:
        st.write_records([rec])
        return 1
    except Exception as e:  # noqa: BLE001
        log.warning("行情輪詢紀錄寫入失敗：%s", e)
        return 0


# --------------------------------------------------------------------------
# 報告
# --------------------------------------------------------------------------
def catalog_text() -> str:
    """來源評估（程式與文件同源）。"""
    L = ["=== 行情來源評估（P3）===",
         "四種行情必須分開：即時（realtime）／延遲（delayed）／盤後（eod）／"
         "歷史（historical）。本專案目前的正式來源是**盤後**日K。", ""]
    L.append(f"  {'來源':<16}{'種類':<9}{'已實作':<7}{'預設啟用':<9}認證")
    for s in SOURCE_CATALOG:
        L.append(f"  {s.name:<16}{QuoteKind(s.kind).label:<9}"
                 f"{'是' if s.implemented else '否':<7}"
                 f"{'是' if s.enabled_by_default else '否':<9}{s.auth[:40]}")
    L.append("")
    for s in SOURCE_CATALOG:
        L += [f"[{s.name}] {s.label}",
              f"  種類     {QuoteKind(s.kind).label}（{s.kind}）",
              f"  認證     {s.auth}",
              f"  限流     {s.rate_limit}",
              f"  延遲     {s.delay}",
              f"  條款     {s.terms}",
              f"  本階段查證 {s.verified}"]
        if s.caveat:
            L.append(f"  ⚠️ {s.caveat}")
        if s.docs_url:
            L.append(f"  文件     {s.docs_url}")
        L.append("")
    L.append("結論：**沒有已查證、可無條件使用的免費官方台股盤中即時來源。**")
    L.append("  - 交易所 MIS 端點技術上可取得，但 TWSE 網站條款禁止未經同意的自動化"
             "擷取，正式行情需另訂契約付費 → 本專案預設停用，需明確 opt-in。")
    L.append("  - FinMind 分鐘資料為本專案既有來源，可立即使用，但官方未載明交付延遲"
             "→ 一律歸為「延遲」，不得稱為即時。")
    L.append("  - Fugle 免費層有明確即時文件，但需使用者自行註冊 API key（本階段不代為註冊）。")
    L.append("  - 券商 API（Shioaji／富邦）需券商帳戶 → BLOCKED。")
    return "\n".join(L)


def status_text(now: dt.datetime | None = None) -> str:
    """目前行情來源設定、健康狀態與最新已寫入行情（**不連網**）。"""
    now = now or dt.datetime.now()
    L = [catalog_text(), "", "=== 本機行情狀態 ==="]
    env = os.environ.get("TWSTOCK_QUOTE_SOURCE") or ""
    src = resolve_source()
    L.append(f"  TWSTOCK_QUOTE_SOURCE = {env or '（未設定）'}")
    if src is not None:
        ok, why = src.available()
        L.append(f"  來源可用   {'是' if ok else '否'}"
                 + (f"（{why}）" if why else ""))
    L.append(f"  唯讀模式   {'是（不寫入）' if is_readonly() else '否'}")
    L.append(f"  現在時段   {session_state(now).label}")
    df = latest_quotes(source=src.name if src else None)
    rep = quote_health(df, source=src.name if src else None,
                       kind=src.info.kind if src else None, now=now,
                       configured=src is not None)
    L.append("")
    L.append(rep.to_text())
    if df is not None and len(df):
        L.append("")
        L.append(f"  最新行情（{len(df)} 檔，最多列 10 檔）")
        L.append(f"    {'股票':<8}{'市場時間':<21}{'價':>8}{'累積量':>12}"
                 f"{'種類':<7}{'時間來源':<8}狀態")
        for r in df.sort_values("stock_id").head(10).itertuples(index=False):
            L.append(f"    {str(r.stock_id):<8}{str(r.market_time):<21}"
                     f"{(f'{r.price:g}' if r.price is not None else '—'):>8}"
                     f"{(f'{r.total_volume:g}' if r.total_volume is not None else '—'):>12}"
                     f"{QuoteKind(r.kind).label if r.kind in QuoteKind._value2member_map_ else r.kind:<7}"
                     f"{str(r.market_time_source):<8}{r.status}")
    else:
        L.append("  （尚未寫入任何行情）")
    ok, why = intraday_signal_gate(rep)
    L.append("")
    L.append(f"  盤中訊號閘門 {'允許' if ok else '禁止'}"
             + (f"：{why}" if why else ""))
    L.append("  注意：本階段訊號仍為**盤後**產生；此閘門供未來盤中訊號路徑使用。")
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def selftest_now(spec: str | None = None,
                 now: dt.datetime | None = None) -> tuple[dt.datetime, str]:
    """自檢用的時間基準：回傳 ``(時間, 說明)``。

    ``selftest`` 走的是**完整管線**，而管線會把非交易時段的時間戳判為無效。
    因此在非交易時段（例如晚上跑自動測試）自檢拿不到任何通過驗證的行情。
    解法是明確改用**上一個交易日的盤中時間**，並把這件事**打印出來**——
    這是「合成時間戳」，不是把取得時間冒充成市場時間。
    """
    if spec:
        at = dt.datetime.fromisoformat(spec)
        return at, f"使用者指定 {at.isoformat(timespec='seconds')}"
    real = now or dt.datetime.now()
    if session_state(real) is SessionState.OPEN:
        return real.replace(microsecond=0), "實際時間（交易時段內）"
    d = real.date()
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    at = dt.datetime.combine(d, dt.time(10, 0))
    return at, (f"**合成時間戳** {at.isoformat(timespec='seconds')}"
                f"（現在 {real.strftime('%H:%M')} 非交易時段，"
                "此時間只用來驅動自檢管線，不是真實市場時間）")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="行情來源介面與健康狀態（P3）")
    ap.add_argument("cmd", nargs="?", default="status",
                    choices=["status", "poll", "selftest", "catalog"],
                    help="status（預設）| poll（實際抓取一次）| selftest（假來源）| catalog")
    ap.add_argument("--source", default=None,
                    help="來源名稱（未給則讀 TWSTOCK_QUOTE_SOURCE）")
    ap.add_argument("--stocks", nargs="*", default=None, help="指定代碼")
    ap.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_S,
                    help="行情有效年齡門檻（秒）")
    ap.add_argument("--at", default=None,
                    help="selftest 用的時間基準（ISO，預設在非交易時段自動改用最近盤中時間）")
    args = ap.parse_args(argv)

    if args.cmd == "catalog":
        print(catalog_text())
        return 0
    if args.cmd == "status":
        print(status_text())
        return 0

    if args.cmd == "selftest":
        at, why = selftest_now(args.at)
        src = FakeQuoteSource(kind=QuoteKind.DELAYED.value, now=lambda: at)
        # 自檢一律寫入**沙盒暫存 DB**：合成資料絕不得進入任何正式資料庫
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            sandbox = Path(td) / "selftest.db"
            real_db, real_dir = store.DB_PATH, store.DATA_DIR
            store.DB_PATH, store.DATA_DIR = sandbox, Path(td)
            try:
                res = poll_once(args.stocks, source=src, now=at,
                                max_age_s=args.max_age, run_id="selftest")
            finally:
                store.DB_PATH, store.DATA_DIR = real_db, real_dir
        print("⚠️ 以下為**合成（假）行情**，僅驗證管線，不代表任何真實市場資料。")
        print(f"⚠️ 時間基準：{why}（寫入沙盒暫存 DB，不碰正式資料庫）")
        print(res.label())
        print(res.health.to_text() if res.health else "（無健康狀態）")
        ok, why = intraday_signal_gate(res.health) if res.health else (False, "無狀態")
        print(f"盤中訊號閘門：{'允許' if ok else '禁止'}{'｜' + why if why else ''}")
        return 0 if res.write.written > 0 else 1

    src = resolve_source(args.source)
    if src is None:
        print("未設定行情來源：請設 TWSTOCK_QUOTE_SOURCE（可用："
              + "、".join(sorted(SOURCE_TYPES)) + "）後再執行 poll。")
        print("本階段**不會**替你選擇或註冊任何外部服務。")
        return 2
    ok, why = src.available()
    if not ok:
        print(f"來源 {src.name} 不可用：{why}")
        return 2
    res = poll_once(args.stocks, source=src, max_age_s=args.max_age)
    print(res.label())
    if res.health is not None:
        print(res.health.to_text())
    if res.fetch is not None and not res.fetch.ok and res.fetch.error:
        print(f"⚠️ 抓取失敗（{res.fetch.error_class}）：{res.fetch.error}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
