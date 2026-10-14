"""資料擷取狀態治理：讓「成功／空結果／部分成功／失敗／未執行／不可追溯」可被區分。

問題背景（實測本專案發現的無聲失敗）
  1. ``pipeline.ingest`` 對新聞一律寫 ``ok=True``，抓取失敗被記為成功。
  2. ``feedparser.parse(url)`` 在網路錯誤時**不拋例外**，回傳空 entries，
     使「抓取失敗」與「真的沒有新聞」在資料層無法區分。
  3. ``store.write_df`` 在欄位無交集時回傳 0 並只寫 log，呼叫端仍記成功。
  4. Worker 只以「價量列數 > 0」判定整輪成敗，把「來源正常但當日無資料」
     誤判為失敗，反之亦無法從紀錄看出部分失敗。

設計原則
  - **失敗不可降級成空結果**：``classify`` 的判定順序讓任何非 NONE 的錯誤分類
    一律得到 ``FAILED``，不會落成 ``SUCCESS_EMPTY``。
  - **不可追溯就標 unknown**：舊資料或無結構化回傳一律記錄為 ``UNKNOWN``，
    絕不補猜成功。
  - **每次執行可重建**：來源、任務、開始／結束時間、狀態、可重現的錯誤分類、
    取得／解析／驗證／寫入筆數、資料涵蓋日期、重試次數、來源版本與設定識別。
  - **不記錄機密**：所有錯誤訊息寫入前先經 ``redact``。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum

import pandas as pd

from . import store

log = logging.getLogger(__name__)

# 資料庫中代表「來源與任務」的識別
DATASET_NEWS = "news"
DATASET_PRICE = "price"
INGEST_RUN_TABLE = "ingest_run"

RECORDED = "recorded"      # 本階段之後寫入、狀態可追溯
UNKNOWN_PROVENANCE = "unknown"   # 本階段之前的舊資料，狀態不可追溯


class IngestStatus(str, Enum):
    """單一 (執行, 來源, 任務, 標的) 的擷取結果。"""

    SUCCESS_WITH_DATA = "success_with_data"   # 成功且取得有效資料
    SUCCESS_EMPTY = "success_empty"           # 來源正常回應，但沒有符合條件的資料
    PARTIAL_SUCCESS = "partial_success"       # 部分成功、另有可辨識的失敗
    FAILED = "failed"                         # 請求／解析／驗證／寫入失敗
    SKIPPED = "skipped"                       # 依明確規則未執行
    UNKNOWN = "unknown"                       # 舊資料或無法追溯的執行紀錄

    @property
    def label(self) -> str:
        return {
            "success_with_data": "成功（有資料）",
            "success_empty": "成功（無資料）",
            "partial_success": "部分成功",
            "failed": "失敗",
            "skipped": "未執行",
            "unknown": "不可追溯",
        }[self.value]

    @property
    def is_failure(self) -> bool:
        return self is IngestStatus.FAILED

    @property
    def has_data(self) -> bool:
        return self is IngestStatus.SUCCESS_WITH_DATA

    @property
    def is_usable(self) -> bool:
        """來源端有正常結論者（含「確實沒有資料」）。失敗與不可追溯不算。"""
        return self in (IngestStatus.SUCCESS_WITH_DATA,
                        IngestStatus.SUCCESS_EMPTY,
                        IngestStatus.PARTIAL_SUCCESS)


class ErrorClass(str, Enum):
    """可重現的錯誤分類（供重試與健檢使用，非自由文字）。"""

    NONE = "none"
    NETWORK = "network"
    TIMEOUT = "timeout"
    HTTP_STATUS = "http_status"
    PARSE = "parse"
    VALIDATION = "validation"
    WRITE = "write"
    EMPTY = "empty"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {
            "none": "無",
            "network": "網路連線失敗",
            "timeout": "逾時",
            "http_status": "HTTP 狀態異常",
            "parse": "解析失敗",
            "validation": "驗證失敗",
            "write": "寫入失敗",
            "empty": "無符合條件資料",
            "unknown": "未分類",
        }[self.value]


# 失敗類別（不可降級為「成功但沒資料」）
_HARD_FAILURES = frozenset({
    ErrorClass.NETWORK.value, ErrorClass.TIMEOUT.value,
    ErrorClass.HTTP_STATUS.value, ErrorClass.PARSE.value,
    ErrorClass.VALIDATION.value, ErrorClass.WRITE.value,
})


def classify(*, error_class: str = ErrorClass.NONE.value, written: int = 0,
             n_ok: int = 0, n_fail: int = 0) -> str:
    """由可觀測事實判定狀態（純函式）。

    判定順序即語意保證：
      1. 任何硬性失敗分類 → ``FAILED``（**永不**落成 ``SUCCESS_EMPTY``）。
      2. 只有 ``EMPTY`` 錯誤分類 → ``SUCCESS_EMPTY``。
      3. 有成功也有失敗 → ``PARTIAL_SUCCESS``。
      4. 有寫入列數 → ``SUCCESS_WITH_DATA``。
      5. 其餘（成功但 0 列）→ ``SUCCESS_EMPTY``。
    """
    if error_class in _HARD_FAILURES:
        return IngestStatus.FAILED.value
    if error_class == ErrorClass.EMPTY.value:
        return IngestStatus.SUCCESS_EMPTY.value
    if n_fail > 0 and n_ok > 0:
        return IngestStatus.PARTIAL_SUCCESS.value
    if n_fail > 0:
        return IngestStatus.FAILED.value
    if written > 0:
        return IngestStatus.SUCCESS_WITH_DATA.value
    return IngestStatus.SUCCESS_EMPTY.value


def error_class_from_exception(exc: BaseException | None) -> str:
    """把例外映射到可重現的分類（供重試決策與健檢統計）。"""
    if exc is None:
        return ErrorClass.NONE.value
    if isinstance(exc, TimeoutError):
        return ErrorClass.TIMEOUT.value
    if isinstance(exc, (ConnectionError, OSError)):
        return ErrorClass.NETWORK.value
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return ErrorClass.VALIDATION.value
    return ErrorClass.UNKNOWN.value


# --------------------------------------------------------------------------
# 機密遮蔽
# --------------------------------------------------------------------------
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(token|api[_-]?key|apikey|secret|password|passwd|pwd)"
               r"\s*[=:]\s*[^\s,;&)\]}]+"),
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(https?://[^\s?&]+)[?&](token|api_key|apikey|key)="
               r"[^\s&]*"),
)
MAX_ERROR_CHARS = 500


def redact(text: str | None) -> str:
    """移除疑似機密並截斷長度。所有寫入 DB 的錯誤訊息都必須先過此函式。"""
    if not text:
        return ""
    out = str(text)
    for pat in _SECRET_PATTERNS:
        out = pat.sub(lambda m: (m.group(1) if m.lastindex else "") + "«redacted»",
                      out)
    out = re.sub(r"\s+", " ", out).strip()
    return out[:MAX_ERROR_CHARS]


# --------------------------------------------------------------------------
# 紀錄
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class IngestRecord:
    """一次 (執行, 來源, 任務, 標的) 的完整可重建紀錄。"""

    run_id: str
    dataset: str
    run_at: str
    source: str = ""
    task: str = ""
    stock_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    status: str = IngestStatus.UNKNOWN.value
    error_class: str = ErrorClass.NONE.value
    error: str = ""
    rows_fetched: int = 0
    rows_parsed: int = 0
    rows_validated: int = 0
    rows_written: int = 0
    data_start: str = ""
    data_end: str = ""
    attempts: int = 0
    source_version: str = ""
    config_id: str = ""
    provenance: str = RECORDED
    notes: str = ""

    def as_row(self) -> dict:
        row = asdict(self)
        row["error"] = redact(row["error"])
        row["notes"] = redact(row["notes"])
        return row

    @property
    def duration_s(self) -> float | None:
        try:
            a = dt.datetime.fromisoformat(self.started_at)
            b = dt.datetime.fromisoformat(self.ended_at)
        except (TypeError, ValueError):
            return None
        return max(0.0, (b - a).total_seconds())


DEFAULT_CONFIG_ID = "config:default"


def config_id(pool: dict | None = None, lookback_years: float | None = None,
              extra: dict | None = None) -> str:
    """設定識別：可重現的短雜湊（不含任何機密）。"""
    payload = {
        "pool": sorted((pool or {}).keys()),
        "lookback_years": lookback_years,
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return "cfg:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def new_run_id(now: dt.datetime | None = None) -> str:
    """每輪唯一且可排序的執行識別。"""
    now = now or dt.datetime.now()
    return now.strftime("%Y%m%dT%H%M%S")


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def write_records(records: list[IngestRecord]) -> int:
    """把紀錄寫入 ``ingest_run``（依主鍵去重，重跑不產生重複資料）。"""
    if not records:
        return 0
    df = pd.DataFrame([r.as_row() for r in records])
    return store.write_df(df, INGEST_RUN_TABLE,
                          key_cols=("run_id", "dataset", "stock_id"))


# --------------------------------------------------------------------------
# 每輪彙總
# --------------------------------------------------------------------------
@dataclass
class IngestSummary:
    """一輪 ingest 的結果彙總。

    為向後相容，仍可當成 ``{stock_id: price_rows}`` 的 mapping 使用
    （``worker`` 舊介面），但成敗判定**必須**改看 ``statuses``。
    """

    run_id: str = ""
    run_at: str = ""
    started_at: str = ""
    ended_at: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)   # stock -> price 狀態
    records: list[IngestRecord] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)

    # ---- mapping 相容介面 ----
    def __getitem__(self, k: str) -> int:
        return self.counts[k]

    def __contains__(self, k: object) -> bool:
        return k in self.counts

    def __len__(self) -> int:
        return len(self.counts)

    def __iter__(self):
        return iter(self.counts)

    def keys(self):
        return self.counts.keys()

    def values(self):
        return self.counts.values()

    def items(self):
        return self.counts.items()

    # ---- 狀態查詢 ----
    def by_dataset(self, dataset: str) -> list[IngestRecord]:
        return [r for r in self.records if r.dataset == dataset]

    def status_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    @property
    def usable_count(self) -> int:
        """有正常來源結論的紀錄數（含確實無資料）。"""
        return sum(1 for r in self.records
                   if IngestStatus(r.status).is_usable)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.records
                   if IngestStatus(r.status).is_failure)

    @property
    def unknown_count(self) -> int:
        return sum(1 for r in self.records
                   if IngestStatus(r.status) is IngestStatus.UNKNOWN)

    @property
    def has_data(self) -> bool:
        return any(IngestStatus(r.status).has_data for r in self.records)

    @property
    def all_failed(self) -> bool:
        """整輪完全無可用結果（全數硬性失敗）→ 才可視為整輪失敗。

        只要有任何紀錄是 success_with_data / success_empty / partial_success，
        就不算整輪失敗——「來源正常但沒有資料」不是失敗。
        """
        if not self.records:
            return True
        return self.usable_count == 0

    def label(self) -> str:
        c = self.status_counts()
        parts = [f"{IngestStatus(k).label}={v}" for k, v in sorted(c.items())]
        return f"run {self.run_id}：" + ("、".join(parts) if parts else "無紀錄")

    @property
    def overall_status(self) -> str:
        """整輪整體狀態。

        重點：
          - 只有「全數失敗」才回 ``failed``，部分失敗回 ``partial_success``。
          - 「來源正常且確實無資料」貢獻 ``success_empty``，不是失敗。
          - 全部不可追溯時回 ``unknown``，不補猜成功。
        """
        vals = list(self.statuses.values())
        if not vals:
            return IngestStatus.UNKNOWN.value
        n = len(vals)
        has_data = any(v == IngestStatus.SUCCESS_WITH_DATA.value for v in vals)
        has_empty = any(v == IngestStatus.SUCCESS_EMPTY.value for v in vals)
        has_partial = any(v == IngestStatus.PARTIAL_SUCCESS.value for v in vals)
        n_fail = sum(1 for v in vals if v == IngestStatus.FAILED.value)
        n_unk = sum(1 for v in vals if v == IngestStatus.UNKNOWN.value)
        if n_fail and (has_data or has_empty or has_partial):
            return IngestStatus.PARTIAL_SUCCESS.value
        if n_fail == n:
            return IngestStatus.FAILED.value
        if has_partial:
            return IngestStatus.PARTIAL_SUCCESS.value
        if has_data:
            return IngestStatus.SUCCESS_WITH_DATA.value
        if has_empty:
            return IngestStatus.SUCCESS_EMPTY.value
        if n_unk == n:
            return IngestStatus.UNKNOWN.value
        return IngestStatus.SUCCESS_EMPTY.value


def empty_summary() -> IngestSummary:
    return IngestSummary()
