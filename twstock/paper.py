"""P4：模擬交易引擎（Paper Trading Engine）。

**本模組只做虛擬交易**：所有資金是模擬的、所有成交是模擬的。
不得接入券商 API、不得執行真實下單——本模組**不存在任何券商 SDK 或
網路呼叫路徑**（測試含「無券商套件引用」檢查）。本階段支援**盤後（EOD
訊號）**整股模擬交易；不自動進入 P5。

成交模擬原則（沿用回測的時間正確性）
  - 訊號以 **T 日收盤**資料計算 → 最早可成交時點為 **T+1 開盤**（與
    ``backtest._schedule_entries`` 同一約定）。引擎直接讀正式 price
    table 在 signal_date 之後的第一個行情列，以該列**開盤價**成交；
    **不得**以 T 日已知收盤價成交（訂單必須攜帶 ``signal_date``，
    成交日由行情決定，可稽核）。
  - 成交價內含滑價（買 +slippage／賣 −slippage，沿用 ``costs.CostModel``），
    另存滑價前原始行情價（``raw_price_utwd``）供稽核。
  - **不可成交情境**（記錄原因、推進行情光標、不假設成交成功）：缺價
    （行情無列）、停牌（成交量=0）、漲停鎖死（買不進）、跌停鎖死
    （賣不掉）、無前收盤價（漲跌停參考不可得）。訂單逾
    ``good_til_days`` 未成交 → 取消標注 ``expired``。
  - **部分成交**：支援。單筬可成交量 ≤ 當日成交量 × ``participation_rate``
    （沿用 ``CostModel.max_fill_qty``）；殘量保留至下一交易日繼續；
    盤中分筆撮合屬 P5 範圍（如實標注，非悄悄近似）。
  - **整股**：台股現股最小交易單位 1 張＝1000 股；數量與成交量一律
    1000 的倍數（**零股交易暫不支援**）。
  - 成交模擬只是對真實交易的**近似**，不得宣稱與實際券商成交一致。

金額精確性
  **所有帳務金額以「微元」整數（utwd，1 utwd = 10^-6 TWD）累積**，
  不依賴浮點數累積；數量為整數股。費率可設定（costs.CostModel）、
  每張成交列記錄費率情境，**不得硬編碼未確認的現行費率**。

帳務閉環（可重播、可核對）
  - 訂單入庫（pending_new → accepted）→ 成交→ 帳本→ 持倉→ 訂單狀態，
    全部在**同一個 SQLite 交易**（BEGIN IMMEDIATE）內完成；
    任何一處失敗全部回滾，不留半套資料。
  - 冪等防護：``client_order_id`` 唯一；``fill_id=(訂單 id, 行情日)`` 唯一；
    帳本 ``ref`` 唯一。重播不會重複扣款或重複持倉。
  - **資金事實＝ledger**：``cash = initial + Σ ledger.amount``；
    **持倉事實＝成交序列**（WAC 含買入費用分攤）；已實現與未實現損益
    **分開**，皆可由成交序列重放重建。
  - ``reconcile`` 核對：帳本重建現金＝帳戶 cash；重建持倉＝持倉快取；
    每筆成交金額＝notional+fee+tax 與帳本逐筆核对；餘額鏈一致；
    快照 equity = cash + market_value。
  - **資料庫完整性錯誤 → 立即停止後續交易**（``IntegrityBroken``）。

風險整合（訂單進入成交流程**之前**執行）
  沿用 ``risk.DEFAULT_LIMITS`` 與 ``risk.KillSwitch``。
  行情過期 **禁止建立新訂單**（既定未完成訂單仍可依已存行情成交）；
  每次拒單／取消／過期／成交留下 ``paper_event`` 稽核紀錄。
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Iterator

import pandas as pd

from . import gate as gate_mod
from . import risk as risk_mod
from .config import CONFIG, is_readonly
from .costs import CostModel, get_cost_model
from .data import store

log = logging.getLogger(__name__)

ACCOUNT_NAME_DEFAULT = "paper"

#: 1 微元（utwd）= 10^-6 TWD；帳務一律整數運算
MICRO = 10 ** 6
#: 價格最小跳動單位近似（模擬層說明使用）
TICK = Decimal("0.01")
#: 台股現股整股 1 張 = 1000 股
LOT = 1000

#: 台股現股收盤時間（成交模擬只用「已發生」的行情：當日 bar 需已過 13:30）
_EOD_CUTOFF = (13, 30)

ORDER_TYPE_MARKET = "market"
ORDER_TYPES = (ORDER_TYPE_MARKET,)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


STATUS_TRANSITIONS: dict[str, set[str]] = {
    OrderStatus.PENDING_NEW.value: {
        OrderStatus.ACCEPTED.value,
        OrderStatus.REJECTED.value,
        OrderStatus.CANCELLED.value,
    },
    OrderStatus.ACCEPTED.value: {
        OrderStatus.PARTIALLY_FILLED.value,
        OrderStatus.FILLED.value,
        OrderStatus.CANCELLED.value,
        OrderStatus.EXPIRED.value,
    },
    OrderStatus.PARTIALLY_FILLED.value: {
        OrderStatus.FILLED.value,
        OrderStatus.CANCELLED.value,
        OrderStatus.EXPIRED.value,
    },
    OrderStatus.FILLED.value: set(),
    OrderStatus.CANCELLED.value: set(),
    OrderStatus.EXPIRED.value: set(),
    OrderStatus.REJECTED.value: set(),
}


class PaperError(RuntimeError):
    """模擬交易引擎錯誤基底。"""


class InvalidTransition(PaperError):
    """非法訂單狀態轉移（生命週期保護）。"""


class IntegrityBroken(PaperError):
    """資料庫完整性錯誤 —— 引擎必須停止交易而非繼續執行。"""


class LiveModeForbidden(PaperError):
    """在 live 模式（含人工確認）下呼叫模擬交易 —— 直接拒絕。"""


def assert_transition(cur: str, nxt: str) -> None:
    if cur == nxt or nxt not in STATUS_TRANSITIONS.get(cur, set()):
        raise InvalidTransition(f"訂單狀態轉移不合法：{cur} → {nxt}")


OPEN_STATUSES = (OrderStatus.PENDING_NEW.value, OrderStatus.ACCEPTED.value,
                 OrderStatus.PARTIALLY_FILLED.value)
FINAL_STATUSES = (OrderStatus.FILLED.value, OrderStatus.CANCELLED.value,
                  OrderStatus.EXPIRED.value, OrderStatus.REJECTED.value)


# ---------------------------------------------------------------------------
# 金額與數量（全部整數微元運算）
# ---------------------------------------------------------------------------
def to_micro(x: float | Decimal | str) -> int:
    """金額 → 微元整數（ROUND_HALF_UP 至分位）。"""
    return int((Decimal(str(x)) * MICRO).to_integral_value(
        rounding=ROUND_HALF_UP))


def from_micro(x_micro: int | None) -> float:
    if x_micro is None:
        return 0.0
    return float(x_micro) / MICRO


def f_twd(x_micro: int | None, *, signed: bool = False) -> str:
    v = from_micro(x_micro or 0)
    return f"{v:+,.2f}" if signed else f"{v:,.2f}"


def round_tick(p: Decimal | float | str) -> Decimal:
    """價格以 0.01 近似捨入（模擬層說明，非真實 tick 表）。"""
    return Decimal(str(p)).quantize(TICK, rounding=ROUND_HALF_UP)


def _lots(qty: Decimal | float | int) -> int:
    """向下取整到 1000 的倍數（台股整股）。"""
    q = int(qty)
    return (q // LOT) * LOT if q > 0 else 0


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _today(x: dt.datetime | dt.date | str | None) -> str:
    if isinstance(x, dt.datetime):
        return x.date().isoformat()
    if isinstance(x, dt.date):
        return x.isoformat()
    if not x:
        return ""
    return str(x)[:10]


def fee_for(notional_micro: int, *, cost: CostModel, side: str
            ) -> tuple[int, int]:
    """回傳 (手續費, 證交稅)；證交稅僅賣出。金額微元整數。"""
    comm = max(to_micro(from_micro(notional_micro)
                        * cost.commission_effective), to_micro(cost.min_fee))
    tax = (to_micro(from_micro(notional_micro) * cost.tax_rate)
           if side == OrderSide.SELL.value else 0)
    return comm, tax


def fill_price(raw_price: Decimal | float, *, side: str,
               cost: CostModel) -> Decimal:
    """成交價＝行情價 ± 滑價，並作 0.01 tick 近似捨入。"""
    slip = Decimal(str(cost.slippage))
    adj = slip if side == OrderSide.BUY.value else -slip
    return round_tick(Decimal(str(raw_price)) * (Decimal("1") + adj))


def _cash_required(price: Decimal | float, qty: int, cost: CostModel) -> int:
    """買進所需現金（含費用；滑價上限前置，保守檢查）。"""
    notional_slipped = (Decimal(str(price)) * Decimal(int(qty))
                        * (Decimal("1") + Decimal(str(cost.slippage))))
    n = to_micro(notional_slipped)
    comm, _ = fee_for(n, cost=cost, side=OrderSide.BUY.value)
    return n + comm


# ---------------------------------------------------------------------------
# 持倉重建（成交序列 → 持倉；與寫入路徑共用 → 重播一致）
# ---------------------------------------------------------------------------
@dataclass
class PositionState:
    qty: int = 0
    cost_basis: int = 0      # 剩餘持股總成本基礎（含買入費用分攤；微元）
    realized: int = 0        # 累計已實現損益（微元）

    @property
    def avg_cost_share(self) -> int:
        return 0 if self.qty <= 0 else self.cost_basis // self.qty


def _proportional_cost(basis: int, share: int, qty: int) -> int:
    """按比例分攤成本（floor；多處共用本函式保證重播一致）。"""
    if qty <= 0:
        return 0
    return basis * share // qty


def replay_positions(fills: pd.DataFrame) -> dict[str, PositionState]:
    """由成交序列按寫入序（rowid）重建持倉與已實現損益。"""
    pos: dict[str, PositionState] = {}
    for _, r in fills.sort_values("rowid").iterrows():
        sid = str(r["stock_id"])
        st = pos.setdefault(sid, PositionState())
        q = int(r["qty"])
        px = int(r["price_utwd"])          # 微元整數（schema：paper_fill）
        fee = int(r["fee_utwd"] or 0)
        tax = int(r["tax_utwd"] or 0)
        if r["side"] == OrderSide.BUY.value:
            st.qty += q
            st.cost_basis += px * q + fee
        else:
            removed = _proportional_cost(st.cost_basis, q, max(1, st.qty))
            st.realized += (px * q) - fee - tax - removed
            st.qty -= q
            st.cost_basis -= removed
        if st.qty <= 0:
            st.qty = 0
            st.cost_basis = 0
    return pos


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
class PaperEngine:
    """模擬交易引擎（帳務閉環；重啟從 DB 完整恢復）。"""

    def __init__(self, *, cost_model: CostModel | None = None,
                 limits: risk_mod.RiskLimits | None = None,
                 killswitch: risk_mod.KillSwitch | None = None,
                 good_til_days: int = 5,
                 top_n: int | None = None,
                 hold_days: int | None = None,
                 buy_threshold: float | None = None,
                 account_name: str = ACCOUNT_NAME_DEFAULT) -> None:
        self.cm = cost_model or get_cost_model()
        self.limits = limits or risk_mod.DEFAULT_LIMITS
        self.killswitch = killswitch or risk_mod.KillSwitch()
        self.good_til_days = int(good_til_days)
        self.top_n = int(top_n or CONFIG.top_n)
        self.hold_days = int(hold_days or CONFIG.hold_days)
        self.buy_threshold = (float(buy_threshold)
                              if buy_threshold is not None else 0.2)
        self.account_name = account_name

    # ------------------------------------------------------------------
    # 交易閉環（所有帳務寫入都在此 context 內）
    # ------------------------------------------------------------------
    @contextmanager
    def _ro(self) -> Iterator[sqlite3.Connection]:
        """唯讀連線 context（dashboard/唯讀模式安全）。"""
        conn = self._conn_read()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        if is_readonly():
            raise PaperError(
                "唯讀模式（TWSTOCK_READONLY）不允許模擬交易寫入")
        try:
            conn = store._connect()
        except Exception as e:
            raise IntegrityBroken(f"DB 連線不可用：{e}") from e
        try:
            conn.isolation_level = None
            conn.row_factory = sqlite3.Row
            store._init(conn)
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as e:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise IntegrityBroken(f"DB 完整性錯誤（已回滾）：{e}") from e
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 帳戶
    # ------------------------------------------------------------------
    def ensure_account(self, *, initial_cash: float = 1_000_000.0,
                       name: str | None = None,
                       overwrite: bool = False,
                       notes: str = "") -> int:
        """建立（冪等）虛擬帳戶；初始資金寫入帳本（capital 事件）。"""
        name = name or self.account_name
        micro = to_micro(initial_cash)
        if micro <= 0:
            raise ValueError("初始資金必須為正")
        with self._tx() as conn:
            row = conn.execute(
                "SELECT account_id FROM paper_account WHERE name=?",
                (name,)).fetchone()
            if row and not overwrite:
                return int(row["account_id"])
            if row:
                aid = int(row["account_id"])
                conn.execute("DELETE FROM paper_ledger WHERE account_id=?",
                             (aid,))
                conn.execute("DELETE FROM paper_fill WHERE account_id=?",
                             (aid,))
                conn.execute("DELETE FROM paper_position WHERE account_id=?",
                             (aid,))
                conn.execute("DELETE FROM paper_order WHERE account_id=?",
                             (aid,))
                conn.execute(
                    "UPDATE paper_account SET initial_cash_utwd=?, cash_utwd=?"
                    " WHERE account_id=?", (micro, 0, aid))
            else:
                cur = conn.execute(
                    "INSERT INTO paper_account (name, mode, status,"
                    " initial_cash_utwd, cash_utwd, fee_scenario, created_at,"
                    " notes) VALUES (?, 'paper', 'active', ?, ?, ?, ?, ?)",
                    (name, micro, micro, self.cm.name, _now(), notes))
                aid = int(cur.lastrowid)
            self._ledger(conn, aid, "capital", micro, f"a{aid}:capital",
                         note=f"初始虛擬資金 {f_twd(micro)}",
                         ignore_base=True)
            return aid

    def account(self) -> dict[str, Any] | None:
        df = store.read_sql(
            "SELECT * FROM paper_account WHERE name=?", (self.account_name,))
        return None if df.empty else df.iloc[0].to_dict()

    def _aid(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT account_id FROM paper_account WHERE name=?",
            (self.account_name,)).fetchone()
        if row is None:
            raise PaperError("尚未建立 paper 帳戶（請先執行 paper init）")
        return int(row["account_id"])

    # ------------------------------------------------------------------
    # 帳本與持倉（同一 _tx 內一致更新；重播＝重建）
    # ------------------------------------------------------------------
    def _ledger(self, conn: sqlite3.Connection, aid: int, kind: str,
                amount: int, ref: str, *, stock_id: str = "",
                order_id: str = "", fill_id: str = "", note: str = "",
                ignore_base: bool = False
                ) -> None:
        """寫一筆有號現金事件並同步帳戶 cash（ref 唯一 → 重播防護）。"""
        acc = conn.execute(
            "SELECT cash_utwd FROM paper_account WHERE account_id=?",
            (aid,)).fetchone()
        if acc is None:
            raise PaperError(f"帳戶 {aid} 不存在")
        base = int(acc["cash_utwd"]) if not ignore_base else 0
        balance = base + int(amount)
        conn.execute(
            "INSERT INTO paper_ledger (account_id, ts, kind, amount_utwd,"
            " balance_utwd, order_id, fill_id, stock_id, note, ref) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (aid, _now(), kind, int(amount), balance, order_id, fill_id,
             stock_id, note[:400], ref))
        conn.execute(
            "UPDATE paper_account SET cash_utwd=? WHERE account_id=?",
            (balance, aid))

    @staticmethod
    def cash_and_positions(aid: int
                           ) -> tuple[int, dict[str, PositionState]]:
        """由帳本＋成交序列重建事實（重啟後即完整恢復、可核對）。"""
        cash = int(store.read_sql(
            "SELECT COALESCE(SUM(amount_utwd),0) AS c FROM paper_ledger "
            "WHERE account_id=?", (aid,)).iloc[0]["c"])
        fills = store.read_sql(
            "SELECT rowid AS rowid, * FROM paper_fill WHERE account_id=?",
            (aid,))
        return cash, replay_positions(fills)

    def _state(self, conn: sqlite3.Connection, aid: int
               ) -> tuple[int, dict[str, PositionState]]:
        cash = int(conn.execute(
            "SELECT cash_utwd FROM paper_account WHERE account_id=?",
            (aid,)).fetchone()["cash_utwd"])
        rows = pd.read_sql_query(
            "SELECT rowid AS rowid, * FROM paper_fill WHERE account_id=?",
            conn, params=(aid,))
        return cash, replay_positions(rows)

    def _equity_with_market(self, conn: sqlite3.Connection, aid: int,
                            cash: int,
                            positions: dict[str, PositionState]) -> float:
        """equity = cash + Σ(qty × 最新收盤)；缺行情以成本計（保守）。"""
        mv_int = 0
        for sid, st in positions.items():
            if st.qty <= 0:
                continue
            _, pc = self._last_close(conn, sid)
            mv_int += ((to_micro(pc) if pc and pc > 0 else st.avg_cost_share)
                       * st.qty)
        return from_micro(cash + mv_int)

    def _open_orders(self, conn: sqlite3.Connection, aid: int) -> int:
        return int(conn.execute(
            "SELECT COUNT(*) FROM paper_order WHERE account_id=? AND status"
            " IN ('pending_new','accepted','partially_filled')",
            (aid,)).fetchone()[0])

    def _orders_today(self, conn: sqlite3.Connection, aid: int,
                      day: str) -> int:
        return int(conn.execute(
            "SELECT COUNT(*) FROM paper_order WHERE account_id=? AND "
            "substr(created_at,1,10)=?", (aid, day)).fetchone()[0])

    def _day_pnl(self, conn: sqlite3.Connection, now: dt.datetime) -> float:
        """相對上一「不同交易日」快照的當日回報（無快照 → 0，保守）。"""
        day = _today(now)
        aid = self._aid(conn)
        rows = conn.execute(
            "SELECT as_of, equity_utwd FROM paper_snapshot WHERE account_id=?"
            " ORDER BY snapshot_id DESC LIMIT 30", (aid,)).fetchall()
        cur = prev = None
        for r in rows:
            if cur is None:
                cur = int(r["equity_utwd"])
            elif _today(r["as_of"]) != day:
                prev = int(r["equity_utwd"])
                break
        if prev is None or prev <= 0 or cur is None:
            return 0.0
        return (cur - prev) / prev

    def _drawdown(self, conn: sqlite3.Connection,
                  equity_now: float) -> float:
        """回撤＝(歷史最高 equity − 現在) / 歷史最高。"""
        peak = conn.execute(
            "SELECT MAX(equity_utwd) AS p FROM paper_snapshot WHERE"
            " account_id=?", (self._aid(conn),)).fetchone()
        # 快照存微元整數；此處與 equity_now（TWD float）同單位比較
        p = max(from_micro(int(peak["p"] or 0)) if peak else 0.0,
                equity_now)
        return 0.0 if p <= 0 else (p - equity_now) / p

    # ------------------------------------------------------------------
    # 行情輔助（唯一輸入＝正式 price table，不推估）
    # ------------------------------------------------------------------
    def _last_close(self, conn: sqlite3.Connection, stock_id: str
                    ) -> tuple[str, Decimal]:
        row = conn.execute(
            "SELECT date, close FROM price WHERE stock_id=? ORDER BY date"
            " DESC LIMIT 1", (stock_id,)).fetchone()
        if row is None or row["close"] is None:
            return "", Decimal("0")
        c = float(row["close"])
        if c <= 0:
            return "", Decimal("0")
        return str(row["date"]), Decimal(str(c))

    def _last_trade_date(self, conn: sqlite3.Connection) -> str | None:
        row = conn.execute("SELECT MAX(date) AS m FROM price").fetchone()
        return str(row["m"]) if row and row["m"] else None

    def data_staleness_days(self, now: dt.datetime | None = None) -> int:
        now = now or dt.datetime.now()
        df = store.read_sql("SELECT MAX(date) AS m FROM price", None)
        last = df.iloc[0]["m"] if not (df.empty or pd.isna(
            df.iloc[0]["m"])) else None
        if not last:
            return 99
        return (now.date() - dt.date.fromisoformat(str(last))).days

    def limit_up_down(self, prev_close: float | None
                      ) -> tuple[Decimal | None, Decimal | None]:
        """漲跌停價（10% ± 0.01 tick 近似；請見文件限制）。"""
        if prev_close is None or prev_close <= 0:
            return None, None
        base = Decimal(str(prev_close))
        return (round_tick(base * Decimal("1.10")),
                round_tick(base * Decimal("0.90")))

    def _bar_after(self, stock_id: str, after_date: str
                   ) -> pd.Series | None:
        df = store.read_sql(
            "SELECT stock_id, date, open, high, low, close, volume FROM "
            "price WHERE stock_id=? AND date>? ORDER BY date LIMIT 1",
            (stock_id, after_date))
        return None if df.empty else df.iloc[0]

    def _prev_close(self, stock_id: str, before_date: str) -> float | None:
        df = store.read_sql(
            "SELECT close FROM price WHERE stock_id=? AND date<? ORDER BY "
            "date DESC LIMIT 1", (stock_id, before_date))
        if df.empty or pd.isna(df.iloc[0]["close"]):
            return None
        v = float(df.iloc[0]["close"])
        return v if v > 0 else None

    def _price_hint(self, conn: sqlite3.Connection, stock_id: str
                    ) -> float | None:
        _, pc = self._last_close(conn, stock_id)
        return float(pc) if pc and pc > 0 else None

    # ------------------------------------------------------------------
    # 下單（風控在成交之前；拒單也入庫供稽核）
    # ------------------------------------------------------------------
    def place_order(self, *, stock_id: str, side: str, qty: int,
                    client_order_id: str, signal_date: str,
                    order_type: str = ORDER_TYPE_MARKET,
                    price_hint: float | None = None, reason: str = "",
                    now: dt.datetime | None = None) -> dict[str, Any]:
        """建立模擬訂單（拒絕也入庫 status=rejected 供稽核）。"""
        now = now or dt.datetime.now()
        if risk_mod.current_mode() is risk_mod.Mode.LIVE or \
                risk_mod.live_enabled():
            raise LiveModeForbidden(
                "模擬交易引擎不得在 live 模式（含人工確認）下執行")
        sig_day = _today(signal_date)
        with self._tx() as conn:
            aid = self._aid(conn)
            acc = conn.execute(
                "SELECT status FROM paper_account WHERE account_id=?",
                (aid,)).fetchone()
            if acc is None or acc["status"] != "active":
                raise PaperError("帳戶不存在或未啟用")
            # 冪等：同一 client_order_id 只對應一張訂單
            dup = conn.execute(
                "SELECT order_id, status FROM paper_order WHERE "
                "client_order_id=?", (client_order_id,)).fetchone()
            if dup:
                return {"accepted": dup["status"] in OPEN_STATUSES,
                        "order_id": str(dup["order_id"]),
                        "status": dup["status"], "duplicate": True}

            reasons: list[str] = []
            if side not in (OrderSide.BUY.value, OrderSide.SELL.value):
                reasons.append("side 必須是 BUY 或 SELL")
            if order_type not in ORDER_TYPES:
                reasons.append(
                    f"訂單類型 {order_type} 不支援（一期僅 market）")
            if qty <= 0:
                reasons.append("數量必須為正")
            elif qty % LOT != 0:
                reasons.append(
                    "台股整股：數量必須是 1000 股的倍數（零股暫不支援）")
            if not sig_day:
                reasons.append("缺少 signal_date（T 日，防 look-ahead）")
            elif sig_day > _today(now):
                reasons.append(
                    f"signal_date({sig_day}) 不得為未來（T+1 約束）")
            elif (sig_day == _today(now)
                  and (now.hour, now.minute) < _EOD_CUTOFF):
                reasons.append(
                    "signal_date 為今日：訊號需以今日收盤資料計算，"
                    f"請於 {_EOD_CUTOFF[0]:02d}:{_EOD_CUTOFF[1]:02d} "
                    "收盤後下單")
            if stock_id not in CONFIG.pool:
                reasons.append(f"{stock_id} 不在股票池")
            if self.killswitch.engaged():
                reasons.append(
                    f"Kill Switch：{self.killswitch.reason() or '已被啟動'}")

            cash, positions = self._state(conn, aid)
            equity = self._equity_with_market(conn, aid, cash, positions)
            stale = self.data_staleness_days(now)
            if stale > self.limits.max_data_staleness_days:
                reasons.append(
                    f"行情過期 {stale} 天（> "
                    f"{self.limits.max_data_staleness_days}），"
                    "禁止新交易")
            if cash <= 0 and side == OrderSide.BUY.value:
                reasons.append("可用現金 ≤ 0，不得買進")

            price_ref = (price_hint if price_hint is not None
                         else self._price_hint(conn, stock_id))
            acc_state = risk_mod.AccountState(
                equity=equity,
                position_value_by_stock={
                    sid: from_micro(st.avg_cost_share) * st.qty
                    for sid, st in positions.items() if st.qty > 0},
                open_orders=self._open_orders(conn, aid),
                orders_today=self._orders_today(conn, aid, _today(now)),
                day_pnl_pct=self._day_pnl(conn, now),
                drawdown_pct=self._drawdown(conn, equity),
                data_staleness_days=stale,
                liquidity_20d_value=0.0,
                realized_position_reconciles=True,
            )
            if price_ref is None:
                reasons += risk_mod.preflight(
                    acc_state, limits=self.limits).reasons
                reasons.append("無參考價（price_hint／最近收盤皆不可得）")
            else:
                rd = risk_mod.check_order(
                    acc_state, stock_id=stock_id, price=float(price_ref),
                    qty=int(qty), side=side, limits=self.limits)
                reasons += rd.reasons
                if side == OrderSide.SELL.value:
                    held = positions.get(stock_id, PositionState()).qty
                    if int(qty) > held:
                        reasons.append(
                            f"持倉不足（持有 {held} 股，賣 {int(qty)}）")
                if side == OrderSide.BUY.value:
                    need = _cash_required(price_ref, int(qty), self.cm)
                    if need > cash:
                        reasons.append(
                            f"資金不足（含費用約需 {f_twd(need)}，"
                            f"可用 {f_twd(cash)}）")

            # --- P5-B 行情品質閘門（最後一道；fail-closed）---
            # 涵蓋：行情過期/無資料、盤中無即時行情、非交易日、帳務不一致、
            # Kill Switch。任一不過即拒單（拒單也入庫可稽核）。
            dec = gate_mod.evaluate(now=now, engine=self,
                                    limits=self.limits,
                                    killswitch=self.killswitch)
            for r in dec.reasons:
                if r not in reasons:
                    reasons.append(r)

            if reasons:
                return self._reject_in_tx(conn, aid, stock_id, side, qty,
                                          reasons, client_order_id, reason)
            oid = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO paper_order (order_id, account_id, "
                "client_order_id, stock_id, side, qty, order_type, status,"
                " filled_qty, last_checked_date, signal_date, created_at,"
                " updated_at, reason, reject_reason"
                ") VALUES (?,?,?,?,?,?,?,?,0,'',?,?,?,?,'')",
                (oid, aid, client_order_id, stock_id, side, int(qty),
                 order_type, OrderStatus.PENDING_NEW.value, sig_day, _now(),
                 _now(), reason[:200]))
            self._transition(conn, oid, OrderStatus.PENDING_NEW.value,
                             OrderStatus.ACCEPTED.value,
                             reason[:80] or "order_placed")
            self._event(conn, aid, "order_created", oid,
                        f"{side} {stock_id} {qty} 股 訊號日 {sig_day} "
                        f"{reason}")
            return {"accepted": True, "order_id": oid,
                    "status": OrderStatus.ACCEPTED.value,
                    "duplicate": False}

    def _reject_in_tx(self, conn: sqlite3.Connection, aid: int,
                      stock_id: str, side: str, qty: int,
                      reasons: list[str], client_order_id: str,
                      submitted_reason: str) -> dict[str, Any]:
        """拒單也入庫（status=rejected；建立時即終態）。"""
        oid = str(uuid.uuid4())
        reasons_txt = "；".join(reasons)
        conn.execute(
            "INSERT INTO paper_order (order_id, account_id, client_order_id,"
            " stock_id, side, qty, order_type, status, filled_qty,"
            " last_checked_date, signal_date, created_at, updated_at,"
            " reason, reject_reason"
            ") VALUES (?,?,?,?,?,?,?,?,0,'',NULL,?,?,?,?)",
            (oid, aid, client_order_id, stock_id, side, int(qty),
             ORDER_TYPE_MARKET, OrderStatus.REJECTED.value, _now(), _now(),
             submitted_reason[:200], reasons_txt[:400]))
        self._event(conn, aid, "order_rejected", oid, reasons_txt[:400])
        return {"accepted": False, "order_id": oid,
                "status": OrderStatus.REJECTED.value, "reasons": reasons}

    def _event(self, conn: sqlite3.Connection, aid: int, kind: str,
               ref: str, data: str) -> None:
        conn.execute(
            "INSERT INTO paper_event (account_id, ts, kind, ref, data) "
            "VALUES (?,?,?,?,?)",
            (aid, _now(), kind, ref, data[:400]))

    def _transition(self, conn: sqlite3.Connection, oid: str, cur: str,
                    nxt: str, note: str) -> None:
        """生命週期：合法轉移＋樂觀鎖（where 狀態一致）；不得直接覆寫。"""
        assert_transition(cur, nxt)
        row = conn.execute(
            "SELECT status, account_id FROM paper_order WHERE order_id=?",
            (oid,)).fetchone()
        if row is None:
            raise PaperError(f"訂單不存在：{oid}")
        if row["status"] != cur:
            raise InvalidTransition(
                f"訂單 {oid} 狀態為 {row['status']}（要求 {cur} → {nxt}）")
        conn.execute(
            "UPDATE paper_order SET status=?, updated_at=? WHERE order_id=?"
            " AND status=?", (nxt, _now(), oid, cur))
        self._event(conn, int(row["account_id"]), "order_transition", oid,
                    f"{cur} → {nxt}：{note}")

    # ------------------------------------------------------------------
    # 成交模擬
    # ------------------------------------------------------------------
    def process_pending(self, *, now: dt.datetime | None = None
                        ) -> list[dict[str, Any]]:
        """對所有未完成訂單嘗試成交（每張訂單單一交易）。

        不可成交情境只推進 ``last_checked_date`` 並記事件；
        ``IntegrityBroken`` 直接停止後續交易（如實拋出）。
        """
        now = now or dt.datetime.now()
        open_orders = store.read_sql(
            "SELECT order_id, stock_id, side, qty, filled_qty, order_type,"
            " signal_date, last_checked_date, created_at, status FROM "
            "paper_order WHERE status IN ('pending_new','accepted',"
            "'partially_filled') ORDER BY created_at, order_id", None)
        results: list[dict[str, Any]] = []
        for _, row in open_orders.iterrows():
            results.append(self._attempt_fill(row.to_dict(), now))
        return results

    def _attempt_fill(self, order: dict[str, Any],
                      now: dt.datetime) -> dict[str, Any]:
        oid = str(order["order_id"])
        stock = str(order["stock_id"])
        side = str(order["side"])
        filled = int(order["filled_qty"])
        pending = int(order["qty"]) - filled
        sig_day = _today(order["signal_date"])
        last_checked = _today(order["last_checked_date"] or "")
        with self._tx() as conn:
            aid = self._aid(conn)
            cur = str(conn.execute(
                "SELECT status FROM paper_order WHERE order_id=?",
                (oid,)).fetchone()["status"])
            if cur not in OPEN_STATUSES:
                return {"order_id": oid, "filled": 0, "idempotent": True,
                        "status": cur, "reason": "already_final"}
            if self.killswitch.engaged():
                # 緊急停止：既定未成交訂單也不得再成交（可稽核）
                self._event(conn, aid, "no_fill", oid,
                            f"Kill Switch：{self.killswitch.reason()}")
                return {"order_id": oid, "status": cur, "filled": 0,
                        "reason": "kill_switch_engaged"}
            # P5-B：帳務不一致等「停止交易」條件也擋成交（只回報不修帳）。
            # 行情過期/非交易時段**不**擋既定訂單成交（P4 既有語意）。
            halted = [r for r in gate_mod.fill_block_reasons(
                engine=self, killswitch=self.killswitch)
                if not r.startswith("Kill Switch")]
            if halted:
                self._event(conn, aid, "no_fill", oid,
                            f"交易暫停：{halted[0]}")
                return {"order_id": oid, "status": cur, "filled": 0,
                        "reason": "trading_halted", "detail": halted[0]}
            after = max(sig_day, last_checked)
            bar = self._bar_after(stock, after)
            if bar is None or _today(bar["date"]) > _today(now) or (
                    _today(bar["date"]) == _today(now)
                    and (now.hour, now.minute) < _EOD_CUTOFF):
                # 當日 bar 只有在收盤（13:30）後才可視作已發生（避免用
                # 當日尚未發生的開盤價「提前」成交）
                return {"order_id": oid, "status": cur, "filled": 0,
                        "reason": "no_market_data_yet", "market_date": None}
            bar_date = str(bar["date"])
            created_day = _today(order["created_at"])
            expiry = dt.date.fromisoformat(created_day) + \
                dt.timedelta(days=self.good_til_days)
            if dt.date.fromisoformat(bar_date) > expiry:
                self._transition(conn, oid, cur, OrderStatus.EXPIRED.value,
                                 f"逾 {self.good_til_days} 日未成交")
                self._event(conn, aid, "order_expired", oid,
                            f"{bar_date} 逾未成交，取消殘量")
                return {"order_id": oid, "status": OrderStatus.EXPIRED.value,
                        "filled": 0, "reason": "expired",
                        "market_date": bar_date}
            vol = bar["volume"]
            open_raw = bar.get("open")
            if (vol is None or float(vol) <= 0 or open_raw is None
                    or pd.isna(open_raw)):
                return self._postpone(conn, aid, oid, cur, bar_date,
                                      "suspended" if (vol is not None
                                                      and float(vol) <= 0)
                                      else "missing_price")
            prev_close = self._prev_close(stock, bar_date)
            if prev_close is None:
                return self._postpone(conn, aid, oid, cur, bar_date,
                                      "unknown_limit_reference")
            up, dn = self.limit_up_down(prev_close)
            open_px = Decimal(str(open_raw))
            if up is not None and open_px >= up:
                return self._postpone(conn, aid, oid, cur, bar_date,
                                      "limit_up_no_fill")
            if dn is not None and open_px <= dn:
                return self._postpone(conn, aid, oid, cur, bar_date,
                                      "limit_down_no_fill")
            raw = round_tick(open_px)
            px = fill_price(raw, side=side, cost=self.cm)
            cap = self.cm.max_fill_qty(float(vol), float(raw), pending)
            fill_qty = _lots(min(int(cap), pending))
            if fill_qty <= 0:
                return self._postpone(conn, aid, oid, cur, bar_date,
                                      "no_liquidity")
            notional = to_micro(px) * fill_qty
            comm, tax = fee_for(notional, cost=self.cm, side=side)
            if side == OrderSide.BUY.value:
                cash_before = int(conn.execute(
                    "SELECT cash_utwd FROM paper_account WHERE account_id=?",
                    (aid,)).fetchone()["cash_utwd"])
                if notional + comm > cash_before:
                    while fill_qty >= LOT:
                        n = to_micro(px) * fill_qty
                        c, _ = fee_for(n, cost=self.cm, side=side)
                        if n + c <= cash_before:
                            notional, comm, tax = n, c, 0
                            break
                        fill_qty -= LOT
                    if fill_qty < LOT:
                        return self._postpone(conn, aid, oid, cur,
                                              bar_date, "insufficient_cash")
            partial = pending > fill_qty
            return self._apply_fill(conn, aid, order, bar_date, raw, px,
                                    fill_qty, comm, tax, partial)

    def _postpone(self, conn: sqlite3.Connection, aid: int, oid: str,
                  cur: str, market_date: str, reason: str) -> dict[str, Any]:
        conn.execute(
            "UPDATE paper_order SET last_checked_date=?, updated_at=? WHERE"
            " order_id=? AND status=?", (market_date, _now(), oid, cur))
        self._event(conn, aid, "no_fill", oid, f"{market_date}：{reason}")
        return {"order_id": oid, "status": cur, "filled": 0,
                "reason": reason, "market_date": market_date}

    def _apply_fill(self, conn: sqlite3.Connection, aid: int,
                    order: dict[str, Any], fill_date: str, raw: Decimal,
                    px: Decimal, fill_qty: int, comm: int, tax: int,
                    partial: bool) -> dict[str, Any]:
        """成交入庫（訂單→成交→帳本→持倉→事件，同一 DB 交易）。"""
        oid = str(order["order_id"])
        stock = str(order["stock_id"])
        side = str(order["side"])
        notional = to_micro(px) * int(fill_qty)
        fill_id = f"{oid}:{fill_date}"
        if conn.execute("SELECT 1 FROM paper_fill WHERE fill_id=?",
                        (fill_id,)).fetchone():
            return {"order_id": oid, "filled": 0, "idempotent": True,
                    "reason": "fill_already_exists", "market_date": fill_date}
        conn.execute(
            "INSERT INTO paper_fill (fill_id, account_id, order_id, stock_id,"
            " side, qty, fill_date, price_utwd, raw_price_utwd, fee_utwd,"
            " tax_utwd, order_type, price_source, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fill_id, aid, oid, stock, side, int(fill_qty), fill_date,
             to_micro(px), to_micro(raw), comm, tax, ORDER_TYPE_MARKET,
             f"price_table:{fill_date}｜slip {self.cm.slippage}"
             f"｜fee {self.cm.name}", _now()))
        # --- 帳本（單筆有號事件；ref 唯一 → 重播防護）---
        delta = (-(notional + comm) if side == OrderSide.BUY.value
                 else (notional - comm - tax))
        self._ledger(conn, aid, "fill", delta, f"{fill_id}:cash",
                     stock_id=stock, order_id=oid, fill_id=fill_id,
                     note=f"{side} {fill_qty} 股 @ {px}｜notional "
                          f"{f_twd(notional)} fee {f_twd(comm)} "
                          f"tax {f_twd(tax)}")
        # --- 持倉（成交序列為事實；與 replay_positions 同一公式）---
        rows = pd.read_sql_query(
            "SELECT rowid AS rowid, * FROM paper_fill WHERE account_id=?",
            conn, params=(aid,))
        positions = replay_positions(rows)
        for sid, st in positions.items():
            conn.execute(
                "INSERT INTO paper_position (account_id, stock_id, qty,"
                " cost_basis_utwd, avg_cost_share_utwd, realized_utwd,"
                " updated_at) VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(account_id, stock_id) DO UPDATE SET"
                " qty=excluded.qty,"
                " cost_basis_utwd=excluded.cost_basis_utwd,"
                " avg_cost_share_utwd=excluded.avg_cost_share_utwd,"
                " realized_utwd=excluded.realized_utwd,"
                " updated_at=excluded.updated_at",
                (aid, sid, st.qty, st.cost_basis, st.avg_cost_share,
                 st.realized, _now()))
        # --- 訂單狀態（生命週期；不得直接覆寫）---
        total_qty = int(order["qty"])
        new_filled = int(order["filled_qty"]) + int(fill_qty)
        cur = str(conn.execute(
            "SELECT status FROM paper_order WHERE order_id=?",
            (oid,)).fetchone()["status"])
        if new_filled >= total_qty:
            nxt = OrderStatus.FILLED.value
        else:
            nxt = OrderStatus.PARTIALLY_FILLED.value
        if nxt != cur:
            self._transition(conn, oid, cur, nxt,
                             f"成交 {fill_qty} 股 @ {px}（{fill_date}）")
        conn.execute(
            "UPDATE paper_order SET filled_qty=?, last_checked_date=?,"
            " updated_at=? WHERE order_id=?",
            (new_filled, fill_date, _now(), oid))
        self._event(conn, aid, "fill", oid,
                    f"{side} {fill_qty} 股 @ {px}（{fill_date}）"
                    f" raw {raw}；fee {f_twd(comm)} tax {f_twd(tax)}"
                    f"{'（部分成交）' if partial else ''}")
        return {"order_id": oid, "status": nxt, "filled": int(fill_qty),
                "filled_qty": new_filled, "price": float(px),
                "raw_price": float(raw), "fee": from_micro(comm),
                "tax": from_micro(tax), "partial": partial,
                "market_date": fill_date}

    # ------------------------------------------------------------------
    # 取消
    # ------------------------------------------------------------------
    def cancel_order(self, order_id: str,
                     reason: str = "manual_cancel") -> dict[str, Any]:
        with self._tx() as conn:
            aid = self._aid(conn)
            row = conn.execute(
                "SELECT status, account_id FROM paper_order WHERE order_id=?",
                (order_id,)).fetchone()
            if row is None:
                raise PaperError(f"訂單不存在：{order_id}")
            if int(row["account_id"]) != aid:
                raise PaperError("訂單屬其它帳戶")
            cur = row["status"]
            if cur in FINAL_STATUSES:
                raise InvalidTransition(
                    f"訂單 {order_id} 已終態（{cur}），不可取消")
            self._transition(conn, order_id, cur,
                             OrderStatus.CANCELLED.value, reason)
            self._event(conn, aid, "order_cancelled", order_id, reason[:200])
        return {"order_id": order_id,
                "status": OrderStatus.CANCELLED.value, "reason": reason}

    # ------------------------------------------------------------------
    # 快照 / 帳務核對
    # ------------------------------------------------------------------
    def snapshot(self, *, write: bool = True, now: dt.datetime | None = None,
                 note: str = "") -> dict[str, Any]:
        """帳戶快照：equity = cash + Σ持倉市值；已實現／未實現分離。"""
        acc = self.account()
        if acc is None:
            raise PaperError(f"帳戶 {self.account_name} 不存在")
        now = now or dt.datetime.now()
        aid = int(acc["account_id"])
        cash, positions = self.cash_and_positions(aid)
        market_value = 0
        cost_total = 0
        realized = sum(st.realized for st in positions.values())
        positions_rows: list[dict[str, Any]] = []
        market_date = None
        ctx = self._tx() if write else self._ro()
        with ctx as conn:
            for sid, st in positions.items():
                if st.qty <= 0:
                    continue
                d, pc = self._last_close(conn, sid)
                px = pc if (pc and pc > 0) else \
                Decimal(str(st.avg_cost_share))
                mv = to_micro(px) * st.qty
                market_value += mv
                cost_total += st.cost_basis
                positions_rows.append({
                    "stock_id": sid, "qty": st.qty,
                    "avg_cost_share": from_micro(st.avg_cost_share),
                    "market_price": (float(pc) if pc and pc > 0 else None),
                    "market_date": d, "market_value": from_micro(mv),
                    "unrealized": from_micro(mv - st.cost_basis)})
            market_date = self._last_trade_date(conn)
            equity = cash + market_value
            unrealized = market_value - cost_total
            peak_row = conn.execute(
                "SELECT MAX(equity_utwd) AS p FROM paper_snapshot WHERE"
                " account_id=?", (aid,)).fetchone()
            peak = max(int(peak_row["p"] or 0), equity)
            drawdown = 0.0 if peak <= 0 else (peak - equity) / peak
            if write:
                self._snapshot_write(conn, aid, as_of=now.isoformat(
                    timespec="seconds"),
                                     cash=cash, market_value=market_value,
                                     equity=equity, realized=realized,
                                     unrealized=unrealized, peak=peak,
                                     drawdown=drawdown,
                                     market_date=market_date, note=note)
        return {"account_id": aid, "cash": from_micro(cash),
                "market_value": from_micro(market_value),
                "equity": from_micro(equity), "realized": from_micro(realized),
                "unrealized": from_micro(unrealized),
                "drawdown_pct": drawdown,
                "market_date": market_date, "positions": positions_rows}

    def _snapshot_write(self, conn, aid: int, *, as_of: str, cash: int,
                        market_value: int, equity: int, realized: int,
                        unrealized: int, peak: int, drawdown: float,
                        market_date: str | None, note: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO paper_snapshot (account_id, as_of,"
            " cash_utwd, market_value_utwd, equity_utwd, realized_utwd,"
            " unrealized_utwd, max_equity_utwd, drawdown_pct, market_date,"
            " note) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (aid, as_of, cash, market_value, equity, realized, unrealized,
             peak, drawdown, market_date, note[:200]))

    def reconcile(self) -> dict[str, Any]:
        """帳務一致性檢查（不變條件；發現失誤即回報，不修復也不繼續交易）。"""
        acc = self.account()
        if acc is None:
            raise PaperError("尚未建立 paper 帳戶")
        aid = int(acc["account_id"])
        problems: list[str] = []
        cash, positions = self.cash_and_positions(aid)
        if cash != int(acc["cash_utwd"]):
            problems.append(
                f"帳本重建現金 {cash} ≠ 帳戶 cash_utwd {int(acc['cash_utwd'])}")
        if cash < 0:
            problems.append(f"現金為負（{f_twd(cash)}）")
        for sid, st in positions.items():
            if st.qty < 0:
                problems.append(f"持倉為負 {sid}：{st.qty}")
        ledger = store.read_sql(
            "SELECT ledger_id, kind, amount_utwd, balance_utwd, ref,"
            " fill_id, order_id, stock_id, note FROM paper_ledger WHERE"
            " account_id=? ORDER BY ledger_id", (aid,))
        if not ledger.empty:
            prev_balance = 0
            for _, r in ledger.iterrows():
                if int(r["balance_utwd"]) != (prev_balance
                                              + int(r["amount_utwd"])):
                    problems.append(
                        f"帳本 #{int(r['ledger_id'])} 餘額鏈不一致")
                prev_balance = int(r["balance_utwd"])
            if ledger.iloc[0]["kind"] != "capital":
                problems.append("帳本首筆必須是 capital（初始資金）")
            unexpected = set(ledger["kind"]) - {"capital", "fill"}
            if unexpected:
                problems.append(f"帳本含非預期事件：{sorted(unexpected)}")
            if int(ledger.iloc[-1]["balance_utwd"]) != cash:
                problems.append("帳本末筆餘額 ≠ 帳本重建現金")
        fills = store.read_sql(
            "SELECT fill_id, order_id, qty, price_utwd, side, fee_utwd,"
            " tax_utwd FROM paper_fill WHERE account_id=? ORDER BY rowid",
            (aid,))
        ledger_fills = set(ledger[ledger["kind"] == "fill"]["fill_id"]) \
            if not ledger.empty else set()
        if len(ledger_fills) != len(fills) or \
                not set(fills["fill_id"]).issubset(ledger_fills):
            problems.append("成交數與帳本成交事件數不一致")
        for _, f in fills.iterrows():
            r = ledger[(ledger["fill_id"] == f["fill_id"])
                       & (ledger["kind"] == "fill")]
            if r.empty:
                continue
            amt = int(r.iloc[0]["amount_utwd"])
            n = (int(f["price_utwd"]) * int(f["qty"]) + int(f["fee_utwd"])
                 + int(f["tax_utwd"]))
            sup = (-n if f["side"] == OrderSide.BUY.value
                   else (int(f["price_utwd"]) * int(f["qty"])
                         - int(f["fee_utwd"]) - int(f["tax_utwd"])))
            # buy：−（名目+fee）；sell：+（名目−fee−tax）
            if amt != sup:
                problems.append(
                    f"成交 {f['fill_id']} 帳本金額 {amt} ≠ {sup}")
        pos_db = store.read_sql(
            "SELECT stock_id, qty, cost_basis_utwd, avg_cost_share_utwd,"
            " realized_utwd FROM paper_position WHERE account_id=?", (aid,))
        for _, p in pos_db.iterrows():
            st = positions.get(str(p["stock_id"]))
            if (st is None
                    or st.qty != int(p["qty"])
                    or st.cost_basis != int(p["cost_basis_utwd"])
                    or st.realized != int(p["realized_utwd"])):
                problems.append(f"持倉快取與成交重放不一致：{p['stock_id']}")
        if not problems:
            pos_ids = {str(x) for x in pos_db["stock_id"]} \
                if not pos_db.empty else set()
            for sid, st in positions.items():
                if st.qty > 0 and sid not in pos_ids:
                    problems.append(f"持倉快取缺列：{sid}")
        orders = store.read_sql(
            "SELECT order_id, status, filled_qty FROM paper_order WHERE"
            " account_id=?", (aid,))
        for _, o in orders.iterrows():
            f = fills[fills["order_id"] == o["order_id"]]
            total = int(f["qty"].sum()) if not f.empty else 0
            if int(o["filled_qty"]) != total:
                problems.append(
                    f"訂單 {o['order_id']} filled_qty 與成交紀錄不一致")
            if o["status"] == "cancelled" and total > 0:
                problems.append(
                    f"訂單 {o['order_id']} 標示 cancelled 但已有成交")
        snaps = store.read_sql(
            "SELECT cash_utwd, market_value_utwd, equity_utwd FROM "
            "paper_snapshot WHERE account_id=?", (aid,))
        for _, s in snaps.iterrows():
            if int(s["equity_utwd"]) != (int(s["cash_utwd"])
                                         + int(s["market_value_utwd"])):
                problems.append("快照 equity 不等於 cash + market_value")
                break
        return {"account_id": aid, "ok": not problems,
                "problems": problems, "cash": from_micro(cash),
                "positions": {sid: st.qty for sid, st in positions.items()
                              if st.qty > 0},
                "fills": int(len(fills)), "ledger": int(len(ledger)),
                "snapshots": int(len(snaps))}

    # ------------------------------------------------------------------
    # 風險狀態 / 恢復
    # ------------------------------------------------------------------
    def risk_status(self, *, now: dt.datetime | None = None
                    ) -> dict[str, Any]:
        """風控狀態（唯讀；唯讀模式亦可執行）。"""
        now = now or dt.datetime.now()
        conn = self._conn_read()
        try:
            aid = self._aid(conn)
            cash, positions = self._state(conn, aid)
            equity = self._equity_with_market(conn, aid, cash, positions)
            acc_state = risk_mod.AccountState(
                equity=equity,
                position_value_by_stock={
                    sid: from_micro(st.avg_cost_share) * st.qty
                    for sid, st in positions.items() if st.qty > 0},
                open_orders=self._open_orders(conn, aid),
                orders_today=self._orders_today(conn, aid, _today(now)),
                day_pnl_pct=self._day_pnl(conn, now),
                drawdown_pct=self._drawdown(conn, equity),
                data_staleness_days=self.data_staleness_days(now),
                liquidity_20d_value=0.0,
                realized_position_reconciles=True,
            )
            last = self._last_trade_date(conn)
        finally:
            conn.close()

        pf = risk_mod.preflight(acc_state, limits=self.limits)
        return {
            "equity": equity, "cash": from_micro(cash),
            "open_orders": acc_state.open_orders,
            "orders_today": acc_state.orders_today,
            "day_pnl_pct": acc_state.day_pnl_pct,
            "drawdown_pct": acc_state.drawdown_pct,
            "data_staleness_days": acc_state.data_staleness_days,
            "last_trade_date": last,
            "kill_switch_engaged": self.killswitch.engaged(),
            "kill_switch_reason": self.killswitch.reason(),
            "max_order_value": self.limits.max_order_value,
            "max_position_pct": self.limits.max_position_pct,
            "max_drawdown_pct": self.limits.max_drawdown_pct,
            "max_daily_loss_pct": self.limits.max_daily_loss_pct,
            "preflight_ok": pf.ok, "preflight_reasons": pf.reasons,
        }

    def recover(self) -> dict[str, Any]:
        """重啟恢復：全部由 DB 重建；回傳帳務與未成交訂單摘要。"""
        rec = self.reconcile()
        open_orders = store.read_sql(
            "SELECT COUNT(*) AS n FROM paper_order WHERE status IN "
            "('accepted','partially_filled')", None)
        return {"reconcile_ok": rec["ok"], "problems": rec["problems"],
                "open_orders": int(open_orders.iloc[0]["n"]),
                "cash": rec["cash"], "fills": rec["fills"]}

    # ------------------------------------------------------------------
    # 週期流程（EOD 訊號 → 決策 → 訂單 → 成交 → 快照）
    # ------------------------------------------------------------------
    def run_paper_cycle(self, *, now: dt.datetime | None = None,
                        run_id: str = "") -> dict[str, Any]:
        """執行一次模擬交易週期。

        0. **P5-B 行情品質閘門先行**（``gate.evaluate``）：行情過期/無資料、
           盤中無可驗證即時行情、非交易日、帳務不一致、Kill Switch 任一成立
           → **不建立任何新訂單**（決策仍保留供稽核），並把判定寫進
           ``results["gate"]``；既有未完成訂單是否續成交依 ``allow_fills``。
        1. 讀最新 signals（T 日）→ 規則（沿用 strategies 慣例）：
           ``score >= buy_threshold`` 買入（top_n、未持有、無未完成買單）；
           持倉滿 ``hold_days`` 賣出全部（不對稱門檻的策略邏輯屬 P5；
           本期為 fixed-hold + threshold，與回測 default 相同）。
        2. 成交模擬（``process_pending``）；未完成訂單過期者取消。
        3. 快照更新（``run_id`` 寫入快照備註，供追查）。

        ``run_id`` 只進 ``reason``/備註；**不得**進 ``client_order_id``
        （那會破壞跨輪冪等，重跑就會重複下單）。
        """
        now = now or dt.datetime.now()
        aid_acc = self.account()
        if aid_acc is None:
            raise PaperError(f"帳戶 {self.account_name} 不存在")
        aid = int(aid_acc["account_id"])
        sig = store.read_sql(
            "SELECT date, stock_id, score, close FROM signals WHERE date ="
            " (SELECT MAX(date) FROM signals)", None)
        decisions: list[dict[str, Any]] = []
        results: dict[str, Any] = {"orders": [], "fills": [],
                                   "decisions": decisions,
                                   "skipped": []}
        # --- 閘門（新單）：不通過則整輪不下新單，但仍成交既有訂單與快照 ---
        gdec = gate_mod.evaluate(now=now, engine=self, limits=self.limits,
                                 killswitch=self.killswitch)
        results["gate"] = gdec.as_dict()
        if not gdec.allow_new_orders:
            results["blocked"] = list(gdec.reasons)
            log.warning("paper cycle 新單被閘門阻止（%s）：%s", gdec.kind,
                        "；".join(gdec.reasons))
        cash, positions = self.cash_and_positions(aid)
        with self._ro() as conn:
            equity = self._equity_with_market(conn, aid, cash, positions)
            open_by_stock = {
                str(r["stock_id"]): str(r["side"]) for _, r in
                pd.read_sql_query(
                    "SELECT stock_id, side FROM paper_order WHERE "
                    "account_id=? AND status IN ('pending_new','accepted',"
                    "'partially_filled')", conn,
                    params=(aid,)).iterrows()}

        # --- 賣出（持倉滿 hold_days）---
        for sid, st in positions.items():
            if st.qty <= 0:
                continue
            first = self._first_open_fill_date(sid)
            if not first:
                continue
            after = store.read_sql(
                "SELECT COUNT(*) AS n FROM price WHERE stock_id=? AND"
                " date>?", (sid, first))
            days_after = int(after.iloc[0]["n"])
            if days_after < self.hold_days:
                continue
            with self._ro() as conn:
                d, pc = self._last_close(conn, sid)
            ref_date = d or _today(now)
            if pc is None or pc <= 0:
                decisions.append({"action": "SKIP", "stock_id": sid,
                                  "reason": "no_ref_price_for_sell"})
                continue
            decisions.append({
                "action": "SELL", "stock_id": sid, "qty": st.qty,
                "price_hint": float(pc), "ref_date": ref_date})
        # --- 買入（最新訊號 top_n，未持有且無未完成買單）---
        if not sig.empty:
            scored = sig[sig["score"] >= self.buy_threshold].copy()
            scored = scored.sort_values("score", ascending=False)
            ref_date = str(sig["date"].iloc[0])
            budget = min(equity * self.limits.max_position_pct,
                         self.limits.max_order_value)
            chosen = 0
            for _, s in scored.iterrows():
                sid = str(s["stock_id"])
                if chosen >= self.top_n:
                    break
                if sid in positions and positions[sid].qty > 0:
                    continue
                if open_by_stock.get(sid) == OrderSide.BUY.value:
                    continue
                px = (float(s["close"])
                      if pd.notna(s["close"]) else None)
                if px is None or px <= 0:
                    decisions.append({"action": "SKIP", "stock_id": sid,
                                      "reason": "no_ref_price"})
                    continue
                qty = _lots(budget / px)
                if qty <= 0:
                    decisions.append({"action": "SKIP", "stock_id": sid,
                                      "reason": "budget_too_small"})
                    continue
                decisions.append({"action": "BUY", "stock_id": sid,
                                  "qty": int(qty), "price_hint": px,
                                  "ref_date": ref_date})
                chosen += 1
        for dec in decisions:
            if dec["action"] == "SKIP":
                results["skipped"].append(dec)
                continue
            if not gdec.allow_new_orders:
                # 閘門關閉：決策保留供稽核，但一張新單都不建立
                results["skipped"].append({**dec,
                                           "reason": "gate_blocked",
                                           "gate_kind": gdec.kind})
                continue
            res = self.place_order(
                stock_id=dec["stock_id"], side=dec["action"],
                qty=int(dec["qty"]), client_order_id=(
                    f"cycle:{dec['ref_date']}:{dec['action']}:"
                    f"{dec['stock_id']}"),
                signal_date=dec["ref_date"],
                price_hint=dec.get("price_hint"),
                reason=(f"paper_cycle:{run_id}" if run_id
                        else "paper_cycle"), now=now)
            results["orders"].append(res)
        fill_res = self.process_pending(now=now)
        results["fills"] = fill_res
        results["snapshot"] = self.snapshot(
            now=now, note=(f"paper_cycle:{run_id}" if run_id
                           else "paper_cycle"))
        return results

    def _aid_public(self) -> int:
        acc = self.account()
        if acc is None:
            raise PaperError("尚未建立 paper 帳戶")
        return int(acc["account_id"])

    def _first_open_fill_date(self, stock_id: str) -> str | None:
        """目前持有世代的第一次買入日期（持倉滿 hold_days 用）。"""
        df = store.read_sql(
            "SELECT rowid AS rowid, side, qty, fill_date FROM paper_fill "
            "WHERE stock_id=? ORDER BY rowid", (stock_id,))
        qty = 0
        first: str | None = None
        for _, r in df.iterrows():
            qty += int(r["qty"]) if r["side"] == OrderSide.BUY.value \
                else -int(r["qty"])
            if first is None and qty > 0:
                first = str(r["fill_date"])
            elif qty <= 0:
                first = None
        return first

    # ------------------------------------------------------------------
    # 查詢 view（CLI / dashboard）
    # ------------------------------------------------------------------
    def orders_view(self, *, status: str | None = None,
                    limit: int = 50) -> pd.DataFrame:
        sql = ("SELECT order_id, client_order_id, stock_id, side, qty,"
               " order_type, status, filled_qty, signal_date, created_at,"
               " reason, reject_reason FROM paper_order WHERE account_id=?")
        params: list[Any] = [self._aid_public()]
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC, order_id DESC LIMIT ?"
        params.append(int(limit))
        return store.read_sql(sql, tuple(params))

    def fills_view(self, *, limit: int = 50) -> pd.DataFrame:
        return store.read_sql(
            "SELECT fill_id, order_id, stock_id, side, qty, fill_date,"
            " price_utwd, raw_price_utwd, fee_utwd, tax_utwd, price_source,"
            " created_at FROM paper_fill WHERE account_id=?"
            " ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (self._aid_public(), int(limit)))

    def ledger_view(self, *, limit: int = 50) -> pd.DataFrame:
        return store.read_sql(
            "SELECT ledger_id, ts, kind, amount_utwd, balance_utwd,"
            " order_id, fill_id, stock_id, note FROM paper_ledger WHERE"
            " account_id=? ORDER BY ledger_id DESC LIMIT ?",
            (self._aid_public(), int(limit)))

    def positions_view(self) -> pd.DataFrame:
        aid = self._aid_public()
        cash, positions = self.cash_and_positions(aid)
        rows: list[dict[str, Any]] = []
        with self._ro() as conn:
            for sid, st in positions.items():
                if st.qty <= 0:
                    continue
                d, pc = self._last_close(conn, sid)
                px = pc if (pc and pc > 0) else \
                    Decimal(str(st.avg_cost_share))
                rows.append({
                    "stock_id": sid, "qty": st.qty,
                    "avg_cost_share": from_micro(st.avg_cost_share),
                    "market_price": float(px), "market_date": d,
                    "market_value": from_micro(to_micro(px) * st.qty),
                    "unrealized": from_micro(to_micro(px) * st.qty
                                             - st.cost_basis)})
        return pd.DataFrame(rows)

    def _conn_read(self) -> sqlite3.Connection:
        conn = store._connect()
        conn.row_factory = sqlite3.Row
        if not is_readonly():
            store._init(conn)
        return conn

    # ------------------------------------------------------------------
    # CLI（tz 文件式 account_text 供 dashboard／CLI 顯示）
    # ------------------------------------------------------------------
    def account_text(self) -> str:
        snap = self.snapshot(write=False)
        rec = self.reconcile()
        L = [f"=== Paper 模擬帳戶（{self.account_name}）"
             " —— 所有資金與成交皆為**模擬**，非真實交易 ===",
             f"  總資產     TWD {snap['equity']:,.2f}",
             f"  現金       TWD {snap['cash']:,.2f}",
             f"  持倉市值   TWD {snap['market_value']:,.2f}",
             f"  已實現損益 TWD {snap['realized']:+,.2f}",
             f"  未實現損益 TWD {snap['unrealized']:+,.2f}",
             f"  回撤       {snap['drawdown_pct']:.2%}",
             f"  行情基準日 {snap['market_date'] or '—'}",
             f"  帳務一致   {'✓ 是' if rec['ok'] else '✗ 否'}"]
        if not rec["ok"]:
            for p in rec["problems"]:
                L.append(f"    ⚠️ {p}")
        for p in snap["positions"]:
            L.append(
                f"  - {p['stock_id']:5s} × {p['qty']:>5d} 股 "
                f"@ 成本 {p['avg_cost_share']:>9.2f} "
                f"現價 {(p['market_price'] or 0):>9.2f}　"
                f"未實現 {p['unrealized']:+,.2f}")
        return "\n".join(L)

    def risk_text(self) -> str:
        r = self.risk_status()
        ks = "是（" + r["kill_switch_reason"] + "）" \
            if r["kill_switch_engaged"] else "否"
        L = ["=== Paper 風控狀態（模擬交易；限於成交流程之前） ===",
             f"  Kill Switch      {ks}",
             f"  總資產（模擬）    TWD {r['equity']:,.2f}",
             f"  可用現金         TWD {r['cash']:,.2f}",
             f"  未完成訂單數     {r['open_orders']}（上限 "
             f"{risk_mod.DEFAULT_LIMITS.max_open_orders}）",
             f"  今日訂單數       {r['orders_today']}（上限 "
             f"{risk_mod.DEFAULT_LIMITS.max_orders_per_day}）",
             f"  當日損益         {r['day_pnl_pct'] * 100:+.2f}%"
             f"（單日損失上限 {r['max_daily_loss_pct'] * 100:.1f}%）",
             f"  回撤             {r['drawdown_pct'] * 100:+.2f}%"
             f"（上限 {r['max_drawdown_pct'] * 100:.1f}%）",
             f"  行情新鮮度       {r['data_staleness_days']} 日"
             f"（上限 {risk_mod.DEFAULT_LIMITS.max_data_staleness_days} 日，"
             f"最近交易日 {r['last_trade_date'] or '—'}）",
             f"  新交易可下單     {'✓ 是' if r['preflight_ok'] else '✗ 否'}"]
        for reason in r["preflight_reasons"]:
            L.append(f"    ⛔ {reason}")
        return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI（python -m twstock.cli paper <cmd>）
# ---------------------------------------------------------------------------
SUBCMDS = ("init", "account", "orders", "fills", "positions", "ledger",
           "cycle", "risk", "reconcile", "recover", "kill", "resume")


def main(argv: list[str] | None = None) -> int:
    ap_desc = ("模擬交易引擎（paper；**不下真單**）。"
               "子命令：" + ", ".join(SUBCMDS))
    ap = argparse.ArgumentParser(description=ap_desc)
    ap.add_argument("cmd", nargs="?", default="account")
    ap.add_argument("--cash", type=float, default=None,
                    help="初始資金（TWD；init 用，預設 1,000,000）")
    ap.add_argument("--limit", type=int, default=20,
                    help="orders/fills/ledger 顯示筆數")
    ap.add_argument("--status", default=None,
                    help="orders 檢視時過濾狀態")
    ap.add_argument("--reason", default=None,
                    help="kill（緊急停止）原因")
    args, _rest = ap.parse_known_args(argv)

    if args.cmd == "kill":
        reason = args.reason or "emergency_stop"
        ks = risk_mod.KillSwitch()
        ks.engage(f"paper: {reason}")
        print(f"已啟用緊急停止（Kill Switch）：{reason}")
        return 0
    if args.cmd == "resume":
        risk_mod.KillSwitch().release()
        print("已解除緊急停止（Kill Switch）。")
        return 0

    eng = PaperEngine()
    if args.cmd == "init":
        cash = args.cash if args.cash is not None else 1_000_000.0
        aid = eng.ensure_account(initial_cash=cash,
                                 notes=f"init@{_now()}")
        print(f"已建立（或取得）paper 帳戶：account_id={aid}，"
              f"初始資金 TWD {cash:,.2f}")
        return 0

    # 其餘子命令都要求帳戶存在
    if eng.account() is None:
        print("尚未建立 paper 帳戶——請先執行："
              "python -m twstock.cli paper init [--cash 1000000]")
        return 2
    if args.cmd == "account":
        print(eng.account_text())
        return 0
    if args.cmd == "orders":
        df = eng.orders_view(status=args.status, limit=args.limit)
        cols = ["order_id", "stock_id", "side", "qty", "status", "filled_qty"]
        show = df[[c for c in cols if c in df.columns]]
        print(show.to_string(index=False) if not df.empty else "（無訂單）")
        return 0
    if args.cmd == "fills":
        df = eng.fills_view(limit=args.limit)
        if df.empty:
            print("（無成交）")
            return 0
        df = df.copy()
        df["price"] = df["price_utwd"].map(from_micro)
        df["raw_price"] = df["raw_price_utwd"].map(from_micro)
        df["fee"] = df["fee_utwd"].map(from_micro)
        df["tax"] = df["tax_utwd"].map(from_micro)
        out = ["fill_id", "order_id", "stock_id", "side", "qty", "fill_date",
               "price", "raw_price", "fee", "tax", "price_source"]
        print(df[[c for c in out if c in df.columns]].to_string(index=False))
        return 0
    if args.cmd == "positions":
        df = eng.positions_view()
        out = ["stock_id", "qty", "avg_cost_share", "market_price",
               "market_value", "unrealized"]
        show = df[[c for c in out if c in df.columns]]
        print(show.to_string(index=False) if df.empty is False else "（無持倉）")
        return 0
    if args.cmd == "ledger":
        df = eng.ledger_view(limit=args.limit)
        if df.empty:
            print("（帳本為空）")
            return 0
        df = df.copy()
        df["amount"] = df["amount_utwd"].map(
            lambda x: f_twd(int(x), signed=True))
        df["balance"] = df["balance_utwd"].map(f_twd)
        out = ["ledger_id", "ts", "kind", "amount", "balance", "order_id",
               "fill_id", "note"]
        print(df[[c for c in out if c in df.columns]].to_string(index=False))
        return 0
    if args.cmd == "cycle":
        res = eng.run_paper_cycle()
        snap = res["snapshot"]
        ginfo = res.get("gate") or {}
        print(f"=== paper cycle 完成（{len(res['orders'])} 張訂單指令） ===")
        if ginfo and not ginfo.get("allow_new_orders", True):
            print("  閘門：禁止新增模擬交易（fail-closed）")
            for gr in ginfo.get("reasons", []):
                print(f"    ⛔ {gr}")
        for o in res["orders"]:
            tag = "✔ 已建立" if o.get("accepted") else (
                "↺ 冪等既存在" if o.get("duplicate") else "✘ 拒單")
            reasons = ("：" + "；".join(o.get("reasons", []))
                       if not o.get("accepted") else "")
            print(f"  {tag} order_id={o.get('order_id')}{reasons}")
        for f in res["fills"]:
            if f.get("filled"):
                tag = "部分成交" if f.get("partial") else "成交"
                print(f"  {tag}：{f.get('market_date')} {f.get('filled')} 股"
                      f" @ {f.get('price')}（fee {f.get('fee'):,.2f}）")
            else:
                print(f"  未成交：{f.get('reason')}（{f.get('market_date')}）")
        print(f"  快照：equity TWD {snap.get('equity'):,.2f}；"
              f"现金 TWD {snap.get('cash'):,.2f}；"
              f"回撤 {snap.get('drawdown_pct'):.2%}")
        return 0
    if args.cmd == "risk":
        print(eng.risk_text())
        return 0
    if args.cmd == "reconcile":
        rec = eng.reconcile()
        if rec["ok"]:
            print(f"✓ 帳務一致（fills={rec['fills']}, "
                  f"ledger={rec['ledger']}, snapshots={rec['snapshots']}）")
            return 0
        print("✗ 帳務不一致：")
        for p in rec["problems"]:
            print(f"  ⚠ {p}")
        return 1
    if args.cmd == "recover":
        rec = eng.recover()
        print(f"重啟恢復：cash={rec['cash']:,.2f}，"
              f"未完成訂單 {rec['open_orders']} 張；"
              f"帳務一致：{'✓' if rec['reconcile_ok'] else '✗'}")
        if rec["problems"]:
            for p in rec["problems"]:
                print(f"  ⚠ {p}")
            return 1
        return 0
    print(f"未知子命令：{args.cmd}（可用：{', '.join(SUBCMDS)}）")
    return 2


import argparse  # noqa: E402  # 位於 main 之後的共用 import（Python 允許）


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
