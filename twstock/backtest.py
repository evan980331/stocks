"""回測：每期打分排序選 top-N，等權買入並持有 hold_days 個交易日，含台股交易成本。

時間正確性（關鍵）：
  訊號以 **T 日收盤**資料計算 → 必須在 **T+1 開盤**成交，否則就是 look-ahead bias。
  出場：持有 hold_days 個交易日後，於到期日**開盤**賣出。
  漲停無法買進、跌停無法賣出（以開盤價相對前一日收盤判斷），據以跳過或順延。

成本：手續費（2.8 折，低消 20 元）＋ 證交稅（賣出 0.3%）＋ 雙邊滑價 0.1%。

不支援：放空、融資、借券。整股/零股皆以 1 股為單位（零股模式，利於小資金回測）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CONFIG

log = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    turnover: float = 0.0
    cost_total: float = 0.0
    exposure: float = 0.0
    periods_per_year: int = TRADING_DAYS
    # 基準（買進持有）
    benchmark_total_return: float = 0.0
    benchmark_cagr: float = 0.0
    benchmark_max_drawdown: float = 0.0
    benchmark_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def excess_return(self) -> float:
        return self.total_return - self.benchmark_total_return

    def summary(self) -> str:
        lines = [
            f"策略：年化 {self.cagr:+.2%}  總報酬 {self.total_return:+.2%}  "
            f"Sharpe {self.sharpe:.2f}  Sortino {self.sortino:.2f}  "
            f"MDD {self.max_drawdown:.2%}  勝率 {self.win_rate:.2%}  "
            f"交易 {len(self.trades)} 筆",
            f"風險/成本：獲利因子 {self.profit_factor:.2f}  "
            f"平均獲利 {self.avg_win:+.2%}  平均虧損 {self.avg_loss:+.2%}  "
            f"週轉率 {self.turnover:.2f}x  總成本 {self.cost_total:,.0f}  曝險 {self.exposure:.1%}",
        ]
        if len(self.benchmark_equity) > 1:
            lines.append(
                f"基準 {CONFIG.benchmark} 買進持有：總報酬 {self.benchmark_total_return:+.2%}  "
                f"年化 {self.benchmark_cagr:+.2%}  MDD {self.benchmark_max_drawdown:.2%}  "
                f"超額 {self.excess_return():+.2%}")
        lines.append(
            "⚠️ 回測結果不代表未來報酬；成本/成交假設見 twstock/backtest.py 與 plan.md。")
        return "\n".join(lines)


def _prepare(panel: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """把每檔日K整理成可供 T+1 查找的結構。"""
    out: dict[str, dict] = {}
    for sid, df in panel.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        d = d.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        for col in ("open", "close"):
            if col not in d.columns:
                d[col] = np.nan
        d = d.dropna(subset=["open", "close"])
        if d.empty:
            continue
        out[sid] = {
            "dates": d["date"].tolist(),
            "idx": {t: i for i, t in enumerate(d["date"])},
            "open": d["open"].to_numpy(dtype=float),
            "close": d["close"].to_numpy(dtype=float),
        }
    return out


def _schedule_entries(scores: pd.DataFrame, book: dict[str, dict],
                      hold_days: int) -> dict[pd.Timestamp, list[dict]]:
    """把每個訊號換算成 (進場日, 進場價 index, 出場日)。以 T+1 開盤進場。"""
    schedule: dict[pd.Timestamp, list[dict]] = {}
    for row in scores.itertuples(index=False):
        sid = getattr(row, "stock_id")
        bk = book.get(sid)
        if bk is None:
            continue
        sig_date = pd.Timestamp(getattr(row, "date")).normalize()
        i = bk["idx"].get(sig_date)
        if i is None:
            continue
        entry_i = i + 1                      # T+1
        exit_i = entry_i + hold_days
        if exit_i >= len(bk["dates"]):       # 無法完整評估（資料不足）
            continue
        entry_date = bk["dates"][entry_i]
        schedule.setdefault(entry_date, []).append({
            "stock_id": sid,
            "signal_date": sig_date,
            "score": float(getattr(row, "score", 0.0) or 0.0),
            "entry_i": entry_i,
            "exit_i": exit_i,
            "exit_date": bk["dates"][exit_i],
        })
    return schedule


def run_backtest(scores: pd.DataFrame,
                 price_panel: dict[str, pd.DataFrame],
                 top_n: int = 5,
                 hold_days: int = 5,
                 initial_capital: float = 1_000_000.0,
                 benchmark: pd.DataFrame | None = None,
                 buy_threshold: float | None = None,
                 limit_pct: float | None = None) -> BacktestResult:
    """scores: date, stock_id, score(, close)；price_panel: sid -> 日K。"""
    cfg = CONFIG
    threshold = 0.2 if buy_threshold is None else buy_threshold
    limit = cfg.limit_pct if limit_pct is None else limit_pct

    if scores is None or scores.empty:
        return BacktestResult()

    scores = scores.copy()
    scores["date"] = pd.to_datetime(scores["date"]).dt.normalize()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")

    book = _prepare(price_panel)
    if not book:
        return BacktestResult()

    # 交易日曆：所有價格日期的聯集（含基準）
    calendar = sorted({t for bk in book.values() for t in bk["dates"]})
    if benchmark is not None and not benchmark.empty:
        bench = benchmark.copy()
        bench["date"] = pd.to_datetime(bench["date"]).dt.normalize()
        calendar = sorted(set(calendar) | set(bench["date"].tolist()))
    if not calendar:
        return BacktestResult()

    schedule = _schedule_entries(scores, book, hold_days)

    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    equity_pts: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    invested_sum = 0.0
    prev_equity = float(initial_capital)
    cost_total = 0.0
    notional_traded = 0.0
    last_close: dict[str, float] = {}

    def mtm(day: pd.Timestamp) -> float:
        total = cash
        for sid, pos in positions.items():
            px = last_close.get(sid, pos["entry_price"])
            total += pos["qty"] * px
        return total

    for day in calendar:
        # 0) 更新各股最後收盤（供評價）
        for sid, bk in book.items():
            i = bk["idx"].get(day)
            if i is not None:
                last_close[sid] = float(bk["close"][i])

        # 1) 到期出場（開盤）
        for sid in list(positions.keys()):
            pos = positions[sid]
            if pos["exit_date"] != day:
                continue
            bk = book[sid]
            i = bk["idx"].get(day)
            if i is None:            # 停牌：順延到下一個有價日
                continue
            px = float(bk["open"][i])
            prev_close = float(bk["close"][i - 1]) if i > 0 else px
            # 跌停無法賣出 → 順延
            if prev_close > 0 and px <= prev_close * (1 - limit) + 1e-9:
                pos["exit_date"] = bk["dates"][i + 1] if i + 1 < len(bk["dates"]) else day
                continue
            notional = px * pos["qty"]
            fee = max(notional * cfg.commission_effective, cfg.buy_fixed_fee)
            tax = notional * cfg.tax_rate
            slip = notional * cfg.slippage
            proceeds = notional - fee - tax - slip
            cash += proceeds
            cost_total += fee + tax + slip
            notional_traded += notional
            trades.append({
                "entry_date": pos["entry_date"], "exit_date": day,
                "stock_id": sid, "qty": pos["qty"],
                "entry_price": pos["entry_price"], "exit_price": px,
                "ret": proceeds / pos["cost"] - 1,
            })
            del positions[sid]

        # 2) 進場（開盤），資金以「前一日收盤評價」為準，避免用到當日尚未發生的資訊
        for cand in sorted(schedule.get(day, []), key=lambda c: -c["score"]):
            sid = cand["stock_id"]
            if len(positions) >= top_n or sid in positions:
                continue
            if cand["score"] < threshold:
                continue
            bk = book[sid]
            i = cand["entry_i"]
            px = float(bk["open"][i])
            if px <= 0:
                continue
            prev_close = float(bk["close"][i - 1]) if i > 0 else px
            # 漲停無法買進 → 放棄此訊號
            if prev_close > 0 and px >= prev_close * (1 + limit) - 1e-9:
                continue
            budget = prev_equity * cfg.max_position_pct
            if budget < px:
                continue
            qty = int(budget // px)
            if qty <= 0:
                continue
            fee = max(px * qty * cfg.commission_effective, cfg.buy_fixed_fee)
            slip = px * qty * cfg.slippage
            cost = px * qty + fee + slip
            if cost > cash:
                qty = int((cash - cfg.buy_fixed_fee) // (px * (1 + cfg.slippage)))
                if qty <= 0:
                    continue
                fee = max(px * qty * cfg.commission_effective, cfg.buy_fixed_fee)
                slip = px * qty * cfg.slippage
                cost = px * qty + fee + slip
                if cost > cash:
                    continue
            cash -= cost
            cost_total += fee + slip
            notional_traded += px * qty
            positions[sid] = {
                "entry_price": px, "qty": qty, "cost": cost,
                "entry_date": day, "exit_date": cand["exit_date"],
            }

        # 3) 收盤評價
        eq = mtm(day)
        equity_pts.append((day, eq))
        invested = eq - cash
        invested_sum += invested
        prev_equity = eq

    equity = pd.Series(dict(equity_pts), name="equity").sort_index().astype(float)
    trades_df = pd.DataFrame(trades)

    res = BacktestResult(equity=equity, trades=trades_df)
    if len(equity) > 1:
        res.total_return = equity.iloc[-1] / equity.iloc[0] - 1
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        res.cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
        daily = equity.pct_change().dropna()
        if daily.std() > 0:
            res.sharpe = float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS))
        downside = daily[daily < 0]
        if len(downside) and downside.std() > 0:
            res.sortino = float(daily.mean() / downside.std() * np.sqrt(TRADING_DAYS))
        peak = equity.cummax()
        res.max_drawdown = float(((equity - peak) / peak).min())
        res.exposure = float(invested_sum / (len(equity_pts) * max(equity.mean(), 1e-9))) \
            if equity_pts else 0.0

    if not trades_df.empty:
        res.win_rate = float((trades_df["ret"] > 0).mean())
        wins = trades_df.loc[trades_df["ret"] > 0, "ret"]
        losses = trades_df.loc[trades_df["ret"] <= 0, "ret"]
        res.avg_win = float(wins.mean()) if len(wins) else 0.0
        res.avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        res.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    res.cost_total = cost_total
    res.turnover = notional_traded / max(initial_capital, 1.0)

    # 基準（買進持有 0050）
    if benchmark is not None and not benchmark.empty:
        b = benchmark.copy()
        b["date"] = pd.to_datetime(b["date"]).dt.normalize()
        b = b.sort_values("date").drop_duplicates("date", keep="last")
        b = b[(b["date"] >= equity.index[0]) & (b["date"] <= equity.index[-1])] \
            if len(equity) else b
        if len(b) > 1:
            beq = pd.Series(b["close"].to_numpy(dtype=float), index=b["date"], name="bench")
            res.benchmark_equity = beq
            res.benchmark_total_return = float(beq.iloc[-1] / beq.iloc[0] - 1)
            byears = (beq.index[-1] - beq.index[0]).days / 365.25
            res.benchmark_cagr = float((beq.iloc[-1] / beq.iloc[0]) ** (1 / byears) - 1) \
                if byears > 0 else 0.0
            peak = beq.cummax()
            res.benchmark_max_drawdown = float(((beq - peak) / peak).min())

    return res
