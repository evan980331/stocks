"""回測：每期打分排序選 top-N 等權買入持有 hold_days 日，含台股交易成本。

台股整股為 1 張（1000 股），高價股（台股 0050 成分）可能買不到一張。
支援：a) 整張 b) 盤中零股（odd lot，可 1 股起）。此版本用零股模式以利小資金回測。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CONFIG

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    cagr: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0

    def summary(self) -> str:
        return (
            f"策略年化 {self.cagr:+.2%}  總報酬 {self.total_return:+.2%}  "
            f"Sharpe {self.sharpe:.2f}  MDD {self.max_drawdown:.2%}  "
            f"勝率 {self.win_rate:.2%}  交易次數 {len(self.trades)}"
        )


def run_backtest(scores: pd.DataFrame,
                 price_panel: dict[str, pd.DataFrame],
                 top_n: int = 5,
                 hold_days: int = 5,
                 initial_capital: float = 1_000_000.0) -> BacktestResult:
    """scores: date, stock_id, score, close；price_panel: sid -> 日K DataFrame。"""
    cfg = CONFIG

    scores = scores.copy()
    scores["date"] = pd.to_datetime(scores["date"]).dt.normalize()
    dates = sorted(scores["date"].unique())
    price_lookup: dict[tuple[str, pd.Timestamp], tuple[float, float]] = {}

    for sid, df in price_panel.items():
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        for _, r in df.iterrows():
            price_lookup[(sid, r["date"])] = (r["open"], r["close"])

    cash = initial_capital
    positions: dict[str, dict] = {}  # sid -> {entry_price, qty, days_held, entry_date}
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []

    def mark_to_market(day, cash_val, positions_val) -> float:
        total = cash_val
        for sid, pos in positions_val.items():
            px = price_lookup.get((sid, day), (pos["last_price"], pos["last_price"]))[1]
            total += pos["qty"] * px
        return total

    for day in dates:
        # 0) 更新持倉市值
        for sid, pos in positions.items():
            px = price_lookup.get((sid, day), (None, None))[1]
            if px:
                pos["last_price"] = px

        # 1) 到期/平倉
        for sid in list(positions.keys()):
            pos = positions[sid]
            pos["days_held"] += 1
            pos["last_price"] = price_lookup.get((sid, day), (None, None))[1] or pos["last_price"]
            if pos["days_held"] >= hold_days:
                px = pos["last_price"]
                if px is None:
                    continue
                fee_out = max(px * pos["qty"] * cfg.commission_effective, cfg.buy_fixed_fee)
                tax = px * pos["qty"] * cfg.tax_rate
                slip = px * pos["qty"] * cfg.slippage
                proceeds = px * pos["qty"] - fee_out - tax - slip
                cash += proceeds
                ret = (proceeds / pos["cost"] - 1)
                trades.append({
                    "date": day, "stock_id": sid,
                    "entry_price": pos["entry_price"], "exit_price": px,
                    "ret": ret,
                })
                del positions[sid]

        # 2) 訊號進場（僅在無持倉時考慮 top-N）
        day_scores = scores[scores["date"] == day]
        if not day_scores.empty:
            top = day_scores.sort_values("score", ascending=False).head(top_n)
            equity_now = mark_to_market(day, cash, positions)
            for _, row in top.iterrows():
                sid = row["stock_id"]
                if sid in positions or row["score"] < 0.2:
                    continue
                open_px = price_lookup.get((sid, day), (None, None))[0]
                if not open_px:
                    day_stock = scores[(scores["stock_id"] == sid) &
                                       (scores["date"] == day)]
                    close_fallback = day_stock["close"].iloc[0] if not day_stock.empty else None
                    if close_fallback:
                        open_px = close_fallback
                    else:
                        log.debug("%s %s 無價格可進場", day, sid)
                        continue
                # 等權：每檔 max_position_pct 佔時總資產（零股模式，1 股起）
                budget = equity_now * cfg.max_position_pct
                if budget < open_px:
                    log.debug("%s %s 預算不足 1 股", day, sid)
                    continue
                qty = int(budget // open_px)
                if qty <= 0:
                    continue
                fee_in = max(open_px * qty * cfg.commission_effective, cfg.buy_fixed_fee)
                slip = open_px * qty * cfg.slippage
                cost = open_px * qty + fee_in + slip
                if cost > cash:
                    qty = int((cash - cfg.buy_fixed_fee) // (open_px * (1 + cfg.slippage)))
                    if qty <= 0:
                        continue
                    fee_in = max(open_px * qty * cfg.commission_effective, cfg.buy_fixed_fee)
                    cost = open_px * qty + fee_in + open_px * qty * cfg.slippage
                    if cost > cash:
                        continue
                cash -= cost
                positions[sid] = {
                    "entry_price": open_px, "last_price": open_px,
                    "qty": qty, "cost": cost, "days_held": 0,
                    "entry_date": day,
                }

        equity_curve.append((day, mark_to_market(day, cash, positions)))

    equity = pd.Series(
        {d: v for d, v in equity_curve}, name="equity").sort_index()
    trades_df = pd.DataFrame(trades)

    result = BacktestResult(equity=equity, trades=trades_df)
    if len(equity) > 1:
        result.total_return = equity.iloc[-1] / equity.iloc[0] - 1
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        result.cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
        daily_ret = equity.pct_change().dropna()
        if daily_ret.std() > 0:
            result.sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        peak = equity.cummax()
        result.max_drawdown = ((equity - peak) / peak).min()
    if not trades_df.empty:
        result.win_rate = (trades_df["ret"] > 0).mean()
    return result
