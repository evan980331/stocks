"""回測：依策略規格選股並持有，含可調整的台股交易成本。

時間正確性（關鍵）
  訊號以 **T 日收盤**資料計算 → 必須在 **T+1 開盤**成交，否則就是 look-ahead bias。
  出場：``fixed`` 政策在到期日開盤賣出；``hysteresis`` 政策的出場決策也只用
  「當日之前最近一個交易日的分數」，同樣在當日開盤成交。
  漲停無法買進、跌停無法賣出（以開盤價相對前一日收盤判斷），據以跳過或順延。

週轉率定義（重要，避免與交易次數混淆）
  令 ``N_buy`` = 買進名目金額合計、``N_sell`` = 賣出名目金額合計、
  ``notional_traded = N_buy + N_sell``（**買賣各計一次**）：

  - ``turnover``（累積、雙邊）= ``notional_traded / initial_capital``
    這是本專案先前回報的 94.72x：**未年化**，且**買賣都算**，因此一次完整進出
    會計入約 2 倍名目金額。它**不等於**交易次數，也不是年化值。
  - ``turnover_one_way`` = ``(notional_traded / 2) / initial_capital``（累積、單邊）
  - ``turnover_annualized`` = ``turnover / years``
  - ``portfolio_turnover_annualized`` = ``(notional_traded / 2) / mean_equity / years``
    此為業界常用的「年化單邊週轉率」，與資金規模無關，適合跨策略比較。

成本：由 `twstock.costs.CostModel` 提供（預設 "typical"）。
流動性：``participation_rate < 1`` 時，單筆數量受當日成交量上限限制
（``participation_rate = 1.0`` 代表未設額外流動性限制）。

不支援：放空、融資、借券。整股/零股皆以 1 股為單位（零股模式，利於小資金回測）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CONFIG
from .costs import CostModel, get_cost_model
from .strategies import StrategySpec, baseline

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
    cost_total: float = 0.0
    exposure: float = 0.0
    periods_per_year: int = TRADING_DAYS
    # 週轉率（定義見模組 docstring）
    notional_traded: float = 0.0
    turnover: float = 0.0
    turnover_one_way: float = 0.0
    turnover_annualized: float = 0.0
    portfolio_turnover_annualized: float = 0.0
    round_trips: int = 0
    avg_holding_days: float = 0.0
    years: float = 0.0
    # 元資料
    strategy_name: str = ""
    cost_model_name: str = "typical"
    initial_capital: float = 0.0
    skipped_limit_up: int = 0
    skipped_liquidity: int = 0
    skipped_no_price: int = 0
    # 基準（買進持有）
    benchmark_total_return: float = 0.0
    benchmark_cagr: float = 0.0
    benchmark_max_drawdown: float = 0.0
    benchmark_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def excess_return(self) -> float:
        return self.total_return - self.benchmark_total_return

    def cost_drag(self, gross: "BacktestResult") -> float:
        """成本造成的報酬差距（含成本 − 不含成本）。"""
        return self.total_return - gross.total_return

    def summary(self) -> str:
        lines = [
            f"策略 {self.strategy_name or '-'}｜成本情境 {self.cost_model_name}"
            f"｜期間 {self.years:.2f} 年",
            f"策略：年化 {self.cagr:+.2%}  總報酬 {self.total_return:+.2%}  "
            f"Sharpe {self.sharpe:.2f}  Sortino {self.sortino:.2f}  "
            f"MDD {self.max_drawdown:.2%}  勝率 {self.win_rate:.2%}  "
            f"交易 {self.round_trips} 筆",
            f"風險/成本：獲利因子 {self.profit_factor:.2f}  "
            f"平均獲利 {self.avg_win:+.2%}  平均虧損 {self.avg_loss:+.2%}  "
            f"總成本 {self.cost_total:,.0f}  曝險 {self.exposure:.1%}  平均持有 "
            f"{self.avg_holding_days:.1f} 交易日",
            f"週轉率：累積雙邊 {self.turnover:.2f}x  累積單邊 {self.turnover_one_way:.2f}x  "
            f"年化雙邊 {self.turnover_annualized:.2f}x  年化單邊(資金) "
            f"{self.portfolio_turnover_annualized:.2f}x",
        ]
        if len(self.benchmark_equity) > 1:
            lines.append(
                f"基準 {CONFIG.benchmark} 買進持有：總報酬 {self.benchmark_total_return:+.2%}  "
                f"年化 {self.benchmark_cagr:+.2%}  MDD {self.benchmark_max_drawdown:.2%}  "
                f"超額 {self.excess_return():+.2%}")
        lines.append(
            "⚠️ 回測結果不代表未來報酬；成本/成交假設見 twstock/costs.py 與 plan.md。")
        return "\n".join(lines)


def _prepare(panel: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """把每檔日K整理成可供 T+1 查找的結構（含成交量）。"""
    out: dict[str, dict] = {}
    for sid, df in panel.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        d = d.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        for col in ("open", "close", "volume"):
            if col not in d.columns:
                d[col] = np.nan
            d[col] = pd.to_numeric(d[col], errors="coerce")
        # 非正數價格／成交量一律視為缺值（實測 2317 曾有整列 0 的錯誤資料，
        # 當成 0 元會讓市值歸零並產生假回撤）
        for col in ("open", "close"):
            d.loc[d[col] <= 0, col] = np.nan
        d.loc[d["volume"] <= 0, "volume"] = np.nan
        d = d.dropna(subset=["open", "close"])
        if d.empty:
            continue
        out[sid] = {
            "dates": d["date"].tolist(),
            "idx": {t: i for i, t in enumerate(d["date"])},
            "open": d["open"].to_numpy(dtype=float),
            "close": d["close"].to_numpy(dtype=float),
            "volume": d["volume"].to_numpy(dtype=float),
        }
    return out


def _schedule_entries(scores: pd.DataFrame, book: dict[str, dict],
                      hold_days: int) -> dict[pd.Timestamp, list[dict]]:
    """把每個訊號換算成 (進場日, 進場索引, 最晚出場日)。以 T+1 開盤進場。"""
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
                 limit_pct: float | None = None,
                 cost_model: CostModel | None = None,
                 strategy: StrategySpec | None = None) -> BacktestResult:
    """scores: date, stock_id, score(, close)；price_panel: sid -> 日K。

    ``strategy`` 提供時會覆蓋 top_n / hold_days / 門檻 / 漲跌停 / 持倉政策。
    """
    spec = strategy or baseline().with_updates(
        top_n=top_n, hold_days=hold_days,
        buy_threshold=(0.2 if buy_threshold is None else buy_threshold),
        limit_pct=(CONFIG.limit_pct if limit_pct is None else limit_pct))
    spec.validate()

    cm = cost_model or get_cost_model("typical")
    threshold = spec.buy_threshold
    limit = spec.limit_pct

    if scores is None or scores.empty:
        return BacktestResult(strategy_name=spec.name, cost_model_name=cm.name,
                              initial_capital=initial_capital)

    scores = scores.copy()
    scores["date"] = pd.to_datetime(scores["date"]).dt.normalize()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    # 以日期索引分數，避免逐日掃描整張表（驗證流程會跑多次回測）
    scores_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    for r in scores.dropna(subset=["score"]).itertuples(index=False):
        scores_by_date.setdefault(r.date, {})[r.stock_id] = float(r.score)

    book = _prepare(price_panel)
    if not book:
        return BacktestResult(strategy_name=spec.name, cost_model_name=cm.name,
                              initial_capital=initial_capital)

    # 交易日曆：所有價格日期的聯集（含基準）
    calendar = sorted({t for bk in book.values() for t in bk["dates"]})
    if benchmark is not None and not benchmark.empty:
        bench = benchmark.copy()
        bench["date"] = pd.to_datetime(bench["date"]).dt.normalize()
        calendar = sorted(set(calendar) | set(bench["date"].tolist()))
    if not calendar:
        return BacktestResult(strategy_name=spec.name, cost_model_name=cm.name,
                              initial_capital=initial_capital)

    schedule = _schedule_entries(scores, book, spec.hold_days)

    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    equity_pts: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    invested_sum = 0.0
    prev_equity = float(initial_capital)
    cost_total = 0.0
    notional_traded = 0.0
    last_close: dict[str, float] = {}
    last_score: dict[str, float] = {}
    skipped_limit_up = skipped_liquidity = skipped_no_price = 0

    def mtm() -> float:
        total = cash
        for sid, pos in positions.items():
            total += pos["qty"] * last_close.get(sid, pos["entry_price"])
        return total

    for cal_i, day in enumerate(calendar):
        # 0) 更新收盤價（供評價與隔日判斷）
        for sid, bk in book.items():
            i = bk["idx"].get(day)
            if i is not None:
                last_close[sid] = float(bk["close"][i])

        # 1) 出場（開盤）：決策只用「當日之前」已知的分數
        for sid in list(positions.keys()):
            pos = positions[sid]
            bk = book[sid]
            i = bk["idx"].get(day)
            if i is None:
                continue                       # 停牌／無價：順延
            days_held = i - pos["entry_i"]

            if spec.policy == "fixed":
                if day != pos["exit_date"]:
                    continue
            else:
                max_reached = day >= pos["exit_date"]
                sig = last_score.get(sid)
                below = sig is not None and sig < float(spec.exit_threshold)
                if not (max_reached or (days_held >= spec.min_hold_days and below)):
                    continue

            px = float(bk["open"][i])
            prev_close = float(bk["close"][i - 1]) if i > 0 else px
            # 跌停無法賣出 → 順延（fixed 政策改寫到期日；hysteresis 隔日會重新評估）
            if prev_close > 0 and px <= prev_close * (1 - limit) + 1e-9:
                if spec.policy == "fixed" and i + 1 < len(bk["dates"]):
                    pos["exit_date"] = bk["dates"][i + 1]
                continue

            notional = px * pos["qty"]
            sell_cost = cm.sell_cost(notional)
            proceeds = notional - sell_cost
            cash += proceeds
            cost_total += sell_cost
            notional_traded += notional
            trades.append({
                "entry_date": pos["entry_date"], "exit_date": day,
                # 訊號日（供歸因量測訊號→成交的延遲），不影響任何計算
                "signal_date": pos.get("signal_date"),
                "stock_id": sid, "qty": pos["qty"],
                "entry_price": pos["entry_price"], "exit_price": px,
                "holding_days": days_held,
                "gross_pnl": (px - pos["entry_price"]) * pos["qty"],
                "cost": pos["entry_cost"] + sell_cost,
                "ret": proceeds / pos["cost"] - 1,
            })
            del positions[sid]

        # 2) 進場（開盤）：再平衡頻率以交易日曆為準
        rebalance_ok = (cal_i % spec.rebalance_days == 0)
        if rebalance_ok:
            for cand in sorted(schedule.get(day, []), key=lambda c: -c["score"]):
                sid = cand["stock_id"]
                if len(positions) >= spec.top_n or sid in positions:
                    continue
                if cand["score"] < threshold:
                    continue
                bk = book[sid]
                i = cand["entry_i"]
                px = float(bk["open"][i])
                if px <= 0:
                    skipped_no_price += 1
                    continue
                prev_close = float(bk["close"][i - 1]) if i > 0 else px
                if prev_close > 0 and px >= prev_close * (1 + limit) - 1e-9:
                    skipped_limit_up += 1          # 漲停無法買進
                    continue
                budget = prev_equity * CONFIG.max_position_pct
                if budget < px:
                    continue
                qty = int(budget // px)
                if qty <= 0:
                    continue
                # 流動性上限（僅在成本模型明確設定 participation_rate < 1 時生效）
                if cm.participation_rate < 1.0:
                    capped = cm.max_fill_qty(float(bk["volume"][i]), px, qty)
                    if capped <= 0:
                        skipped_liquidity += 1
                        continue
                    qty = capped

                buy_cost = cm.buy_cost(px * qty)
                cost = px * qty + buy_cost
                if cost > cash:
                    qty = int((cash - cm.min_fee) // (px * (1 + cm.slippage)))
                    if qty <= 0:
                        continue
                    if cm.participation_rate < 1.0:
                        qty = cm.max_fill_qty(float(bk["volume"][i]), px, qty)
                        if qty <= 0:
                            skipped_liquidity += 1
                            continue
                    buy_cost = cm.buy_cost(px * qty)
                    cost = px * qty + buy_cost
                    if cost > cash:
                        continue
                cash -= cost
                cost_total += buy_cost
                notional_traded += px * qty
                positions[sid] = {
                    "entry_price": px, "qty": qty, "cost": cost,
                    "entry_cost": buy_cost,
                    "entry_date": day, "exit_date": cand["exit_date"],
                    "signal_date": cand["signal_date"],
                    "entry_i": i,
                }

        # 3) 收盤評價
        eq = mtm()
        equity_pts.append((day, eq))
        invested_sum += eq - cash
        prev_equity = eq

        # 4) 更新「最近已知分數」（今日收盤後才可得）
        for sid, s in scores_by_date.get(day, {}).items():
            last_score[sid] = s

    equity = pd.Series(dict(equity_pts), name="equity").sort_index().astype(float)
    trades_df = pd.DataFrame(trades)

    res = BacktestResult(equity=equity, trades=trades_df,
                         strategy_name=spec.name, cost_model_name=cm.name,
                         initial_capital=initial_capital,
                         skipped_limit_up=skipped_limit_up,
                         skipped_liquidity=skipped_liquidity,
                         skipped_no_price=skipped_no_price)
    res.round_trips = len(trades_df)
    res.cost_total = cost_total
    res.notional_traded = notional_traded
    res.turnover = notional_traded / max(initial_capital, 1.0)
    res.turnover_one_way = (notional_traded / 2) / max(initial_capital, 1.0)

    if len(equity) > 1:
        res.years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
        res.turnover_annualized = res.turnover / res.years
        mean_eq = float(equity.mean())
        res.portfolio_turnover_annualized = (
            (notional_traded / 2) / max(mean_eq, 1e-9) / res.years
            if mean_eq > 0 else 0.0)
        res.total_return = equity.iloc[-1] / equity.iloc[0] - 1
        years = res.years
        res.cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
        daily = equity.pct_change().dropna()
        if daily.std() > 0:
            res.sharpe = float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS))
        downside = daily[daily < 0]
        if len(downside) and downside.std() > 0:
            res.sortino = float(daily.mean() / downside.std() * np.sqrt(TRADING_DAYS))
        peak = equity.cummax()
        res.max_drawdown = float(((equity - peak) / peak).min())
        res.exposure = float(invested_sum / (len(equity_pts) * max(mean_eq, 1e-9))) \
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
        res.avg_holding_days = float(trades_df["holding_days"].mean())

    # 基準（買進持有）
    if benchmark is not None and not benchmark.empty:
        b = benchmark.copy()
        b["date"] = pd.to_datetime(b["date"]).dt.normalize()
        b = (b.sort_values("date").drop_duplicates("date", keep="last")
             .dropna(subset=["close"]))
        if len(equity) and len(b) > 1:
            b = b[(b["date"] >= equity.index[0]) & (b["date"] <= equity.index[-1])]
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


def run_cost_scenarios(scores: pd.DataFrame,
                       price_panel: dict[str, pd.DataFrame],
                       strategy: StrategySpec,
                       scenario_names: list[str],
                       *, benchmark: pd.DataFrame | None = None,
                       initial_capital: float = 1_000_000.0,
                       ) -> dict[str, BacktestResult]:
    """以多組成本情境跑同一策略，回傳 {情境名: 結果}。"""
    out: dict[str, BacktestResult] = {}
    for name in scenario_names:
        out[name] = run_backtest(
            scores, price_panel, strategy=strategy, benchmark=benchmark,
            cost_model=get_cost_model(name), initial_capital=initial_capital)
    return out
