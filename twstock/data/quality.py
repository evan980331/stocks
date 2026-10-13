"""價格資料品質：無效值清理與公司行動（除權息／分割）還原調整。

問題背景（實測本機 DB 發現）
  1. **無效價格列**：2317（鴻海）2025-07-30 的 open/high/low/close/volume 全為 0。
     把 0 當成價格會讓市值瞬間歸零、產生假回撤，並讓報酬率變成 ±inf。
     → 一律視為**缺值**（missing），不是 0、也不是 -100% 報酬。
  2. **未調整的公司行動**：0050 於 2025-06-18 由 188.65 變 47.57（1 拆 4），
     未調整的序列會顯示 -74.8% 的假崩盤；2603 的除息缺口亦超過 ±10% 漲跌停。
     → 以 FinMind 的 `TaiwanStockSplitPrice` 與 `TaiwanStockDividend` 事件
     重建還原（total-return）序列。

還原方法（標準乘法後調）
  對每個除權息日 D 計算一個調整因子 ``a``，並將 **D 之前**的所有價格乘以 ``a``
  （最新價格維持真實價，越舊的價格被等比縮放）：

  - 分割／減資：``a = after_price / before_price``
  - 現金股利：``a = (D 前一交易日收盤 − 現金股利) / D 前一交易日收盤``
  - 股票股利：``a = 1 / (1 + 每股股票股利)``
  - 同日多項事件：因子相乘

  成交量：僅在「股數變動」事件（分割／股票股利）時以 ``1/a`` 反向調整，
  純現金股利不調整成交量。此為還原序列，非原始申報值。

限制
  - 無法取得前一交易日收盤價的事件會被跳過並記錄，不臆測因子。
  - FinMind 免費層不提供 `TaiwanStockPriceAdj`（實測 status=400），故自行還原。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRICE_COLS = ("open", "high", "low", "close")
TWD_LIMIT_PCT = 0.10          # 台股一般股票單日漲跌停幅度
JUMP_TOLERANCE = 0.11         # 超過此幅度即視為疑似未調整公司行動


@dataclass
class QualityReport:
    stock_id: str
    invalid_rows: int = 0
    invalid_dates: list[str] = field(default_factory=list)
    detected_jumps: list[str] = field(default_factory=list)

    def label(self) -> str:
        parts = [f"{self.stock_id}"]
        if self.invalid_rows:
            parts.append(f"無效價格 {self.invalid_rows} 列（{', '.join(self.invalid_dates[:3])}…）")
        if self.detected_jumps:
            parts.append(f"疑似未調整公司行動 {len(self.detected_jumps)} 筆 "
                         f"（{', '.join(self.detected_jumps[:3])}…）")
        if len(parts) == 1:
            parts.append("無異常")
        return "：".join(parts)


# --------------------------------------------------------------------------
# 無效值
# --------------------------------------------------------------------------
def find_invalid_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    """回傳含有非正數或缺失 OHLC 的列（視為資料錯誤，不是 0 元成交）。"""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in PRICE_COLS if c in df.columns]
    if not cols:
        return pd.DataFrame()
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    bad = (num <= 0) | num.isna()
    return df[bad.any(axis=1)]


def clean_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """移除無效價格列。回傳 (乾淨資料, 被移除的列)。

    被移除的交易日會成為**缺口**（缺值），而非 0；下游必須以缺值處理。
    """
    if df is None or df.empty:
        return df, pd.DataFrame()
    bad = find_invalid_price_rows(df)
    if bad.empty:
        return df.copy(), pd.DataFrame()
    mask = ~df.index.isin(bad.index)
    return df[mask].copy(), bad.copy()


def find_unadjusted_jumps(df: pd.DataFrame,
                          tolerance: float = JUMP_TOLERANCE) -> pd.DataFrame:
    """找出超出漲跌停幅度的單日變動 → 疑似未調整的除權息或分割。"""
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    ret = pd.to_numeric(d["close"], errors="coerce").pct_change()
    hit = ret.abs() > tolerance
    return d.loc[hit, ["date", "close"]].assign(ret=ret[hit])


def audit_prices(df: pd.DataFrame, stock_id: str = "") -> QualityReport:
    """單一標的的價格品質摘要。"""
    rep = QualityReport(stock_id=stock_id or "?")
    if df is None or df.empty:
        return rep
    bad = find_invalid_price_rows(df)
    rep.invalid_rows = len(bad)
    if not bad.empty and "date" in bad.columns:
        rep.invalid_dates = [str(pd.Timestamp(d).date()) for d in bad["date"].head(5)]
    jumps = find_unadjusted_jumps(df)
    if not jumps.empty:
        rep.detected_jumps = [
            f"{pd.Timestamp(r.date).date()}({r.ret:+.1%})" for r in jumps.itertuples()]
    return rep


# --------------------------------------------------------------------------
# 公司行動 → 調整因子
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CorporateAction:
    stock_id: str
    date: pd.Timestamp
    kind: str            # split | cash_dividend | stock_dividend
    factor: float        # 該事件對「事件日之前」價格的乘數
    price_scale: bool    # 是否調整成交量（股數變動者為 True）
    detail: str = ""

    def __post_init__(self) -> None:
        if not (self.factor > 0):
            raise ValueError(f"調整因子必須為正數，得到 {self.factor}")


def split_factor(before_price: float, after_price: float) -> float:
    """分割／減資因子 = after / before。"""
    before = float(before_price or 0)
    after = float(after_price or 0)
    if before <= 0 or after <= 0:
        raise ValueError("分割前後價格必須為正數")
    return after / before


def cash_dividend_factor(prev_close: float, cash: float) -> float:
    """現金股利因子 = (除息前收盤 − 現金股利) / 除息前收盤。"""
    prev = float(prev_close or 0)
    c = float(cash or 0)
    if prev <= 0:
        raise ValueError("除息前收盤價必須為正數")
    if c < 0 or c >= prev:
        raise ValueError(f"現金股利不合理：{c} vs 收盤 {prev}")
    return (prev - c) / prev


def stock_dividend_factor(stock_ratio: float) -> float:
    """股票股利因子 = 1 / (1 + 每股股票股利)。"""
    r = float(stock_ratio or 0)
    if r < 0:
        raise ValueError("股票股利不得為負")
    return 1.0 / (1.0 + r)


def _prev_close_before(price: pd.DataFrame, date: pd.Timestamp) -> float | None:
    d = price.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    prior = d[d["date"] < pd.Timestamp(date).normalize()]
    if prior.empty:
        return None
    close = pd.to_numeric(prior.sort_values("date")["close"].iloc[-1], errors="coerce")
    return float(close) if pd.notna(close) and close > 0 else None


def raw_return_at(price: pd.DataFrame, date: pd.Timestamp) -> float | None:
    """事件日相對前一交易日的**原始**報酬；找不到價格回 None。"""
    if price is None or price.empty:
        return None
    d = price.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values("date").reset_index(drop=True)
    target = pd.Timestamp(date).normalize()
    hit = d.index[d["date"] == target]
    if len(hit) == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return None
    prev = pd.to_numeric(d["close"].iloc[i - 1], errors="coerce")
    cur = pd.to_numeric(d["close"].iloc[i], errors="coerce")
    if pd.isna(prev) or pd.isna(cur) or prev <= 0 or cur <= 0:
        return None
    return float(cur / prev - 1)


def prune_actions_by_observed_move(price: pd.DataFrame,
                                   actions: list[CorporateAction],
                                   ) -> tuple[list[CorporateAction], list[str]]:
    """只保留「能真正降低價格序列不連續」的調整動作。

    檢驗方式：套用因子後，事件日報酬由 ``r`` 變為 ``(1+r)/factor − 1``。
    若套用後反而離 0 更遠，代表該事件在序列中**並未被反映**
    （例如資料源的事件日或比率與實際不符），此時套用會製造假跳動 → 捨棄並記錄。

    這能避免盲目信任事件檔而破壞正確的價格序列。
    """
    kept: list[CorporateAction] = []
    skipped: list[str] = []
    for a in actions:
        r = raw_return_at(price, a.date)
        if r is None:
            skipped.append(f"{a.stock_id} {a.date:%Y-%m-%d} {a.kind}：事件日無有效價格，"
                           f"無法驗證，不予調整")
            continue
        after = (1.0 + r) / a.factor - 1.0
        if abs(after) < abs(r):
            kept.append(a)
        else:
            skipped.append(
                f"{a.stock_id} {a.date:%Y-%m-%d} {a.kind}：原始報酬 {r:+.2%} 未反映"
                f"事件（factor {a.factor:.4f}，套用後變 {after:+.2%}），捨棄")
    return kept, skipped


def build_corporate_actions(price: pd.DataFrame, stock_id: str,
                            splits: pd.DataFrame | None = None,
                            dividends: pd.DataFrame | None = None,
                            ) -> tuple[list[CorporateAction], list[str]]:
    """由 FinMind 的分割與股利事件建立調整動作。回傳 (actions, 警告)。

    無法取得必要價格（如除息前收盤）的事件會被跳過並記錄，**不臆測因子**。
    """
    actions: list[CorporateAction] = []
    warnings: list[str] = []
    price = price.copy() if price is not None else pd.DataFrame()
    if not price.empty and "date" in price.columns:
        price["date"] = pd.to_datetime(price["date"]).dt.normalize()

    if splits is not None and not splits.empty:
        for r in splits.itertuples(index=False):
            try:
                factor = split_factor(getattr(r, "before_price", 0),
                                      getattr(r, "after_price", 0))
            except ValueError as e:
                warnings.append(f"{stock_id} {getattr(r, 'date', '?')} 分割事件無法計算：{e}")
                continue
            kind = str(getattr(r, "type", "") or "分割")
            actions.append(CorporateAction(
                stock_id, pd.Timestamp(getattr(r, "date")).normalize(),
                "split", factor, True, kind))

    if dividends is not None and not dividends.empty:
        for r in dividends.itertuples(index=False):
            cash = (float(getattr(r, "CashEarningsDistribution", 0) or 0)
                    + float(getattr(r, "CashStatutorySurplus", 0) or 0))
            stock_div = (float(getattr(r, "StockEarningsDistribution", 0) or 0)
                         + float(getattr(r, "StockStatutorySurplus", 0) or 0))
            ex_cash = getattr(r, "CashExDividendTradingDate", "") or ""
            ex_stock = getattr(r, "StockExDividendTradingDate", "") or ""

            if cash > 0 and ex_cash:
                ex_date = pd.Timestamp(ex_cash).normalize()
                prev = _prev_close_before(price, ex_date)
                if prev is None:
                    warnings.append(f"{stock_id} {ex_cash} 無除息前收盤價，跳過現金股利還原")
                else:
                    try:
                        actions.append(CorporateAction(
                            stock_id, ex_date, "cash_dividend",
                            cash_dividend_factor(prev, cash), False,
                            f"現金股利 {cash:.4f}"))
                    except ValueError as e:
                        warnings.append(f"{stock_id} {ex_cash} 現金股利因子異常：{e}")

            if stock_div > 0 and ex_stock:
                ex_date = pd.Timestamp(ex_stock).normalize()
                actions.append(CorporateAction(
                    stock_id, ex_date, "stock_dividend",
                    stock_dividend_factor(stock_div), True,
                    f"股票股利 {stock_div:.4f}"))

    # 只保留能實際降低序列不連續的動作（避免事件檔與實際價格不符時製造假跳動）
    actions, skipped = prune_actions_by_observed_move(price, actions)
    warnings.extend(skipped)
    return actions, warnings


def apply_corporate_actions(df: pd.DataFrame,
                            actions: list[CorporateAction]) -> pd.DataFrame:
    """對價格序列套用後調（back-adjust）：事件日之前的價格乘以累積因子。

    最新價格維持真實成交價，越舊的價格被等比縮放，使序列連續。
    """
    if df is None or df.empty or not actions:
        return df.copy() if df is not None else pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    price_cols = [c for c in PRICE_COLS if c in out.columns]
    has_volume = "volume" in out.columns
    out["_adj_price"] = 1.0
    out["_adj_volume"] = 1.0

    for a in sorted(actions, key=lambda x: x.date):
        before = out["date"] < a.date
        if not before.any():
            continue
        out.loc[before, "_adj_price"] *= a.factor
        if a.price_scale and has_volume:
            out.loc[before, "_adj_volume"] *= a.factor

    for c in price_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce") * out["_adj_price"]
    if has_volume:
        out["volume"] = (pd.to_numeric(out["volume"], errors="coerce")
                         .fillna(0.0) / out["_adj_volume"])
    return out.drop(columns=["_adj_price", "_adj_volume"])


def adjusted_close_series(price: pd.DataFrame,
                          actions: list[CorporateAction]) -> pd.Series:
    """回傳以日期為索引的還原收盤價（供基準報酬計算）。"""
    adj = apply_corporate_actions(price, actions)
    if adj.empty:
        return pd.Series(dtype=float)
    adj = adj.sort_values("date")
    return pd.Series(adj["close"].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(adj["date"]), name="close")
