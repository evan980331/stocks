"""價格載入：清理無效列並套用除權息／分割還原。

所有分析（特徵、回測、基準比較）都應透過此模組取得價格，確保
「策略」與「基準」使用**一致的除權息處理方式**；`adjusted=False` 可取得原始
未還原序列供稽核比對。
"""
from __future__ import annotations

import logging

import pandas as pd

from . import store
from .quality import (CorporateAction, apply_corporate_actions, clean_prices,
                      prune_actions_by_observed_move)

log = logging.getLogger(__name__)


def load_actions_map() -> dict[str, list[CorporateAction]]:
    """由 DB 讀取公司行動並轉為調整動作（依 stock_id 分組）。"""
    df = store.read_sql(
        "SELECT stock_id, date, kind, factor, price_scale, detail "
        "FROM corporate_actions")
    out: dict[str, list[CorporateAction]] = {}
    if df.empty:
        return out
    for r in df.itertuples(index=False):
        try:
            action = CorporateAction(
                stock_id=str(r.stock_id),
                date=pd.Timestamp(r.date).normalize(),
                kind=str(r.kind),
                factor=float(r.factor),
                price_scale=bool(r.price_scale),
                detail=str(r.detail or ""),
            )
        except (ValueError, TypeError) as e:  # 因子不合法者略過，不臆測
            log.warning("略過不合法公司行動 %s %s: %s", r.stock_id, r.date, e)
            continue
        out.setdefault(action.stock_id, []).append(action)
    return out


def new_stats() -> dict:
    """載入統計計數器（供報告如實呈現資料處理量）。"""
    return {"rows_dropped": 0, "actions_applied": 0, "actions_skipped": 0,
            "stocks": 0}


def load_price_panel(adjusted: bool = True,
                     columns: str = "*",
                     stats: dict | None = None) -> pd.DataFrame:
    """讀取全部價格並清理 / 還原。回傳長表（含 stock_id）。

    `adjusted=True`：移除無效價格列並套用除權息／分割還原。
    `stats`：傳入 `new_stats()` 的 dict 以累計處理量（移除列數、套用/捨棄事件數）。
    """
    raw = store.read_sql(f"SELECT {columns} FROM price")
    if raw.empty:
        return raw
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

    actions = load_actions_map() if adjusted else {}
    parts: list[pd.DataFrame] = []
    for sid, g in raw.groupby("stock_id"):
        g = g.sort_values("date").reset_index(drop=True)
        clean, dropped = clean_prices(g)
        if not dropped.empty:
            log.warning("%s 移除 %d 列無效價格（%s）", sid, len(dropped),
                        ", ".join(str(pd.Timestamp(d).date())
                                  for d in dropped["date"].head(3)))
            if stats is not None:
                stats["rows_dropped"] += len(dropped)
        if stats is not None:
            stats["stocks"] += 1
        if adjusted and sid in actions and not clean.empty:
            # 縱深防護：即使 DB 內有未經驗證的事件，讀取時再驗一次
            kept, skipped = prune_actions_by_observed_move(clean, actions[sid])
            for s in skipped:
                log.warning("略過調整：%s", s)
            if stats is not None:
                stats["actions_applied"] += len(kept)
                stats["actions_skipped"] += len(skipped)
            clean = apply_corporate_actions(clean, kept)
        parts.append(clean)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.sort_values(["stock_id", "date"]).reset_index(drop=True)


def panel_dict(price_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """長表 → {stock_id: 日K DataFrame}。"""
    if price_long is None or price_long.empty:
        return {}
    return {sid: g.reset_index(drop=True)
            for sid, g in price_long.groupby("stock_id")}


def benchmark_series(price_long: pd.DataFrame, benchmark_id: str) -> pd.DataFrame:
    """由已還原的價格長表取出基準（date, close）。"""
    if price_long is None or price_long.empty:
        return pd.DataFrame(columns=["date", "close"])
    b = price_long[price_long["stock_id"] == benchmark_id][["date", "close"]]
    return b.sort_values("date").reset_index(drop=True)
