"""P2.8 §6：前瞻研究紀錄（凍結研究版本、每日可稽核切點、最低觀察期間）。

本階段**只建立紀錄與收集流程**，不宣稱已完成新的樣本外驗證。

三類資料必須明確分開，不得互相混用：
  - ``research``      研究期資料：用來開發因子與策略。
  - ``consumed_test`` 已反覆使用的測試期（B17 那批）：**已消耗**，
                      不得再當成乾淨樣本，也不得為了盡快取得結果而重複使用。
  - ``prospective``   凍結之後才開始累積、從未用於任何決策的新資料。

最低觀察期間必須**事前**宣告、理由一併凍結；不得在看到未來績效之後
回頭挑選期間（那就是事後選參數）。前瞻資料未達最低期間前，
**只報告資料品質與覆蓋率**，不宣稱策略有效。

設計原則
  - 切點「首次寫入即定案」：重跑不得改寫「當初知道什麼」。
  - 凍結不會被靜默覆寫：規格改變時回報 drift，需明確 force 才更新，
    避免研究版本在不知不覺中漂移。
  - 切點保存當時的訊號摘要，使「事後改寫歷史訊號」可被稽核。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import subprocess
from dataclasses import fields

import pandas as pd

from . import signals as sig_mod
from .backtest import BacktestResult
from .config import CONFIG, PROJECT_ROOT
from .costs import get_cost_model
from .data import pointintime, store
from .data import status as st
from .sentiment import news_rss
from .strategies import baseline

log = logging.getLogger(__name__)

FREEZE_META_KEY = "research_freeze"
CUTOFF_TABLE = "research_cutoff"

SPEC_SCHEMA_VERSION = 1

# ---- 最低觀察期間（事前宣告；理由見 MIN_OBSERVATION_RATIONALE）----
# 單位：交易日。此值在**任何前瞻結果產生之前**即隨程式碼凍結。
MIN_OBSERVATION_TRADING_DAYS = 120
# 較嚴格、建議用於正式結論的期間（一年，涵蓋季節性循環）。
PREFERRED_OBSERVATION_TRADING_DAYS = 250

MIN_OBSERVATION_RATIONALE = (
    "120 個交易日（約 6 個月，依台股約每年 246 個交易日）"
    "不是依績效挑選，而是依統計檢定力與季節性覆蓋："
    "(1) 現行策略持有 5 日、每日再平衡，觀察期須累積足夠的**獨立**回合"
    "（以既有換手率約每日 5 個回合計，120 日約 600 回合，"
    "遠高於 validation.MIN_TRADES_FOR_CONCLUSION＝20 的最低門檻）；"
    "(2) 至少涵蓋一次連續 3 個月的月營收週期與一季財報，"
    "否則營收／新聞因子只在一種季節性下被檢驗；"
    "(3) 約為現有 walk-forward 摺數可比長度的兩倍，"
    "降低單一市場狀態主宰結果的機會。"
    "期間在任何前瞻結果產生之前即寫入本研究紀錄並隨程式碼凍結；"
    f"{PREFERRED_OBSERVATION_TRADING_DAYS} 日（約一年）為建議用於正式結論的期間。"
)

# 評估指標（凍結以免事後換指標）；僅保留 BacktestResult 真正存在的欄位。
_BACKTEST_FIELDS = {f.name for f in fields(BacktestResult)}
EVAL_METRICS = tuple(
    m for m in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown",
                "win_rate", "profit_factor", "exposure",
                "turnover_annualized", "round_trips")
    if m in _BACKTEST_FIELDS)
assert len(EVAL_METRICS) >= 8, "評估指標清單與 BacktestResult 欄位不一致"

# 每日稽核的資料品質旗標（可稽核的「當時已知」判準）
QUALITY_FLAGS = (
    "has_price", "news_fetched_known", "news_published_known",
    "news_untraceable_absent",
)


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def code_sha() -> str:
    """目前程式碼的 git SHA；取不到時回空字串（不臆測）。

    以 ``git rev-parse HEAD`` 於專案根目錄取得；測試環境無 git 時不擲錯。
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT.parent), capture_output=True, text=True,
            timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pass
    return ""


def canonical_hash(obj) -> str:
    """穩定雜湊：排序鍵、不留空白，讓同一語意得到同一摘要。"""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def research_spec(*, frozen_at: str | None = None,
                  sha: str | None = None) -> dict:
    """目前研究版本的完整凍結規格（可重建、可比較）。

    內容涵蓋任務要求：程式 SHA、因子公式（權重與方向）、股票池、
    成本假設、交易規則與評估指標。``spec_hash`` 為其餘欄位的穩定摘要。
    """
    cost = get_cost_model("typical")
    strat = baseline()
    body = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "code_sha": code_sha() if sha is None else sha,
        "benchmark": CONFIG.benchmark,
        "pool": {"n": len(CONFIG.pool),
                 "ids": sorted(CONFIG.pool.keys())},
        "factor_weights": dict(sorted(sig_mod.WEIGHTS.items())),
        "factor_formula": (
            "score = Σ(weight_i × direction_i × value_i) / Σ(weight_i)"
            "（僅計入當列可用的因子；缺值不補 0）"),
        "feature_params": {
            "momentum_windows": list(CONFIG.momentum_windows),
            "rsi_window": CONFIG.rsi_window,
            "volume_ma_window": CONFIG.volume_ma_window,
            "sentiment_decay_days": CONFIG.sentiment_decay_days,
        },
        "cost_model": {
            "name": cost.name, "commission_rate": cost.commission_rate,
            "commission_discount": cost.commission_discount,
            "tax_rate": cost.tax_rate, "slippage": cost.slippage,
            "min_fee": cost.min_fee,
            "participation_rate": cost.participation_rate,
        },
        "trading_rules": {
            "strategy": strat.name, "policy": strat.policy,
            "top_n": strat.top_n, "hold_days": strat.hold_days,
            "buy_threshold": strat.buy_threshold,
            "exit_threshold": strat.exit_threshold,
            "min_hold_days": strat.min_hold_days,
            "rebalance_days": strat.rebalance_days,
            "limit_pct": strat.limit_pct,
            "max_position_pct": CONFIG.max_position_pct,
            "fusion_method": CONFIG.fusion_method,
            "fusion_weight_rule": CONFIG.fusion_weight_rule,
            "fusion_weight_ml": CONFIG.fusion_weight_ml,
        },
        "eval_metrics": list(EVAL_METRICS),
        "news_ruleset_version": news_rss.NEWS_RULESET_VERSION,
        "min_observation": {
            "trading_days": MIN_OBSERVATION_TRADING_DAYS,
            "preferred_trading_days": PREFERRED_OBSERVATION_TRADING_DAYS,
            "rationale": MIN_OBSERVATION_RATIONALE,
        },
    }
    body["spec_hash"] = canonical_hash(body)
    body["frozen_at"] = frozen_at or _now_iso()
    return body


def load_freeze() -> dict | None:
    """讀回已凍結的研究版本；尚未凍結時回 None。"""
    raw = store.get_meta(FREEZE_META_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        log.warning("research_freeze 內容無法解析（保留原值，不覆寫）")
        return None


def freeze(*, force: bool = False) -> dict:
    """凍結目前研究版本。回傳 ``{"status", "spec", "drift"}``。

    - 尚未凍結 → 寫入並回 ``status='created'``。
    - 已凍結且摘要相同 → 回 ``status='unchanged'``（不重寫時間）。
    - 已凍結但摘要不同 → 回 ``status='drift'`` 且**不覆寫**；
      模型／權重／成本改變時必須明確 ``force=True``，以免研究版本靜默漂移。
    """
    spec = research_spec()
    current = load_freeze()
    if current is None:
        store.set_meta(FREEZE_META_KEY, json.dumps(spec, ensure_ascii=False))
        return {"status": "created", "spec": spec, "drift": {}}
    if current.get("spec_hash") == spec["spec_hash"]:
        return {"status": "unchanged", "spec": current, "drift": {}}
    drift = _diff(current, spec)
    if force:
        store.set_meta(FREEZE_META_KEY, json.dumps(spec, ensure_ascii=False))
        return {"status": "forced", "spec": spec, "drift": drift}
    return {"status": "drift", "spec": current, "drift": drift}


def _diff(old: dict, new: dict) -> dict:
    """列出凍結後改變的欄位（僅比對同鍵值）。"""
    out = {}
    for k in sorted(set(old) | set(new)):
        if k in ("frozen_at", "spec_hash"):
            continue
        if old.get(k) != new.get(k):
            out[k] = {"frozen": old.get(k), "now": new.get(k)}
    return out


# --------------------------------------------------------------------------
# 每日可稽核的時間切點
# --------------------------------------------------------------------------
def _digest_signals(upto: str) -> str:
    """cutoff 之前訊號列的內容摘要（(date, stock_id, score) 排序後雜湊）。

    用途：若有人事後改寫歷史訊號，重算摘要會與切點記錄不符，
    使「因當日結果不好而回頭修改歷史訊號」可被稽核。
    """
    try:
        df = store.read_sql(
            "SELECT date, stock_id, score FROM signals WHERE date<=? "
            "ORDER BY date, stock_id", (upto,))
    except Exception as e:  # noqa: BLE001 - 舊 DB 可能無 signals 表
        log.warning("讀取 signals 失敗（摘要視為不可得）：%s", e)
        return ""
    if df.empty:
        return canonical_hash([])
    rows = [[str(r.date), str(r.stock_id), round(float(r.score), 6)]
            for r in df.itertuples(index=False)]
    return canonical_hash(rows)


def snapshot(cutoff_date: str | None = None) -> dict:
    """建立某日「當時已知」的資料與品質快照（不寫入）。"""
    cutoff = cutoff_date or dt.date.today().isoformat()
    snap = {
        "cutoff_date": cutoff, "recorded_at": _now_iso(),
        "code_sha": code_sha(),
        "price_min": "", "price_max": "",
        "n_price_rows": 0, "n_price_stocks": 0,
        "n_news_published_upto": 0, "n_news_fetched_upto": 0,
        "n_news_untraceable": 0, "n_news_fetched_unknown": 0,
        "last_run_id": "", "n_ingest_records": 0, "n_ingest_failed": 0,
        "signals_digest": _digest_signals(cutoff), "notes": "",
    }
    spec = load_freeze()
    snap["spec_hash"] = (spec or {}).get("spec_hash", "")

    try:
        price = store.read_sql(
            "SELECT date, stock_id FROM price WHERE date<=?", (cutoff,))
    except Exception as e:  # noqa: BLE001
        log.warning("讀取 price 失敗（快照視為無價量）：%s", e)
        price = pd.DataFrame()
    if not price.empty:
        d = pd.to_datetime(price["date"], errors="coerce").dropna()
        if not d.empty:
            snap["price_min"] = str(d.min().date())
            snap["price_max"] = str(d.max().date())
        snap["n_price_rows"] = int(len(price))
        snap["n_price_stocks"] = int(price["stock_id"].nunique())

    avail = pointintime.news_availability()
    if not avail.empty:
        cut = pd.Timestamp(cutoff)
        pub = pd.to_datetime(avail["published"], errors="coerce")
        # 只有「在切點當日之前已可取得」的新聞才算已知（point-in-time）；
        # 之後才取得的新聞不得回填進這個切點。
        fet = pd.to_datetime(avail["usable_from"], errors="coerce")
        snap["n_news_published_upto"] = int((pub <= cut).sum())
        snap["n_news_fetched_upto"] = int((fet <= cut).sum())
        snap["n_news_untraceable"] = int(
            (avail["reason"] == "legacy_untraceable").sum())
        snap["n_news_fetched_unknown"] = int(
            avail["reason"].isin(["legacy_untraceable",
                                  "fetched_missing"]).sum())

    try:
        ir = store.read_sql(
            "SELECT run_id, status FROM ingest_run WHERE run_at<=?", (cutoff,))
    except Exception as e:  # noqa: BLE001
        log.warning("讀取 ingest_run 失敗：%s", e)
        ir = pd.DataFrame()
    if not ir.empty:
        snap["n_ingest_records"] = int(len(ir))
        snap["n_ingest_failed"] = int(
            (ir["status"] == st.IngestStatus.FAILED.value).sum())
        snap["last_run_id"] = str(ir["run_id"].iloc[-1])

    flags = []
    if snap["n_price_rows"]:
        flags.append("has_price")
    if snap["n_news_fetched_upto"]:
        flags.append("news_fetched_known")
    if snap["n_news_published_upto"]:
        flags.append("news_published_known")
    if snap["n_news_untraceable"] == 0:
        flags.append("news_untraceable_absent")
    snap["notes"] = "flags=" + ",".join(flags)
    return snap


def register_cutoff(cutoff_date: str | None = None) -> dict:
    """寫入某日的時間切點。**首次寫入即定案**，重跑不得改寫。

    回傳 ``{"registered": bool, "snapshot": dict}``；``registered=False``
    表示該日已有切點（可能是先前程式或先前執行所寫），一律保留原值。
    """
    snap = snapshot(cutoff_date)
    n = store.write_df(pd.DataFrame([snap]), CUTOFF_TABLE,
                       key_cols=("cutoff_date",), if_conflict="ignore")
    return {"registered": bool(n), "snapshot": snap}


def cutoffs() -> pd.DataFrame:
    """所有時間切點（依日期排序）。"""
    try:
        df = store.read_sql(
            f"SELECT * FROM {CUTOFF_TABLE} ORDER BY cutoff_date")
    except Exception as e:  # noqa: BLE001
        log.warning("讀取時間切點失敗：%s", e)
        return pd.DataFrame()
    return df


def verify_cutoffs() -> list[str]:
    """檢查歷史訊號是否與切點記錄的摘要一致（偵測事後改寫）。

    回傳不一致的描述清單；空清單代表一致或無可比對的切點。
    """
    df = cutoffs()
    problems = []
    if df.empty:
        return problems
    for r in df.itertuples(index=False):
        digest = getattr(r, "signals_digest", "") or ""
        if not digest:
            continue
        now = _digest_signals(str(r.cutoff_date))
        if now and now != digest:
            problems.append(
                f"{r.cutoff_date} 的歷史訊號摘要不一致"
                f"（記錄 {digest}、現為 {now}）：歷史訊號可能被改寫")
    return problems


# --------------------------------------------------------------------------
# 觀察期狀態
# --------------------------------------------------------------------------
def observation_status(today: str | None = None) -> dict:
    """前瞻觀察期狀態；未達最低期間前**不得**宣稱策略有效。"""
    df = cutoffs()
    freeze_rec = load_freeze()
    start = ""
    if not df.empty:
        start = str(pd.to_datetime(df["cutoff_date"]).min().date())
    elif freeze_rec:
        start = str(freeze_rec.get("frozen_at", ""))[:10]
    today = today or dt.date.today().isoformat()
    elapsed = 0
    if start:
        try:
            d0 = dt.date.fromisoformat(start)
            d1 = dt.date.fromisoformat(today)
            # 以日曆日/7×5 近似交易日；精確計算需交易日曆，故標示為近似
            elapsed = max(0, int((d1 - d0).days * 5 / 7))
        except ValueError:
            elapsed = 0
    sufficient = elapsed >= MIN_OBSERVATION_TRADING_DAYS
    return {
        "start_date": start,
        "today": today,
        "elapsed_trading_days_approx": elapsed,
        "min_required": MIN_OBSERVATION_TRADING_DAYS,
        "preferred": PREFERRED_OBSERVATION_TRADING_DAYS,
        "sufficient": sufficient,
        "n_cutoffs": int(len(df)),
        "frozen": freeze_rec is not None,
    }


def _obs_line(obs: dict) -> str:
    if not obs["start_date"]:
        return ("  尚未開始累積前瞻資料（無凍結紀錄或時間切點）→ "
                "只報告資料品質與覆蓋率")
    if obs["sufficient"]:
        return (f"  已達最低觀察期間（約 {obs['elapsed_trading_days_approx']} "
                f"≧ {obs['min_required']} 交易日）")
    return (f"  尚未達最低觀察期間（約 {obs['elapsed_trading_days_approx']} / "
            f"{obs['min_required']} 交易日）→ **只報告資料品質與覆蓋率**，"
            "不宣稱策略有效")


def to_text(today: str | None = None) -> str:
    """研究紀錄的人類可讀摘要（供 CLI 與文件引用）。

    ``today`` 可覆寫「今天」以便重現與測試；預設為系統日期。
    """
    frozen = load_freeze()
    obs = observation_status(today=today)
    lines = ["=== 前瞻研究紀錄（P2.8 §6） ==="]
    if frozen is None:
        lines.append("尚未凍結研究版本（執行 freeze 以建立）")
    else:
        lines += [
            f"研究版本 spec_hash   {frozen.get('spec_hash')}",
            f"凍結時間             {frozen.get('frozen_at')}",
            f"程式 SHA             {frozen.get('code_sha') or '（不可得）'}",
            f"股票池               {frozen['pool']['n']} 檔",
            f"成本模型             {frozen['cost_model']['name']}"
            f"（折扣 {frozen['cost_model']['commission_discount']}、"
            f"稅 {frozen['cost_model']['tax_rate']}、"
            f"滑價 {frozen['cost_model']['slippage']}）",
            f"交易規則             {frozen['trading_rules']['strategy']}"
            f"｜top_n={frozen['trading_rules']['top_n']}"
            f"｜持有 {frozen['trading_rules']['hold_days']} 日"
            f"｜門檻 {frozen['trading_rules']['buy_threshold']}",
            f"新聞規則版本         {frozen.get('news_ruleset_version')}",
        ]
    lines += [
        "",
        f"最低觀察期間         {MIN_OBSERVATION_TRADING_DAYS} 交易日"
        f"（建議 {PREFERRED_OBSERVATION_TRADING_DAYS}）",
        f"已記錄時間切點       {obs['n_cutoffs']} 日",
        _obs_line(obs),
    ]
    problems = verify_cutoffs()
    if problems:
        lines += ["", "⚠️ 稽核警告（歷史訊號可能被改寫）"] + \
            [f"  - {p}" for p in problems]
    lines += [
        "",
        "三類資料分開：research（研究期）／consumed_test（已消耗的測試期，"
        "不得重用）／prospective（凍結後新累積）。",
        "累積期只報告資料品質與覆蓋率；不宣稱新聞因子或策略具備可交易優勢。",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI：``python -m twstock.research [freeze|cutoff|status]``。"""
    import argparse
    ap = argparse.ArgumentParser(description="前瞻研究紀錄")
    ap.add_argument("cmd", nargs="?", default="status",
                    choices=["status", "freeze", "cutoff"],
                    help="status（預設）| freeze（凍結目前版本）| cutoff（記錄今日切點）")
    ap.add_argument("--date", default=None, help="cutoff 用的日期（YYYY-MM-DD）")
    ap.add_argument("--force", action="store_true",
                    help="freeze 時即使偵測到 drift 也覆寫（需明確指定）")
    ap.add_argument("--today", default=None,
                    help="觀察期計算用的日期（預設為系統日期）")
    args = ap.parse_args(argv)

    if args.cmd == "freeze":
        res = freeze(force=args.force)
        print(f"凍結狀態：{res['status']}")
        if res["status"] == "drift":
            print("偵測到研究版本漂移（未覆寫）：")
            for k, v in res["drift"].items():
                print(f"  {k}: {v['frozen']} → {v['now']}")
            print("若確定要更新版本，請加 --force。")
    elif args.cmd == "cutoff":
        res = register_cutoff(args.date)
        d = res["snapshot"]
        print(("已記錄" if res["registered"] else "已存在，保留原值")
              + f"：{d['cutoff_date']}（價量 {d['n_price_rows']} 列、"
              f"可見新聞 {d['n_news_fetched_upto']} 則、{d['notes']}）")
    print()
    print(to_text(today=args.today))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
