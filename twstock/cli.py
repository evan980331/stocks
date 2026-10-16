"""CLI 唯一進入點。

資料/分析：
  ingest | run | backtest | train | notify | experts | ai
  validate（樣本外驗證） | datacheck（資料品質與擷取狀態）
  research（前瞻研究紀錄） | stress（空頭市況壓力測試）

本地服務生命週期：
  dashboard（前景） | worker（前景）
  start <dashboard|worker> | stop [name|all] | restart <name> | status | health

範例：
  python -m twstock.cli ingest
  python -m twstock.cli run
  python -m twstock.cli datacheck
  python -m twstock.cli stress
  python -m twstock.cli start worker
  python -m twstock.cli health
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DATA_CMDS = ["ingest", "run", "backtest", "validate", "notify", "dashboard",
             "train", "all", "experts", "ai", "advice", "datacheck",
             "research", "stress"]
OPS_CMDS = ["start", "stop", "restart", "status", "health", "worker"]


def main() -> int:
    ap = argparse.ArgumentParser(description="台股本地量化系統 CLI")
    ap.add_argument("cmd", choices=DATA_CMDS + OPS_CMDS)
    ap.add_argument("name", nargs="?",
                    help="服務元件名稱（start/stop/restart 用，可為 dashboard|worker|all）")
    ap.add_argument("--stocks", nargs="*", default=None,
                    help="指定股票代碼（預設為整個 pool）")
    ap.add_argument("--host", default="127.0.0.1",
                    help="儀表板綁定位址（預設僅本機 127.0.0.1）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--interval", type=int, default=1800,
                    help="Worker 每輪間隔秒數")
    ap.add_argument("--once", action="store_true", help="Worker 只執行一次")
    args = ap.parse_args()

    # ---- 服務生命週期 ----
    if args.cmd == "start":
        from . import ops
        if not args.name:
            ap.error("start 需要元件名稱：dashboard | worker")
        if args.name == "all":
            for comp in ops.COMPONENTS:
                ops.start(comp, ["--interval", str(args.interval)] if comp == "worker" else [])
            return 0
        extra = ["--port", str(args.port)] if args.name == "dashboard" else \
            ["--interval", str(args.interval)]
        ops.start(args.name, extra)
        return 0

    if args.cmd == "stop":
        from . import ops
        if args.name in (None, "all"):
            ops.stop_all()
        else:
            ops.stop(args.name)
        return 0

    if args.cmd == "restart":
        from . import ops
        if not args.name or args.name == "all":
            ap.error("restart 需要元件名稱：dashboard | worker")
        ops.stop(args.name)
        extra = ["--port", str(args.port)] if args.name == "dashboard" else \
            ["--interval", str(args.interval)]
        ops.start(args.name, extra)
        return 0

    if args.cmd == "status":
        from . import ops
        print("=== 服務狀態 ===")
        ops.print_status()
        return 0

    if args.cmd == "health":
        from . import ops
        ops.print_health()
        return 0

    if args.cmd == "worker":
        from .worker import run_forever
        return run_forever(interval=args.interval, force_once=args.once)

    # ---- 資料與分析 ----
    if args.cmd in ("ingest", "all"):
        from .pipeline import ingest
        ingest(args.stocks)
    if args.cmd in ("run", "all"):
        from .pipeline import run_scores
        df = run_scores(args.stocks)
        print(df.tail(20) if not df.empty else "無資料")

    if args.cmd == "backtest":
        from .pipeline import run_backtest_cli
        print(run_backtest_cli())

    if args.cmd == "datacheck":
        # 資料擷取狀態與新聞品質（只描述可信度，不產生任何績效結論）
        from .data import report
        print(report.build_report_text())

    if args.cmd == "research":
        # P2.8 §6：前瞻研究紀錄（凍結版本、時間切點、觀察期狀態）
        from . import research
        return research.main([args.name] if args.name else [])

    if args.cmd == "stress":
        # P2.9：空頭市況壓力測試（期間辨識、基準比較、訊號檢驗）
        from . import stress
        return stress.main([])

    if args.cmd == "validate":
        # 樣本外驗證（P2.1）：時間序列切分 + 基準檢查 + 策略比較 + 成本敏感度
        from .validation import run_validation
        report = run_validation()
        print(report.to_text())
        import datetime as _dt
        from .config import REPORTS_DIR
        REPORTS_DIR.mkdir(exist_ok=True)
        out = REPORTS_DIR / f"validation_{_dt.datetime.now():%Y%m%d_%H%M%S}.txt"
        out.write_text(report.to_text() + "\n", encoding="utf-8")
        print(f"\n（報告已寫入：{out}）")

    if args.cmd in ("notify", "all"):
        from .pipeline import daily_notify
        daily_notify()

    if args.cmd == "dashboard":
        import uvicorn
        from .app import app
        uvicorn.run(app, host=args.host, port=args.port)

    if args.cmd == "experts":
        from .pipeline import ingest_experts
        print(f"專家觀點 {ingest_experts()} 則")

    if args.cmd in ("ai", "advice"):
        from .ai import gemini
        if gemini.has_key():
            res = gemini.ask(" ".join(args.stocks) if args.stocks else None)
            print(res["text"] if res["ok"] else f"失敗：{res['error']}")
        else:
            print(gemini.rule_based_advice())

    if args.cmd == "train":
        from .pipeline import load_benchmark, load_panel
        from .ml_model import train_walk_forward
        panel, _ = load_panel()
        model, meta = train_walk_forward(panel, load_benchmark())
        print(meta)

    return 0


if __name__ == "__main__":
    sys.exit(main())
