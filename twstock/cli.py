"""CLI: python cli.py {ingest|run|backtest|notify|dashboard|train}"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description="台股預測系統 CLI")
    ap.add_argument("cmd", choices=["ingest", "run", "backtest", "notify",
                                    "dashboard", "train", "all",
                                    "experts", "ai", "advice"])
    ap.add_argument("--stocks", nargs="*", default=None,
                    help="指定股票代碼（預設為整個 pool）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.cmd in ("ingest", "all"):
        from .pipeline import ingest
        ingest(args.stocks)
    if args.cmd in ("run", "all"):
        from .pipeline import run_scores
        df = run_scores(args.stocks)
        print(df.tail(20) if not df.empty else "無資料")

    if args.cmd in ("backtest",):
        from .pipeline import run_backtest_cli
        print(run_backtest_cli())

    if args.cmd in ("notify", "all"):
        from .pipeline import daily_notify
        daily_notify()

    if args.cmd in ("dashboard",):
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
        from .pipeline import load_panel
        from .ml_model import train_walk_forward
        panel, _ = load_panel()
        model, meta = train_walk_forward(panel)
        print(meta)

    return 0


if __name__ == "__main__":
    sys.exit(main())
