"""Vercel Serverless 入口：暴露 FastAPI app。"""
import os
import sys

# 專案根目錄加入 path，讓 twstock 可被 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Vercel 上檔案系統唯讀，DB 指向打包的唯讀快照
os.environ.setdefault("TWSTOCK_READONLY", "1")
os.environ.setdefault(
    "TWSTOCK_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "twstock", "data_store", "twstock_v3.db"),
)

from twstock.app import app  # noqa: E402
