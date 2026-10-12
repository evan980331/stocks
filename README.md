# 台股預測與交易訊號系統（TW Stock Predictor）

一套「資料管線 → 多因子演算法 → 新聞情緒 → 回測 → 訊號通知 → 儀表板」的台股波段系統。

```
twstock/
  config.py       # 設定（股票池、參數、交易成本）
  data/           # FinMind 客戶端 + SQLite 儲存
  sentiment/      # Google News RSS + 情緒打分（LLM 或規則）
  features.py     # 技術指標 + 籌碼特徵
  signals.py      # 規則式打分（baseline）+ LightGBM（可選）
  ml_model.py     # LightGBM walk-forward 訓練/預測
  backtest.py     # 含手續費/證交稅/滑價的回測
  notify.py       # Telegram 推送
  app.py          # FastAPI + 本地儀表板
  pipeline.py     # 每日流程：ingest → features → score → notify
  cli.py          # CLI 進入點
dashboard/        # Next.js 儀表板（Phase 1 後半）
data_store/       # SQLite 資料庫（自動建立）
reports/          # 回測與每日报告輸出
```

## 快速開始

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # 填入 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID（可選）

# 1) 抓資料（日K、法人、融資券、月營收、新聞）
python cli.py ingest

# 2) 計算特徵與訊號
python cli.py run

# 3) 回測（含台股交易成本）
python cli.py backtest

# 4) 推送訊號到 Telegram（需 .env）
python cli.py notify

# 5) 啟動儀表板 http://localhost:8000
python cli.py dashboard
```

ML 模式（安裝 lightgbm 後自動啟用）：`python cli.py train`

## 訊號邏輯
- 因子：動能、均線排列、RSI、量能異常、法人連續買超、融券回補壓力、月營收 YoY、新聞情緒（3日衰減）
- Baseline 為規則式打分排序 top-N；ML 層以 LightGBM 預測未來 5 日超額報酬，兩者整合輸出最終訊號
- 所有決策以回測績效為準，不迷信單一模型

## 風險聲明
本專案僅供研究與教育用途，不構成投資建議。回測績效不代表未來報酬。
