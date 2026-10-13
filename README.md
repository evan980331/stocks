# 台股本地量化系統（TW Stock Quant, local-first）

一套在**單一本地電腦**運作的台股量化系統：
`資料取得 → 清理/驗證 → 特徵工程 → 規則策略 →（可選）ML → 融合 → 回測 → 監控 Worker → 本地儀表板`。

> **本地優先**：不部署 Vercel／Cloudflare，不使用 Tunnel、公開網域或反向代理，
> 不需要公開入口才能運作。GitHub 僅作原始碼版本管理與備份。
>
> **風險聲明**：本專案僅供研究與教育用途，不構成投資建議；回測績效不代表未來報酬。
> **正式交易預設關閉**（`TWSTOCK_MODE=paper`），需人工核准才會啟用。

---

## 快速開始（Windows）

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

或直接雙擊 **`run.bat`** 使用互動選單。

### 一次性資料與分析

```bat
python -m twstock.cli ingest      :: 抓日K／法人／融資券／月營收／新聞（含 0050 基準）
python -m twstock.cli run         :: 特徵 + 規則打分（+ 可選 ML）
python -m twstock.cli backtest    :: 回測（含交易成本與基準比較）
python -m twstock.cli notify      :: Telegram 通知（可選）
```

### 服務生命週期

```bat
python -m twstock.cli start dashboard   :: 儀表板 http://127.0.0.1:8000
python -m twstock.cli start worker      :: 監控 Worker（收盤後自動更新）
python -m twstock.cli status            :: 服務狀態
python -m twstock.cli health            :: 系統健康狀態
python -m twstock.cli stop all          :: 安全停止所有服務
```

停止流程會先送出停止旗標／可攔截訊號讓行程收尾，逾時才強制終止；
資料庫寫入為交易式且冪等，不會因重新啟動而重複寫入或遺失既有紀錄。

---

## 專案結構

```
twstock/
  config.py            # 設定：股票池、交易成本、融合權重（單一來源）
  utils.py             # 共用工具（retry）
  risk.py              # 風險限額、執行模式、Kill Switch（正式交易預設關閉）
  data/
    finmind_client.py  # FinMind API v4（失敗回 None，與「空資料」區分）
    store.py           # SQLite：INSERT OR REPLACE 冪等寫入，永不 DROP 表
  sentiment/           # Google News RSS + 規則式情緒；專家觀點來源
  features.py          # 技術指標（RSI/KD/MACD/布林/K線型態）+ 籌碼 + 營收 + 情緒
  signals.py           # 規則式多因子打分 + 尺度無關的訊號融合
  ml_model.py          # LightGBM walk-forward（可選，未安裝時明確停用）
  backtest.py          # T+1 開盤成交、含成本、含 0050 基準與完整指標
  pipeline.py          # 每日流程 + 失敗追蹤（ingest_log）
  worker.py            # 本地監控 Worker（心跳／安全停止／崩潰恢復）
  ops.py               # start / stop / status / health
  app.py               # FastAPI 本地儀表板
  cli.py               # CLI 唯一進入點
tests/                 # pytest：功能、資料品質、時間正確性、風控
plan.md                # 工程計畫與進度（活的文件）
docs/STATUS_REPORT.md  # 專案現況盤點報告
```

資料庫、報告、PID 與日誌都放在 `twstock/data_store/`（已被 Git 忽略）。

---

## 資料來源與限制

| 來源 | 內容 | 更新 | 限制 |
|---|---|---|---|
| FinMind API v4 | 日K、法人、融資券、月營收、0050 基準 | 盤後 | 免費版限流；`FINMIND_TOKEN` 可選 |
| Google News RSS | 個股新聞 | 每次 ingest | 公開 RSS，非正式 API |
| YouTube RSS / Google News | 專家觀點 | 手動 | 公開 RSS |

**行情為延遲/收盤後資料，並非即時行情**。儀表板一律顯示「最新行情日期」與延遲天數。

缺值／失敗處理原則：抓取失敗一律記為失敗（不回填 0），資料不足時標記「無法判斷」，
不產生看似正常的訊號。

---

## 啟用 ML 層（可選）

```bat
pip install lightgbm scikit-learn
```

ML 需要 0050 基準資料（`ingest` 會一併抓取）。未安裝時系統會明確顯示
「ML 停用（lightgbm_not_installed）」，不會以 0 冒充模型判讀。

---

## 測試

```bat
python -m pytest -q
```

涵蓋：特徵計算、規則打分、情緒、寫入冪等（不覆蓋其他標的）、ingest 失敗處理、
ML 標籤、訊號融合尺度、回測時間正確性、風險限制與 Kill Switch。

---

## 相關文件

- [`plan.md`](plan.md)：工程計畫、進度、已知問題與下一步。
- [`docs/STATUS_REPORT.md`](docs/STATUS_REPORT.md)：專案現況盤點與缺陷證據。
