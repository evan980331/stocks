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
python -m twstock.cli ingest      :: 抓日K／法人／融資券／月營收／新聞／公司行動（含 0050 基準）
python -m twstock.cli run         :: 特徵 + 規則打分（+ 可選 ML）
python -m twstock.cli backtest    :: 回測（含交易成本與基準比較）
python -m twstock.cli validate    :: 樣本外驗證 + 成本敏感度（產出完整報告）
python -m twstock.cli notify      :: Telegram 通知（可選）
```

### 樣本外驗證與成本敏感度

```bat
python -m twstock.cli validate
python -m twstock.validation --quick      :: 只比較基準策略，較快
```

報告包含：時間序列切分與 purge/embargo、樣本與交易數、報酬與 MDD、Sharpe、
勝率與盈虧比、交易成本前後績效、與 0050 基準的比較、不同市場期間表現、
資料完整度與測試限制，以及**週轉率與成本假設定義**。
輸出於 `twstock/reports/validation_*.txt`。

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
  costs.py             # 交易成本模型與 6 組情境（含流動性上限）
  strategies.py        # 策略規格：持有期間、進出場門檻、再平衡頻率
  validation.py        # 樣本外驗證：切分、purge/embargo、基準檢查、比較報告
  data/
    finmind_client.py  # FinMind API v4（失敗回 None，與「空資料」區分）
    store.py           # SQLite：INSERT OR REPLACE 冪等寫入，永不 DROP 表
    quality.py         # 無效價格清理、除權息／分割還原與驗證守門
    loader.py          # 價格載入（清理 + 還原，策略與基準一致）
  sentiment/           # Google News RSS + 規則式情緒；專家觀點來源
  features.py          # 技術指標（RSI/KD/MACD/布林/K線型態）+ 籌碼 + 營收 + 情緒
  signals.py           # 規則式多因子打分 + 尺度無關的訊號融合
  ml_model.py          # LightGBM walk-forward（可選，未安裝時明確停用）
  backtest.py          # T+1 開盤成交、成本模型、週轉率指標、0050 基準
  pipeline.py          # 每日流程 + 失敗追蹤（ingest_log）
  worker.py            # 監控 Worker（單一實例鎖、重試、逾時、健康狀態機）
  ops.py               # start / stop / status / health
  app.py               # FastAPI 本地儀表板
  cli.py               # CLI 唯一進入點
tests/                 # pytest：功能、資料品質、時間正確性、成本、驗證、Worker
plan.md                # 工程計畫與進度（活的文件）
docs/STATUS_REPORT.md  # 專案現況盤點報告
```

## 價格與成本處理

- 價格一律經 `data/loader.load_price_panel(adjusted=True)`：非正數價格視為缺值，
  並以 FinMind 分割／股利事件對事件日之前價格後調（策略與基準一致）。
  與價格序列不符的事件會被捨棄並記錄（保守處理）。
- 成本由 `costs.py` 的 `CostModel` 管理（手續費、證交稅、滑價、低消、
  成交量佔比上限），可切換多組情境做敏感度分析。
- **週轉率**＝`(買進名目 + 賣出名目) / 初始資金`（累積、未年化、買賣各計一次）；
  另提供年化與單邊版本。週轉率**不等於**交易次數。

資料庫、報告、PID 與日誌都放在 `twstock/data_store/`（已被 Git 忽略）。

---

## 資料來源與限制

| 來源 | 內容 | 更新 | 限制 |
|---|---|---|---|
| FinMind API v4 | 日K、法人、融資券、月營收、0050 基準 | 盤後 | 免費版限流；`FINMIND_TOKEN` 可選 |
| FinMind（公司行動） | 分割／減資（`TaiwanStockSplitPrice`）、股利（`TaiwanStockDividend`） | 事件發生時 | 免費層可用；`TaiwanStockPriceAdj` 還原股價為付費層 |
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
