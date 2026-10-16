# plan.md — 台股本地量化系統工程計畫

> 本文件是**反映實際進度的工作文件**，不是願景文件。
> 每次完成一個可獨立驗證的工作單位就更新本檔，並以可重現證據區分 `DONE` 與 `VERIFIED`。
> 最後更新：2026-10-13（Phase P0/P1 首輪）

---

## 1. 專案目標與系統邊界

**目標**：一套可在**單一本地電腦**持續運作的台股量化系統：資料取得 → 清理/驗證 → 特徵工程 →
策略/模型 → 回測/樣本外驗證 → 模擬交易 → 監控 Worker → 風險控制 → 本地儀表板。

**系統邊界（明確不做）**：

- 不部署到 Vercel / Cloudflare（Workers / Pages / Tunnel / 反向代理 / 公開網域）。
- 不把常駐監控或交易 Worker 放到任何 Serverless 平台。
- 不把券商憑證暴露給前端、瀏覽器、Git 或外部服務。
- 不宣稱本機在關機或斷網時仍能交易（本地優先 = 本地負責可用性）。
- 正式交易（Live Trading）**預設關閉**，未經使用者明確核准不得啟用。

**核心原則**：本地優先、數據可信、風險優先、可驗證、可恢復、可追蹤。

**版本管理**：GitHub 僅作為原始碼版本管理與備份（`https://github.com/evan980331/stocks`），
不負責執行交易程式。

---

## 2. 現有架構與模組關係

```
twstock/
  config.py       # 設定：股票池、交易成本、參數（單一來源）
  utils.py        # 共用工具（retry_requests、today_str）
  data/
    finmind_client.py  # FinMind API v4 抓取：日K/法人/融資券/月營收
    store.py           # SQLite 儲存層（price/institution/margin/revenue/news/signals/experts/meta）
  sentiment/
    news_rss.py        # Google News RSS 抓取＋規則式情緒打分
    sentiment.py       # 新聞入庫聚合
    experts.py         # 名嘴/專家觀點來源（YouTube RSS / Google News）
  features.py     # 技術指標（RSI/KD/MACD/布林/K線型態）+ 籌碼 + 營收 + 新聞情緒特徵
  signals.py      # 規則式多因子打分（baseline）；pick_top_n
  ml_model.py     # LightGBM walk-forward（可選；未安裝時停用）
  backtest.py     # 含台股成本的回測
  notify.py       # Telegram 推送
  pipeline.py     # 每日流程：ingest → features → score → (ML) → 入庫 → notify
  ai/gemini.py    # Gemini 側邊建議（可選；無金鑰時規則式降級）
  app.py          # FastAPI 本地儀表板
  cli.py          # CLI 唯一進入點
```

**資料流**：`FinMind / Google News` → `finmind_client` / `news_rss` → `SQLite(store)`
→ `features.build_features` → `signals.score_stocks`（+ 可選 `ml_model`）
→ `signals` 表 → `app.py` 儀表板 / `notify.py` Telegram / `backtest.py` 回測。

**邏輯分層 ≠ 微服務**：目前規模適合「單一 Python 應用 + 獨立 Worker 行程」，
不拆分微服務。Worker 由 `twstock/ops.py` 管理（見 §3）。

---

## 3. 本地執行方式（Windows 優先）

```bat
:: 一次性安裝
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

:: 資料與分析
python -m twstock.cli ingest        :: 抓資料（含 0050 基準）
python -m twstock.cli run           :: 特徵 + 規則打分 + （可選）ML
python -m twstock.cli backtest      :: 回測（含成本與基準比較）
python -m twstock.cli notify        :: Telegram（可選）

:: 服務生命週期（PID 檔管理，可安全重啟）
python -m twstock.cli start dashboard
python -m twstock.cli start worker
python -m twstock.cli status
python -m twstock.cli health
python -m twstock.cli stop
```

Windows 一鍵包裝：`run.bat`。安全停止會寫入 `data_store/state/*.stop` 讓 Worker 於
檢查點自行收尾，不會在交易狀態未落地時被強制中止。

---

## 4. 資料來源與更新策略

| 來源 | 內容 | 更新頻率 | 授權/限制 | 失敗處理 |
|---|---|---|---|---|
| FinMind API v4（`TaiwanStockPrice`） | 日K、成交值 | 每日盤後 | 免費版有限流；`FINMIND_TOKEN` 可選 | 失敗回 `None`（非空表），計入失敗紀錄 |
| FinMind（`TaiwanStockInstitutionalInvestorsBuySell`） | 三大法人買賣 | 每日盤後 | 同上 | 同上 |
| FinMind（`TaiwanStockMarginPurchaseShortSale`） | 融資融券 | 每日盤後 | 同上 | 同上 |
| FinMind（`TaiwanStockMonthRevenue`） | 月營收 YoY | 每月 10 日前後 | 同上 | 同上 |
| Google News RSS | 個股新聞標題 | 每次 ingest | 公開 RSS，非正式 API，可能限流 | 空清單＝無新聞（**不覆蓋其他股票**） |
| YouTube RSS / Google News | 專家觀點 | 手動 | 公開 RSS | 略過 |
| 基準 0050 | 超額報酬基準 | 每日 | 同 FinMind | 失敗時 ML 標籤標記為不可用 |

**行情延遲揭露**：本系統使用**延遲/收盤後**資料，不是即時行情。儀表板須顯示
「最後資料日期」與延遲狀態，不得冒充即時。

**增量與可重跑**：所有寫入改為「先刪除該鍵、再 append」，**禁止 `if_exists="replace"`**
（會整表 DROP 造成跨股票資料遺失）。同一任務重跑不得破壞其他標的資料。

---

## 5. 模型、策略及回測設計

### 5.1 策略分層（各自可獨立啟停、比較）

`features` → `signals（規則多因子）` → `ml_model（LightGBM 超額報酬）` → `fusion` → 風險檢查 → 執行。

### 5.2 訊號融合（尺度問題）

規則分數定義於 `[-1, 1]`；ML 輸出為**預測未來 5 日超額報酬率**（約 ±0.03）。
兩者尺度不同，**不得直接加權相加**。改為：

1. `signals.fuse_scores()`：先把每個訊號做**當日橫斷面排名百分位**映射到 `[-1, 1]`，
   再加權合併（權重正規化）。預設 `w_rule=0.6, w_ml=0.4`。
2. 融合方法必須與單一策略做**樣本外比較**後才可作為預設（目前 ML 未安裝 → 無法驗證，見 §10）。

### 5.3 特徵可用時間（point-in-time）

- 技術/價量特徵：以當日**收盤後**才算得，訊號僅能在**次一交易日**執行（回測亦同）。
- 月營收：以**實際公告日**為可用時間，不用「資料所屬月份」（目前以 `date` 欄位為公告日）。
- 新聞情緒：以發佈日 + 3 日衰減權重（1.0 / 0.66 / 0.4）。
  **P2.7 起**：發布時間必須來自來源（`timestamp_source='source'`）；取不到時
  `published` 記為 NULL 並排除於衰減窗口（**不得**以抓取日代替，否則等於憑空
  生成時間戳）。來源時間晚於抓取時間者亦不採用。風散方向只往未來（lag ≥ 0），
  故發布日之前的交易日不會看到該新聞。

### 5.4 回測時間正確性（已修正）

- 訊號於 T 日收盤產生 → **T+1 開盤進場**（避免 look-ahead）。
- 持有 `hold_days` 個交易日 → 於到期日**開盤出場**。
- 漲停無法成交：`open >= prev_close * (1 + limit_pct)` 時跳過進場。
- 成本：手續費（2.8 折，低消 20 元）＋證交稅（賣出 0.3%）＋雙邊滑價 0.1%。
- 提供**買進持有（0050）基準**比較。

### 5.5 評估指標

累積報酬、CAGR、Sharpe、Sortino、MDD、勝率、平均獲利/虧損、獲利因子、
交易次數、週轉率、成本佔比、曝險比例、基準比較、樣本外表現、參數敏感度。

**週轉率定義**（`backtest.py` docstring 與報告 [8] 節）：
令 `notional_traded = 買進名目 + 賣出名目`（買賣各計一次）：
`turnover = notional_traded / initial_capital`（累積、未年化），
`turnover_one_way = turnover / 2`，`turnover_annualized = turnover / 年數`，
`portfolio_turnover_annualized = (notional/2) / 平均權益 / 年數`。
**週轉率不等於交易次數**（後者另列）。

### 5.6 除權息／分割還原（策略與基準一致）

價格一律經 `data/loader.load_price_panel(adjusted=True)`：
非正數價格→缺值；以 FinMind 分割／股利事件建立調整因子，對事件日之前價格後調。
若事件與價格序列不符（套用後反而增加不連續）則**捨棄並記錄**，保守處理。

> 即使通過回測也不代表未來獲利。不得只呈現最佳結果。

---

## 6. 風險管理及交易安全規範

**執行模式區分**（各自獨立設定與權限界線）：

| 模式 | 說明 | 下單能力 |
|---|---|---|
| `research` | 特徵/打分/分析 | 無 |
| `backtest` | 歷史回測 | 無 |
| `paper` | 模擬交易（預設） | 模擬券商 |
| `live` | 正式交易（**預設關閉**） | 券商 API，需人工核准 |

**風險限制（集中於 `twstock/risk.py`，數值由設定管理，未經使用者確認不得放寬）**

| 項目 | 預設值 | 說明 |
|---|---|---|
| `max_order_value` | 100,000 TWD | 單筆委託金額上限 |
| `max_position_pct` | 0.10 | 單一股票部位上限（佔總資產） |
| `max_gross_exposure` | 0.60 | 整體持倉上限 |
| `max_daily_loss_pct` | 0.03 | 單日損失上限（觸發熔斷） |
| `max_drawdown_pct` | 0.15 | 最大回撤限制 |
| `max_orders_per_day` | 20 | 單日委託數上限 |
| `min_liquidity_value` | 10,000,000 TWD | 最低 20 日均成交值 |
| `max_data_staleness_days` | 3 | 可接受行情延遲 |
| `max_open_orders` | 10 | 最大未完成委託數 |

**強制停止（Kill Switch）條件**：行情過期／來源失效、券商連線異常、資金或持倉無法確認、
本地與券商持倉不一致、疑似重複委託、風險額度超限、交易狀態不明、DB 無法可靠寫入、
系統時間或交易時段判斷異常。

Kill Switch 只「禁止新增正式委託」，**不刪除交易紀錄、不自動大量平倉**。

**冪等與狀態**：每筆委託使用 `client_order_id`（唯一鍵），重啟後先查詢券商狀態，
**狀態不明時不得重送**。

**正式交易權限界線（Agent 自我約束）**：不得自行啟用 live、不得提高風險限額、
不得關閉熔斷、不得用未驗證策略下單、不得將憑證提交 Git、不得為測試送真實委託。

---

### 5.7 擷取狀態與資料可追溯性（P2.7）

每次擷取寫入 `ingest_run`，區分六種狀態：`success_with_data`、`success_empty`、
`partial_success`、`failed`、`skipped`、`unknown`。

- **失敗不得降級成空結果**：`status.classify` 的判定順序讓任何硬性錯誤分類
  （network／timeout／http_status／parse／validation／write）一律得到 `failed`。
- **「來源正常但沒有資料」不是失敗**：舊 `worker.run_cycle` 以「價量列數 > 0」
  判定整輪成敗，會把這種情況誤判為失敗，反之亦無法看出部分失敗。
- **不可追溯就標 `unknown`**：舊資料或非結構化回傳絕不補猜成功。
- 紀錄不含任何 API 金鑰；所有錯誤訊息寫入前經 `status.redact`。

---

## 7. 各階段開發計畫（P0 → P8）

| 階段 | 內容 | 完成條件 | 狀態 |
|---|---|---|---|
| **P0** | 現況盤點（程式/資料/模型/回測/Git） | 產出 `docs/STATUS_REPORT.md` 並標明證據 | `VERIFIED` |
| **P1** | 資料品質與基礎架構（覆蓋、時間戳、重跑、錯誤處理） | 新聞不再互相覆蓋、ingest 失敗不得記為成功、寫入可重跑 | `VERIFIED`（W3／W4；擷取狀態治理於 P2.7／W23 完成） |
| **P2** | 策略與回測可信度 | 無 look-ahead、成本模型合理、有基準比較、可重現 | `VERIFIED`（P2.1 樣本外 + P2.2 成本已驗證） |
| **P3** | 本地監控 Worker | 可持續執行、可觀測、可重啟恢復 | `VERIFIED`（P3.1 硬化完成） |
| **P4** | 風險控制與模擬券商 | 風控、委託狀態機、持倉同步、冪等、Kill Switch | `TODO` |
| **P5** | 模擬交易與故障測試 | 完整生命週期 + 失敗情境測試 | `TODO` |
| **P6** | 正式券商 API 整合準備 | 券商評估、介面、憑證安全（live 仍關閉） | `BLOCKED`（需使用者選券商） |
| **P7** | 人工核准與受控正式交易 | 使用者明確核准 + 驗收通過 + 保守限額 | `BLOCKED`（需人工核准） |
| **P8** | 長期監控與持續改善 | 漂移/衰退偵測、自動停用失效策略 | `TODO` |

---

## 8. 已完成 / 已驗證工作紀錄

### W1 — 專案盤點 ✅ `VERIFIED`
- 執行：閱讀全部 `twstock/**` 原始碼、`README`、`requirements.txt`、`vercel.json`、`api/index.py`、4 個測試檔；檢查 Git 狀態；檢視 3 個 SQLite DB 實際內容。
- 結果：見 `docs/STATUS_REPORT.md`。基礎測試 **14 passed**。

### W2 — 移除雲端部署依賴 ✅ `VERIFIED`
- 移除 `vercel.json`、`api/index.py`、`.vercelignore`、`public/robots.txt` 與本機 `.vercel/`。
- `config.is_readonly()` 不再偵測 `VERCEL` 環境變數（保留通用 `TWSTOCK_READONLY` 保護模式）。
- **附帶發現**：`.vercel/.env.preview.local` 內含 `VERCEL_OIDC_TOKEN` 憑證，已一併移除
  （該目錄本就被 Git 忽略，未進入版控）。
- 驗證：測試全綠、儀表板本機啟動 `GET /` 200。

### W8 — 修正 `cli run` 崩潰 ✅ `VERIFIED`
- **問題**：`load_panel()` 存取不存在的 `CONFIG.vol_ma_window`（正確為 `volume_ma_window`），
  導致 `python -m twstock.cli run` 直接 `AttributeError`，主要分析流程無法執行。
- **修復**：更正欄位名。實跑 `cli run` 成功寫入 9700 筆訊號。

### W9 — 儀表板建議動作一致性 ✅ `VERIFIED`
- **問題**：`signals.action` 從未落庫（全為 NULL），HTML 一律 `fillna("HOLD")`，
  即使分數 +0.58 仍顯示「持有」。
- **修復**：新增 `app._ensure_actions()` 即時計算，HTML 與 `/api/signals` 一致。
- **驗證**：`3711 0.5793 BUY`。

### W10 — 本地服務生命週期與風險控制 ✅ `VERIFIED`
- 新增 `twstock/ops.py`（start/stop/restart/status/health，PID 檔管理）、
  `twstock/worker.py`（心跳、安全停止旗標、崩潰恢復）、`twstock/risk.py`（限額、模式、Kill Switch）、
  `run.bat`（Windows 一鍵選單）。
- **驗證**：`start dashboard` → `status` 顯示執行中 → `GET /` 200 → `GET /api/health` 回傳真實狀態
  → `stop` 後埠關閉；`tests/test_ops_worker.py`、`tests/test_risk.py` 通過。

### W3 — 新聞覆蓋 bug 修復 ✅ `VERIFIED`
- **問題**：`sentiment.update_news_for()` 使用 `to_sql(if_exists="replace")`，每抓一檔股票就
  **DROP 整個 `news` 表**，導致先前所有股票新聞遺失。實測 v3 DB `news` 僅剩 `3231` 5 筆。
- **修復**：`store.write_df()` 改為「依鍵刪除 + append」，永不 DROP 表。
- **驗證**：`tests/test_store_write.py` 證明連續寫入兩檔股票後兩者資料皆存在。

### W4 — ingest 失敗不得記為成功 ✅ `VERIFIED`
- **問題**：`fetch_finmind` 失敗回空表，`ingest` 仍無條件 `set_meta("last_ingest", today)`，
  系統會誤判資料新鮮。
- **修復**：`finmind_client` 失敗回 `None`（區別「失敗」與「真的沒資料」）；`ingest` 逐檔記錄
  成功/失敗，寫入 `ingest_log`，僅在有實際成功時更新 `last_ingest`。
- **驗證**：`tests/test_ingest_status.py`。

### W5 — 0050 基準與 ML 標籤修復 ✅ `DONE`（ML 未安裝，未能端到端驗證）
- **問題**：`ml_model._make_labels` 依賴 `0050` 基準，但 `ingest`/`load_panel` 從不抓/納入 0050
  → 標籤全 NaN → ML 永遠無法訓練（實測 price 表 20 檔，無 0050）。
- **修復**：`ingest` 一併抓取並儲存 0050；`load_panel` 附帶回傳基準序列；
  `ml_model` 明確回報 `enabled/reason/rows`，資料不足時不靜默失敗。
- **驗證**：`tests/test_ml_labels.py`（合成資料）；**端到端待安裝 lightgbm**。

### W6 — 訊號融合尺度修復 ✅ `VERIFIED`
- **問題**：`run_scores` 直接 `rule*0.6 + ml*0.4`，rule∈[-1,1]、ml≈±0.03 → ML 實質被忽略，
  且缺值以 0 填入冒充中性。
- **修復**：新增 `signals.fuse_scores()`，以**當日橫斷面排名百分位**映射到 [-1,1] 再加權；
  缺 ML 時明確退化為規則分數。
- **驗證**：`tests/test_fusion.py`（尺度不變性、缺值處理、單檔當日）。

### W7 — 回測 look-ahead 修復 + 基準與完整指標 ✅ `VERIFIED`
- **問題**：原回測以 T 日收盤特徵訊號卻在 **T 日開盤**進場（look-ahead）；無基準比較。
- **修復**：改為 **T+1 開盤進場 / 到期日開盤出場**、漲停跳過、新增 0050 買進持有基準，
  以及 Sharpe/Sortino/MDD/勝率/獲利因子/成本/曝險等指標。
- **驗證**：`tests/test_backtest_timing.py`（進場價必須等於次一日開盤）。

---

### W11 — 公司行動未還原（除權息／分割）✅ `VERIFIED`
- **問題**：價格表為 FinMind 原始收盤價，**完全未還原公司行動**。實測 0050 於
  2025-06-18 因 1 拆 4 出現 **−74.8%** 的假崩盤；以該序列算出的基準報酬為
  **−69.94%**（實際為 +22.76%），連帶讓「超額報酬」出現 +67% 的假優勢。
  另有 2603 除息缺口 −13.8%、−12.0% 超過漲跌停幅度。
- **修復**：新增 `twstock/data/quality.py` 與 `loader.py`：
  以 FinMind `TaiwanStockSplitPrice`／`TaiwanStockDividend` 建立調整因子
  （分割 `after/before`、現金股利 `(前收−股利)/前收`、股票股利 `1/(1+配股)`），
  對事件日之前價格做乘法後調；策略與基準**採一致處理**。
- **驗證守門**：只在「能真正降低序列不連續」時才套用因子；與價格序列不符的事件
  一律捨棄並記錄（實測富邦金的股票股利事件屬此類，避免製造 +20.8% 假跳動）。
- **驗證**：`tests/test_data_quality.py` 16 項；0050 分割日報酬由 −74.8% 修正為 +0.9%。

### W12 — 無效價格列（0 元）污染回測 ✅ `VERIFIED`
- **問題**：2317（鴻海）2025-07-30 之 open/high/low/close/volume **全為 0**。
  舊版以 0 為價格，會使市值瞬間歸零、產生假回撤與 ±inf 報酬。
- **修復**：`clean_prices()` 在寫入與讀取兩端將非正數價格視為**缺值**（不是 0）；
  `backtest._prepare` 與 `features.build_features` 同步防護。
- **驗證**：`tests/test_costs_turnover.py::test_zero_price_row_does_not_corrupt_equity`。

### W13 — P2.1 樣本外驗證流程 ✅ `VERIFIED`
- 新增 `twstock/validation.py`：時間序列切分（**不洗牌**）、purge/embargo
  （移除標籤期間跨界的樣本）、可記錄窗口的 walk-forward、只以訓練資料擬合的
  `FeatureScaler`、基準缺失明確回報、`HoldoutGuard` 防止重複使用測試集。
- 報告含：切分期間、樣本/交易數、報酬、MDD、Sharpe、勝率與盈虧比、
  成本前後績效、基準比較、不同市場期間、資料完整度與限制，
  以及**週轉率與成本假設定義**（可稽核）。
- **驗證**：`tests/test_validation.py`；`python -m twstock.validation` 產出完整報告。

### W14 — P2.2 成本模型與週轉率定義 ✅ `VERIFIED`
- 新增 `twstock/costs.py`：`CostModel` + 6 組成本情境（zero/ideal/typical/
  conservative/adverse/severe），含可設定的手續費折扣、低消、證交稅、滑價，
  以及 `participation_rate` 成交量上限。
- **週轉率定義釐清**：舊回報的 94.72x 實為 `(買進名目+賣出名目)/初始資金`，
  **未年化且買賣各計一次**；已改為四項指標（累積雙邊／累積單邊／年化雙邊／
  年化單邊資金），並在回測 docstring 與報告 [8] 節寫明。
- **驗證**：`tests/test_costs_turnover.py`（公式、稅僅賣出、低消、流動性上限、
  成本單調性）。

### W15 — 降低換手的策略變體 ✅ `VERIFIED`
- 新增 `twstock/strategies.py`：`StrategySpec` 支援 `rebalance_days`（再平衡頻率）、
  `buy_threshold`（進場門檻）、`exit_threshold` + `min_hold_days`（訊號緩衝區 +
  最短持有）、`hold_days`（最長持有）。
- 回測新增 `hysteresis` 持倉政策：出場決策只用「當日之前最近一個交易日的分數」，
  仍於當日開盤成交，維持時間正確性。
- **驗證**：`tests/test_costs_turnover.py`（提前出場、最短持有、再平衡降低交易數）。

### W16 — P3.1 Worker 硬化 ✅ `VERIFIED`
- 新增單一實例鎖（PID 檔、死鎖自動回收）、有上限的指數退避重試、
  任務逾時偵測（`task_started_at` 殘留 → 判定上次被中斷）、
  以及 9 種健康狀態機 `derive_state()`。
- **停止語意修正**：原 `ops.stop()` 只要 PID 消失就回報「已安全停止」，
  但實際可能是被訊號直接殺掉（未收尾）。改為：worker 先寫停止旗標等它自行收尾，
  只有逾時才送訊號／強制終止，並**區分回報「已安全停止」與「已強制停止」**；
  行程結束後清理 PID 檔、停止旗標與殘留鎖。
- **驗證**：`tests/test_worker_hardening.py`、`tests/test_ops_lifecycle.py`；
  實機 `start worker` → 狀態「任務執行中／正常運作」→ `stop worker` **1.77 秒**內
  自行收尾，心跳 status=`stopped`、鎖與旗標皆已清除。

### W17 — P2.3 訊號資訊量診斷 ✅ `VERIFIED`
- **成果**：新增 `twstock/diagnostics.py`（`python -m twstock.diagnostics`）。
  逐一登記 12 個底層因子的公式／觀察期間／可得時間／方向／缺值處理，
  計算 Pearson IC 與 Spearman Rank IC（1/5/10 日前瞻報酬）、五分組報酬、
  隨機排序基準、Newey-West 穩健 t、移動區塊自助法信賴區間與 BH FDR 校正。
- **關鍵設計**：前瞻報酬一律 `fwd_h(T) = open(T+1+h)/open(T+1) - 1`
  （T 日收盤得訊號 → T+1 開盤是第一個可成交價），統計單位是**每日橫斷面 IC**
  而不是股票日；`signals.py` 抽出 `_factor_parts` 讓診斷與 `score_row` 共用
  同一份因子計算（重構後 9 項既有測試不變即通過）。
- **驗證**：`tests/test_diagnostics.py`（36 項）+ 真實資料報告 2 份指令退出碼 0。

### W18 — P2.4 策略失效歸因 ✅ `VERIFIED`
- **成果**：新增 `twstock/attribution.py`（`python -m twstock.attribution`）。
  三個控制變因（真實訊號／隨機選股／完美前瞻上限）只改變分數，交易規則、
  成本與資金限制完全相同；另拆解訊號→持倉落差、進場延遲、持有期間、
  週轉來源、曝險、集中度、成本與流動性。
- **配套修正**：`backtest.py` 的每筆交易新增 `signal_date`（供量測延遲），
  不改任何計算結果。
- **驗證**：`tests/test_attribution.py`（22 項）；真實資料報告退出碼 0。

### W19 — P2.5 資料來源比較與遷移方案 ✅ `DONE`（遷移 `BLOCKED`）
- **成果**：`docs/DATA_SOURCES.md`。量化三項已可稽核的限制（survivorship 無法量化、
  新聞因子實質失效並稀釋其他因子、樣本期間不足以支持穩定結論），
  並以已查證來源比較 FinMind／TWSE 官方／TEJ-TQuant／CMoney／Fugle／XQ。
- **結論**：TEJ／TQuant Lab 是唯一已查證同時提供下市股與逐日台灣 50 成分旗標者；
  **未在評估授權前購買任何資料**（依任務要求）。

### W20 — P3.2 Worker 子程序架構評估與原型 ✅ `VERIFIED`（全面遷移 `TODO`）
- **成果**：`twstock/jobs.py` 原型 + `docs/WORKER_SUBPROCESS.md`。
- **實測修正（重要）**：Windows 的 `CTRL_BREAK_EVENT` 不是可攔截中斷，
  而是立即終止（實測子程序以 `0xC000013A` 結束，連 `except KeyboardInterrupt`
  的任務也照被殺）；若當成優雅停止就是製造假的安全停止，故**移除該訊號**，
  改以合作式取消檔為第一段。另實測：對不在獨立程序群組的子程序送該訊號
  會殺掉整個主控台群組（含父程序）。
- **資料完整性實測**：交易中寫入 300 筆後被硬殺 → 交易完整回復、
  `COUNT(*)=0`、`integrity_check=ok`。
- **驗證**：`tests/test_jobs.py`（23 項，含「停止一個任務不誤殺其他任務」）。

### W21 — 新聞因子稀釋與缺值政策 ✅ `DONE`（已界定，待實作修正）
- **發現**：`news` 僅涵蓋 8 個日曆日（237 筆），歷史面板的 `news_sentiment`
  幾乎恆為 0（`fillna(0.0)`），卻仍佔 `WEIGHTS` 的 0.10 分母。
- **影響**：P2.3 診斷顯示該因子有效交易日 0~2 天（判定「樣本不足」），
  且合成 `score` 的 IC（+0.0705）低於單一最強因子月營收 YoY（+0.0987）。
- **狀態**：已登錄為 B17；**P2.6 已完成消融驗證，結論是不建議立即修改**（見 W22）。

### W22 — P2.6 B17 新聞因子消融驗證與合成分數因果診斷 ✅ `DONE`（**不修改正式策略**）
- **成果**：`twstock/ablation.py` + `tests/test_ablation.py`（19 項）+
  `docs/ABLATION_B17.md`；報告 `python -m twstock.ablation -o ablation_b17.txt`。
- **四方案**：A 現行 / B 移除＋重新標準化 / C 固定權重不標準化 / D 可用性閘門；
  另設 **A\* 分解控制組**（僅移除分子、分母維持 baseline），非第五個候選方案。
- **關鍵結構事實（實測推翻先驗直覺）**：`W_A` 在 **483/486** 個交易日跨股票變動
  （0.53~1.05），故分母從 `W_A` 改為 `W_A − 0.10` **不是**單一正比例常數
  （同日散布均值 0.01343）→ **排名真的會改變**（A→B 僅 393/486 日相同，80.9%）。
  已用合成資料寫反例測試驗證。
- **兩個機制必須分開**：
  - **分子**（`fillna(0)`）：情緒非 0 僅 **29 列**，且落在樣本最末端 →
    **該區間 0 天有可用 5 日前瞻報酬** → 分子效應在本資料中**不可估計**（非「弱」）。
  - **分母**（0.10 佔位）：佔據**全部 9,719 列**（100%，不只有新聞的 61 列）→
    全樣本可觀測，但 Rank IC 影響僅 0.0001~0.0009（比 IC 本身小兩個數量級）。
- **統計結論**：86 個配對假設（IC 差異 38、價差差異 38、交易層級日報酬 10）
  → **未校正顯著 0 個、BH 校正後 0 個**；A−B 的 Rank IC 差在 train +0.0006、
  valid −0.0009（**符號翻轉、方向不一致**）。
- **交易層級**：test 段 **A ≡ B ≡ D 完全相同**（持倉 122/122、Jaccard 1.000、
  總報酬 25.63%、120 筆）；train/valid 點估計略優但配對日報酬差異皆不顯著
  （p=0.897 / 0.551）。
- **驗收**：依「只用 train/valid、需一致且可解釋的改善」→ **不達修改門檻，
  保留現行策略並記錄不確定性**。亦未以 test 挑選任何方案。
- **驗證**：`pytest -q` → 253 passed（含新檔 19 項）；
  `ablation`（完整）、`diagnostics --quick`、`cli validate` 皆退出碼 0。
- **新增欄位**：`features.build_features` 新增 `news_present`（僅在有新聞分支加入，
  **向後相容**），使「有記錄 / 情緒 0 / 缺值」三態可分。

### W23 — P2.7 資料擷取狀態治理與新聞資料品質修復 ✅ `VERIFIED`
- **背景**：本階段**不修改策略、不調因子權重、不追求績效**；目標是讓「成功／
  空結果／部分成功／失敗」在被區分，停止無聲失敗污染研究結果。
- **盤點出 6 條失敗路徑**（全部都有程式碼位置，非推測）：
  1. `pipeline.ingest` 對新聞硬編碼 `log_ingest(..., ok=True)` → 抓取失敗記為成功。
  2. `news_rss` 直接呼叫 `feedparser.parse(url)`；feedparser 在網路失敗時**不拋例外**，
     只設 `bozo` 並回傳空 entries → 失敗偽裝成「沒新聞」。
  3. 發布時間取不到時以 `today_str()` 代替 → 憑空生成日期。
  4. `sentiment.update_news_for` 在「沒有新聞」與「抓取失敗」都回 `0` → 四態塌縮成兩態。
  5. `store.write_df` 在欄位無交集時只寫 log 並回 `0`，呼叫端仍記 `ok=True`。
  6. `worker.run_cycle` 只數 `counts > 0` → 混同「來源正常但無資料」與「全部失敗」。
- **解方**：新增 `data/status.py`（狀態詞彙、錯誤分類、`IngestRecord`、機密遮蔽）、
  `data/report.py`（資料品質報告）、`ingest_run` 資料表、`cli datacheck`；
  `news` 表新增 5 個溯源欄位；新聞改用 `requests` 顯式取得 HTTP 後才解析。
- **量測（真實本機 DB，於副本上比較）**：

  | 指標 | 修復前 | 修復後 |
  |---|---|---|
  | `news` 欄位數 | 6 | 11 |
  | 可判定**抓取成敗**的列 | **0** | 237（但全部 = `unknown`） |
  | 可判定**時間戳來源**的列 | **0** | 237（全部 `unknown`） |
  | 新聞列數／正·零·負分數 | 237／16·183·38 | 237／16·183·38 |
  | 特徵面板列數與 `sha` | 9,719／`c0fa6357e4af` | **完全相同** |
  | `signals` 表 `sha` | `ebfb28c2618c` | **完全相同** |
  | 面板新聞覆蓋率 | 0.628% | **0.628%**（不變） |

- **關鍵結論**：**特徵與訊號逐位元不變**→ schema 變更沒有造成非預期差異，
  也沒有、也不該提升任何預測能力。真正改變的是「**可信度**」：
  修復前對抓取成敗與時間戳來源的可判定性都是 **0**。
- **無聲失敗的直接證據**：`ingest_log` 163 筆**全部 `ok=1`**（其中新聞 20/20 皆為成功），
  這是舊程式硬編碼的結果而非證據 → 回填 `ingest_run` 時**一律標為 `unknown`**。
- **驗證**：`pytest -q` → **303 passed**（上階段 253，+50 新增檔）；
  `cli datacheck`／`validate`／`health`／`status`／`--help` 退出碼皆為 **0**。
- **額外修正（本階段發現）**：測試套件原本會以可寫模式開啟**正式 DB**
  （`test_should_run_outside_trading_window` 未使用 `temp_db`）→ 執行 `pytest`
  就會對使用者資料觸發遷移（實測時間戳 `2026-10-15T00:11:04`）。
  已改為 autouse 隔離，並實測確認跑完 `pytest` 後真實 DB 雜湊**不變**。

### W24 — P2.9 空頭市況壓力測試與防禦能力評估 ✅ `VERIFIED`
- **成果**：`twstock/stress.py`（新）、`tests/test_stress.py`（30 項）、
  `docs/BEAR_MARKET_STRESS_TEST.md`、`cli stress`；
  報告 `reports/stress_*.txt`。**未修改任何策略參數**。
- **資料可行性（實測）**：個股 9,719 列 / 20 檔 / **2024-10-14 ~ 2026-10-13
  （486 交易日）**；0050 529 列 / 2024-08-01 起 → **基準比個股多 48 個交易日**，
  該段無法做策略比較（歷史缺口，未補造）。每日有效檔數最少 19（僅 1 天）、
  價格品質違規 0、公司行動套用 41／捨棄 0。
- **推翻既有敘述（最重要）**：`validation`／`diagnostics`／`attribution` 過去一律
  輸出「樣本僅涵蓋單一多頭市況，未包含空頭／高波動」。實測**不成立**：
  0050 自高點最大回撤 **−27.48%**（2025-04-09）、20 日年化波動最高 **62.09%**。
  已改為由實測推導（`validate` 現在直接印出 −27.48%），並指向本報告。
- **壓力期間（門檻事前固定，辨識為事後描述、期間內交易只用當時已知資訊）**：
  回撤 ≥10% 共 4 段（其中 ≥15% 2 段：2025-01-07→2025-04-09 −27.48%、
  2026-06-22→2026-07-30 −15.88%）；高波動（20 日年化 ≥30%）129 日、
  構成 4 段；空頭狀態（20 日與 60 日同時為負）39 日 1 段。
  **在可評估範圍內獨立的嚴重事件只有 2 段**，不以期間數累加證據。
- **核心量測（同一規則／成本／資金）**：下跌段 3 段策略 −7.97%／−4.42%／−1.26%，
  皆優於 0050（+19.51／+6.42／+14.62pp），策略 MDD −8.79% ~ −3.95%（0050 −27.48%）；
  **但平均曝險僅 43.5%** → 以「等曝險基準」（同比例持有 0050）拆解後，
  **選股效果僅 +5.16／+0.11／+5.95pp 且落在隨機選股分布內（百分位 27／36／70.5）**。
  抗跌主要來自低曝險，**不是選股**。
- **代價在急彈段**：4/4 高波動期間全部落後 0050，最嚴重 2026-03-09~2026-05-06
  （策略 +4.94% vs 0050 +30.10%，超額 **−25.16pp**）；全段（前高→復原）3 段僅 1 段為正。
- **訊號檢驗**：合成分數 Rank IC（h=5）在**下跌段 −0.043（NW t=−1.18、p=0.237，不顯著）**、
  在**非壓力日 +0.103（t=3.88、p<0.001）**；五分組價差由 +1.125% 轉為 −0.574%
  （扣來回成本 −1.154%）。→ **空頭期間沒有可辨識的選股資訊**。
  高波動各段符號互相矛盾（`rev_yoy` +0.239 與 −0.113 並存）→ 不穩定，不可宣稱結論。
- **統計紀律**：21 個檢定屬同一組多重檢定，未做 FDR 校正者不得單獨宣稱顯著；
  短窗年化一律 N/A，Sharpe 僅供排序；每視窗尾端 120 列訊號因無法在窗內完成
  一回合而排除（截斷，非前視偏差）。
- **驗證**：`pytest -q` → **392 passed**（上階段 362，+30）；`compileall` 退出碼 0；
  `stress`／`datacheck`／`validate`／`research`／`status`／`health` 全部退出碼 0；
  真實 DB md5 前後**皆為 `3523521e48acbe28ff4f6ea5423c5106`**（未變更，未做 migration）。

## 9. 測試方式及驗證結果

```bat
python -m pytest -q
```

| 測試檔 | 涵蓋 |
|---|---|
| `tests/test_features.py` | 技術指標、特徵合併、新聞衰減、`news_present` 三態（W22） |
| `tests/test_ablation.py` | B17 消融：方案 A ≡ `score_row`、代數恆等式、方案 D 三態、
  **W 同日常數 → 排名相同**、**W 跨股變動 → 排名改變**、配對差異檢定、
  分組價差、持倉/交易差異、門檻翻轉精確計數（W22，19 項） |
| `tests/test_signals.py` | 規則打分、排序 |
| `tests/test_sentiment.py` | 情緒打分、連結解碼 |
| `tests/test_backtest.py` | 回測基本行為 |
| `tests/test_store_write.py` | 寫入不覆蓋其他標的（W3） |
| `tests/test_ingest_status.py` | 失敗不得記為成功（W4） |
| `tests/test_ml_labels.py` | 標籤基準與缺失處理（W5） |
| `tests/test_fusion.py` | 融合尺度（W6） |
| `tests/test_backtest_timing.py` | 無 look-ahead（W7） |
| `tests/test_data_quality.py` | 無效價格、除權息／分割還原與驗證守門（W11/W12） |
| `tests/test_validation.py` | 切分、purge/embargo、轉換不洩漏、基準處理（W13） |
| `tests/test_costs_turnover.py` | 成本模型、週轉率公式、成本單調性、換手變體（W14/W15） |
| `tests/test_worker_hardening.py` | 鎖、重試、逾時、健康狀態、生命週期（W16） |
| `tests/test_ops_lifecycle.py` | 停止語意與狀態清理（W16） |
| `tests/test_ingest_governance.py` | 擷取狀態治理：六態區分、新聞四態、網路／逾時／解析失敗、
  部分成功、寫入重試、幂等性、舊資料 `unknown`、未來新聞不外洩、時間不明
  不得進訊號、Worker 不得把失敗記成完整成功、機密遮蔽、以及真實
  `pipeline.ingest()` 的紀錄完整性（W23，50 項） |
| `tests/test_stress.py` | 空頭壓力測試：期間辨識（分級門檻、非重疊、**前綴穩定性**）、
  退化輸入、等曝險基準數學、年化與最差 k 日、**加入未來資料不改變視窗結果**、
  T+1 開盤進場、無資料視窗不得回報 0%、尾端截斷計數、訊號遮罩不得用
  [起,訖] 區間、報告不得輸出 `nan`、**不得連外且不得寫入 DB**（W24，30 項） |

**現況**：P0 基準 `14 passed` → 首輪修復後 **`55 passed`** → P2.7 後 **`303 passed`**
→ P2.8 後 **`362 passed`** → P2.9 後 **`392 passed`**。

實際執行項目與結果：

| 指令 | 結果 |
|---|---|
| `python -m pytest -q` | `55 passed` |
| `python -m compileall -q twstock tests` | OK |
| `python -m twstock.cli run` | 成功，寫入 9700 筆訊號 |
| `python -m twstock.cli backtest` | 年化 +11.07% / Sharpe 0.71 / MDD -18.18% / 459 筆 |
| `python -m twstock.cli health` | 正確顯示延遲 1 天、ML 停用、Kill Switch 未啟用 |
| `start/status/GET /`/`stop dashboard` | 200 OK，停止後埠關閉 |

---

## 10. 已知問題、阻塞因素及技術債

| # | 項目 | 嚴重度 | 狀態 |
|---|---|---|---|
| B1 | **LightGBM / scikit-learn 未安裝** → ML 層完全未啟用，融合的 ML 分支無法驗證 | 高 | `BLOCKED`（需使用者決定是否安裝） |
| B2 | 未選定台灣券商 → 交易執行層無法實作，P6–P7 阻塞 | 高 | `BLOCKED`（需使用者決策） |
| B3 | 行情為**延遲/收盤後**資料，非即時；盤中監控能力受限 | 中 | `TODO`（需即時行情授權） |
| B11 | ~~樣本僅約 1.93 年且屬單一市況（多頭），未涵蓋空頭~~ → **P2.9 實測推翻**：期間內 0050 自高點最大回撤 **−27.48%**、20 日年化波動最高 62.09%，**確有**空頭與高波動期間；但僅約 1.93 年、可評估的獨立嚴重事件只有 2 段 → 無統計檢定力 | 高 | 量測 `DONE`（W24）；仍需更長歷史才能推論 |
| B12 | 策略在所有區段皆**未勝過買進持有 0050**（超額 −8% ~ −30%） | 高 | 已實測並記錄；不宣稱具獲利能力 |
| B13 | 部分公司行動因與價格序列不符而被捨棄（保守、略低估報酬） | 低 | `DONE`（有記錄） |
| B14 | `TaiwanStockPriceAdj` 為 FinMind 付費層，故自行還原 | 低 | `DONE`（改用事件檔） |
| B4 | 股票池為寫死的 20 檔，未處理下市/新上市/成分變動 → survivorship 風險 | 中 | `TODO` |
| B5 | 月營收點時（point-in-time）僅以 `date` 近似，未用公告時間 | 中 | `TODO` |
| B6 | Worker 已具單實例鎖／逾時／重試／九態健康，但**未實作跨程序崩潰後的自動接續**（重啟後需人工確認未完成任務） | 中 | `DONE`（P3.1 硬化）；殘餘風險見 B15 |
| B7 | 尚無模擬券商 / 委託狀態機 / Kill Switch | 高 | `TODO`（P4） |
| B8 | 樣本外切分與參數敏感度報告 | 中 | `DONE`（W13/W14，`cli validate` 可重跑） |
| B9 | `store.upsert()` 為死碼且含 `if_exists="replace"` 危險寫法 | 低 | 已移除（W3） |
| B10 | Vercel 相關痕跡（`.vercel/` 本機目錄） | 低 | 已清理設定檔 |
| B15 | Worker 逾時偵測為**事後判定**，無法真正中止已在執行的耗時任務（Python 執行緒無法安全 kill） | 中 | `DONE`（有記錄）；需改為子程序架構才能強制中止 |
| B16 | 公司行動僅以事件檔還原，無獨立還原來源可交叉驗證（校正後跳動已與序列一致） | 低 | `TODO`（付費 `TaiwanStockPriceAdj` 可交叉比對） |
| B17 | **新聞情緒因子實質失效且稀釋其他因子**：資料僅涵蓋 8 個日曆日，歷史面板中 `news_sentiment` 近乎恆為 0（`fillna(0.0)`），卻佔 `WEIGHTS` 的 0.10 分母 | 高 | **已消融驗證（W22）**：分子效應在本資料中**不可估計**（29 列且無前瞻報酬）；分母效應全樣本可觀測但無一致顯著改善 → **保留現行策略、不修改**；待補跨期新聞資料後重評 |
| B18 | **訊號在樣本外不穩定**：`score` 的 IC 由 train +0.1170（p=0.0010）退到 valid +0.0502（p=0.30）、test −0.0104（p=0.74） | 高 | 已實測（W17）；目前證據不支持投入調參 |
| B19 | Worker 逾時仍**無法中止**已執行的任務；子程序原型已備但**尚未接進** `worker.run_forever` | 中 | 原型 `VERIFIED`（W20）；接線 `TODO` |
| B20 | 沒有 CI：測試、驗證與診斷報告均靠手動執行，缺乏回歸防護 | 中 | `TODO` |
| B21 | **歷史新聞的抓取成敗不可追溯**：237 則全部寫在狀態治理之前，且舊程式會用「今天」
偽造缺失的發布時間 → 無法分辨哪些日期是真的 | 高 | 已標為 `unknown`（W23）；**不可重建**，需靠新增的跨期資料重建可信歷史。P2.8 已將此類列排除於訊號路徑（不可作為可見時間的證據） |
| B22 | **新聞來源為公開 RSS（非正式 API）**：無 SLA、可能變更格式或封鎖；
現已可區分失敗（parse／http_status），但仍無備援來源 | 中 | `TODO`（`DATA_SOURCES.md` 列有交叉驗證來源） |
| B23 | 僅能取得「最近 `days` 日」的新聞，**無法回溯歷史**（RSS 特性）→ 新聞因子的
跨期評估先天受限 | 高 | 收集流程 `DONE`（P2.8 每日冪等累積 + 可稽核切點）；**分析**仍 `TODO`——須累積至最低觀察期間才可評估 |
| B24 | `signals.py` 的 `WEIGHTS` 註解寫「總和 = 1.0」，實測總和為 **1.05** | 低 | `TODO`（註解錯誤，不影響分數） |
| B25 | **前視偏差（P2.8 修正）**：`load_panel` 曾以 `published` 當可用日，
使事後才抓到的新聞回填到較早交易日 | 高 | `DONE`（`data/pointintime.py`，14 測試；`ABLATION_B17.md` 不再重現已明確標示） |
| B26 | **公開 RSS 缺穩定識別碼與可靠發布時間**：`news_uid` 採 guid→link→content
三層回退，發布時間來源未提供時一律 NULL | 中 | `DONE`（可稽核、不臆測）；限制已記錄於 `docs/DATA_SOURCES.md` |
| B27 | **「抗跌」主要來自低曝險而非選股**（P2.9 實測）：下跌段平均曝險僅 43.5%，
拆解出同曝險基準後選股效果 +0.11~+5.95pp 且落在隨機選股分布內；
空頭期間合成分數 IC −0.043（p=0.237，不顯著），非壓力日 +0.103（p<0.001）；
代價是 4/4 高波動段落後 0050（最差 −25.16pp） | 高 | 已實測記錄（W24）；
**未據此調整曝險或權重**，後續報告一律並列同曝險基準 |

---

## 11. Git commit / push 狀態

| 項目 | 值 |
|---|---|
| branch | `main` |
| remote | `origin` → `https://github.com/evan980331/stocks` |
| P0 起始 HEAD | `794a0ff chore: init TW stock predictor project` |
| 本輪 commit | 見文末「本輪 commit（P2.8）」 |
| push | 依使用者授權執行（見 §12 回報） |

---

## 12. 最後一次成功驗證的內容

- `python -m pytest -q` → 全部通過（見 §9 / §11）。
- 本地儀表板可啟動並回應 `GET /` 200（`http://127.0.0.1:8000`）。
- 見文末「最後驗證（P2.8）」。
- `python -m twstock.cli datacheck` → 退出碼 0（雙涵蓋區間、失敗與空結果分開、來源與關聯覆蓋率）。
- `python -m twstock.cli research` → 退出碼 0（research 模組 14 測試）。

---

## 13. 下一個最優先執行的任務

依 P2.3 / P2.4 / P2.6 / P2.9 的證據排序（**不再以「調參救績效」為優先**）：

0. **P2.9 定調（B27，先於以下各項）**：下跌段的相對抗跌**主要來自低曝險**
   （平均 43.5%），剝離曝險後選股效果不顯著；空頭期間 IC 不顯著；
   但低曝險的代價在急彈段完全浮現（4/4 高波動段落後 0050）。
   因此：(a) **不得以「抗跌」為理由調高／調低曝險或權重**——
   現有證據不支持任何方向的調整；(b) 後續任何績效報告一律並列
   **同曝險基準**，避免把曝險效果誤讀成選股能力；
   (c) 同一批空頭期間不得重跑多次當成新證據；
   (d) 開放問題改為「補資料」，而非「調參」——見 `docs/BEAR_MARKET_STRESS_TEST.md` §7。
1. **補跨期新聞資料（原「B17 修正」已由 P2.6 證偽）**：P2.6 實測顯示
   分子效應**不可估計**（非 0 列皆落在無前瞻報酬的樣本末端）、分母效應雖
   全樣本可觀測但 86 個配對假設無一顯著、test 段 A ≡ B ≡ D 完全相同。
   **因此不改權重**；先取得跨期（含空頭）新聞資料，才能真正評估新聞因子。
2. **B18 定調**：目前證據支持「訊號在樣本外不穩定」，因此**先不要**投入更多
   持倉參數調整；若要繼續，應先擴充樣本（B11）而不是增加變體數量。
3. **B11/B4 資料**：依 `docs/DATA_SOURCES.md` 的 A 階段（免費）先延長歷史至
   2018 以後並接入 TWSE 官方下市清單；B 階段（TEJ/TQuant）待使用者決定。
   這是取得**新 test 段**的唯一途徑（現 test 已消耗）。
4. **B19 接線**：把 `JobRunner` 接進 `worker.run_forever`，先只接 `ingest`。
5. **B20 CI**：把 `pytest` 與重跑報告納入自動化。
6. **P4**：模擬券商 + 委託狀態機（`risk.py` 已有檢查層）——但**只有在訊號
   真的有樣本外優勢時才值得做**，否則只是把一個沒有優勢的策略自動化。
7. **B1 決策**：是否安裝 `lightgbm`/`scikit-learn`（需使用者決定）。
8. **B23 累積已就緒、只差時間**：P2.8 已完成按日冪等累積、point-in-time 可用性、
   可稽核時間切點與最低觀察期間（120／建議 250 交易日）。
   **目前唯一該做的事是每天讓擷取照常執行並記錄切點**，不要在期間未滿前
   偷看前瞻結果並調整策略（B21 的 237 則舊新聞已終結，不可用來重評）。
9. **B22 備援新聞來源**：公開 RSS 無 SLA；失敗現在可觀測，但沒有第二來源可交叉驗證。
10. **小修（低優先）**：`signals.py` 的 `WEIGHTS` 註解寫「總和 = 1.0」，
   實測總和為 **1.05**（不影響分數，因會再標準化，但註解錯誤）。

---

## 本輪 commit（P2.9，2026-10-16，已 push 至 `origin/main`）

起始 HEAD = `6d0ea32`（= `origin/main`）；結束 HEAD 見文末。無 force push、
未覆蓋任何未提交修改。**本階段未修改因子權重、曝險、持有期間或交易規則；
未開真實下單、未購買資料服務、未部署至 Vercel／Cloudflare。**

| 順序 | SHA | 訊息 | 內容 |
|---|---|---|---|
| 1 | （見 §11） | `feat(stress): 空頭市況壓力測試、基準曝險拆解與空頭訊號檢驗（P2.9）` | stress.py（新）、cli.py（`stress`）、validation.py／diagnostics.py／attribution.py（更正「單一市況」敘述為實測值）+ `test_stress.py`（30） |
| 2 | （本 commit） | `docs: 記錄 P2.9 空頭壓力測試結果與抗跌來源拆解` | plan.md、BEAR_MARKET_STRESS_TEST.md、STATUS_REPORT.md |

### 本階段的關鍵發現（實測，非推論）

1. **「沒有空頭資料」是錯的**：既有報告一律寫「樣本僅涵蓋單一多頭市況」，
   但實測 0050 期間內最大回撤 **−27.48%**，正是無法回答本題的原因。
2. **抗跌 ≠ 選股能力**：下跌段平均曝險 43.5%，以等曝險基準拆解後，
   選股效果不顯著（隨機百分位 27／36／70.5）。
3. **代價在急彈段**：4/4 高波動段落後 0050（最差 −25.16pp），
   全段（前高→復原）3 段僅 1 段為正。
4. **空頭期間訊號失效**：IC 由非壓力日的 +0.103（p<0.001）萎為
   下跌段的 −0.043（p=0.237），五分組價差轉負且扣成本後更負。
5. **測試抓出真實程式缺陷**：`drawdown_episodes` 的 `severe` 誤用**最低**分級
   （`min(tiers)`）而非最高分級，導致所有 ≥10% 的回撤都被標為「較嚴重」；
   已修正並由 `test_drawdown_tier_boundary_is_exclusive_of_severe` 鎖定。
6. **報告不得輸出 `nan`**：無個股資料的視窗（2024-08 缺口）原本會把
   「沒資料」印成「0% 報酬」（回測引擎對空分數提早回傳，基準欄維持預設 0）；
   已改為自行計算基準並將策略欄一律標示 N/A。

## 上一輪 commit（P2.8，2026-10-16，已 push 至 `origin/main`）

起始 HEAD = `6acee94`（`origin/main`）；結束 HEAD 見文末。無 force push、
未覆蓋任何未提交修改。**本階段未修改因子權重、曝險、持有期間或交易規則；
未開真實下單、未購買資料服務、未部署。**

| 順序 | SHA | 訊息 | 內容 |
|---|---|---|---|
| 1 | `5cff235` | `feat(news): 建立可追溯新聞模型與首次寫入即定案的累積寫入（P2.8 §2-§3）` | store.py（+5 溯源欄位、4 索引、`if_conflict="ignore"`）、news_rss.py（news_uid／content_hash／normalize_link／link_basis）、sentiment.py + `test_news_accumulation.py`（16） |
| 2 | `d56bcb9` | `fix(features): 新聞改用 point-in-time 可用性，發布時間不再冒充可見時間（P2.8 §5）` | data/pointintime.py（新）、`data/__init__.py`、pipeline.py、ablation.py（凍結報告提醒）+ `test_pointintime.py`（14）、`test_ablation.py`（+2） |
| 3 | `1d9e5d3` | `feat(report): datacheck 區分「當日新增／當日發布」並新增來源與關聯覆蓋率（P2.8 §4）` | data/report.py、`test_news_report.py`（11） |
| 4 | `e726e39` | `feat(research): 凍結研究版本、每日稽核切點與事前宣告的最低觀察期間（P2.8 §6）` | research.py（新）、store.py（`research_cutoff`）、cli.py（`research`）+ `test_research.py`（14） |
| 5 | （本 commit） | `docs: 記錄 P2.8 新聞累積、point-in-time 與前瞻資料集規則` | plan.md、STATUS_REPORT.md、DATA_SOURCES.md |

### 本階段發現的真實缺陷與研究影響（實測，非推論）

1. **前視偏差（B25，最嚴重）**：`pipeline.load_panel` 原本以
   `SELECT published as date` 直接把發布時間當成可用時間。一則「發布於 D、
   但在 D 之後才被抓到」的新聞會被回填到 D 及其衰減窗口——等於把事後才知道的
   資訊塞進當時的訊號。已改為 `max(published, fetched_at)`，並把舊資料
   （`fetched_at` 不可得）排除於訊號路徑之外。
2. **分母效應被量化並固定成測試**：舊資料被排除後，`news_sentiment` 欄位
   整個**消失**（不是變成 0）→ 打分分母少掉 0.10 權重。兩者不可混同：
   前者是「沒有可追溯的新聞」，後者是「有新聞但情緒中性」。
3. **凍結研究產物不再重現（已明確告知）**：`docs/ABLATION_B17.md` 的數字
   算在 P2.8 之前的面板上，`ablation` 重跑**不會**重現它；已在報告中
   以醒目提醒標出，避免把差異誤讀成新發現。
4. **唯讀模式對未遷移舊 DB 會擲錯**：`report` 與 `pointintime` 在唯讀模式下
   若 DB 缺少新欄位／表會直接失敗。已改為缺欄位視為不可追溯、缺表視為無紀錄。

### 最低觀察期間（§6.7；事前宣告，不得依未來績效反向選擇）

- **最低：120 個交易日（約 6 個月）；建議用於正式結論：250 個交易日（約一年）。**
- 理由（依檢定力與季節性覆蓋，不是依績效）：
  1. 現行策略持有 5 日、每日再平衡，以既有換手率（約每日 5 回合）計，
     120 日約 600 回合，遠高於 `validation.MIN_TRADES_FOR_CONCLUSION＝20`。
  2. 至少涵蓋一次連續 3 個月的月營收週期與一季財報，
     否則營收／新聞因子只在一種季節性下被檢驗。
  3. 約為現有 walk-forward 摺數可比長度的兩倍，降低單一市場狀態主宰結果的機會。
- 評估方式：只用凍結後新累積的 `prospective` 資料；`consumed_test`（B17 那批）
  已消耗，**不得重用**。未達最低期間前**只報告資料品質與覆蓋率**，
  不宣稱新聞因子或策略具備可交易優勢。
- 落地方式：`python -m twstock.cli research freeze` 凍結研究版本、
  `python -m twstock.cli research cutoff` 每日記錄可稽核切點（冪等，首次寫入即定案）、
  `python -m twstock.cli research` 檢視觀察期狀態。

## 最後驗證（P2.9，2026-10-16）

| 命令 | 結果 | 退出碼 |
|---|---|---:|
| `python -m pytest -q` | **392 passed**（上階段 362，本階段 +30） | **0** |
| `python -m compileall -q -f twstock tests` | 無警告 | **0** |
| `python -m twstock.cli stress` | 期間辨識／基準比較／訊號檢驗（產出本報告全部數據） | **0** |
| `python -m twstock.cli datacheck` | 資料品質與擷取狀態 | **0** |
| `python -m twstock.cli validate` | 已更正「單一市況」敘述，現印出實測 −27.48% | **0** |
| `python -m twstock.cli research` | 凍結 `unchanged`（研究版本無漂移）、觀察期 0/120 日 | **0** |
| `python -m twstock.cli status` / `health` | — | **0** |
| 正式 DB md5（驗證前後） | `3523521e48acbe28ff4f6ea5423c5106` → **相同** | — |

## 最後驗證（P2.8，2026-10-16）

| 命令 | 結果 | 退出碼 |
|---|---|---:|
| `python -m pytest -q` | **362 passed**（上階段 303，本階段 +59） | **0** |
| `python -m twstock.cli datacheck` | 新增雙涵蓋區間、每日新增 vs 發布、來源與關聯覆蓋率 | **0** |
| `python -m twstock.cli research` | 前瞻研究紀錄與觀察期狀態 | **0** |
| `python -m twstock.cli validate` | 既有驗證未受影響 | **0** |
| `python -m twstock.cli status` / `health` | — | **0** |
| `python -m compileall -q -f twstock tests` | 全部編譯成功 | **0** |
| 真實 DB 完整性（完整 `pytest` 前後） | 不變（見 STATUS_REPORT.md） | — |

**禁止事項遵守**：未修改因子權重／曝險／持有期間／交易規則；未真實下單；
未購買資料服務；未部署至 Vercel／Cloudflare；不宣稱策略具備可交易優勢。

---

## 上一輪 commit（P2.7，2026-10-15，已 push 至 `origin/main`）

起始 HEAD = `47fcec8`；結束 HEAD 見文末。無 force push、未覆蓋任何未提交修改。

| 順序 | SHA | 訊息 | 內容 |
|---|---|---|---|
| 1 | `7cd9f3c` | `feat(data): 擷取狀態治理，讓失敗不再降級成「沒有資料」` | data/status.py、data/report.py、store.py（`ingest_run`＋news 溯源欄位＋`WriteRejected`）、`conftest.py`（autouse 隔離） |
| 2 | `c08721c` | `fix(sentiment): 新聞抓取失敗不再偽裝成沒有新聞，時間戳不臆測` | news_rss.py（`fetch_feed`）、sentiment.py、experts.py、pipeline.py |
| 3 | `de6b82b` | `fix(worker): 依結構化狀態判定成敗，並新增資料品質檢查指令` | worker.py、ops.py、cli.py（`datacheck`）、`test_ingest_governance.py`（43） |
| 4 | `5f4c1d7` | `docs: 記錄 P2.7 擷取狀態治理結果、資料品質統計與技術債` | README.md、plan.md、STATUS_REPORT.md、DATA_SOURCES.md |
| 5 | `770abd0` | `refactor(data): 移除未使用的狀態與報告輔助函式` | status.py、sentiment.py、report.py、experts.py（不改變行為） |
| 6 | `2347fc6` | `docs(plan): 補上本輪 commit 表與最終驗證紀錄` | plan.md |
| 7 | `6dcd595` | `fix(worker): 根資料集全失敗時仍須視為整輪失敗` | status.py、report.py（+每日筆數）、worker.py、`test_ingest_governance.py`（+7） |
| 8 | （本 commit） | `docs: 同步 P2.7 最終數字與自我複核結果` | plan.md、STATUS_REPORT.md、README.md |

**每個 commit 皆在 detached worktree 檢出完整快照獨立驗證過**（非只在最終狀態跑一次）：

| commit | 驗證 | 結果 |
|---|---|---|
| `7cd9f3c` | worktree 檢出該快照後 `pytest -q` | **253 passed**（退出碼 0） |
| `c08721c` | 同上 | **253 passed**（退出碼 0，且無 SyntaxWarning） |
| `de6b82b` | 同上 | **296 passed**（退出碼 0；當時新增檔為 43 項） |

第 2 個 commit 因 `experts.py` 改了回傳型態，其呼叫端 `pipeline.ingest_experts`
必須同一個 commit 落地（否則該快照的 `ingest_experts` 會拋 `ValueError`）——
這是實際驗證時發現的問題，因此調整了切分方式而非默認它能獨立運作。

## 最後驗證（P2.7，2026-10-15）

| 命令 | 結果 | 退出碼 |
|---|---|---:|
| `python -m pytest -q` | **303 passed**（上階段 253，本階段 +50） | **0** |
| `python -m twstock.cli datacheck` | 新聞覆蓋率、抓取成功率／失敗率、狀態分佈 | **0** |
| `python -m twstock.cli validate` | 既有驗證未受影響 | **0** |
| `python -m twstock.cli health` | 新增顯示最近一輪各資料集擷取狀態 | **0** |
| `python -m twstock.cli status` | — | **0** |
| `python -m twstock.cli --help` | 列出 `datacheck` | **0** |
| `python -m compileall -q -f twstock tests` | 全部編譯成功、**無 SyntaxWarning** | **0** |
| 真實 DB 雜湊（跑完完整 `pytest` 前後） | **不變**（`md5 e4970d6f…`） | — |
| 移除死碼後重跑 `pytest` ＋五個 CLI | 同上全數通過（受影響檔案已重驗） | **0** |

**研究影響檢查（任務第 6 節，實測而非推論）**：

1. **既有 features 與 signals 沒有非預期差異**：在真實本機 DB 的副本上比較
   「遷移前 vs 遷移後」，特徵面板（9,719 列）與 `signals`（9,720 列）的
   `sha256` 前 12 碼**完全相同**（`c0fa6357e4af`／`ebfb28c2618c`）。
2. **新聞有效覆蓋率不變**：`news_present=1` 皆為 61 列／**0.628%**。
   狀態治理**沒有**、也不應該提升預測能力。
3. **不重複使用已消耗的 test 段**：本階段未執行任何回測、未重跑 `validate` 的
   test 段結論、未宣稱任何新的樣本外優勢（僅確認既有報告仍可重現）。
4. **不可判定者一律標 `unknown`**：237 則新聞、163 筆舊擷取紀錄全部如實標示，
   不補造成功狀態。

**本階段結論（一句話）**：修復後**策略與訊號完全沒變**；變的是
「我們現在知道哪些資料可信、哪些不知道」而已——而過去連「不知道」都看不出來。

**本階段自行複核後補修的項目**（不是一次就寫對，如實記錄）：
1. 新增的 docstring 使用 `\(` 造成 `SyntaxWarning`（已修正，並確認全部檔案在
   `-W error` 下編譯無警告）。
2. 第 2 個 commit 原本會讓 `pipeline.ingest_experts` 暫時壞掉（`collect_experts`
   改了回傳型態）→ worktree 驗證時發現，已調整切分使該快照可獨立運作。
3. 五個只有定義、沒有呼叫端的符號，以及一個未使用的匯入（已移除）。
4. 文件初稿以簡體／異體字寫入（刪、拋）→ 已修正為繁體，並以字元掃描確認。
5. **自己的改動造成安全性倒退（最嚴重的一項）**：`IngestSummary` 原本只看
   「所有資料集是否都失敗」，所以**價量全數失敗、僅新聞成功**時不會拋錯，
   會繼續用陳舊價格打分並產生看似正常的訊號；舊版程式在此情境是會中止的。
   已改為以**根資料集（價量）** 判定（`root_all_failed()`），並補上針對這個
   情境的回歸測試。
6. 任務第 3.6 節要求報告「明確顯示每日新聞筆數」，我只計算卻**沒有輸出** →
   已補上每日明細表格（`daily_news_counts()` 為唯一算法來源）。

> 這六項都是交付前自行複核（包含對真實 DB 副本量測、對自己的紀錄做欄位完整性
> 檢查、以及清點只有定義沒有呼叫端的符號）才發現的，不是一次就寫對。

---

## 上一輪 commit（P2.6，2026-10-14，已 push 至 `origin/main`）

起始 HEAD = `293c34d`；結束 HEAD = `origin/main` = `47fcec8`。無 force push。

| 順序 | SHA | 訊息 | 內容 |
|---|---|---|---|
| 1 | `e8b97b1` | `feat(quant): B17 新聞因子消融驗證，分離分子與分母效應` | ablation.py、features.py（+`news_present`）、`test_ablation.py`（19） |
| 2 | `107cabc` | `docs: 記錄 P2.6 消融結果，修正 B17 的量化描述與處置` | ABLATION_B17.md、STATUS_REPORT.md、plan.md |
| 3 | `8d40939` | `docs(plan): 補上本輪 commit 表與最終驗證紀錄` | plan.md |
| 4 | `47fcec8` | `docs: 同步 P2.6 結論至 README 與資料來源方案` | README.md、DATA_SOURCES.md |

每個 commit 皆為可獨立驗證的工作單位：`features.py` 的 `news_present` 為**純新增**
（預設行為不變，既有測試不變即通過），方案實作與測試同單位落地。

## 最後驗證（P2.6，2026-10-14）

- `python -m pytest -q` → **253 passed**（正式退出碼 0；上階段 234，本階段 +19）
- `python -m twstock.ablation` → 退出碼 0（完整報告，sims=500，14 個章節）
- `python -m twstock.ablation --quick` → 退出碼 0
- `python -m twstock.diagnostics --quick` → 退出碼 0（既有診斷未受影響）
- `python -m twstock.cli validate` → 退出碼 0（既有驗證流程未受影響）
- `python -m twstock.cli --help` → 退出碼 0
- 工作目錄乾淨（`git status` 無輸出）
- 簡體字掃描：變更檔案僅命中**共用字形**（影響／排序／涵蓋等）與正當的「核准」，
  無簡體專用字。

**本階段結論（回報第 5 項）**：B17 **會**改變排名（A→B 僅 80.9% 日相同）與成交
（train/valid 有差），但在 train/valid **未**呈現一致且可解釋的改善（86 個配對假設
校正後無一顯著、Rank IC 差符號翻轉、test 段 A ≡ B ≡ D 完全相同）
→ **保留現行策略、記錄不確定性，不提出修改建議**。

---

## 上一輪 commit（P2.3 / P2.4 / P2.5 / P3.2，2026-10-14，已 push 至 `origin/main`）

起始 HEAD = `3ae68ee`；結束 HEAD = `origin/main` = `2791fd0`。無 force push。

| 順序 | SHA | 訊息 | 內容 |
|---|---|---|---|
| 1 | `96261a9` | `feat(quant): 新增訊號資訊量診斷（IC、分組、隨機基準、多重檢定）` | signals.py（抽出 `_factor_parts`）、diagnostics.py、`test_diagnostics.py`（36） |
| 2 | `f6e72dc` | `feat(quant): 策略失效歸因，以隨機選股與完美前瞻拆解瓶頸` | backtest.py（交易新增 `signal_date`）、attribution.py、`test_attribution.py`（22） |
| 3 | `63888e7` | `feat(worker): 耗時任務子程序隔離原型，修正 CTRL_BREAK 造成的假優雅停止` | jobs.py、`test_jobs.py`（23） |
| 4 | `6df3e4b` | `docs: 記錄 P2.3/P2.4/P2.5/P3.2 結果、資料來源方案與子程序架構` | README、plan.md、STATUS_REPORT、DATA_SOURCES、WORKER_SUBPROCESS |
| 5 | `f8ac339` | `docs(plan): 補上本輪 commit 表與最終驗證紀錄` | plan.md |
| 6 | `2791fd0` | `feat(quant): 補齊分組穩健性、分期間一致性與持有期間選擇理由` | diagnostics.py、`test_diagnostics.py`、STATUS_REPORT（+ 本表） |

每個 commit 皆為可獨立驗證的工作單位：`signals.py` 的重構在不改行為下
（既有 9 項測試不變即通過）完成；`backtest.py` 只新增欄位、不動計算。

## 最後驗證（P2.3 / P2.4 / P2.5 / P3.2，2026-10-14）

- `python -m pytest -q` → **234 passed**（正式退出碼 0；起始為 145，本階段 +89）
- `python -m twstock.diagnostics` → 退出碼 0，報告
  `twstock/reports/diagnostics_20261014_020427.txt`（486 交易日、12 因子 × 3 持有期間，
  含價差穩健性與分期間一致性）
- `python -m twstock.attribution --sims 200` → 退出碼 0，報告
  `twstock/reports/attribution_20261014_010205.txt`（測試段 122 交易日、200 次隨機控制）
- 工作目錄乾淨（`git status` 無輸出）；`HEAD = origin/main = 2791fd0`
- 複核任務要求時發現三項未完整涵蓋的檢查並已補齊（持有期間選擇理由、
  分組排序的跨時期一致性、價差是否只由少數交易日主導），各自有測試與報告段落；
  結果強化了原結論：**沒有任何訊號在三個區段都維持單調排序**。

### 前兩輪 commit

## 上一輪 commit（P2.1 / P2.2 / P3.1，2026-10-13，已 push 至 `origin/main`）

| 順序 | SHA | 訊息 | 內容 |
|---|---|---|---|
| 1 | `76c64ee` | `fix(data): 還原除權息與分割，並把無效價格視為缺值` | data/quality.py、data/loader.py、pipeline、features、store、finmind_client + `test_data_quality.py` |
| 2 | `80a32c2` | `feat(quant): 樣本外驗證流程、成本敏感度分析與精確週轉率` | costs.py、strategies.py、backtest.py、validation.py、signals.py + `test_costs_turnover.py`、`test_validation.py` |
| 3 | `3ff2b3a` | `fix(worker): 單實例鎖、逾時與重試、明確健康狀態，並修正「假安全停止」` | worker.py、ops.py、cli.py + `test_worker_hardening.py`、`test_ops_lifecycle.py` |
| 4 | `b180ffa` | `docs: 記錄 P2.1/P2.2/P3.1 結果、週轉率公式與成本假設` | plan.md、docs/STATUS_REPORT.md、README.md |

每個 commit 皆為可獨立驗證的工作單位（各自於提交前以對應 `pytest` 確認）。

### 更早的 commit

| 順序 | SHA | 訊息 |
|---|---|---|
| 1 | `56b63b6` | `fix(quant): 修正資料完整性與回測 look-ahead…` |
| 2 | `a5f4b5d` | `feat(local): 新增風險控制、監控 Worker 與本地服務生命週期` |
| 3 | `d150b58` | `chore: 移除雲端部署依賴，改為本地優先` |

## 最後驗證（P2.1 / P2.2 / P3.1，2026-10-13）

- `python -m pytest -q` → **140 passed**（正式退出碼 0）
- `python -m twstock.validation` → 產出完整報告
  （`twstock/reports/validation_*.txt`；摘要見 `docs/STATUS_REPORT.md`）
- Worker 實機：`start worker` → 健康狀態「任務執行中」→「正常運作」
  （`run_cycle 成功（嘗試 1 次，126.4s）`，ingest 20/20 成功、9719 筆訊號）
  → `stop worker` **1.77 秒**自行收尾，心跳 `stopped`、鎖與停止旗標已清除
- 資料還原實測：0050 分割日報酬 −74.8% → +0.9%；2317 無效列已剔除；
  公司行動共 41 筆套用
- 全期（2.20 年）`backtest`：年化 +14.63%、Sharpe 1.18、MDD −14.54%、456 筆、
  週轉率累積雙邊 **96.16x**；同期 0050 買進持有年化 +51.56% → 超額 **−114.48%**
- 待驗證基準 94.72x 已對上（同一累積雙邊定義；差異來自還原後價格與剔除無效列）
- 正式退出碼：`pytest` 0、`cli validate` 0、`cli backtest` 0

### 前一輪驗證（P0/P1/P2 基礎，2026-10-13）

- `python -m pytest -q` → 55 passed；`compileall` OK
- `python -m twstock.cli backtest` → 修正 look-ahead 後年化由 +96.38% 降為 +11.07%
- 儀表板：`start dashboard` → `GET /` 200、`GET /api/health` 200 → `stop` 後連線關閉
- **仍未驗證**：ML 端到端（lightgbm 未安裝）
