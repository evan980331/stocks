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

## 7. 各階段開發計畫（P0 → P8）

| 階段 | 內容 | 完成條件 | 狀態 |
|---|---|---|---|
| **P0** | 現況盤點（程式/資料/模型/回測/Git） | 產出 `docs/STATUS_REPORT.md` 並標明證據 | `VERIFIED` |
| **P1** | 資料品質與基礎架構（覆蓋、時間戳、重跑、錯誤處理） | 新聞不再互相覆蓋、ingest 失敗不得記為成功、寫入可重跑 | `DONE`（見 §8） |
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

## 9. 測試方式及驗證結果

```bat
python -m pytest -q
```

| 測試檔 | 涵蓋 |
|---|---|
| `tests/test_features.py` | 技術指標、特徵合併、新聞衰減 |
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

**現況**：P0 基準 `14 passed` → 首輪修復後 **`55 passed`**（新增 41 項回歸測試）。

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
| B11 | 樣本僅約 1.93 年且屬單一市況（多頭），未涵蓋空頭 | 高 | `TODO`（需更長歷史） |
| B12 | 策略在所有區段皆**未勝過買進持有 0050**（超額 −8% ~ −30%） | 高 | 已實測並記錄；不宣稱具獲利能力 |
| B13 | 部分公司行動因與價格序列不符而被捨棄（保守、略低估報酬） | 低 | `DONE`（有記錄） |
| B14 | `TaiwanStockPriceAdj` 為 FinMind 付費層，故自行還原 | 低 | `DONE`（改用事件檔） |
| B4 | 股票池為寫死的 20 檔，未處理下市/新上市/成分變動 → survivorship 風險 | 中 | `TODO` |
| B5 | 月營收點時（point-in-time）僅以 `date` 近似，未用公告時間 | 中 | `TODO` |
| B6 | 無 Worker 持續排程與崩潰恢復的完整實作 | 中 | `IN_PROGRESS`（P3） |
| B7 | 尚無模擬券商 / 委託狀態機 / Kill Switch | 高 | `TODO`（P4） |
| B8 | 未實作樣本外切分與參數敏感度報告 | 中 | `TODO`（P2 後半） |
| B9 | `store.upsert()` 為死碼且含 `if_exists="replace"` 危險寫法 | 低 | 已移除（W3） |
| B10 | Vercel 相關痕跡（`.vercel/` 本機目錄） | 低 | 已清理設定檔 |

---

## 11. Git commit / push 狀態

| 項目 | 值 |
|---|---|
| branch | `main` |
| remote | `origin` → `https://github.com/evan980331/stocks` |
| P0 起始 HEAD | `794a0ff chore: init TW stock predictor project` |
| 本輪 commit | 見文末「本輪 commit」 |
| push | 依使用者授權執行（見 §12 回報） |

---

## 12. 最後一次成功驗證的內容

- `python -m pytest -q` → 全部通過（見 §9 / §11）。
- 本地儀表板可啟動並回應 `GET /` 200（`http://127.0.0.1:8000`）。
- 見文末「最後驗證」。

---

## 13. 下一個最優先執行的任務

1. **P4**：模擬券商 + 委託狀態機 + 持倉同步 + 冪等控制（`risk.py` 已有檢查層）。
2. **B11/B12 回應**：延長歷史與納入空頭期間；先解釋「為何輸給買進持有」
   （多頭樣本 + 每日再平衡的高換手成本），再決定是否繼續發展策略。
3. **B4 硬化**：以 FinMind 全市場清單 + 下市清單處理 survivorship bias。
4. **B1 決策**：是否安裝 `lightgbm`/`scikit-learn` 以啟用並驗證 ML 層（需使用者決定）。
5. **參數敏感度**：對 hold_days / rebalance_days 做網格分析並以 walk-forward 驗證，
   但**不得**以同一測試集挑選參數。

---

## 本輪 commit

| 順序 | SHA | 訊息 | 內容 |
|---|---|---|---|
| 1 | `56b63b6` | `fix(quant): 修正資料完整性與回測 look-ahead…` | store/sentiment/finmind/pipeline/ml_model/signals/backtest/config + 6 個回歸測試 |
| 2 | `a5f4b5d` | `feat(local): 新增風險控制、監控 Worker 與本地服務生命週期` | risk/worker/ops/cli/app + run.bat + 2 個測試 |
| 3 | 本文件所在 commit | `chore: 移除雲端部署依賴，改為本地優先` | 刪除 vercel 相關檔、更新 README/.env.example/.gitignore、plan.md、docs/ |

每個 commit 皆為可獨立驗證的工作單位（以 `pytest` 確認）。

## 最後驗證（P2.1 / P2.2 / P3.1，2026-10-13）

- `python -m pytest -q` → **140 passed**（正式退出碼 0）
- `python -m twstock.validation` → 產出完整報告
  （`twstock/reports/validation_*.txt`；摘要見 `docs/STATUS_REPORT.md`）
- Worker 實機：`start worker` → 健康狀態「任務執行中」→「正常運作」
  （`run_cycle 成功（嘗試 1 次，126.4s）`，ingest 20/20 成功、9719 筆訊號）
  → `stop worker` **1.77 秒**自行收尾，心跳 `stopped`、鎖與停止旗標已清除
- 資料還原實測：0050 分割日報酬 −74.8% → +0.9%；2317 無效列已剔除；
  公司行動共 41 筆套用

### 前一輪驗證（P0/P1/P2 基礎，2026-10-13）

- `python -m pytest -q` → 55 passed；`compileall` OK
- `python -m twstock.cli backtest` → 修正 look-ahead 後年化由 +96.38% 降為 +11.07%
- 儀表板：`start dashboard` → `GET /` 200、`GET /api/health` 200 → `stop` 後連線關閉
- **仍未驗證**：ML 端到端（lightgbm 未安裝）
