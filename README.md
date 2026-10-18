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
python -m twstock.cli datacheck   :: 資料擷取狀態 + 新聞品質（不連網、不改資料）
python -m twstock.cli research    :: 前瞻研究紀錄：凍結版本、每日切點、觀察期狀態
python -m twstock.cli stress      :: 空頭市況壓力測試：期間辨識、基準比較、空頭訊號檢驗
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

### 訊號資訊量診斷（訊號有沒有資訊？）

```bat
python -m twstock.diagnostics            :: 全部持有期間
python -m twstock.diagnostics --quick    :: 只跑 5 日、較少模擬次數
```

對 12 個底層因子逐一計算 IC / Rank IC（1/5/10 日前瞻報酬）、五分組報酬、
隨機排序基準、Newey-West 穩健 t、區塊自助法信賴區間與 Benjamini-Hochberg
多重檢定校正，並依證據判定每個訊號是「初步獲支持 / 未獲支持 / 樣本不足」。
前瞻報酬一律由 **T+1 開盤**起算（與回測成交假設一致）。
輸出於 `twstock/reports/diagnostics_*.txt`。

### 策略失效歸因（為什麼沒轉化成超額報酬？）

```bat
python -m twstock.attribution            :: 測試段；預設 200 次隨機控制
python -m twstock.attribution --sims 500
```

三個控制變因（真實訊號 / **隨機選股** / **完美前瞻上限（不可交易）**）
只改變分數，交易規則、成本與資金限制完全相同；另拆解訊號→持倉落差、
進場延遲、週轉來源、曝險、集中度與成本。
輸出於 `twstock/reports/attribution_*.txt`。

### B17 新聞因子消融（該不該改權重？）

```bat
python -m twstock.ablation           :: 完整；含交易層級配對回測
python -m twstock.ablation --quick   :: 少量模擬次數（快速檢查）
```

在同一面板、成本與交易規則下配對比較四個方案（A 現行 / B 移除並重新標準化 /
C 固定權重 / D 可用性閘門）加一個分解控制組 A\*，並驗證**排名是否只是被乘上常數**
（實測：可用權重跨股變動 → 不是常數，排名真的會改變）。
另以配對日報酬的 Newey-West 檢定提供交易層級不確定性。
**結論：不修改正式策略**（86 個配對假設校正後無一顯著，test 段成交完全相同）。
完整結果見 `docs/ABLATION_B17.md`，逐行輸出於 `twstock/reports/ablation_*.txt`。

### 資料品質與擷取狀態（這批資料可不可信？）

```bat
python -m twstock.cli datacheck      :: 每日新增 vs 發布／雙涵蓋區間／情緒覆蓋率／來源與關聯覆蓋率
python -m twstock.cli research       :: status（預設）｜freeze（凍結版本）｜cutoff（記錄今日切點）
python -m twstock.cli stress         :: 四種壓力期間定義 + 等曝險／現金／隨機選股對照（詳見 docs/BEAR_MARKET_STRESS_TEST.md）
python -m twstock.cli health         :: 含最近一輪各資料集的擷取狀態
```

### 行情來源與健康狀態（P3）

```bat
python -m twstock.cli quotes              :: 來源評估與健康狀態（不連網）
python -m twstock.cli quotes catalog      :: 8 個來源的分類、認證、限流、授權與查證狀態
python -m twstock.cli quotes selftest     :: 假來源自檢（寫入沙盒暫存 DB，不碰正式資料庫）
python -m twstock.cli quotes poll         :: 以設定的來源實際抓取一次（需 TWSTOCK_QUOTE_SOURCE）
```

行情來源介面 `twstock/data/quotes.py`：每筆行情保留**來源市場時間**（含原樣字串）
與**系統取得時間**，兩者不得互相冒充；重複去重、**舊不得覆蓋新**；
來源失敗不得降級成「正常無資料」；健康狀態分正常／延遲／失敗／未知／非交易時段；
過期或異常一律禁止通過盤中訊號閘門（未來盤中路徑用；目前訊號仍為盤後產生）。

> ⚠️ **本專案目前沒有即時行情**：唯一預設啟用的行情來源是 FinMind 分鐘資料
>（**延遲**、延遲未載明故不稱即時）；TWSE MIS 端點受條款限制需 opt-in，
> 預設停用；Fugle／券商 API 需註冊或帳戶，未實作。完整評估見
> [`docs/QUOTE_SOURCE_ASSESSMENT.md`](docs/QUOTE_SOURCE_ASSESSMENT.md)。

每次擷取都會留下**結構化狀態**（`ingest_run` 資料表），明確區分六種結果：
`success_with_data`（成功且有資料）、`success_empty`（來源正常但確實沒資料）、
`partial_success`、`failed`、`skipped`、`unknown`（舊資料不可追溯）。
**失敗不會被降級成「沒有資料」**；記錄含來源、任務、起訖時間、可重現的錯誤分類、
取得／解析／驗證／寫入筆數、資料涵蓋日期範圍、重試次數與來源版本／設定識別，
且寫入前一律遮蔽 API 金鑰與密碼。

新聞另外區分四態：**沒有新聞**、**有新聞但情緒中性**、**有有效情緒分數**、
**抓取失敗**；發布時間取不到時記為 NULL（不以「今天」代替），
情緒規則版本以指紋記錄，使跨版本的分数混用可被稽核。

> ⚠️ 本機現有 237 則新聞全部寫在狀態治理之前，當時**是否抓取成功無從判定**，
> 已一律標為 `unknown`（不可追溯）。分數欄位本身可觀察，但不得視為抓取成功。

### 模擬交易（P4 paper；純模擬、不下真單）

```bat
python -m twstock.cli paper init --cash 1000000  :: 建立虛擬帳戶（冪等）
python -m twstock.cli paper account             :: 資產／現金／持倉／帳務一致性
python -m twstock.cli paper cycle               :: 執行一次模擬交易循環（T 日訊號→T+1 成交）
python -m twstock.cli paper orders|fills|ledger :: 訂單／成交／帳本查詢
python -m twstock.cli paper risk                :: 風控狀態與停止原因
python -m twstock.cli paper reconcile           :: 帳務一致性核對（不一致→退出碼 1）
python -m twstock.cli paper recover             :: 重啟恢復檢查
python -m twstock.cli paper kill / resume       :: 緊急停止／解除
```

模擬交易引擎 `twstock/paper.py`：**所有資金與成交皆為模擬**，不接券商、
不存在真實下單路徑（測試以 AST 掃描強制）。金額以微元整數入帳（無浮點
帳務誤差）；現金可由帳本重建、持倉可由成交序列重放，`reconcile` 逐項核對；
成交用 **T+1 開盤價 ± 滑價**（不用 T 日收盤價成交，防 look-ahead）；
缺價／停牌／漲跌停／行情過期一律「不成交＋記錄原因」；風險限制（單筆/
部位/現金/單日損失/回撤/行情過期/Kill Switch）全部在成交之前執行。
儀表板另有唯讀 API `/api/paper/*`（只有 GET，無下單端點）。

> ⚠️ 模擬成交是對真實交易的近似（開盤價±滑價，不模擬排隊與逐檔），
> **不代表策略具備樣本外優勢**，也**不具備真實自動交易能力**。
> 盤中即時交易屬 P5，本階段未實作。詳見
> [`docs/PAPER_TRADING.md`](docs/PAPER_TRADING.md)。

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
  diagnostics.py       # 訊號資訊量診斷：IC／分組／隨機基準／多重檢定
  attribution.py       # 策略失效歸因：隨機選股與完美前瞻控制組
  ablation.py          # B17 消融：四方案配對比較與排名不變性驗證
  jobs.py              # 耗時任務子程序隔離（合作式取消，不用 CTRL_BREAK）
  data/
    finmind_client.py  # FinMind API v4（失敗回 None，與「空資料」區分）
    store.py           # SQLite：INSERT OR REPLACE 冪等寫入，永不 DROP 表
    status.py          # 擷取狀態詞彙、錯誤分類、紀錄結構與機密遮蔽
    report.py          # 資料品質報告：新聞覆蓋率、抓取成功率／失敗率、狀態分佈
    quality.py         # 無效價格清理、除權息／分割還原與驗證守門
    loader.py          # 價格載入（清理 + 還原，策略與基準一致）
  data/
    quotes.py         # 統一行情介面：來源目錄、驗證、健康狀態、訊號閘門（P3）
  sentiment/           # Google News RSS + 規則式情緒；專家觀點來源（共用狀態化取得）
  features.py          # 技術指標（RSI/KD/MACD/布林/K線型態）+ 籌碼 + 營收 + 情緒
  signals.py           # 規則式多因子打分 + 尺度無關的訊號融合
  ml_model.py          # LightGBM walk-forward（可選，未安裝時明確停用）
  backtest.py          # T+1 開盤成交、成本模型、週轉率指標、0050 基準
  pipeline.py          # 每日流程 + 結構化擷取狀態（ingest_run，另保留 ingest_log）
  worker.py            # 監控 Worker（單一實例鎖、重試、逾時、健康狀態機）
  ops.py               # start / stop / status / health（含擷取狀態）
  app.py               # FastAPI 本地儀表板
  cli.py               # CLI 唯一進入點
tests/                 # pytest：功能、資料品質、時間正確性、成本、驗證、Worker、
                       #        擷取狀態治理（autouse 隔離，不會寫入真實 DB）
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
| Google News RSS | 個股新聞 | 每次 ingest | 公開 RSS，非正式 API；僅涵蓋最近 `days` 日 |
| YouTube RSS / Google News | 專家觀點 | 手動 | 公開 RSS |

**行情為延遲/收盤後資料，並非即時行情**。儀表板一律顯示「最新行情日期」與延遲天數。
**沒有已查證、可無條件使用的免費官方台股盤中即時來源**（評估與條款查證見
[`docs/QUOTE_SOURCE_ASSESSMENT.md`](docs/QUOTE_SOURCE_ASSESSMENT.md)）；
本專案的行情介面已就緒，但真實盤中來源尚未接通。

缺值／失敗處理原則：抓取失敗一律記為失敗（不回填 0），資料不足時標記「無法判斷」，
不產生看似正常的訊號。

**新聞資料的可追溯性限制（實測）**：本機既有 237 則新聞（2026-10-06 ~ 2026-10-13，
影響面板 61/9,719 列 ≈ 0.63%）全部寫在狀態治理之前，且舊程式在發布時間取不到時
會以**當天日期**代替，因此：

- 無法判定任何一則當時是否真的抓取成功 → 全部標為 `unknown`；
- 無法分辨哪些日期是來源真實發布日、哪些是程式補上的日期 → `timestamp_source` 全部 `unknown`；
- 舊系統的抓取紀錄（`ingest_log`）163 筆**全部 `ok=1`**，其中新聞 20/20 全為成功，
  但那是舊程式硬編碼的結果，**不能當成證據**。

因此新聞因子的任何評估都必須先聲明此限制；本階段**不宣稱**新聞因子具備預測能力。

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
ML 標籤、訊號融合尺度、回測時間正確性、風險限制與 Kill Switch，
擷取狀態治理（六態區分、新聞四態、未來新聞不外洩、舊資料不可追溯、
Worker 不得把失敗記成完整成功），以及行情介面（去重、舊不得覆蓋新、
時間戳分離、健康狀態、訊號閘門、Worker 行情輪詢不影響 EOD）、
模擬交易 17 類（帳務閉環、T+1 成交、防重播、風控、重啟恢復、無真實
下單路徑）。目前 **489 項**。

測試一律將儲存層隔離到暫存 DB（`tests/conftest.py` 的 autouse 夾具），
**不會**寫入或遷移 `twstock/data_store/` 的真實資料。

---

## 相關文件

- [`plan.md`](plan.md)：工程計畫、進度、已知問題與下一步。
- [`docs/STATUS_REPORT.md`](docs/STATUS_REPORT.md)：專案現況盤點與缺陷證據。
- [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)：歷史資料偏差改善方案、
  survivorship bias 的量化限制與資料來源比較（含授權與價格查證）。
- [`docs/WORKER_SUBPROCESS.md`](docs/WORKER_SUBPROCESS.md)：Worker 子程序架構評估、
  停止階梯與失敗情境測試（尚未全面遷移）。
- [`docs/PAPER_TRADING.md`](docs/PAPER_TRADING.md)：P4 模擬交易引擎設計、
  帳務閉環、成交模擬假設、風險控制與**未實作限制**。
