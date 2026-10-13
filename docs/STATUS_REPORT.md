# 專案現況報告（P0 盤點）

- 專案：`evan980331/stocks`（本機工作目錄 `D:\股市預測`）
- 盤點日期：2026-10-13
- 起始 HEAD：`794a0ff chore: init TW stock predictor project`（branch `main`）
- 盤點方式：完整閱讀 `twstock/**`、`README`、`requirements.txt`、`vercel.json`、`api/index.py`、4 個測試檔；
  檢視 3 個 SQLite DB 實際內容；實跑 pytest 與 CLI 流程。

---

## 1. 目前真正可執行的功能（有證據）

| 功能 | 狀態 | 證據 |
|---|---|---|
| 抓取台股日K／法人／融資券／月營收（FinMind） | 可用（需網路） | DB `twstock_v3.db`：price 9700 筆／20 檔、institution 48480、margin 9700、revenue 2793 |
| 新聞抓取與規則式情緒打分 | 可用 | DB `news` 表（但見缺陷 D1） |
| 技術指標與特徵工程 | 可用 | `tests/test_features.py` 4 項通過 |
| 規則式多因子打分 | 可用 | `tests/test_signals.py` 3 項通過 |
| 儀表板（FastAPI，本機） | 可用 | `GET /` 200；`GET /api/health` 200；畫面截圖確認 |
| 回測（含台股成本） | 可用（時間模型已修） | `python -m twstock.cli backtest` 產出報告 |
| Telegram 通知 | 可用（需 token） | 未設定 token 時明確略過 |
| 專家觀點抓取 | 可用（需網路） | DB `experts` 125 筆 |

## 2. 已完成但尚未驗證的功能

- **ML（LightGBM）層**：程式碼存在，但 `lightgbm`／`scikit-learn` **未安裝**，
  且標籤所需的 0050 基準**從未被抓取**（見缺陷 D2）。因此 ML 從未真正跑過。
- **Vercel 部署路徑**：`vercel.json`／`api/index.py` 存在，但與本地功能無關；本次已移除。

## 3. 無法執行的功能與原因

| 功能 | 原因 |
|---|---|
| ML 訓練／預測 | 未安裝 lightgbm；且缺 0050 基準（已修，仍待安裝套件驗證） |
| 任何交易執行（模擬或正式） | **完全不存在**交易/委託/持倉模組 |
| 即時行情 | 僅有延遲/日線資料來源 |
| 持續監控 | 原無 Worker；本次新增 |

---

## 4. 確認的缺陷與修復（附證據）

### D1 — 新聞寫入會覆蓋其他所有股票資料　【嚴重】
- **證據**：`sentiment.update_news_for()` 使用 `to_sql(..., if_exists="replace")`，
  該參數會 **DROP 整張 `news` 表**再重建。實測 `twstock_v3.db` 的 `news` 只有
  **`3231` 5 筆**（其餘 19 檔全部消失），證明每抓一檔就清空先前所有新聞。
- **影響**：新聞情緒特徵大面積缺失 → 所有含情緒因子的分數失真。
- **修復**：`store.write_df()` 改為 `INSERT OR REPLACE`（依主鍵），永不 DROP；
  已由 `tests/test_store_write.py` 驗證。

### D2 — ML 標籤所需的 0050 基準從不存在　【高】
- **證據**：`ml_model._make_labels()` 從 panel 找 `stock_id==0050`，但
  `ingest()` 只抓 `CONFIG.pool`、`load_panel()` 也只組股票池；實測 `price` 表
  20 檔中**沒有 0050**。標籤因此全為 NaN，模型永遠訓練不出東西卻不報錯。
- **修復**：`ingest()` 一併抓取並儲存 0050；`make_labels()` 改為明確接收基準序列；
  `train_walk_forward()` 回傳 `enabled/reason/rows`，缺基準時明確回報
  `missing_benchmark`。已由 `tests/test_ml_labels.py` 驗證。

### D3 — 規則分數與 ML 預測尺度不同卻直接加權相加　【高】
- **證據**：`pipeline.run_scores()` 為 `score*0.6 + ml_score.fillna(0)*0.4`。
  規則分數在 `[-1,1]`，ML 輸出是**預測未來 5 日超額報酬率**（量級約 ±0.03）。
  0.4 權重實際只貢獻約 ±0.012，**ML 形同被忽略**；且缺值填 0 等同冒充「中性」。
- **修復**：新增 `signals.fuse_scores()`，先做**當日橫斷面排名百分位**映射到 `[-1,1]`
  再加權；ML 全缺時明確退化為規則分數。已由 `tests/test_fusion.py` 驗證（含尺度不變性）。

### D4 — 回測存在 look-ahead bias（結果嚴重高估）　【嚴重】
- **證據**：原 `backtest.run_backtest()` 以 **T 日收盤**特徵產生訊號，卻在 **T 日開盤**進場；
  且部分情況以「當日收盤」當備援成交價。原回測報告顯示
  **年化 +96.38%、Sharpe 3.45、MDD -3.82%**（不合理地好）。
- **修復**：改為 **T+1 開盤進場／到期日開盤出場**，並加入漲停無法成交判斷、
  0050 買進持有基準與完整指標。修正後同一份資料的結果：
  **年化 +11.07%、Sharpe 0.71、MDD -18.18%、勝率 47.93%、459 筆交易**——
  與修正前的 +96.38% 差距懸殊，證實高估來自 look-ahead。
- 已由 `tests/test_backtest_timing.py` 驗證（進場價必須等於次一日開盤）。

### D5 — 抓取失敗被當成成功，資料新鮮度造假　【高】
- **證據**：`finmind_client.fetch_finmind()` 失敗時回**空表**，與「真的沒資料」無法區分；
  `pipeline.ingest()` 無論成敗都執行 `set_meta("last_ingest", today)`。
  因此 API 全掛時系統仍顯示「今日已更新」。
- **修復**：失敗回 `None`；`ingest()` 逐檔記錄成功／失敗到新的 `ingest_log` 表，
  **只有真的有價量成功才更新 `last_ingest`**。已由 `tests/test_ingest_status.py` 驗證。

### D6 — `python -m twstock.cli run` 直接崩潰　【高】
- **證據**：`pipeline.load_panel()` 存取 `CONFIG.vol_ma_window`，但 `Config` 的欄位名為
  `volume_ma_window` → `AttributeError`。實跑 `cli run` 重現。
- **影響**：主要分析流程完全無法執行（DB 中的 signals 為更早版本所產生）。
- **修復**：更正為 `CONFIG.volume_ma_window`；實跑 `cli run` 成功寫入 9700 筆訊號。

### D7 — 儀表板訊號一律顯示「持有」　【中】
- **證據**：`app.index()` 只做 `action.fillna("HOLD")`，而 `signals` 表的 `action` 欄
  從未落庫（實測全為 NULL）→ 即使分數 +0.58 仍顯示「持有」。
- **修復**：新增 `_ensure_actions()`，未落庫時即時計算；`/api/signals` 與 HTML 一致。
  驗證：`3711 0.5793 BUY`。

### D8 — `store.upsert()` 死碼且含 `if_exists="replace"`　【低】
- 全專案無呼叫者，但若被使用同樣會 DROP 表。已隨 store 改寫移除。

### D9 — 雲端部署依賴　【依需求移除】
- `vercel.json`、`api/index.py`、`.vercelignore`、`public/robots.txt` 已刪除；
  `config.is_readonly()` 不再偵測 `VERCEL` 環境變數。
- **附帶發現**：`.vercel/.env.preview.local` 內含 **`VERCEL_OIDC_TOKEN` 憑證**，
  已隨 `.vercel/` 目錄一併移除（該目錄本就被 Git 忽略，未進入版控）。

---

## 5. 資料完整度

| 資料 | 現況 | 缺口 |
|---|---|---|
| 日K | 20 檔 × 約 485 交易日（2024-10-14 ~ 2026-10-12） | 缺 0050 基準（已修，需重新 ingest 才會有資料） |
| 三大法人 | 48480 筆 | 覆蓋完整 |
| 融資融券 | 9700 筆 | 覆蓋完整 |
| 月營收 | 2793 筆 | 2015-01-01 起；採「公告日」近似，未精確至 point-in-time |
| 新聞 | **僅 1 檔**（缺陷 D1） | 修復後需重新 ingest |
| 專家觀點 | 125 筆 | 標題層級，資訊有限 |

---

## 6. 模型、策略與回測現況

- **規則策略**：可運作，12 個因子加權，輸出 `[-1,1]` 分數與中文理由。
- **ML**：未啟用（D2 + lightgbm 未安裝）。
- **融合**：尺度問題已修（D3），但**尚未做樣本外驗證**，不可宣稱優於單一策略。
- **回測**：時間模型與成本假設已修正（D4）。仍缺：樣本外切分、參數敏感度、
  survivorship bias 處理（股票池寫死 20 檔）。

---

## 7. 交易功能安全性

- **交易執行模組完全不存在** → 不存在誤下單風險。
- 本次新增 `twstock/risk.py`（限額、模式、Kill Switch）與 `plan.md` 的權限界線，
  但**尚未接上任何券商或模擬券商**（P4/P5）。
- 正式交易預設關閉：`TWSTOCK_MODE` 預設 `paper`；啟用 live 還需 `TWSTOCK_LIVE_ACK=1`。

---

## 8. 主要風險與技術債

| # | 項目 | 嚴重度 |
|---|---|---|
| B1 | lightgbm 未安裝 → ML 無法驗證 | 高 |
| B2 | 未選券商 → 交易層阻塞（需使用者決策） | 高 |
| B3 | 僅延遲資料，非即時 | 中 |
| B4 | 股票池寫死 20 檔，未處理下市/成分變動 → survivorship bias | 中 |
| B5 | 月營收未用精確公告時間 | 中 |
| B6 | 週轉率偏高（94.72x）→ 成本敏感 | 中 |
| B7 | 尚無模擬券商／委託狀態機／Kill Switch 整合 | 高 |
| B8 | 無樣本外與參數敏感度報告 | 中 |

---

## 9. 結論

系統的**資料管線與儀表板骨架可用**，但先前存在兩個會讓結果「看起來很好卻是錯的」
關鍵缺陷（新聞覆蓋、回測 look-ahead），以及讓 ML 從未真正運作的基準缺口。
本次已修復 D1–D9 並補上對應回歸測試；下一階段應聚焦
**樣本外驗證、Worker 硬化、風險控制與模擬券商**，而非增加策略數量。
