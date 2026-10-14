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
| B6 | 週轉率偏高（對帳後：累積雙邊 96.16x，全期 456 筆）→ 成本敏感 | 中 |
| B7 | 尚無模擬券商／委託狀態機／Kill Switch 整合 | 高 |
| B8 | 無樣本外與參數敏感度報告 | 中 |

---

## 9. 結論

系統的**資料管線與儀表板骨架可用**，但先前存在兩個會讓結果「看起來很好卻是錯的」
關鍵缺陷（新聞覆蓋、回測 look-ahead），以及讓 ML 從未真正運作的基準缺口。
本次已修復 D1–D9 並補上對應回歸測試；下一階段應聚焦
**樣本外驗證、Worker 硬化、風險控制與模擬券商**，而非增加策略數量。

---

# 附錄：第二階段盤點與驗證結果（2026-10-13，P2.1 / P2.2 / P3.1）

起始 HEAD：`d150b58`（branch `main`，與 origin 同步，工作目錄乾淨）
基準測試：`55 passed`

## A. 本階段新發現的缺陷

### D10 — 價格完全未還原公司行動（除權息／分割）　【嚴重】
- **證據**：
  - 0050 於 2025-06-18 由 188.65 → 47.57，原始序列顯示 **−74.8%**（實為 1 拆 4）。
    以該序列計算的基準報酬為 **−69.94%**，實際應為 **+22.76%**，
    導致「超額報酬」出現 **+67%** 的假優勢。
  - 2603（長榮）除息缺口 −13.8%、−12.0%，超過 ±10% 漲跌停幅度。
- **影響**：基準比較完全失真；特徵（動能、均線）與回測報酬在事件日前後被扭曲。
- **修復**：新增 `twstock/data/quality.py`（事件→調整因子）與 `data/loader.py`
  （載入時清理＋還原）。以 FinMind `TaiwanStockSplitPrice`、`TaiwanStockDividend`
  建立因子，對事件日之前價格後調；策略與基準採一致處理（41 筆調整）。
- **驗證守門**：只在「套用後能降低序列不連續」時才調整。實測富邦金股票股利事件
  （因子 0.8）與價格序列不符，套用會製造 +20.8% 假跳動 → 已捨棄並記錄。
- 註：FinMind `TaiwanStockPriceAdj`（還原股價）為付費層（實測 status=400）。

### D11 — 無效價格列（全為 0）污染回測　【嚴重】
- **證據**：2317 於 2025-07-30 的 open/high/low/close/volume 全為 0，amount 為 0。
- **影響**：0 元價格使持倉市值瞬間歸零 → 假回撤、±inf 報酬。2317 在交易池內。
- **修復**：寫入與讀取兩端將非正數價格視為**缺值**（非 0 值）；
  `backtest._prepare`、`features.build_features` 同步防護。

### D12 — 週轉率定義不明確（94.72x 的實際意義）　【中】
- **證據**：原 `turnover = notional_traded / initial_capital`，
  其中 `notional_traded = 買進名目 + 賣出名目`（**買賣各計一次**），且**未年化**。
- **影響**：易被誤讀為年化週轉率或交易次數；跨策略比較不公平。
- **修復**：提供四項指標（累積雙邊／累積單邊／年化雙邊／年化單邊資金），
  並在 `backtest.py` docstring 與驗證報告 [8] 節寫明公式與單位。

### D13 — 停止回報不實（「已安全停止」實為被強制終止）　【中】
- **證據**：原 `ops.stop()` 只要 `pid_alive` 回 False 就印出「已安全停止」，
  但實際可能是 CTRL_BREAK_EVENT 直接殺了行程，`finally` 未執行 →
  停止旗標與鎖殘留、心跳未更新、健康狀態停在 STOPPING。
- **修復**：worker 改為「先寫停止旗標等它自行收尾」，逾時才送訊號、最後才強制終止；
  回報區分「已安全停止」與「已強制停止」；結束後清理 PID 檔、停止旗標與殘留鎖。

## B. 本階段驗證結果（實測）

| 項目 | 結果 |
|---|---|
| `pytest -q` | **140 passed**（正式退出碼 0） |
| `python -m twstock.validation` | 產出完整報告 |
| Worker 實跑一輪 | ingest 價量 **20/20 成功**、訊號 9719 筆、耗時 126.4s、1 次嘗試 |
| Worker 停止 | `stop worker` **1.77 秒**自行收尾；心跳 `stopped`、鎖與旗標已清除 |
| 基準還原 | 0050 分割日報酬 −74.8% → **+0.9%**；基準期間報酬 −69.94% → **+22.76%** |

## C. 樣本外驗證摘要（test 段 2026-04-17 ~ 2026-10-13，122 交易日）

切分：train 2024-10-14~2025-10-01（238 日）、valid 2025-10-13~2026-04-09（116 日）、
test 2026-04-17~2026-10-13（122 日）；purge/embargo = 5 交易日（= hold_days）。

| 策略 | test 總報酬 | 年化 | MDD | Sharpe | 筆數 | 超額（vs 0050） |
|---|---:|---:|---:|---:|---:|---:|
| baseline_rule_h5 | +25.63% | +59.29% | −5.54% | 2.82 | 120 | **−7.94%** |
| rule_h10 | +22.85% | +52.17% | −9.48% | 2.42 | 60 | −10.73% |
| rule_h20 | +12.79% | +27.85% | −9.86% | 1.46 | 30 | −20.78% |
| rule_h5_rebal5 | +8.07% | +17.17% | −8.14% | 1.11 | 115 | −25.50% |
| rule_h5_thresh35 | +24.27% | +55.80% | −5.17% | 2.82 | 115 | −9.30% |
| rule_hysteresis | +12.14% | +26.35% | −6.88% | 1.65 | 28 | −21.43% |

**關鍵結論：策略在所有區段與四個子期間皆未勝過買進持有 0050。**
（train 超額 −15%~−28%、valid −19%~−31%、test −8%~−30%）
這不代表策略「接近可用」——在單一多頭樣本中輸給基準，加上高週轉成本，
不具備可交易優勢的證據。

> **子期間表的讀法（重要）**：四個子期間各為「獨立起始」的單獨回測（各自重新
> 投入初始資金、無跨期持倉），跨界線且無法在該段內完成持有期的部位會被排除，
> 因此合計筆數（100）低於完整測試段的 120 筆，且各列報酬**不可相加或複合**回
> 測試段總報酬。此表只作各子期間內與同期基準的方向性對照（四段皆為負超額）。

## D. 成本敏感度（測試段，baseline_rule_h5）

| 成本情境 | 總報酬 | 年化 | Sharpe | 總成本 |
|---|---:|---:|---:|---:|
| zero（僅對照） | +34.68% | +83.59% | 3.67 | 0 |
| ideal（2 折、0.05% 滑價） | +27.47% | +64.08% | 3.00 | 65,403 |
| typical（2.8 折、0.1% 滑價） | +25.63% | +59.29% | 2.82 | 82,284 |
| conservative（6 折、0.2%） | +21.32% | +48.35% | 2.39 | 121,308 |
| adverse（不打折、0.3%） | +16.74% | +37.13% | 1.91 | 161,773 |
| severe（不打折、0.5%、1% 量限） | +11.28% | +24.38% | 1.33 | 211,570 |

成本造成的報酬差距達 **約 23 個百分點**（zero → severe，122 交易日），
且六項假設由寬到嚴對應單調下降的績效——週轉率是主要成本驅動因素。
費率為市場常見基準，**非特定券商通用費率**（牌告手續費 0.1425%、折扣依帳戶而異、
低消無全市場統一值；一般股票證交稅 0.3% 僅賣出；本策略不適用當沖 0.15%）。

## E. 仍未覆蓋的風險

- 樣本僅約 1.93 年且為單一（多頭）市況，未含空頭與高波動期間。
- 股票池寫死 20 檔，survivorship bias 未處理；結果不代表整個台股市場。
- ML／融合策略未納入比較（lightgbm 未安裝），強行排名會不公平。
- 部分公司行動因與價格序列不符而捨棄 → 保守但略微低估報酬。
- Worker 以同步任務執行，逾時僅「偵測與恢復」，未強制中斷單一長任務。
- 交易執行層（模擬券商／委託狀態機）仍不存在；正式交易維持關閉。

---

# 附錄：第三階段（2026-10-14，P2.3 / P2.4 / P2.5 / P3.2）

## F. 本階段新發現的缺陷

### D14 — 新聞情緒因子在歷史回測中實質失效，並稀釋其他因子　【高】
- **證據**：`news` 表共 237 筆、涵蓋 20 檔，但日期範圍只有
  **2026-10-06 ~ 2026-10-13（8 個日曆日）**；而 `features.build_features()` 的原始
  行為是 `df["news_sentiment"] = df["w_score"].fillna(0.0)`，因此歷史面板中該因子
  **近乎恆為 0**。
- **雙重後果**：
  1. 不提供資訊 —— P2.3 診斷中該因子有效交易日僅 0~2 天，判定「樣本不足」；
  2. **稀釋其他因子** —— `score_row` 的分母 `total_w` 固定包含它的 0.10 權重，
     使其他因子的貢獻被縮小約 10%。
- **佐證**：P2.3 顯示合成 `score` 的 IC（h=5 為 +0.0705）**低於**單一最強因子
  月營收 YoY（+0.0987）。
- **狀態**：已量化並登錄為 B17；**P2.6 已完成消融驗證 → 結論是不修改正式策略**
  （證據見附錄 K）。先前「稀釋約 10%」的說法已修正：0.10 佔據的是
  **全部 9,719 列**的分母（不只有新聞的 61 列），且稀釋倍率逐股不同、
  **不是單一常數**。

### D15 — Windows `CTRL_BREAK_EVENT` 是立即終止，不是可攔截中斷　【中】
- **證據**：以 `CREATE_NEW_PROCESS_GROUP` 啟動子程序後送 `CTRL_BREAK_EVENT`，
  子程序以 **`0xC000013A`（STATUS_CONTROL_C_EXIT）** 結束，且**沒有**拋出可捕捉的
  `KeyboardInterrupt`；連明確 `except KeyboardInterrupt: continue` 的任務
  （`stubborn`、交易中的 `sqlite-atomic`）也照樣被殺。
- **影響**：若把它當成「優雅停止」，就會把硬殺回報成安全停止 —— 與 P3.1 修過的
  D13 同類（假的安全停止）。另外實測：對**不在獨立程序群組**的子程序送該訊號，
  訊號會投遞到整個主控台程序群組，**會殺掉父程序與使用者開的其他行程**
  （實際發生時整條命令無聲消失）。
- **修復**：`twstock/jobs.py` **完全不使用 `os.kill`**，停止階梯改為
  「合作式取消檔 →（僅 POSIX）SIGTERM → 針對自身 PID 的強制終止」，
  並以 `forced` 明確標示強制。已有原始碼層級的測試強制此性質。

## G. 本階段驗證結果（實測）

`pytest -q` → **234 passed**（正式退出碼 0；本階段自 181 增加至 234）。

本階段另補上三項任務明確要求、且原本未完整涵蓋的檢查：
持有期間的**選擇理由事先說明**、**分組排序是否在不同時期維持**、
以及**價差是否只由少數交易日主導**（後兩項各自有對應測試與報告段落）。

## H. P2.3 訊號資訊量診斷摘要（2026-10-14，486 交易日）

前瞻報酬定義 `fwd_h(T) = open(T+1+h)/open(T+1) - 1`；統計單位為每日橫斷面 IC；
顯著性以 Newey-West 穩健 t 與移動區塊自助法檢定；36 個假設以 BH FDR 校正。

| 訊號 | h=5 平均 IC | NW t | p | FDR 校正後最小 p | 扣成本 Q5-Q1 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 月營收 YoY | **+0.0987** | 3.71 | 0.0002 | **0.0037** | **+0.350%** | 初步獲支持 |
| 合成 score | **+0.0705** | 3.02 | 0.0025 | **0.0228** | **+0.167%** | 初步獲支持 |
| 20 日動能 | +0.0545 | 2.04 | 0.0414 | 0.2131 | — | 未獲支持（校正後消失） |
| 均線多頭排列 | +0.0403 | 1.79 | 0.0727 | 0.3131 | — | 未獲支持 |
| K線綜合 | −0.0172 | −0.84 | 0.3987 | 0.6241 | −0.741% | 未獲支持 |
| 60 日動能 | +0.0056 | 0.18 | 0.8591 | 0.9104 | −0.465% | 未獲支持 |
| 三大法人近 3 日淨買超 | −0.0084 | −0.43 | 0.6698 | 0.8848 | −0.672% | 未獲支持 |
| 法人連續買超天數 | −0.0448 | −1.23 | 0.2176 | 0.4843 | — | 未獲支持 |
| RSI 逆勢 | −0.0367 | −1.30 | 0.1927 | 0.4843 | −1.278% | 未獲支持 |
| 券資比 | −0.0008 | −0.04 | 0.9692 | 0.9692 | −1.023% | 未獲支持 |
| 量能方向 | +0.0109 | 0.24 | 0.8123 | 0.8860 | — | 未獲支持 |
| 20 日突破／破線 | −0.0394 | −0.38 | 0.7029 | 0.8848 | — | 未獲支持 |
| 新聞情緒 | N/A（有效 0 天） | — | — | 1.0000 | — | **樣本不足** |

- 未校正即顯著 7 個；**BH 校正後僅 4 個**——多重檢定確實製造了假陽性。
- **全部 13 個訊號 × 3 個持有期間中，只有「月營收 YoY」的 Q5-Q1 完整單調**
  （h=1/5/10 皆為「是」），且其扣成本後價差為正（+0.350% / +1.143%）。
  `score` 的 Q5 最高但 Q2/Q3 並非遞增，屬「端點有效、中段無序」。
- **價差穩健性（最佳 5 日佔價差總和）**：月營收 YoY 為 32.9% / **15.9%** / 9.1%，
  `score` 為 246.3% / **18.5%** / 16.3% —— 兩者的價差分散在多數交易日，
  不是靠少數幾天。相對地，量能方向 h=1 為 **7120%**、均線多頭排列 h=5 為 **645%**、
  20 日動能 h=1 為 **198%**，價差完全由少數交易日主導（數字超過 100% 表示
  少數幾天不只貢獻全部、還被其他天抵銷掉一部分）。
- **分期間一致性（h=5）**：**沒有任何訊號在 train / valid / test 三個區段都維持單調排序**；
  唯一在單一區段單調的是 `score`，而它只在 train 成立。
  即使是最強的月營收 YoY，扣成本後價差在 valid 段為 **−0.230%**，
  在 train / test 才為正（+0.680% / +0.257%）。
- **注意幾個因子的有效樣本極少**（它們在多數交易日為常數，無法計算橫斷面相關）：
  20 日突破/破線 31 天、量能方向 72 天、法人連續買超天數 157 天、新聞情緒 0 天。
  這些因子的 IC 統計量建立在極少的有效日上，不應與其他因子等量齊觀。
- **最重要的樣本外發現**：`score` 的 IC 由 train +0.1170（p=0.0010）
  退到 valid +0.0502（p=0.30）、**test −0.0104（p=0.74）**；
  月營收 YoY 同樣由 train +0.1776（p=0.0022）退到 valid +0.0904（p=0.16）、
  test +0.0527（p=0.43）。**全期顯著主要由訓練段驅動，樣本外無法重現。**

## I. P2.4 策略失效歸因摘要（測試段 2026-04-17 ~ 2026-10-13，122 交易日）

| 情境（機制完全相同，只換分數） | 總報酬 | 年化 | MDD | Sharpe | 筆數 |
|---|---:|---:|---:|---:|---:|
| 真實訊號（策略） | +25.63% | +59.29% | −5.54% | 2.82 | 120 |
| **完美前瞻（不可交易，僅作上限）** | **+189.84%** | +777.08% | −3.14% | 10.63 | 120 |
| 隨機選股（中位數，200 次） | +13.09% | — | — | — | 120 |
| 隨機選股（5%~95%） | +3.11% ~ +23.27% | — | — | — | 120 |
| 0050 買進持有（同期） | +33.57% | — | — | — | — |

- 真實策略在隨機分布中位於 **97.5 百分位**，單尾經驗 p = **0.030**
  → 在此區段選股**優於隨機選股**（但仍不優於買進持有，超額 −7.94%）。
- 完美前瞻證明**機制的上限很高**（+189.84%），瓶頸不是交易規則撐不住訊息。
- **真正的瓶頸是曝險結構**：`top_n=5 × 單檔上限 10%` ⇒ 最大曝險僅 50%；
  實測平均曝險 **49.2%（現金 50.8%）**。在多頭區段，選股 α（約 +12.5 個百分點
  vs 隨機中位數）恰好被「只能吃到一半市場報酬」抵銷，
  毛報酬（+25.63% + 成本 8.23% ≈ +33.9%）與買進持有（+33.57%）幾乎相同。
- **週轉率拆解**：120 筆交易涉及全部 20 檔；**59.2% 是同一標的在出場後 5 日內
  再次進場**（`fixed` 持有 5 日到期就強制出場，即使訊號仍強）。
  累積雙邊週轉率 28.21x，年化單邊 24.63x。
- **訊號→持倉落差**：116 個進場日、候選 580 筆、實際成交 120 筆（**20.7%**）；
  落差來源包含已持有、漲停、資金不足與單檔預算不足。
- 流動性限制與漲停跳過在此區段皆為 0 次（此段未觸發）。

## J. 仍未覆蓋的風險（第三階段更新）

- **survivorship bias 仍在**：20 檔寫死、無下市股與歷史成分；
  改善方案見 `docs/DATA_SOURCES.md`（A 階段免費可做，B 階段待授權決策）。
- **訊號的樣本外優勢無法確立**：P2.3 的 test 段 IC 為負且不顯著；
  P2.4 的隨機控制雖為正（p=0.030）但只有單一區段、120 筆，
  屬於**提示性**而非確證。兩者都不足以支持投入調參。
- **新聞因子不修正（P2.6 消融驗證後的決定）**：四方案 86 個配對假設無一顯著、
  test 段 A ≡ B ≡ D 完全相同；分子效應在本資料中**不可估計**。
  待補跨期新聞資料後重評，不在證據不足時改權重。
- **子程序原型尚未接進 Worker**（B19）：逾時在真實 `ingest` 上仍只能事後偵測。
- **完美前瞻為不可交易的控制組**，不得被引用為績效目標。
- 交易執行層（模擬券商／委託狀態機）仍不存在；**正式交易維持關閉**。

---

## K. P2.6 B17 新聞因子消融驗證摘要（2026-10-14）

> 完整結果見 `docs/ABLATION_B17.md`；可重現命令 `python -m twstock.ablation`。

**方案（執行前固定）**：A 現行 / A\* 分子控制組 / B 移除＋重新標準化 /
C 固定權重 / D 可用性閘門。四方案共用同一面板、成本、`top_n=5`、`hold_days=5`、
`buy_threshold=0.2`，僅分數計算不同。

| 項目 | 實測 |
|---|---|
| 0.10 佔據分母的列數 | **9,719 / 9,719（100%）**；其中有新聞記錄 61 列、分子非 0 **29 列** |
| `W_A` 跨股票變動日 | **483 / 486**（0.53 ~ 1.05）→ 分母改變**非**單一正比例常數 |
| A→B 排名完全相同日 | **393 / 486（80.9%）**；Spearman 0.9996、top-5 重疊 99.6%、最大名次變動 2 |
| A→C / A→D 排名相同日 | 145/486（29.8%）／393/486（80.9%） |
| Rank IC（h=5 全期） | A +0.0705、B +0.0704、C +0.0696、D +0.0704（差 0.0001~0.0009） |
| 有新聞記錄日的可用前瞻報酬 | **0 天**（新聞全在樣本最末端）→ **分子效應不可估計** |
| 代數恆等式（9,719 列，clip 前） | `A−A*=0.10v/W`、`den_b=W−0.10`、`C=B·den_b` 最大誤差 ≤ 4.2e-17 |
| test 段交易結果 | **A ≡ B ≡ D 完全相同**（25.63%、120 筆、持倉 122/122、Jaccard 1.000） |
| train/valid 成本後日報酬配對差 | B 分別 −0.000016 / −0.000162，p=0.897 / 0.551（皆不顯著） |
| **多重檢定** | IC 差異 38 + 價差差異 38 + 交易層級 10 = **86 個假設，未校正顯著 0 個、BH 後 0 個** |

**結論（依驗收標準：只用 train/valid、需一致且可解釋的改善）**：
A−B 的 Rank IC 差在 train +0.0006、valid −0.0009 → **符號翻轉、方向不一致**；
交易層級兩段點估計略優但配對檢定皆不顯著，且 test 差異嚴格為 0。
**→ B17 會改變排名與成交，但不達修改正式策略的門檻：保留現行策略，記錄不確定性。**

**test 消耗聲明**：同一批 test 資料已在 P2.3／P2.4 反覆使用，
本階段 test 數字**不得宣稱為全新樣本外驗證**，且未用於挑選方案。

**驗證**：`pytest -q` → **253 passed**（+19）退出碼 0；
`python -m twstock.ablation`（完整）／`diagnostics --quick`／`cli validate`
／`cli --help` 退出碼皆 **0**。

---

# 附錄：第四階段（2026-10-15，P2.7 資料擷取狀態治理）

> 本階段**不修改交易策略、不調整因子權重、不追求回測績效**。
> 目標只有一個：讓「成功／空結果／部分成功／失敗」可被區分，停止無聲失敗
> 污染研究結果。所有數字皆來自實測（可重現命令見各節）。

## L. 本階段新發現的缺陷

### D16 — 新聞抓取失敗被記為成功　【嚴重】
- **位置**：`pipeline.ingest`（舊版）。
- **事實**：`n_news = sent.update_news_for(...)` 之後直接
  `store.log_ingest(run_at, sid, "news", ok=True, rows=n_news)`，
  `ok` 是**硬編碼常數**，與實際結果完全無關。
  `update_news_for` 在「沒有新聞」與「抓取失敗」都回傳 `0`，故兩者在紀錄上同形。
- **後果**：新聞抓取失敗在資料層**完全不可見**；健檢與失敗重試都以 `ok` 為依據。

### D17 — feedparser 不拋例外，網路失敗偽裝成「沒有新聞」　【嚴重】
- **位置**：`news_rss.fetch_google_news`（舊版）直接 `feedparser.parse(url)`。
- **事實**：feedparser 在網路失敗時**不拋例外**，只設 `bozo` 並回傳空 entries。
  因此「連不上」、「被限流」與「今天真的沒新聞」三者回應值都是 `[]`。
- **後果**：上層無法區分；D16 又把 `[]` 記成成功 → **失敗被寫成空結果並標記成功**。
  相同缺陷重複存在於 `sentiment/experts.py` 的三個來源。

### D18 — 發布時間缺失時以「今天」偽造　【高】
- **位置**：`news_rss.fetch_google_news`（舊版）
  `"published": ... if ts else today_str()`。
- **事實**：取不到來源的 `published_parsed` 時，**以抓取當日的日期代替**。
  舊版 `experts.py` 三個來源同樣如此。
- **後果**：舊資料中無法得知哪些 `published` 是來源真實日期、哪些是程式補上的，
  亦令無法驗證的時間進入情緒衰減窗口。

### D19 — 舊新聞的抓取成敗與時間戳來源不可追溯　【高，**不可修復**】
- **事實**：現有 237 則新聞全部寫入於狀態治理之前。舊 schema 只有 6 個欄位，
  既無抓取狀態也無時間戳來源；且因 D18，連「哪些日期是真的」都不可得。
- **定位**：這是**必須承認的資料損失**，不是可修正的 bug。
  依任務要求一律標為 `unknown`，**不以推測補造成功狀態**。
- 可重建的只有分數本身是否存在（`score` 欄位就是直接證據：
  NULL → `missing`、0 → `neutral`、其餘 → `scored`）。

### D20 — Worker 混同「來源正常但無資料」與「全部失敗」　【中】
- **位置**：`worker.run_cycle`（舊版）`ok = sum(1 for v in counts.values() if v and v > 0)`。
- **事實**：只要所有檔的價量列數都是 0 就拋「ingest 全部失敗」——
  即使資料來源均**正常回應且確實無新資料**；反之，部分失敗也無法從任何欄位看出。
- **後果**：**無法只依整體退出碼判定所有資料均成功**，這正是任務第 4.4 條禁止的情形。

### D21 — `write_df` 欄位無交集時靜默回 0　【中】
- **位置**：`store.write_df`（舊版）：`log.warning(...); return 0`。
- **事實**：有資料但欄位與資料表無交集（或表不存在）時回傳 0，
  呼叫端看到「0 列」便以 `ok=True` 記下。
- **後果**：**完全沒有寫入**卻被記錄為成功。現改擧 `WriteRejected`。

### D22 — 測試套件會以可寫模式開啟正式 DB　【中】
- **發現過程**：本階段新增遷移後，實測發現 `news` 表在未執行任何 `ingest`
  的情況下於 **2026-10-15T00:11:04** 被遷移（`meta.news_legacy_marked`）。
- **原因**：`tests/test_ops_worker.py::test_should_run_outside_trading_window`
  未使用 `temp_db`，卻經由 `worker._should_run` → `store.get_meta` →
  `store.read_sql` → `_init` 以**可寫模式開啟真實 DB**，
  即「跑 pytest 就會遷移使用者資料」。
- **修正**：`tests/conftest.py` 改為 **autouse** 隔離（store DB／DATA_DIR／
  worker 與 ops 的 STATE_DIR 全部導向 `tmp_path`）。已實測：
  完整 `pytest` 跑完後真實 DB 的雜湊**不變**。

## M. 資料流與失敗路徑盤點（排程 → 擷取 → 解析 → 驗證 → 寫入 → 狀態 → features）

```
Worker.run_forever / cli ingest
  └─ _should_run（休市不轮詢）
      └─ pipeline.ingest（一輪 = 一個 run_id）
          ├─ fm.fetch_benchmark/fetch_price   → None=失敗｜空表=成功無資料
          │     └─ _write_price：clean_prices → _write_with_retry → price 表
          ├─ _save(fetch_institutional/margin/revenue)
          ├─ _ingest_corporate_actions → build_corporate_actions → corporate_actions 表
          ├─ sent.update_news_for → news_rss.fetch_feed → news 表
          └─ 逐筆寫入 ingest_run（＋保留舊 ingest_log）
              ↓
          store.set_meta(last_ingest*／last_ingest_status)
      └─ run_scores → features.build_features（讀 news 的 published/score）
                     → signals 表 → worker 心跳（ingest_status/）
```

**已修正的無聲失敗路徑**（全部位於上述流程，皆有對應測試）：

| # | 路徑 | 修正後行為 |
|---|---|---|
| 1 | 新聞狀態硬編碼 `ok=True` | 依 `NewsIngestResult.status` 記錄 |
| 2 | `feedparser` 吞網路失敗 | `requests` 顯式取得 HTTP → network/timeout/http_status/parse 分類 |
| 3 | 缺失時間戳補「今天」 | `published=NULL` ＋ `timestamp_source='unknown'`（排除於衰減窗口） |
| 4 | 失敗與空結果同回 `0` | 回傳結構化結果，四態可分 |
| 5 | `write_df` 無交集回 0 | 擧 `WriteRejected`；呼叫端記 `failed/write` 並重試 |
| 6 | Worker 只看列數 | 依 `IngestSummary.overall_status`；部分失敗寫入心跳 |
| 7 | 舊 `ingest_log` 布林 `ok` | 回填 `ingest_run` 一律 `unknown`（只做一次、冪等） |
| 8 | 測試寫入真實 DB | autouse 隔離（D22） |

## N. 新增的狀態定義與資料表變更

**狀態（`twstock/data/status.py`）**：`success_with_data`／`success_empty`／
`partial_success`／`failed`／`skipped`／`unknown`。
`classify()` 的判定順序即語意保證：任何硬性錯誤分類
（network／timeout／http_status／parse／validation／write）一律得到 `failed`，
**永不會落成 `success_empty`**。

**錯誤分類**：`none`／`network`／`timeout`／`http_status`／`parse`／
`validation`／`write`／`empty`／`unknown`。

**新表 `ingest_run`**（主鍵 `run_id, dataset, stock_id`）：`run_id`／`dataset`／
`stock_id`／`run_at`／`source`／`task`／`started_at`／`ended_at`／`status`／
`error_class`／`error`／`rows_fetched`／`rows_parsed`／`rows_validated`／
`rows_written`／`data_start`／`data_end`／`attempts`／`source_version`／
`config_id`／`provenance`／`notes`。
來源版本（新聞）為情緒規則的詞表指紋，設定識別為股票池與回溯年數的短雜湊，
**皆不含機密**；所有錯誤訊息寫入前經 `status.redact` 遮蔽 token／api_key／Bearer 等。

**`news` 表新增 5 欄**：`published_at`／`fetched_at`／`timestamp_source`／
`score_kind`／`provenance`（CREATE 與 ALTER 兩路皆覆蓋，向後相容）。

## O. 修復前後差異（真實本機 DB，於副本上量測）

| 指標 | 修復前 | 修復後 |
|---|---|---|
| `news` 欄位數 | 6 | 11 |
| 可判定**抓取成敗**的列 | **0** | 237（全部 = `unknown`） |
| 可判定**時間戳來源**的列 | **0** | 237（全部 `unknown`） |
| 新聞列數／正·零·負分數 | 237／16·183·38 | 237／16·183·38（不變） |
| `published` NULL | 0 | 0 |
| 特徵面板列數／`sha` | 9,719／`c0fa6357e4af` | **完全相同** |
| `signals` 表 `sha` | `ebfb28c2618c` | **完全相同** |
| 面板 `news_present=1`／覆蓋率 | 61／0.628% | 61／**0.628%（不變）** |
| `ingest_log` 失敗筆數 | **0**（163 筆全為 `ok=1`，含 news 20/20） | 回填後 163 筆全為 `unknown` |

**這張表最重要的兩列**：特徵面板與訊號表**逐位元不變** →
schema 變更沒有造成非預期差異（任務 6.3）；新聞覆蓋率**不變** →
狀態治理**沒有、也不應該**提升預測能力（任務 6.4）。
真正改變的只有「可信度」：修復前對抓取成敗與時間戳來源的可判定性**都是 0**。

## P. 新聞資料覆蓋率與品質統計

`python -m twstock.cli datacheck`（實測輸出）：

| 項目 | 值 |
|---|---|
| 新聞列數 | 237（可追溯 0、不可追溯 **237**） |
| 發布時間可追溯 | 0（時間不明 237） |
| 有效情緒分數（scored） | 54 |
| 情緒中性（neutral） | 183 |
| 無情緒分數（missing） | 0 |
| 有效情緒覆蓋率 | 全部列 **22.8%**／可追溯列 **無資料** |
| 不可追溯率 | **100.0%** |
| 新聞抓取成功率／失敗率 | **無可判定紀錄**（20 筆狀態全為 `unknown`） |
| 涵蓋日期 | —（因無法確認哪些 `published` 為真） |

**如何解讀**：這不是「修復後變差」，而是**修復前根本沒有這些數字**。
過去唯一的成敗依據 `ingest_log` 顯示 163 筆全部成功、0 筆失敗，
但其中新聞的 `ok` 是硬編碼常數（D16），因此那份「完美紀錄」不可採信。
本階段選擇**如實承認不知道**，而不是繼續用一個假的成功率。

## Q. 資料遷移風險及無法追溯的舊資料

- **已發生的遷移**：本機 `twstock/data_store/twstock_v3.db` 已於
  **2026-10-15T00:11:04** 被遷移（原因為 D22，已修正）。遷移是**純新增**：
  補 5 個欄位、新建 `ingest_run`、把 `news.provenance`／`timestamp_source`
  設為 `unknown`、回填 163 筆舊紀錄為 `unknown`。
  **已驗證 `price`／`news` 分數／`signals` 內容與特徵面板逐位元不變**，
  且沒有任何列被覆寫或刪除。
- **遷移幂等性**：`_mark_legacy_news` 與 `_backfill_ingest_run` 均以 `meta` 鍵守門
  （`news_legacy_marked`、`ingest_run_backfilled`），且 `UPDATE` 只碰 `NULL` 列，
  重複執行不新增也不覆寫（有測試）。
- **不可追溯的資料（明確標記，不補猜）**：
  - 237 則新聞的**抓取成敗**：不可重建（舊 schema 無此資訊 + D16）。
  - 237 則新聞的**時間戳來源**：不可重建（因 D18 曾以「今天」偽造）。
  - 163 筆舊擷取紀錄的**成功有資料／成功無資料**：不可區分（舊 `ok` 只有布林）。
- **風險**：若日後有新程式讀 `news.published` 而未檢查 `timestamp_source`，
  會把「可能是偽造的日期」當成真日期。已在 `README.md` 與 `DATA_SOURCES.md`
  明文警告。

## R. 測試案例、退出碼與 CLI 驗證

新增 `tests/test_ingest_governance.py`（**43 項**，全部合成資料、**不連外**）：

| 任務要求 | 對應測試 |
|---|---|
| 正常取得資料 | `test_feed_success_with_data`、`test_scored_news_has_valid_sentiment` |
| 正常回應但無資料 | `test_feed_normal_response_without_entries_is_empty_not_failed`、`test_news_empty_result_status_is_success_empty` |
| 網路失敗與逾時 | `test_feed_network_failure_is_failed_not_empty`、`test_feed_timeout_is_failed_timeout`、`test_news_fetch_failure_is_recorded_as_failed` |
| HTTP 成功但解析失敗 | `test_http_200_but_unparseable_body_is_parse_failure`、`test_http_error_status_is_classified` |
| 部分資料成功、部分失敗 | `test_ingest_summary_partial_success_overall`、`test_worker_partial_failure_is_visible_not_full_success` |
| 寫入失敗及重試 | `test_write_retries_then_succeeds`、`test_write_failure_after_retries_is_failed_write`、`test_write_df_rejects_no_intersecting_columns` |
| 重複執行與幂等性 | `test_news_write_is_idempotent`、`test_ingest_run_records_do_not_duplicate_on_rerun` |
| 舊資料與 unknown | `test_legacy_ingest_log_backfilled_as_unknown`、`test_legacy_news_marked_untraceable_but_score_kind_derived`、`test_legacy_rows_not_counted_as_fetch_success`、`test_unknown_status_is_not_a_claimed_success` |
| 新聞中性／缺值／無新聞／失敗 | `test_neutral_news_distinguished_from_no_news`、`test_missing_timestamp_is_not_fabricated`、`test_failed_fetch_writes_nothing_and_is_distinguishable` |
| 未來新聞不得外洩 | `test_future_dated_news_is_not_stored`、`test_news_never_affects_dates_before_publication`、`test_unknown_timestamp_news_is_excluded_from_signals` |
| 失敗不得被 worker 記為成功 | `test_worker_raises_when_every_dataset_failed`、`test_worker_accepts_normal_empty_source_as_not_failure`、`test_worker_reports_success_with_data_as_ok`、`test_ingest_summary_all_failed_only_when_everything_failed` |
| 機密不寫入 | `test_secrets_are_redacted_in_records` |
| 「失敗不得降級成空結果」 | `test_hard_failure_never_classified_as_success_empty`（參數化 6 種錯誤分類） |

| 命令 | 結果 | 退出碼 |
|---|---|---:|
| `python -m pytest -q` | **296 passed**（上階段 253，+43） | **0** |
| `python -m twstock.cli datacheck` | 產出資料品質報告 | **0** |
| `python -m twstock.cli validate` | 既有驗證未受影響 | **0** |
| `python -m twstock.cli health` | 顯示擷取狀態（新增） | **0** |
| `python -m twstock.cli status` | — | **0** |
| `python -m twstock.cli --help` | 列出 `datacheck` | **0** |

**每個 commit 皆獨立驗證**：在 detached worktree 檢出各 commit 的完整快照執行 `pytest`
（commit 1 → 253 passed、commit 2 → 253 passed 且無警告、commit 3 → 296 passed）。

## S. 本階段仍然未解決的問題

1. **新聞的抓取成敗永遠不可追溯**（D19）：237 則全部 `unknown`，不可重建。
2. **RSS 無法回溯歷史**：只能取得「最近 `days` 日」，因此新聞因子的跨期評估
   先天受限；在按日累積寫入並長期保存（或改用付費歷史新聞源）之前，
   重評 B17 仍然沒有可信的分母。
3. **公開 RSS 無 SLA、無備援來源**（失敗現已可觀測，但沒有第二來源交叉驗證）。
4. **未驗證真實網路失敗情境**：測試全部以注入的假 getter 驅動（**刻意如此**，
   任務要求測試不得依賴真實網路）。因此 `requests` 與 Google News 的實際
   互動行為（如限流回應）**未在本階段實測**——不得宣稱已驗證。
5. **worker 的部分失敗僅止於「可見」**：心跳會顯示 `partial_success`，
   但尚未依失敗來源自動重試或分流。
6. **未修改交易策略、曝險上限或持有期間；未調因子權重；未安裝 ML 套件；
   未開設正式交易；不宣稱新聞因子或整體策略具備可交易優勢。**
