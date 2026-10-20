# P5：風險閘門與故障排除（Risk Gates & Troubleshooting）

> 核心原則：**fail-closed**——無法確認行情新鮮度、資料來源失敗、資料筆數為零、
> 帳務不一致或 Kill Switch 啟用時，一律停止新增模擬交易並記錄原因。
> 停止交易**不等於清倉**：持倉保留、既有未完成訂單依既有規則處理，
> 本系統**從不**因閘門而自動賣出或自動取消持倉。

## 1. 兩層閘門：新單 vs 既有單成交

`twstock/gate.py::evaluate()` 同時產出兩個獨立判定（互不混用）：

| 判定 | 不通過時 |
|---|---|
| `allow_new_orders`：允許建立**新的**模擬訂單（買與賣） | 拒單（原因入 `paper_order.reject_reason` 與 `paper_event`） |
| `allow_fills`：允許**既有未完成訂單**繼續成交 | 成交暫停（`no_fill` 事件記 `trading_halted`＋原因，訂單維持原狀態不被誤判） |

**只擋新單、不擋既定成交**（沿用 P4 語意）的條件：行情過期、資料筆數不足、
非交易日、盤中無可驗證即時行情。
**新單與成交都擋**的條件：Kill Switch、帳務核對不一致、帳戶不可確認。

## 2. 閘門實際阻擋交易的條件（完整清單）

判斷順序（所有問題都會被記錄，前面失敗不掩蓋後面）：

```
Kill Switch → 日線基礎 → 帳戶/帳務核對 → 時段規則 → 盤中即時基礎
```

### 2.1 Kill Switch（擋新單＋擋成交）
- 檔案旗標 `data_store/state/KILL_SWITCH` 存在即生效；原因寫入檔案內容。
- **不會自動解除**：只能手動 `python -m twstock.cli paper resume`。
- 觸發：`paper kill --reason …`；自動輪次不會繞過它。

### 2.2 日線基礎（擋新單；來自 price 表）
| 條件 | 判定 |
|---|---|
| price 表零列（無行情資料） | 阻擋＋quality 階段 `failed` |
| 最新交易日延遲 > `max_data_staleness_days`（預設 3 天） | 阻擋（`行情過期 N 天`） |
| 最新一列價格異常（缺值/≤0） | 阻擋（`資料品質不合格`） |
| 價量表存在**晚於系統時間**的列 | quality 階段 `failed`（時間不一致；由成交時點規則擋在成交層） |

### 2.3 帳戶與帳務核對（擋新單＋擋成交）
| 條件 | 判定 |
|---|---|
| 未建立 paper 帳戶 | 阻擋（`paper init` 前零訂單） |
| `reconcile()` 發現不一致（現金/餘額鏈/成交金額/持倉重放/訂單成交量/快照恆等式） | 阻擋；**只回報不修帳**（`p5 audit`、`/api/p5/audit`） |
| 核對本身無法執行（DB 錯誤） | fail-closed 阻擋 |

### 2.4 時段規則（擋新單；`kind=policy` 或 `blocked`）
| 時段（本地時間） | 新單 |
|---|---|
| 週末（週六/日） | 拒絕（`policy`；**無假日曆**，國定假日由 2.2/2.5 補防） |
| 盤前 08:30–09:00 | 允許（EOD 基準；成交仍走 T+1 開盤規則） |
| 盤中 09:00–13:30 | **要求可驗證的即時行情**（見 2.5），否則拒絕（`blocked`） |
| 盤後 13:30–次日 08:30 | 允許（本專案設計路徑：EOD 訊號 → T+1 成交） |
| 收盤前（訊號日 = 今日且 < 13:30） | 拒絕（訊號需以今日收盤資料計算；P4 既有規則） |

### 2.5 盤中即時基礎（盤中新單；來自 quote 表）
**只有日線或延遲資料時，不得假裝具備可交易的盤中行情。** 任一條件即拒絕：

1. 未設定 `TWSTOCK_QUOTE_SOURCE`（不臆測預設來源）。
2. 來源種類不是 `realtime`（`finmind_minute` 是**延遲**、`twse_openapi`/日K 是**盤後**
   → 一律不得當作盤中可交易行情）。
3. quote 表零列（來源失敗或從未寫入）。
4. `market_time`（**行情時間**）不可得/不可解析（時間戳未知）。
5. 行情時間晚於系統時間 > 120 秒（**時鐘不一致**）。
6. 行情年齡 > 300 秒（`DEFAULT_MAX_AGE_S`；**行情過期**）。
7. `fetched_at`（**抓取時間**）缺失。
8. 既有 `quote_health` + `intraday_signal_gate` 未通過（含合成資料、來源失敗）。

### 2.6 「已成功抓取」≠「足以支援這次交易」
閘門的 `details.quote` 同時記錄 `fetched_at` 與 `market_time`：
- `fetched_at` 只證明**來源有被呼叫**；
- `market_time` 才決定**資料夠不夠新**。
抓到資料但市場時間過期/未知 → 仍然禁止交易（第 4、6 條）。

### 2.7 輪次層阻擋（協調器，擋整個 paper 階段）
| 條件 | 行為 |
|---|---|
| `ingest`/`quotes`/`quality`/`signals` 任一 `failed` | paper 階段 `skipped(kind=blocked)`：`上游階段失敗，停止新模擬交易（fail-closed）` |
| `p5 run` 於唯讀模式 | 整輪 `skipped`（不寫任何資料） |
| 已有輪次在執行（鎖被存活 PID 持有） | 拒絕第二個協調器（`busy`，退出碼 4） |
| 單輪逾時（`TWSTOCK_P5_TIMEOUT_S`） | 後續階段 `skipped(逾時)`，輪次不會永久卡住 |

## 3. 拒單/成交暫停後的既有風險語義（保留，不改變）

- **持倉保留**：閘門不賣出、不清倉；`paper_position` 只由成交序列重放決定。
- **停止新單**：買與賣都不建立（含拒絕理由可稽核）。
- **既有未完成訂單**：行情問題下仍可依已存行情成交（P4 語意）；只有
  Kill Switch/帳務不一致才暫停成交；過期（`good_til_days`）或人工
  `paper cancel`（P5 未自動新增取消路徑）才會終結。
- **風險上限**（單筆金額、部位占比、單日損失、回撤、開單數、行情過期）
  與 Kill Switch 的雙重檢查（下單前＋成交前）全部沿用 P4，未放寬任何參數。

## 4. 故障排除

| 症狀 | 診斷命令 | 常見原因與處置 |
|---|---|---|
| `p5 run` 退出碼 3、輸出 `weekend`/`before_eod`/`already_done_today` | `p5 status` | 排程冪等屬正常；要強制：`p5 run`（不加 `--due`） |
| 退出碼 4「已有 P5 輪次在執行」 | `type data_store\state\p5run.lock` | 另一輪仍在跑；若 PID 已死，重啟會自動回收殘留鎖 |
| paper 階段顯示 `skipped(blocked)` | `p5 status` + `p5 errors` | 看拒絕原因：上游失敗/閘門/帳戶未建立；對應 §2 |
| 閘門顯示「行情過期 N 天」 | `python -m twstock.cli health` | 執行 `python -m twstock.cli ingest` 更新日K；或等排程 |
| 盤中新單全被拒、理由含「盤中」 | `p5 gate` | 未設定或非即時來源 → 設 `TWSTOCK_QUOTE_SOURCE`（**注意：目前無已驗證的免費即時來源**，延遲來源也仍會被拒） |
| `reconcile` 不一致、`p5 audit` 退出碼 1 | `python -m twstock.cli paper reconcile` | 逐一列出問題；**系統只阻擋新交易、不修帳**——修復需人工檢查後另行決策（本階段無自動修復工具） |
| 心跳很久沒更新但程序活著 | `p5 status` / `health` | `health` 會顯示 `data_stale`（程序活著但資料過期）；檢查網路與排程 |
| 連續失敗 | `p5 errors` | 每階段最多 `TWSTOCK_P5_ATTEMPTS` 次（指數退避）；不會無限重試 |
| 疑似重複訂單/成交 | `p5 audit`（orders 檢查） | `client_order_id`/`fill_id`/帳本 `ref` 三層唯一；重複只會出現於人為繞過引擎直寫 DB |
| Kill Switch 卡住無法交易 | `paper risk` 顯示原因 | 手動 `paper kill --reason` 啟用；**只能手動** `paper resume` 解除（設計如此，防自動解除） |
| Telegram 沒收到事件 | `p5 alerts` 看 `delivered` | `not_configured` = 未設 `TWSTOCK_ALERT_TELEGRAM=1`；`telegram_failed` = token/chat_id 或網路問題。事件本身必已寫入 `run_alert` 表 |

## 5. 唯讀與安全邊界（不可違反）

- 稽核/健康/閘門查詢**預設唯讀**；發現不一致**只回報與阻止新交易**，
  不修改歷史帳務、不靜默修復（測試以封死寫入函式驗證）。
- 監控 API 全部 GET；**無下單/成交端點**（測試掃描 `POST` 路徑為空）。
- 不連券商、不呼叫真實下單 API（`paper.py` 無網路套件 import，AST 掃描強制）。
- `TWSTOCK_MODE=live` 需 `TWSTOCK_LIVE_ACK=1` 才會生效，且 `paper` 引擎在
  live 模式下**直接拒絕執行**（`LiveModeForbidden`）。

## 6. 相關文件

- [`LOCAL_AUTOMATION.md`](LOCAL_AUTOMATION.md)：啟停、排程、設定與環境影響。
- [`PAPER_TRADING.md`](PAPER_TRADING.md)：帳務閉環與成交模擬假設。
- [`QUOTE_SOURCE_ASSESSMENT.md`](QUOTE_SOURCE_ASSESSMENT.md)：行情來源評估
  （**真實行情來源尚未實際連線驗證**，一律標示「未驗證」）。
