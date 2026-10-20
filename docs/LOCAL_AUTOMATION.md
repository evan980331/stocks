# P5：Windows 本機自動化執行與操作手冊（Local Automation）

> 本階段是**工程可靠性**：行情更新 → 資料品質檢查 → 訊號產生 → 模擬交易 →
> 帳務核對 → 健康狀態與異常事件。**全程為模擬交易**（paper），不連券商、
> 不呼叫真實下單 API、不假裝雲端 24/7。所有自動化都跑在單機 + SQLite 上。

## 1. 元件與職責

| 元件 | 職責 | 進入方式 |
|---|---|---|
| `twstock/orchestrator.py` | 輪次協調：七階段依序執行、階段紀錄、冪等續跑、單實例鎖、重試/逾時、事件 | `cli p5 …` / `worker --p5` |
| `twstock/gate.py` | 行情品質閘門（fail-closed）：決定「能不能下新單／既有單還能不能成交」 | `cli p5 gate` |
| `twstock/worker.py` | 常駐監控（收盤後 EOD 更新）；`--p5` 模式改跑協調器輪次 | `cli start worker` / `cli worker --p5` |
| `twstock/ops.py` | 服務生命週期（start/stop/status/health，PID 檔＋停止旗標） | `cli start/stop/status/health` |
| `twstock/paper.py` | 模擬交易引擎（帳務閉環；受閘門約束） | `cli paper …` |

工作順序（`run_round` 的七階段，寫入 `run_stage` 表）：

```
ingest（行情更新）→ quotes（盤中行情輪詢，可略過）→ quality（資料品質）
→ signals（訊號）→ paper（模擬交易）→ reconcile（帳務核對）→ health（健康/事件）
```

每輪唯一識別碼 `run_id`（`p5-YYYYMMDDTHHMMSS-xxxx`）可追查：
`run_stage`（含 ingest run_id、signal_date）→ `paper_order.reason`（含 run_id）
→ `paper_fill` → `paper_ledger`。

## 2. 操作命令一覽

```bat
:: 手動執行單輪（預設強制執行；同日重跑冪等，不會重複下單）
python -m twstock.cli p5 run

:: 依排程判定執行（週末/16:00 前/當日已有成功輪 → 跳過，退出碼 3）
python -m twstock.cli p5 run --due

:: 狀態查詢：最近 3 輪與各階段結果、目前閘門判定
python -m twstock.cli p5 status

:: 查看最近錯誤（失敗/被阻擋的階段 + 異常事件）
python -m twstock.cli p5 errors

:: 帳務稽核（唯讀；不一致 → 退出碼 1，只回報不修帳）
python -m twstock.cli p5 audit

:: 行情品質閘門目前是否允許新模擬交易（不允許 → 退出碼 1）
python -m twstock.cli p5 gate

:: 異常事件紀錄（含實際送達狀態）
python -m twstock.cli p5 alerts

:: 既有服務生命週期（不受 P5 影響）
python -m twstock.cli start worker|dashboard
python -m twstock.cli status
python -m twstock.cli health
python -m twstock.cli stop all
```

`p5 run` 退出碼：`0` success、`1` partial_success、`2` failed、`3` skipped
（含非排程時段/當日已完成）、`4` 已有輪次在執行（拒絕同時啟動）。

唯讀監控 API（全部 GET，**無任何下單端點**）：

| 端點 | 內容 |
|---|---|
| `GET /api/p5/health` | 程序存活＋資料新鮮度＋錯誤率＋帳務＋閘門＋最近輪次 |
| `GET /api/p5/audit` | 帳務稽核（`readonly: true`；不一致只回報） |
| `GET /api/p5/gate` | 閘門判定與原因 |
| `GET /api/p5/runs?limit=5` | 最近輪次與各階段結果 |
| `GET /api/p5/alerts?limit=50` | 異常事件（`delivered` 如實記錄送達狀態） |

## 3. Windows Task Scheduler 排程（兩種方式，可併存）

**方式 A（建議）：Task Scheduler 每日觸發 `p5 run --due`**

```bat
:: 交易日 16:00 執行一輪（--due：週末或當日已完成 → 跳過且不重複下單）
schtasks /Create /TN "TWStock_P5_Round" ^
  /TR "\"C:\Path\To\.venv\Scripts\python.exe\" -m twstock.cli p5 run --due" ^
  /SC DAILY /ST 16:00 /F

schtasks /Run   "TWStock_P5_Round"     :: 手動觸發一次
schtasks /Query "TWStock_P5_Round" /V  :: 查看上次結果
schtasks /Delete "TWStock_P5_Round" /F :: 移除排程
```

**方式 B：常駐 Worker 以 P5 輪次作為任務**

```bat
python -m twstock.cli start worker          :: 預設：既有 EOD ingest+打分
python -m twstock.cli worker --interval 1800 --p5   :: 前景：改跑 P5 七階段輪次
```

`--p5` 模式下，Worker 以 `trigger=worker` 呼叫協調器（同日冪等、同鎖、
同重試）；輪次 `partial_success` **不**計為 Worker 任務失敗（心跳的
`last_cycle_status` 會如實顯示 `partial_success`，不會誤報完整成功）。

> **排程不假設電腦永遠開機**：錯過的時段不會補跑「當日輪」（冪等設計）；
> 下一個交易日 16:00 自然恢復。若要手動補一輪：`p5 run`。

## 4. 範例設定檔（預設即安全模式）

`copy .env.example .env` 後**維持預設**就是安全模式：

```bat
TWSTOCK_MODE=paper        # live 預設關閉（需 TWSTOCK_MODE=live + TWSTOCK_LIVE_ACK=1）
# TWSTOCK_QUOTE_SOURCE=   # 未設定 → 不輪詢行情；盤中新單一律 fail-closed 禁止
# TWSTOCK_ALERT_TELEGRAM= # 未設定 → 事件只寫 run_alert 表與 CLI，不外送
# TWSTOCK_P5_TIMEOUT_S=1800   # 單輪逾時預算（秒），超時階段標記 skipped
# TWSTOCK_P5_ATTEMPTS=3       # 單一階段重試次數上限（含首次，指數退避）
```

**設定缺漏時的安全行為**（一律 fail-closed，且記錄原因）：

| 情況 | 行為 |
|---|---|
| 未建立 paper 帳戶 | paper/reconcile 階段 `skipped(kind=blocked)`，零訂單 |
| `TWSTOCK_READONLY=1` | 整輪直接 `skipped`（不寫任何資料） |
| 未設定 `TWSTOCK_QUOTE_SOURCE` | quotes 階段 `skipped(policy)`；**盤中新單被閘門禁止**，盤後 EOD 單仍以日K為準可執行 |
| 行情來源失敗／筆數為零／過期 | quality `failed` → paper 階段 `skipped(blocked)`，零訂單 |
| 帳務核對不一致 | 閘門同時擋新單與成交；稽核只回報、不修帳 |
| Kill Switch 啟用 | 新單＋成交全停、記錄觸發原因；**只可手動** `paper resume` 解除 |

## 5. 環境影響（如實說明，不假設永遠正常）

| 情境 | 對工作流的影響 | 恢復方式 |
|---|---|---|
| 電腦休眠 | Task Scheduler 通常會在喚醒後補觸發（依 Windows 版本與設定）；**不保證** | 喚醒後 `p5 status` 檢查；必要時 `p5 run` |
| 關機/錯過排程 | 當日輪未執行 → 當日**不補跑**（冪等） | 次一交易日自動恢復；或手動 `p5 run` |
| 網路中斷 | ingest/quotes 失敗 → 階段 `failed`、輪次 `partial_success`/`failed`；**不得把失敗記成成功** | 網路恢復後下輪自動重試（有限次＋退避） |
| FinMind 免費配額（300~600 次/小時） | 來源回限流 → 該階段失敗並記錄錯誤分類；日K整輪仍在配額內 | 隔輪重試；配額問題見 `docs/DATA_SOURCES.md` |
| 非交易日/週末 | `round_due` 判定 `weekend` → 輪次 `skipped`；即使手動 `p5 run`，週末新單也被閘門以 `policy` 理由拒絕 | 交易日恢復 |
| 國定假日 | **沒有假日曆**（已知限制）：白天可能被當成交易時段 | 盤中無新鮮行情 → 閘門 fail-closed；日線過期逾限 → quality failed |
| 行情來源未驗證 | FinMind 分鐘資料為**延遲**、MIS 條款限制未啟用；**本階段未對任何真實行情端點連線驗證** | 見 `docs/QUOTE_SOURCE_ASSESSMENT.md`（標示「未驗證」） |

## 6. 故障恢復（自動）

- **崩潰/強殺後重啟**：未結束的輪次（`ended_at IS NULL`）下次自動續跑；
  已完成階段**不重複執行**（不重複下單、不重複帳務）；執行中的階段重做。
- **殘留鎖**：`state/p5run.lock` 的 PID 已死 → 自動回收；存活 → 拒絕第二個協調器（退出碼 4）。
- **單輪逾時**：超過 `TWSTOCK_P5_TIMEOUT_S` → 後續階段標記 `skipped(逾時)`，輪次不會永久卡住。
- **資料庫交易失敗**：paper 引擎 `BEGIN IMMEDIATE` 整筆回滾，不留半套訂單/成交/帳本；
  協調器只記錄結果，**絕不修改歷史帳務**。
- **Worker 中斷**：既有 `task_started_at` 殘留偵測（重啟標記 `interrupted`）不變。

## 7. 敏感資訊

- API 金鑰/密碼只放本機 `.env`（`.gitignore` 已排除 `.env`、`data_store/`、`*.log`）；
  **不得**提交個人路徑、金鑰或 `.env` 內容到 Git。
- Telegram 通道需同時設定 `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` **並**
  `TWSTOCK_ALERT_TELEGRAM=1` 才會外送；否則 `run_alert.delivered = not_configured`
  （**不假裝已送達**）。
- 事件/錯誤寫入 DB 前一律經 `status.redact` 遮蔽金鑰與密碼。

## 8. 相關文件

- [`RISK_GATES.md`](RISK_GATES.md)：閘門條件、fail-closed 語意與故障排除。
- [`PAPER_TRADING.md`](PAPER_TRADING.md)：模擬交易帳務閉環與限制。
- [`WORKER_SUBPROCESS.md`](WORKER_SUBPROCESS.md)：子程序逾時與停止階梯。
