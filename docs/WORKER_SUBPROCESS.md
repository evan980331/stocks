# Worker 子程序化：架構評估與最小原型（P3.2）

- 文件日期：2026-10-14
- 原型程式：`twstock/jobs.py`；測試：`tests/test_jobs.py`（23 項）
- 狀態：**架構 DONE、原型 DONE + VERIFIED；全面遷移 TODO（本階段刻意不做）**

---

## 1. 為什麼要做（目前架構的實際缺陷）

現行 `worker.py` 在同一個行程內執行耗時任務（`run_cycle` → `ingest` + `run_scores`）。
後果：

| 缺陷 | 現況 |
|---|---|
| 逾時只能事後偵測 | `task_started_at` 殘留 + `derive_state` 判定 `task_running`；**無法中止**已卡住的任務 |
| 不可安全中止 | 同步任務在 Python 內沒有跨平台的強制中止方式 |
| 崩潰無法隔離 | 任務中一個未捕捉例外會影響同一行程的心跳與狀態機 |
| 停止語意模糊 | 舊版以 `CTRL_BREAK_EVENT` 當「優雅停止」，實測證明它其實是立即終止（見 §3） |

## 2. 架構設計

### 責任劃分
| 角色 | 負責 |
|---|---|
| **父程序**（`JobRunner`，在 Worker 內使用） | 單一行實例鎖、排程節奏、心跳、健康狀態；啟動子程序、等待、逾時處理、記錄結果。**不執行任務本身** |
| **子程序**（`python -m twstock.jobs --task …`） | 只做一個任務，並把結果**原子寫入** `--result` JSON |

### 停止階梯（順序固定，且不把強制當優雅）
1. **合作式取消**：建立 `<name>.cancel` 檔；子程序在檢查點自行收尾並寫出
   `{"status": "cancelled"}`。**這是唯一跨平台且真正優雅的機制。**
2. **可攔截訊號（僅 POSIX）**：`SIGTERM`。
3. **強制終止**：只針對本子程序的 PID 行程樹
   （Windows `taskkill /PID <pid> /T /F`；POSIX `SIGKILL`），並在結果標示 `forced=true`。

Python 的 `finally` 區塊是子程序唯一的收尾保證，因此階梯的**第一段必須是
合作式機制**，而不是訊號。

### 邊界與安全
- **單一實例**：仍由 `worker.WorkerLock` 負責；`JobRunner` 不管鎖。
- **防重**：`_should_run` 的 `already_done_today` 仍在上游把關；
  子程序只做「一次任務」，重跑靠 `store.write_df`（INSERT OR REPLACE）冪等。
- **誤殺防護**：模組**完全沒有 `os.kill`**（測試以原始碼檢查強制此性質），
  只操作自己啟動的 `Popen`／PID。`CREATE_NEW_PROCESS_GROUP` 保留，
  用途是讓子程序不因父程序主控台的 Ctrl+C 而意外死亡。
- **憑證**：子程序輸出只進 `<name>.log`，不寫入環境或結果檔。

## 3. 實測發現（本階段最重要的一項修正）

**Windows 的 `CTRL_BREAK_EVENT` 不是可攔截的中斷，而是立即終止。**

實測：對以 `CREATE_NEW_PROCESS_GROUP` 建立的子程序送出 `CTRL_BREAK_EVENT`，
子程序以 `0xC000013A`（`STATUS_CONTROL_C_EXIT`）結束、**沒有**丟出可捕捉的
`KeyboardInterrupt`；連明確 `except KeyboardInterrupt: continue` 的任務
（`stubborn`、`sqlite-atomic`）也照樣被殺。而 `SQLite` 交易中的任務就是這樣被
硬殺的 —— 若把這個訊號當成「優雅停止」，就會回報成安全停止，也就是
**又一個「假安全停止」**（P3.1 已修過同類型缺陷）。

因此本原型**移除**了 `CTRL_BREAK_EVENT`，改以合作式取消為第一段。

另外過程中發現：對**不在獨立程序群組**的子程序送 `CTRL_BREAK_EVENT`，
訊號會投遞到整個主控台程序群組，**會殺掉父程序與使用者開的其他行程**
（實際發生時整條命令無聲消失）。這是移除該訊號的第二個理由。

## 4. 失敗情境與測試對應

| 情境 | 原型行為 | 測試 |
|---|---|---|
| 正常完成 | `ok`，結果檔存在 | `test_successful_task_reports_ok_with_result` |
| 任務例外 | `failed`，錯誤字串完整保留、不吞例外 | `test_task_exception_is_reported_as_failure` |
| 子程序崩潰（無結果檔） | `crashed`；**退出碼 0 但缺結果檔也不當成功** | `test_child_crash_without_result_is_detected`、`test_classify_never_treats_clean_exit_without_result_as_success` |
| 逾時 + 合作式 | `cancelled`、`forced=false` | `test_timeout_on_cooperative_task_stops_gracefully` |
| 逾時 + 不合作 | `timeout`、`forced=true`，程序確實消失 | `test_timeout_on_uncooperative_task_is_forced_and_really_dies` |
| SQLite 交易中被硬殺 | 交易完整回復、`integrity_check=ok`、**0 筆半截資料** | `test_kill_during_sqlite_transaction_leaves_no_partial_rows` |
| 訊號不足時不得變成 timeout | 已自行收尾的取消不得被記成逾時 | `test_classify_does_not_turn_graceful_cancel_into_timeout` |
| 停止一個任務誤殺其他 | 只操作自己的 PID；A 停、B 存活 | `test_stopping_one_job_leaves_other_jobs_running` |
| 依名稱的批次終止 | 原始碼不得出現 `os.kill`／`pkill`／`killall` | `test_jobs_source_proves_no_signal_based_kill` |
| 結果檔原子性 | 先寫 `.tmp` 再 rename，不留殘檔 | `test_write_result_is_atomic_and_leaves_no_tmp` |

### SQLite 中斷復原的實測數字
`sqlite-atomic` 在交易中寫入 **300 筆**後刻意停住（且忽略中斷訊號），
父程序逾時後強制終止。事後以新連線開啟 DB：

- `probe` 表存在（交易外的 `CREATE TABLE` 已提交）
- `SELECT COUNT(*) FROM probe` = **0**（交易完整回復）
- `PRAGMA integrity_check` = `ok`

結論：**交易式寫入 + 冪等 `INSERT OR REPLACE` 可以安全承受中途硬殺**；
不必為了「安全」而避免強制終止。

## 5. 是否要全面遷移？

**建議：先遷移「可安全獨立執行」的長任務，不要一次重寫。**

- 對象：`ingest`（最長、最常卡在網路）與 `run_scores`（CPU 密集）。
- 保留在主行程：排程判斷、心跳、健康狀態、通知。
- 遷移前置：把 `run_cycle` 拆成「純任務函式」與「狀態更新」，讓子程序只依賴前者；
  子程序的結果以 JSON 回傳，由父程序寫心跳。
- 不做全面子程序化的理由：目前 Worker 已能滿足「每 30 分鐘一次、單一實例、
  可停止、可觀測」的需求；重構本身會增加複雜度與新的失敗模式，
  應等到真的出現「卡住的任務需要被中止」的實際案例再遷移。

## 6. 未覆蓋的風險

- **尚未在真實 ingest 上使用子程序**：原型只驗證了合成任務；
  真實 FinMind 抓取尚未以子程序執行過。
- **尚未把 `JobRunner` 接進 `worker.run_forever`**：目前 Worker 仍走同步路徑。
- **逾時門檻未經實測校準**：`DEFAULT_TASK_TIMEOUT=3600` 仍是舊值，
  真實一輪 ingest 的分布尚未量測。
- **子程序層級的資源限制未實作**：無記憶體上限、無 CPU 上限。
- 本階段**仍不涉及任何交易行為**，紙上模式與 Kill Switch 皆未改動。
