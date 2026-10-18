# P4：模擬交易引擎（Paper Trading Engine）設計文件

> **本階段只做虛擬交易。** 所有資金是模擬的、所有成交是模擬的。
> 引擎內**不存在任何券商 API 或網路呼叫路徑**（`tests/test_paper.py`
> 以 AST 掃描強制檢查 import 面），不得接入真實下單，不自動進入 P5。
> 模擬成交只是對真實交易的**近似**，**不得宣稱與實際券商成交完全一致**。

## 1. 定位與架構

```
行情資料（price 日K表） ─┐
訊號（signals，T 日收盤）─┤
                         ▼
              twstock/paper.py  PaperEngine
   ┌──────────────┬──────────────┬──────────────────┐
   │ 下單（前）     │ 成交模擬（T+1）│ 帳務（同一 DB 交易）  │
   │ risk 檢查      │ 行情光標推進    │ 訂單→成交→帳本→持倉   │
   │ 拒單入庫稽核   │ 滑價／費率      │ 任一失敗整筆 ROLLBACK │
   └──────────────┴──────────────┴──────────────────┘
                         │
        CLI（cli paper …）／儀表板唯讀 API（/api/paper/*）
```

- **單一帳務寫入口**：所有資金變動都走 `PaperEngine._tx()`（SQLite
  `BEGIN IMMEDIATE`），不存在繞過帳本的寫入路徑。
- **既有機制沿用**：成交成本 `costs.CostModel`（費率可設定、逐筆記錄於
  `price_source`）、風險限制與 `KillSwitch`（`risk.py`）、資料隔離
  （conftest autouse 暫存 DB）、行情健康閘門概念（P3 `intraday_signal_gate`
  以「資料新鮮度」形式整合）。
- **不重複造輪子**：沒有新增任何與既有功能等價的表或服務；`paper_*`
  表族只承載交易帳務，**與研究資料（price/signals/quote）完全分離**——
  回測結果不得寫成持倉，研究資料不得當交易帳本。

## 2. 資料模型（`store.SCHEMA`，全部 `paper_*` 表）

| 表 | 內容 | 關鍵設計 |
|---|---|---|
| `paper_account` | 帳戶、初始資金、現金、狀態 | `mode` 恆為 `paper`；`name` 唯一 |
| `paper_order` | 訂單生命週期 | `client_order_id` 唯一（重複防護）；`signal_date` 稽核 T+1；拒單也入庫 |
| `paper_fill` | 成交 | `fill_id = order_id:fill_date` 唯一（重播防護）；含 `raw_price_utwd`（滑價前）與 `fee/tax` |
| `paper_position` | 持倉快取 | 與成交重放結果核對；主鍵 (account, stock) |
| `paper_ledger` | 有號現金事件 | `ref` 唯一（重播防護）；`balance_utwd` 餘額鏈 |
| `paper_snapshot` | 資產快照 | `as_of` 唯一；equity = cash + market_value |
| `paper_event` | 稽核事件 | 下單/拒單/轉移/成交/未成交原因/取消/停止 |

**精確數值**：所有金額為「微元」整數（`*_utwd`，1 utwd = 10⁻⁶ TWD，
`MICRO = 10**6`），數量為整數股；**帳務路徑無浮點累積**（浮點只出現在
顯示與行情讀取的瞬間，立即 `Decimal`/微元化）。

## 3. 訂單生命週期

```
pending_new ─→ accepted ─→ partially_filled ─→ filled
    │             │              │
    │             ├─→ cancelled / expired
    └─→ rejected（建立時即終態）
```

- 轉移規則集中在 `STATUS_TRANSITIONS`，`_transition()` 以
  **合法轉移檢查＋`WHERE status=原狀態` 樂觀鎖**實作，**沒有任何
  直接覆寫 status 的路徑**（非法轉移拋 `InvalidTransition`）。
- **部分成交（本期支援）**：單日可成交量 ≤ 當日成交量 ×
  `participation_rate`，且強制 1000 股整張；殘量保留至下一交易日繼續。
  盤中分筆撮合未實作（屬 P5，如實標注而非假裝全額成交）。
- **零股未實作**：整股 1000 股倍數，非整張數量直接拒單（明確限制）。
- **拒絕也入庫**（`status='rejected'` + `reject_reason`），供稽核統計。

## 4. 成交模擬與時間正確性（防 look-ahead）

1. 訊號以 **T 日收盤**資料計算（`signals.date = T`）→ 訂單必須攜帶
   `signal_date=T`；`T > 今日` 拒單；`T == 今日` 必須在 **13:30 收盤後**
   才准下單（訊號需當日收盤資料）。
2. 成交用 **T 之後第一個行情列的開盤價**（與 `backtest._schedule_entries`
   的 T+1 約定一致）；**絕不用 T 日已知收盤價成交**。
3. 成交價 = 開盤價 ± 滑價（`CostModel.slippage`，買加賣減），0.01 tick
   近似捨入；滑價前價格存 `raw_price_utwd` 供稽核。
4. 當日 bar 只有在 **13:30 收盤後**才視為已發生（盤中不提前成交）。
5. **不可成交情境**（記事件、推進行情光標、不假設成功）：
   - 缺價（訊號日後無行情列）→ `no_market_data_yet`
   - 停牌（volume ≤ 0）→ `suspended`
   - 漲停開盤（買）／跌停開盤（賣）→ `limit_up_no_fill` / `limit_down_no_fill`
   - 前收不可得 → `unknown_limit_reference`
   - 成交量不足整張 → `no_liquidity`；現金不足 → `insufficient_cash`
   - 逾 `good_til_days`（預設 5 日）未成交 → 狀態 `expired`
6. **行情過期禁止新下單**：`max_data_staleness_days`（預設 3 日）
   超過即拒單（既定未成交訂單不受影響——它們已承擔風險）。

## 5. 交易成本（費率可設定、可追溯）

- 沿用 `costs.CostModel`：手續費率 × 折扣、證交稅（僅賣出）、最低手續費、
  滑價、參與率；**引擎不硬編碼任何費率**，每筆成交的 `price_source`
  記錄 `fee_scenario`（`paper_account.fee_scenario` 同步）。
- 買入成本 = 名目 + 手續費（**計入持倉成本基礎**）；
  賣出所得 = 名目 − 手續費 − 證交稅；
  已實現損益 = 賣出淨額 − 按比例移除的成本基礎（WAC）。
- 模擬近似聲明：漲跌停以「前收 ±10% ± tick」近似（未接逐檔撮合）；
  成交一律以開盤價成交（不模擬盤中 VWAP/queue）。

## 6. 帳務閉環與不變條件

- **事實來源**：
  - 現金事實 = `Σ paper_ledger.amount_utwd`（帳本）
  - 持倉事實 = `paper_fill` 序列重放（`replay_positions`，寫入與重播
    共用同一公式 → 快取與重建必然可比）
- `reconcile()` 逐項核對（任一不符即 `ok=False`，**只報告不修復、不繼續交易**）：
  1. 帳本重建現金 == 帳戶 `cash_utwd`；現金 ≥ 0、持倉 ≥ 0
  2. 餘額鏈：`balance[n] == balance[n-1] + amount[n]`；首筆必為 `capital`
  3. 每筆成交的帳本金額 == 名目 ± 費用（買 −(名目+fee)，賣 +名目−fee−tax）
  4. 持倉快取 == 成交重放
  5. 訂單 `filled_qty` == 其成交合計；cancelled 訂單不得有成交
  6. 快照 `equity == cash + market_value`
- **總資產 = 現金 + 所有持倉市值**（快照與 `snapshot()` 皆如此計算）。
- **重播防護**：`client_order_id` 唯一、`fill_id` 唯一、帳本 `ref` 唯一——
  同一訂單/成交重放不重複扣款、不重複持倉（有測試）。
- **失敗即回滾**：訂單/成交/帳本/持倉在同一 DB 交易；中途任何異常
  `ROLLBACK`（測試注入 `_ledger` 失敗驗證無半套資料）；
  `sqlite3.IntegrityError` → `IntegrityBroken` → **停止後續交易**（拋出，
  不吞錯、不繼續）。
- **重啟恢復**：無記憶體狀態；`recover()` 由 DB 重建（帳本現金、成交重放
  持倉、未完成訂單清單），重啟後既定訂單可繼續成交（有測試）。

## 7. 風險控制（全部在進入成交流程**之前**）

整合 `risk.py`（P4 擴充向後相容的 `limits` 參數，預設 `DEFAULT_LIMITS`）：

| 控制 | 實作 | 觸發後 |
|---|---|---|
| 單筆金額上限 | `check_order` | 拒單 + `reject_reason` + `paper_event` |
| 單一股票部位上限 | `check_order` | 同上 |
| 可用現金不足 | `place_order`（含滑價的保守估計） | 同上 |
| 賣出持倉不足 | `place_order` | 同上 |
| 單日損失 / 最大回撤 | `preflight`（由快照歷史計算） | 同上 |
| 行情過期 | `preflight` + `place_order` 獨立檢查 | 禁止新單 |
| Kill Switch | `place_order` **與** `process_pending` 雙重檢查 | 拒新單；未成交訂單也停止成交 |
| 盤前下單（訊號未就緒） | `signal_date` 時點檢查 | 拒單 |
| live 模式 | `LiveModeForbidden`（連 DB 都不開） | 拒絕執行 |

- `risk_status()` 提供總覽（含停止原因），儀表板 `/api/paper/risk` 可查。
- 快照歷史驅動的當日損益＝相對**上一不同交易日**快照的回報（無快照保守
  視為 0）；回撤＝(歷史最高 equity − 現在)/歷史最高。
- **預設全新獨立帳戶**：`paper init` 從未建立；`overwrite=True` 才重建，
  回測歷史不會變成持倉。

## 8. CLI 與可觀測性

```
python -m twstock.cli paper init    [--cash 1000000]   # 建立虛擬帳戶
python -m twstock.cli paper account                    # 資產/現金/持倉/一致性
python -m twstock.cli paper orders   [--status accepted] [--limit 20]
python -m twstock.cli paper fills    [--limit 20]
python -m twstock.cli paper positions
python -m twstock.cli paper ledger   [--limit 20]
python -m twstock.cli paper cycle                       # 一次交易循環
python -m twstock.cli paper risk                        # 風控狀態與停止原因
python -m twstock.cli paper reconcile                  # 帳務一致性（不一致→退出碼 1）
python -m twstock.cli paper recover                    # 重啟恢復檢查
python -m twstock.cli paper kill    [--reason …]        # 緊急停止
python -m twstock.cli paper resume                      # 解除停止
```

- 沿用既有 CLI **單層 argparse** 風格（`paper` 為 DATA_CMDS，子指令轉交
  `twstock.paper.main`），未破壞既有介面。
- 所有輸出標頭明示「模擬、非真實交易」；`reconcile` 失敗回非 0。

**儀表板唯讀 API**（`app.py`，全部 GET、無任何下單/成交端點）：
`/api/paper/account`、`/api/paper/positions`、`/api/paper/orders`、
`/api/paper/fills`、`/api/paper/ledger`、`/api/paper/risk`、
`/api/paper/reconcile`。NULL 欄位轉 `None` 避免 NaN 序列化 500。

## 9. 測試（`tests/test_paper.py`，23 項，全套 489 passed）

對應任務書 17 類：

1. 初始帳戶資金（含冪等 init）✅ 2. 買賣帳務（T+1 開盤、滑價、現金閉環）✅
3. 手續費/交易稅（含最低費、場景可追溯）✅ 4. WAC 成本與已實現/未實現分離 ✅
5. 資金不足/持倉不足拒單（拒單入庫）✅ 6. 重複訂單與重複成交防護 ✅
7. 取消與狀態轉移（非法轉移拒絕）✅ 8. 部分成交量（1000 股倍數）✅
9. 行情過期/缺價/停牌/盤前時點 ✅ 10. 漲跌停不成交 ✅
11. 單筆曝險/持倉上限/損失限制（快照驅動）✅ 12. 緊急停止（下單＋成交雙擋）✅
13. 重啟恢復（新實例由 DB 重建、未完成訂單續跑）✅
14. 交易失敗回滾／`IntegrityBroken` 停止交易 ✅
15. 帳本重建與餘額核對（含竄改偵測）✅
16. 正式 DB 隔離＋唯讀模式（寫入拒絕、查詢可用）✅
17. AST 掃描無網路/券商 import ＋ live 模式拒絕 ✅
附加：CLI 子指令退出碼、儀表板唯讀 API、稽核事件、循環冪等 ✅

測試全程使用 conftest autouse 暫存 DB 與假行情，不連外。

## 10. 未實作功能與限制（如實記錄）

- **不接券商、不下真單**：無任何券商 SDK；live 模式下引擎直接拒絕執行。
- **盤中即時交易（P5）未實作**：本階段僅 EOD（T 日訊號 → T+1 成交）；
  盤中分筆撮合、即時停損、intraday 進出皆無。
- **零股交易未實作**（整股 1000 股倍數）；盤中零股行情未接。
- **漲跌停為 ±10% tick 近似**，非證交所逐檔搓合規則；不模擬減幅/擴幅
  個股差異與盤中打開鎖。
- **成交一律用開盤價**（市價單近似）；不模擬 limit order 隊列、
  成交量撮合順序、實際滑價分布。
- **策略面**：cycle 使用固定持有 `hold_days` + `score ≥ threshold`
  （與回測 default 同構）；不對稱進出場門檻屬 P5。
- **未接真實行情閘門的 quote 表**：本階段成交依據為 price 日K表
  （T+1 收盤後已知）；P3 的 quote 表進場閘門與 `intraday_signal_gate`
  的完整串接待 P5。
- **訊號樣本外優勢未確立**（見 `docs/BEAR_MARKET_STRESS_TEST.md`）：
  模擬交易是帳務與風控基礎建設，**不代表策略具備可交易優勢**。
