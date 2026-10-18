# 台股行情來源評估報告（P3）

- 文件日期：2026-10-18
- 對應階段：P3（台股即時行情與資料管線建置）
- 程式同源：`python -m twstock.cli quotes catalog`（本文件與 `twstock/data/quotes.py` 的
  `SOURCE_CATALOG` 一致；數字若有出入以程式輸出為準）
- 狀態：**介面與管線已完成；唯一預設啟用的行情來源是 FinMind 分鐘資料（延遲），
  本階段未實際呼叫任何外部行情 API**（單元測試一律使用注入的假 getter／假來源）

> ⚠️ **本專案目前沒有、也不宣稱擁有「即時交易能力」或「已驗證的即時行情」。**
> 測試通過只證明**介面與管線正確**，不證明任何真實供應商的穩定性。

---

## 1. 四種行情必須分開（不得混稱「即時」）

| 類別 | 定義 | 本專案現況 |
|---|---|---|
| 即時 `realtime` | 來源標示為盤中即時（例如交易所 5 秒快照） | **未接通**（MIS 端點技術可行但授權受限，預設停用） |
| 延遲 `delayed` | 來源明確延遲，或盤中但延遲未載明 → 保守歸此類 | FinMind 分鐘資料（**唯一預設啟用**） |
| 盤後 `eod` | 當日收盤後才產生的日資料 | FinMind 日K（既有正式來源）、TWSE OpenAPI（未實作） |
| 歷史 `historical` | 非當日資料 | FinMind 日K 庫存 |

**不得將定時抓取或盤後資料稱為即時行情**：系統的正式訊號仍是盤後（EOD）產生；
盤中行情路徑只做健康監控，不產生任何交易訊號。

## 2. 來源評估總表（依授權風險排序）

| 來源 | 種類 | 已實作 | 預設啟用 | 認證 | 限流 | 關鍵限制 |
|---|---|---|---|---|---|---|
| `mis`（TWSE MIS） | 即時 | 是 | **否** | 無（觀察事實，非官方保證） | 未查到官方值；自設 5 秒 | ⚠️ TWSE 條款禁止未經同意的自動化擷取；正式行情需另訂契約付費 |
| `finmind_minute` | 延遲 | 是 | **是** | 可選 `FINMIND_TOKEN` | 300 次/小時（無 token）／600（有 token） | 交付延遲未載明 → 一律歸「延遲」；免費層限流嚴格 |
| `fake` | 延遲（合成） | 是 | 否 | 無 | 無 | `is_fabricated=True`；輸出強制標示假資料；selftest 沙盒化 |
| `fugle` | 即時 | **否** | 否 | 需註冊 API key（Basic 免費） | REST 60 次/分鐘 | 需使用者自行註冊（本階段不代為註冊） |
| `shioaji`（永豐） | 即時 | **否** | 否 | 需券商帳戶與 API 金鑰 | 200 檔訂閱、查詢 10 秒 50 次 | **BLOCKED**：需券商帳戶 |
| `fubon_neo`（富邦） | 即時 | **否** | 否 | 需帳戶、憑證、API 同意書 | 盤中 300 次/分鐘 | **BLOCKED**：需券商帳戶 |
| `yahoo`（yfinance） | 延遲 | **否** | 否 | 無官方 key | 無公開配額 | 官方延遲表列台股 **20 分鐘**；條款禁止未許可自動化 |
| `twse_openapi` | 盤後 | **否** | 否 | 無 | 未查到 | 政府開放資料（可商用需標來源）；**不是盤中來源** |

## 3. 結論（直接回答「有沒有免費即時來源」）

**沒有已查證、可無條件使用的免費官方台股盤中即時來源。**

1. **交易所 MIS 端點**：技術上可取得（一般 HTTP GET 回 JSON、無需 API key），
   但 TWSE 網站條款禁止未經同意的自動化下載／擷取，正式行情資訊需另訂契約並付費；
   商業轉散布屬授權問題。→ 本專案**預設停用**，需 `TWSTOCK_QUOTE_ALLOW_MIS=1`
   明確 opt-in（且僅供個人研究場景的風險自負評估）。
2. **FinMind 分鐘資料**：本專案既有來源、可立即使用，但官方**未載明交付延遲**
   → 一律歸「延遲」，不得稱即時；免費層限流 300～600 次/小時。
3. **Fugle**：有明確即時文件與免費層，但需使用者自行註冊 API key
   （本階段不代為註冊、不代為購買）。
4. **券商 API**（Shioaji／富邦 NEO）：需券商帳戶 → BLOCKED（同時是 P4 的前置條件）。

## 4. 時間戳語義（不得互相冒充）

| 欄位 | 意義 |
|---|---|
| `market_time` | 來源表示的市場時間（正規化 ISO，僅供排序／去重） |
| `market_time_raw` | **來源原樣字串**（不得與 `market_time` 互相改寫） |
| `market_time_source` | `source`（來源提供）／`derived`（可驗證推導，如毫秒 epoch）／`unknown` → **unknown 不得當成新行情** |
| `fetched_at` | 本系統實際取得時間（**不得偽裝成市場成交時間**） |

## 5. 可靠性規則（皆有對應測試 `tests/test_quotes.py`）

1. 同一筆行情重複取得不得重複寫入（主鍵＝冪等去重鍵）。
2. **較舊的行情不得覆蓋較新的有效資料**（跨批次與批次內都擋）。
3. 來源失敗（網路／限流／逾時／HTTP／解析）**不得**降級成「正常無行情」。
4. 缺值、非正價格、超出當日漲跌停、代碼不符、日期非當日、非交易時段、
   時間來源不可得 → 一律不得當成有效新行情。
5. 過期行情不得通過 `intraday_signal_gate`（未來盤中訊號必須先過此閘；
   本階段訊號仍是盤後產生，閘門目前不影響既有流程）。
6. 健康狀態五態互斥：`normal`／`delayed`／`failed`／`unknown`／`non_trading_session`。

## 6. 本階段實測（全部離線）

| 項目 | 結果 |
|---|---|
| 假來源自檢（`quotes selftest`，沙盒暫存 DB） | 20 檔解析→驗證→寫入全通；訊號閘門「禁止（合成資料）」 |
| 未設定來源時 `quotes poll` | 明確回報「未設定」exit 2，**不自行臆測來源** |
| 斷線／限流／格式錯誤重試 | 有上限重試＋退避＋節流器（測試以假 Clock 驅動，完全確定） |
| 單元測試 | `tests/test_quotes.py` 74 項；全套 466 項全綠 |
| 正式 DB 保護 | autouse 隔離；selftest 沙盒化；正式 DB hash 前後不變 |

## 7. 只完成介面、尚未實際接通的項目

- `fugle`、`shioaji`、`fubon_neo`、`yahoo`、`twse_openapi`：**只有來源目錄條目
  與分類**（`SOURCE_CATALOG`），沒有 fetch 實作；不影響任何現有流程。
- MIS 盤中輪詢：介面與 opt-in 閘門已完成，**本階段未對真實端點發出任何請求**
  （授權條款考量）；Worker 的行情輪詢預設不啟用（需設 `TWSTOCK_QUOTE_SOURCE`）。

## 8. 相對來源連結

- TWSE MIS 端點（觀察用） <https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_2330.tw>
- FinMind 資料集 <https://finmind.github.io/tutor/TaiwanMarket/DataList/>
- Fugle 行情 API <https://developer.fugle.tw/docs/data/http-api/intraday/quote/>、價格 <https://developer.fugle.tw/docs/pricing/>
- 永豐 Shioaji 行情 <https://sinotrade.github.io/tutor/market_data/streaming/stocks/>
- 富邦 Trade API <https://www.fbs.com.tw/TradeAPI/en/docs/market-data/intraday/quote/>
- Yahoo 延遲說明 <https://help.yahoo.com/kb/SLN2310.html>
- TWSE OpenAPI <https://openapi.twse.com.tw/v1/swagger.json>
