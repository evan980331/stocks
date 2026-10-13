"""系統設定：股票池、資料參數、交易成本。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("TWSTOCK_DATA_DIR") or (PROJECT_ROOT / "data_store"))
REPORTS_DIR = PROJECT_ROOT / "reports"
STATE_DIR = DATA_DIR / "state"        # PID 檔、停止旗標等執行狀態
DB_PATH = Path(os.environ.get("TWSTOCK_DB_PATH") or (DATA_DIR / "twstock_v3.db"))


def is_readonly() -> bool:
    """唯讀保護模式：設定 TWSTOCK_READONLY=1 時只讀不寫（本機安全分析用）。

    與任何雲端平台無關；本專案為本地優先架構。
    """
    return bool(os.environ.get("TWSTOCK_READONLY"))

# 台股前百大市值中的代表池（初版 20 檔，涵蓋電子/傳產/金融）
STOCK_POOL: dict[str, str] = {
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2308": "台達電",
    "2303": "聯電",
    "2882": "國泰金",
    "2881": "富邦金",
    "2886": "兆豐金",
    "2412": "中華電",
    "6505": "台塑化",
    "1301": "台塑",
    "1326": "台化",
    "2002": "中鋼",
    "2603": "長榮",
    "2615": "萬海",
    "3711": "日月光投控",
    "2357": "華碩",
    "2912": "統一超",
    "1216": "統一",
    "3231": "緯創",
}

# 基準指數（0050 ETF）
BENCHMARK_ID = "0050"

FINMIND_API = "https://api.finmindtrade.com/api/v4/data"


@dataclass
class Config:
    pool: dict[str, str] = field(default_factory=lambda: dict(STOCK_POOL))
    benchmark: str = BENCHMARK_ID
    lookback_years: float = 2.0
    # 特徵參數
    momentum_windows: tuple[int, ...] = (20, 60)
    rsi_window: int = 14
    volume_ma_window: int = 20
    sentiment_decay_days: int = 3
    # 交易成本（台股現股）
    commission_rate: float = 0.001425   # 手續費（未打折）
    commission_discount: float = 0.28   # 常見 2.8 折
    tax_rate: float = 0.003             # 證交稅（僅賣出）
    slippage: float = 0.001             # 滑價 0.1%
    buy_fixed_fee: float = 20.0         # 低消 20 元
    # 訊號
    top_n: int = 5
    hold_days: int = 5                  # ML label = 未來 5 日超額報酬
    max_position_pct: float = 0.10      # 單檔部位上限（佔總資產）
    # 回測成交假設
    limit_pct: float = 0.10             # 漲跌停幅度（漲停/跌停當日無法成交）
    # 訊號融合（規則分數與 ML 預測尺度不同，必須先轉成可比較的排名百分位）
    fusion_method: str = "rank_percentile"
    fusion_weight_rule: float = 0.6
    fusion_weight_ml: float = 0.4

    @property
    def commission_effective(self) -> float:
        return self.commission_rate * self.commission_discount

    def roundtrip_cost(self, price: float) -> float:
        """買進+賣出一次循環的成本率（含滑價、稅）。"""
        fee_in = max(price * self.commission_effective, self.buy_fixed_fee) / price
        fee_out = max(price * self.commission_effective, self.buy_fixed_fee) / price
        return fee_in + fee_out + self.tax_rate + 2 * self.slippage


CONFIG = Config()
