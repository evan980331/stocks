"""交易成本模型：可調整、可情境化、可測試。

台股費率（2026 查證；僅為情境基準，**非特定券商通用費率**）：

- 證券交易稅：一般上市櫃股票 **賣出 0.3%**；符合條件的現股當沖 **0.15%**
  （優惠延長至 2027/12/31）；一般股票型 ETF **0.1%**（賣出）。
  本專案持有期間 >= 1 個交易日，**不適用當沖稅率**，一律採 0.3%。
- 券商手續費：市場常見牌告基準 **0.1425%**，**買賣各收一次**；折扣依券商與帳戶而異
  （例如 2.8 折 = 0.0399%、6 折 = 0.0855%）。**低消沒有全市場統一值**
  （常見 20 元，部分方案 1 元），故以可設定參數處理。
- 滑價：無公開費率，屬執行假設，以情境方式呈現。
- 流動性：`participation_rate` 限制單筆可成交數量佔當日成交量的比例，
  避免假設所有交易都能以理想價格完整成交。

來源：財政部稅務入口網（買賣股票、當沖降稅延長）、證交所 ETF 交易規則、
國泰證券與永豐金證券收費頁面。詳見 `docs/STATUS_REPORT.md`。
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import CONFIG


@dataclass(frozen=True)
class CostModel:
    """一組完整的交易成本假設。所有費率以「名目金額比例」表示。"""

    name: str = "typical"
    label: str = "常見（2.8 折手續費、0.3% 稅、0.1% 滑價）"
    commission_rate: float = 0.001425     # 牌告手續費率（買賣各收一次）
    commission_discount: float = 0.28     # 券商折扣
    tax_rate: float = 0.003               # 證交稅（僅賣出；一般股票 0.3%）
    slippage: float = 0.001               # 單邊滑價（買、賣各一次）
    min_fee: float = 20.0                 # 每筆最低手續費
    participation_rate: float = 1.0       # 單筆可成交上限 = 當日成交量 × 此比例

    @property
    def commission_effective(self) -> float:
        return self.commission_rate * self.commission_discount

    def buy_fee(self, notional: float) -> float:
        return max(notional * self.commission_effective, self.min_fee)

    def sell_fee(self, notional: float) -> float:
        return max(notional * self.commission_effective, self.min_fee)

    def buy_cost(self, notional: float) -> float:
        """買進的額外成本（手續費 + 滑價），不含名目金額。"""
        return self.buy_fee(notional) + notional * self.slippage

    def sell_cost(self, notional: float) -> float:
        """賣出的額外成本（手續費 + 證交稅 + 滑價）。"""
        return (self.sell_fee(notional) + notional * self.tax_rate
                + notional * self.slippage)

    def roundtrip_cost_rate(self, price: float, qty: int = 1000) -> float:
        """買進到賣出一次的總成本率（僅供說明，實際回測以逐筆計算為準）。"""
        notional = price * qty
        if notional <= 0:
            return 0.0
        return (self.buy_cost(notional) + self.sell_cost(notional)) / notional

    def max_fill_qty(self, volume: float, price: float,
                     desired_qty: int) -> int:
        """流動性上限：單筆不得超過當日成交量 × participation_rate。

        成交量缺失或 <= 0 時回傳 0（不假設可成交），而非放行。
        """
        if volume is None or volume <= 0:
            return 0
        cap = int(volume * self.participation_rate)
        return max(0, min(desired_qty, cap))


def _from_config(name: str, label: str, *, discount: float, slippage: float,
                 participation: float = 1.0, min_fee: float | None = None,
                 tax_rate: float | None = None) -> CostModel:
    return CostModel(
        name=name, label=label,
        commission_rate=CONFIG.commission_rate,
        commission_discount=discount,
        tax_rate=CONFIG.tax_rate if tax_rate is None else tax_rate,
        slippage=slippage,
        min_fee=CONFIG.buy_fixed_fee if min_fee is None else min_fee,
        participation_rate=participation,
    )


# 成本情境：由「無成本」到「嚴苛」，用以呈現績效對成本假設的敏感度。
COST_SCENARIOS: dict[str, CostModel] = {
    "zero": CostModel(
        name="zero", label="交易成本前（僅供對照，非可實現）",
        commission_rate=0.0, commission_discount=1.0, tax_rate=0.0,
        slippage=0.0, min_fee=0.0),
    "ideal": _from_config("ideal", "理想（2 折、0.05% 滑價）",
                          discount=0.2, slippage=0.0005),
    "typical": _from_config("typical", "常見（2.8 折、0.1% 滑價）",
                            discount=0.28, slippage=0.001),
    "conservative": _from_config("conservative", "保守（6 折、0.2% 滑價）",
                                 discount=0.6, slippage=0.002),
    "adverse": _from_config("adverse", "不利（不打折、0.3% 滑價）",
                            discount=1.0, slippage=0.003),
    "severe": _from_config("severe", "嚴苛（不打折、0.5% 滑價、1% 成交量上限）",
                           discount=1.0, slippage=0.005, participation=0.01),
}


def get_cost_model(name: str | None = None) -> CostModel:
    if name is None:
        return COST_SCENARIOS["typical"]
    try:
        return COST_SCENARIOS[name]
    except KeyError as e:
        raise KeyError(f"未知成本情境 {name}；可用：{sorted(COST_SCENARIOS)}") from e
