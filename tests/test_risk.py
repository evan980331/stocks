"""風險控制：預設不得啟用正式交易、限額檢查、Kill Switch 只擋新委託。"""
from __future__ import annotations

from twstock.risk import (AccountState, KillSwitch, Mode, RiskLimits, check_order,
                          current_mode, kill_switch_decision, live_enabled,
                          preflight)


def test_default_mode_is_not_live(monkeypatch):
    monkeypatch.delenv("TWSTOCK_MODE", raising=False)
    monkeypatch.delenv("TWSTOCK_LIVE_ACK", raising=False)
    assert current_mode() is Mode.PAPER
    assert live_enabled() is False


def test_live_requires_explicit_ack(monkeypatch):
    monkeypatch.setenv("TWSTOCK_MODE", "live")
    monkeypatch.delenv("TWSTOCK_LIVE_ACK", raising=False)
    assert live_enabled() is False            # 只有 live 模式還不夠
    monkeypatch.setenv("TWSTOCK_LIVE_ACK", "1")
    assert live_enabled() is True


def test_kill_switch_engage_release(tmp_path):
    ks = KillSwitch(tmp_path / "KILL_SWITCH")
    assert ks.engaged() is False
    ks.engage("行情過期")
    assert ks.engaged() is True
    assert "行情過期" in ks.reason()
    assert kill_switch_decision(ks).ok is False
    ks.release()
    assert kill_switch_decision(ks).ok is True


def test_preflight_blocks_on_stale_data():
    d = preflight(AccountState(equity=1_000_000, data_staleness_days=10))
    assert d.ok is False
    assert any("過期" in r for r in d.reasons)


def test_preflight_blocks_on_unknown_equity_and_mismatch():
    d = preflight(AccountState(equity=0, realized_position_reconciles=False))
    assert d.ok is False
    assert len(d.reasons) >= 2


def test_preflight_blocks_on_daily_loss_and_drawdown():
    d = preflight(AccountState(equity=1_000_000, day_pnl_pct=-0.05, drawdown_pct=-0.20))
    assert d.ok is False
    assert any("單日" in r for r in d.reasons)
    assert any("回撤" in r for r in d.reasons)


def test_order_value_limit():
    st = AccountState(equity=10_000_000)
    d = check_order(st, stock_id="2330", price=1000.0, qty=1000)   # 1,000,000 > 100,000
    assert d.ok is False
    assert any("單筆金額" in r for r in d.reasons)


def test_position_and_gross_exposure_limits():
    st = AccountState(equity=1_000_000,
                      position_value_by_stock={"2330": 95_000})
    d = check_order(st, stock_id="2330", price=100.0, qty=200)     # 既有 95k + 20k
    assert d.ok is False
    assert any("單一股票部位" in r for r in d.reasons)


def test_valid_order_passes():
    st = AccountState(equity=1_000_000, liquidity_20d_value=5e7)
    d = check_order(st, stock_id="2330", price=100.0, qty=100)     # 10,000
    assert d.ok is True, d.reasons


def test_limits_are_conservative_by_default():
    lim = RiskLimits()
    lim.validate()
    assert lim.max_position_pct <= 0.10
    assert lim.max_gross_exposure <= 0.60
    assert lim.max_data_staleness_days <= 3
