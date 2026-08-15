from __future__ import annotations

from datetime import date
import unittest

from sentiment_analysis.trading.broker import choose_option_contract, option_limit_price
from sentiment_analysis.trading.models import OptionContract


class OptionSelectionTests(unittest.TestCase):
    def test_webull_uses_nickel_ticks_for_premiums_at_three_dollars(self) -> None:
        self.assertEqual(option_limit_price(5.43, penny_program=True), 5.45)
        self.assertEqual(option_limit_price(2.517, penny_program=True), 2.52)

    def test_prefers_liquid_itm_contract_with_small_breakeven_burden(self) -> None:
        contracts = [
            _put(strike=100, bid=5.8, ask=6.0),
            _put(strike=107, bid=10.2, ask=10.5),
            _put(strike=115, bid=16.6, ask=17.0),
        ]

        selected = choose_option_contract(
            contracts,
            max_contract_cost=20_000,
            max_spread_pct=0.12,
            target_itm_pct=0.07,
            target_delta=0.70,
            max_breakeven_move_pct=0.04,
            min_volume=25,
            min_open_interest=100,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.strike, 107)
        self.assertAlmostEqual(selected.breakeven_move_pct, 0.035)

    def test_rejects_contract_below_liquidity_floor(self) -> None:
        selected = choose_option_contract(
            [_put(strike=107, bid=10.2, ask=10.5, volume=3, open_interest=20)],
            max_contract_cost=20_000,
            max_spread_pct=0.12,
            target_itm_pct=0.07,
            target_delta=0.70,
            max_breakeven_move_pct=0.04,
            min_volume=25,
            min_open_interest=100,
        )

        self.assertIsNone(selected)


def _put(
    *,
    strike: float,
    bid: float,
    ask: float,
    volume: int = 500,
    open_interest: int = 500,
) -> OptionContract:
    return OptionContract(
        symbol=f"TEST260821P{int(strike * 1000):08d}",
        ticker="TEST",
        option_type="PUT",
        expiration=date(2026, 8, 21),
        strike=strike,
        bid=bid,
        ask=ask,
        underlying_price=100,
        volume=volume,
        open_interest=open_interest,
    )


if __name__ == "__main__":
    unittest.main()
