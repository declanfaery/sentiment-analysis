from __future__ import annotations

import unittest
from datetime import date

from sentiment_analysis.trading.broker import BrokerError
from sentiment_analysis.trading.models import OptionContract
from sentiment_analysis.trading.webull import (
    _candidate_expirations,
    _latest_option_close_from_history,
    _replacement_order,
)


class WebullOrderTests(unittest.TestCase):
    def test_candidate_expirations_respect_maximum(self) -> None:
        expirations = _candidate_expirations(
            date(2026, 8, 19),
            None,
            date(2026, 9, 2),
        )

        self.assertEqual(expirations, [date(2026, 8, 21), date(2026, 8, 28)])

    def test_option_replacement_includes_required_leg_ids(self) -> None:
        payload = _replacement_order(
            "client-123",
            quantity=2,
            limit_price=4.7,
            order_row={
                "instrument_type": "OPTION",
                "legs": [{"id": "leg-456", "quantity": "1"}],
            },
        )

        self.assertEqual(payload["client_order_id"], "client-123")
        self.assertEqual(payload["quantity"], "2")
        self.assertEqual(payload["limit_price"], "4.70")
        self.assertEqual(payload["legs"], [{"id": "leg-456", "quantity": "2"}])

    def test_option_replacement_rejects_missing_leg_ids(self) -> None:
        with self.assertRaisesRegex(BrokerError, "leg IDs"):
            _replacement_order(
                "client-123",
                quantity=1,
                limit_price=4.7,
                order_row={"instrument_type": "OPTION", "legs": []},
            )

    def test_finds_actual_manual_option_close_in_order_history(self) -> None:
        contract = OptionContract(
            symbol="COHR260821C00307500",
            ticker="COHR",
            option_type="CALL",
            expiration=date(2026, 8, 21),
            strike=307.5,
            bid=0,
            ask=0,
            underlying_price=0,
        )
        payload = [
            {
                "client_order_id": "manual-close",
                "orders": [
                    {
                        "client_order_id": "manual-close",
                        "instrument_type": "OPTION",
                        "side": "SELL",
                        "status": "FILLED",
                        "filled_quantity": "16",
                        "filled_price": "41.20",
                        "filled_time": "1786628218788",
                        "legs": [
                            {
                                "symbol": "COHR",
                                "option_type": "CALL",
                                "option_expire_date": "2026-08-21",
                                "strike_price": "307.50",
                            }
                        ],
                    }
                ],
            }
        ]

        update = _latest_option_close_from_history(payload, contract)

        self.assertIsNotNone(update)
        self.assertEqual(update.client_order_id, "manual-close")
        self.assertEqual(update.filled_quantity, 16)
        self.assertEqual(update.filled_price, 41.20)


if __name__ == "__main__":
    unittest.main()
