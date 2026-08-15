from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from sentiment_analysis.providers.expected_move import expected_move_from_chain


class ExpectedMoveTests(unittest.TestCase):
    def test_expected_move_from_atm_straddle(self) -> None:
        calls = pd.DataFrame(
            [
                {"strike": 95.0, "bid": 8.0, "ask": 9.0},
                {"strike": 100.0, "bid": 5.0, "ask": 6.0},
                {"strike": 105.0, "bid": 3.0, "ask": 4.0},
            ]
        )
        puts = pd.DataFrame(
            [
                {"strike": 95.0, "bid": 3.0, "ask": 4.0},
                {"strike": 100.0, "bid": 4.0, "ask": 5.0},
                {"strike": 105.0, "bid": 6.0, "ask": 7.0},
            ]
        )

        move = expected_move_from_chain(
            calls=calls,
            puts=puts,
            spot_price=101.0,
            expiration=date(2026, 8, 7),
        )

        self.assertIsNotNone(move)
        self.assertEqual(move.strike, 100.0)
        self.assertAlmostEqual(move.expected_move_abs, 10.0)
        self.assertAlmostEqual(move.expected_move_pct, 9.901, places=3)


if __name__ == "__main__":
    unittest.main()

