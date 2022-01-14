""" Tests for logic.py

Copyright PolyAI Limited
"""

import unittest

from utils.logic import Logic


class AndLogicTest(unittest.TestCase):

    def test_min(self):
        logic = Logic()
        inputs_outputs = [
            ([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], 1.0),
            ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.0),
            ([1.0, 1.0, 0.0, 0.0, 0.0, 1.0], 0.0),
            ([1.0, 1.0, 0.0, 0.0, 0.1, 1.0], 0.1),
        ]
        for args, expected in inputs_outputs:
            self.assertEqual(
                logic.combine_scores(*args),
                expected
            )


if __name__ == "__main__":
    unittest.main()
