""" Tests for similarity.py


Copyright PolyAI Limited
"""
import unittest

from utils.similarity import ExactScorer


class PostcodeScorerTest(unittest.TestCase):

    def test_postcode_exact(self):
        scorer = ExactScorer()
        inputs_outputs = [
            ([], 0.0),
            (['CB13PQ'], 1.0),
            (['CB13PQ'], 1.0),
            (['1234', 'CB13PQ'], 1.0),
            (['C b 1 3 P   Q'], 1.0),
            (['Cb 1 3 PQ'], 1.0),
            (['NW12DD'], 0.0),
            (['NW12DD'], 0.0),
            (['NW12DD'], 0.0),
        ]
        for hyps, expected in inputs_outputs:
            self.assertEqual(
                scorer.calc_score(hyps=hyps, ref='C B 13pq'),
                expected
            )


class NameScorerTest(unittest.TestCase):

    def test_name_exact(self):
        scorer = ExactScorer()
        inputs_outputs = [
            ([], 0.0),
            (['John Smith'], 1.0),
            (['John Smith'], 1.0),
            (['A B', 'John Smith'], 1.0),
            (['john   smith'], 1.0),
            (['john   smith'], 1.0),
            (['A B'], 0.0),
            (['A B'], 0.0),
            (['A B'], 0.0),
        ]
        for hyps, expected in inputs_outputs:
            self.assertEqual(
                scorer.calc_score(hyps=hyps, ref='JOHN SMITH'),
                expected
            )


if __name__ == "__main__":
    unittest.main()
