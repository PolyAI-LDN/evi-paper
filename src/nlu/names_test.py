import unittest

from nlu.interface import AbstractValueExtractor
from nlu.names import EviNameValueExtractor


class ContextNameTests(unittest.TestCase):
    def make_extractor(self, strict):
        extractor = EviNameValueExtractor.__new__(EviNameValueExtractor)
        AbstractValueExtractor.__init__(extractor, "en-GB", strict=strict)
        extractor._all_firsts = {"jane"}
        extractor._all_lasts = {"smith"}
        return extractor

    def test_seeking_finds_name_in_a_sentence(self):
        extractor = self.make_extractor(False)
        self.assertIn(("jane", "smith"), extractor.extract("my name is jane smith"))

    def test_cautious_rejects_context(self):
        self.assertEqual(self.make_extractor(True).extract("my name is jane smith"), [])

    def test_full_names_still_work_in_both_modes(self):
        for strict in (False, True):
            self.assertIn(
                ("jane", "smith"), self.make_extractor(strict).extract("jane smith")
            )

    def test_unrecognised_text_is_empty(self):
        self.assertEqual(self.make_extractor(False).extract("hello there"), [])

    def test_context_words_do_not_become_surnames(self):
        results = self.make_extractor(False).extract("jane says her last name is smith")
        self.assertIn(("jane", "smith"), results)
        self.assertNotIn(("jane", "says"), results)
