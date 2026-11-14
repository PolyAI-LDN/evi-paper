import unittest

from nlu.interface import AbstractValueExtractor
from nlu.names import EviNameValueExtractor


class SpelledNameLocaleTests(unittest.TestCase):
    def extractor(self, locale, first):
        extractor = EviNameValueExtractor.__new__(EviNameValueExtractor)
        AbstractValueExtractor.__init__(extractor, locale, strict=True)
        extractor._all_firsts = {first.lower()}
        extractor._all_lasts = {"fox"}
        return extractor

    def test_polish_double_letter(self):
        result = self.extractor("pl-PL", "Anna").extract(
            "a podwójne n a f o x", ["SPELL"]
        )
        self.assertIn(("ANNA", "FOX"), result)

    def test_french_comme_alias(self):
        result = self.extractor("fr-FR", "Anna").extract(
            "a comme anna double n a f o x", ["SPELL"]
        )
        self.assertIn(("ANNA", "FOX"), result)

    def test_english_spelling_is_unchanged(self):
        result = self.extractor("en-GB", "Anna").extract(
            "a double n a f o x", ["SPELL"]
        )
        self.assertIn(("ANNA", "FOX"), result)
