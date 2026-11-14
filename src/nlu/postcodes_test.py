import unittest

from nlu.postcodes import EviPostcodeValueExtractor, _validate_postcode


class SpecialPostcodeTests(unittest.TestCase):
    def test_gir_with_and_without_space(self):
        for value in ("GIR 0AA", "GIR0AA", "gir 0aa"):
            self.assertEqual(_validate_postcode(value, "en-GB"), "GIR0AA")

    def test_cautious_extraction_recognises_gir(self):
        extractor = EviPostcodeValueExtractor("en-GB", strict=True)
        self.assertEqual(extractor.extract("GIR 0AA"), ["GIR0AA"])

    def test_regular_and_invalid_codes(self):
        self.assertEqual(_validate_postcode("SW1A 1AA", "en-GB"), "SW1A1AA")
        self.assertEqual(_validate_postcode("GIR 1AA", "en-GB"), "")
