import os
import tempfile
import unittest

from evi_dataset import load_evi_data


class DatasetLocationTests(unittest.TestCase):
    def test_bundled_data_loads_from_an_unrelated_directory(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                profiles, dialogues = load_evi_data("en_GB")
            finally:
                os.chdir(original)
        self.assertEqual(len(profiles), 10000)
        self.assertEqual(len(dialogues), 1407)
