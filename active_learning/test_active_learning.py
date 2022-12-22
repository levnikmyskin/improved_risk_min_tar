from unittest import TestCase
from active_learning.load_data import filter_files, ALFileInfo, ALPolicy
from datetime import datetime


class TestActiveLearning(TestCase):

    def test_filter_files(self):
        date_created = datetime(420, 4, 20)
        date_before = datetime(320, 4, 20)
        date_searching = datetime(420, 3, 20)
        checkpoint_search = {1000, 5000, 20000}
        fs = [
            ALFileInfo(ALPolicy.RELEVANCE_SAMPLING, 1000, int(date_created.timestamp()), 100_000, ''),
            ALFileInfo(ALPolicy.RELEVANCE_SAMPLING, 5000, int(date_created.timestamp()), 100_000, ''),
            ALFileInfo(ALPolicy.RELEVANCE_SAMPLING, 5000, int(date_before.timestamp()), 100_000, ''),
            ALFileInfo(ALPolicy.RELEVANCE_SAMPLING, 10000, int(date_created.timestamp()), 100_000, ''),
            ALFileInfo(ALPolicy.RELEVANCE_SAMPLING, 20000, int(date_before.timestamp()), 100_000, ''),
        ]

        filters = [
            lambda f: f.checkpoint in checkpoint_search,
            lambda f: f.timestamp > int(date_searching.timestamp())
        ]
        filtered = list(filter_files(fs, filters))
        self.assertEqual(len(filtered), 2)

        checkpoint_search = {10_000, 1000}
        filtered = list(filter_files(fs, filters))
        self.assertEqual(len(filtered), 2)

        checkpoint_search = {20_000}
        filtered = list(filter_files(fs, filters))
        self.assertEqual(len(filtered), 0)
