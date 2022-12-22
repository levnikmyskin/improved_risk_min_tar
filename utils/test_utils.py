import os
import unittest
import numpy as np
from minecore import pairs
from utils.data_utils import random_dataset_with_given_prevalences, flatten, filter_file, ALPolicy, pattern
from utils.misc import AlwaysIn


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng().random((1000, 10))
        self.y = np.random.default_rng().integers(0, 2, (1000,))

    def test_random_dataset_with_given_prevalences(self):
        x_tr, y_tr, x_te, y_te = random_dataset_with_given_prevalences(self.x, self.y, 0.8, 0.2, tr_size=50, te_size=50)
        self.assertTrue(y_tr.mean() == 0.8, f"y_tr prevalence is {y_tr.mean()} instead of 0.8")
        self.assertTrue(y_te.mean() == 0.2, f"y_te prevalence is {y_te.mean()} instead of 0.2")

    def test_filter_file(self):
        # This test is specific to my system. Change your path accordingly if you need to.
        files = os.listdir('.data/active_learning/results/ALvRS/all_classes')
        labels = set(flatten(pairs))
        policy = ALPolicy.RELEVANCE_SAMPLING
        filtered = list(filter(lambda f: filter_file(f, policy, 'SVM', AlwaysIn(), labels=labels), files))
        p = []
        for m in map(lambda f: pattern.match(f), filtered):
            p.append(m.group('label'))
        self.assertSetEqual(labels, set(p))


