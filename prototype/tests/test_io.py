import unittest

import numpy as np
import pandas as pd

from atoms.io import hist_outcome, load_friends, load_impressions


class TestIO(unittest.TestCase):
    def test_train_dedup(self):
        self.assertEqual(len(load_impressions("train_impressions")), 68137 - 685)

    def test_blank_play_seconds_is_nan(self):
        self.assertGreater(load_impressions("train_impressions")["play_seconds"].isna().sum(), 0)

    def test_hist_outcome_rules(self):
        n = np.nan
        rows = [
            # play_now, play_seconds, completed, replays
            (0, 0, 0, 0),
            (1, 10, 0, 0),
            (1, 30, 0, 0),
            (1, 40, 1, 0),
            (1, 40, 1, 1),
            (1, n, 1, 0),
        ]
        df = pd.DataFrame(rows, columns=["play_now", "play_seconds", "completed", "replays"])
        df["shared"] = 0
        df["challenged"] = 0
        self.assertEqual(hist_outcome(df).tolist(), [0, 0, 1, 2, 3, 2])

    def test_friends_symmetric(self):
        fr = load_friends()
        for u, vs in fr.items():
            for v in vs:
                self.assertIn(u, fr[v])


if __name__ == "__main__":
    unittest.main()
