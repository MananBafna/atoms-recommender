import unittest

import numpy as np
import pandas as pd

from atoms.evaluation import ndcg5_per_session, paired_bootstrap, popularity_scores
from atoms.io import DATA_DIR, load_impressions


def _labels():
    df = pd.read_csv(DATA_DIR / "val_labels.csv", dtype={"session_id": str, "game_id": str})
    return df.rename(columns={"relevance": "rel"})


class TestEvaluation(unittest.TestCase):
    def test_matches_official_baseline(self):
        sc = pd.read_csv(DATA_DIR / "baseline_val_rankings.csv", dtype={"session_id": str, "game_id": str})
        n = ndcg5_per_session(sc, _labels())
        self.assertEqual(len(n), 604)
        self.assertAlmostEqual(n.mean(), 0.4032, delta=5e-5)

    def test_popularity_reproduces_baseline(self):
        raw = pd.read_csv(DATA_DIR / "train_impressions.csv", dtype=str, keep_default_na=False)
        for c in ["play_now", "play_seconds", "completed", "replays", "shared", "challenged"]:
            raw[c] = pd.to_numeric(raw[c].replace("", np.nan), errors="coerce")
        slates = load_impressions("val_slates")[["session_id", "game_id"]]
        mine = ndcg5_per_session(popularity_scores(raw, slates), _labels())
        sc = pd.read_csv(DATA_DIR / "baseline_val_rankings.csv", dtype={"session_id": str, "game_id": str})
        base = ndcg5_per_session(sc, _labels())
        self.assertAlmostEqual(mine.mean(), base.mean(), delta=1e-4)

    def test_bootstrap_identical_is_zero(self):
        s = pd.Series([0.2, 0.5, 0.9, 0.4], index=list("abcd"))
        r = paired_bootstrap(s, s)
        self.assertEqual(r["mean_diff"], 0)
        self.assertEqual((r["ci_low"], r["ci_high"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
