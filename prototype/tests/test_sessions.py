import sys
import unittest

import numpy as np

from atoms.io import DATA_DIR, label_relevance, load_impressions
from atoms.sessions import eval_targets, history_for, pseudo_sessions

sys.path.insert(0, str(DATA_DIR))
import common  # noqa: E402


class TestSessions(unittest.TestCase):
    def test_eval_target_counts(self):
        for split, total in (("val", 9576), ("test", 9724)):
            ts = eval_targets(split)
            self.assertEqual(sum(len(t.slate_games) for t in ts), total)
            self.assertTrue(all(len(t.prefix) == 3 for t in ts))

    def test_pseudo_shape(self):
        train = load_impressions("train_impressions")
        targets, labels = pseudo_sessions(train, (7, 19))
        self.assertGreater(len(targets), 1000)
        for t in targets:
            self.assertEqual(len(t.prefix), 3)
            self.assertGreaterEqual(len(t.slate_games), 4)
            self.assertTrue(7 <= t.day <= 19)
        self.assertEqual(len(labels), sum(len(t.slate_games) for t in targets))
        self.assertTrue(labels["rel"].between(0, 3).all())

    def test_val_history_excludes_val_events(self):
        h = set(history_for("val")["session_id"])
        vp = set(load_impressions("val_prefix")["session_id"])
        ve = set(load_impressions("val_events")["session_id"])
        self.assertEqual(h & (ve - vp), set())
        hist = history_for("val")
        val_ids = vp | ve
        self.assertFalse(((hist["position"] > 3) & hist["session_id"].isin(val_ids)).any())
        self.assertEqual(len(hist), len(load_impressions("train_impressions"))
                         + len(load_impressions("val_prefix")))

    def test_label_matches_common_relevance(self):
        rows = common.read("val_prefix.csv")[:2442]
        df = load_impressions("val_prefix")
        want = [common.relevance(r) for r in rows]
        got = label_relevance(load_impressions("val_prefix")).tolist()
        # val_prefix has no exact duplicates, so row order is preserved
        self.assertEqual(len(want), len(got))
        self.assertEqual(want, got)

    def test_history_for_holdout_is_train(self):
        self.assertEqual(len(history_for("holdout")), 68137 - 685)


if __name__ == "__main__":
    unittest.main()
