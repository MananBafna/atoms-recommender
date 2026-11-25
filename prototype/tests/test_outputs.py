import json
import pathlib
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def _check_rankings(tc, rankings, slates, rows):
    if not rankings.exists():
        tc.skipTest(f"{rankings} not found; run `python run.py` first")
    r = pd.read_csv(rankings)
    tc.assertEqual(list(r.columns), ["session_id", "game_id", "score"])
    tc.assertEqual(len(r), rows)
    tc.assertFalse(r.duplicated(["session_id", "game_id"]).any())
    tc.assertTrue(np.isfinite(r["score"].to_numpy(dtype=float)).all())
    s = pd.read_csv(slates)
    tc.assertEqual(set(zip(r.session_id, r.game_id)), set(zip(s.session_id, s.game_id)))


class TestOutputs(unittest.TestCase):
    def test_test_rankings_complete(self):
        _check_rankings(self, ROOT / "submission" / "rankings_test.csv", DATA / "test_slates.csv", 9724)

    def test_val_rankings_complete(self):
        _check_rankings(self, ROOT / "prototype" / "out" / "rankings_val.csv", DATA / "val_slates.csv", 9576)

    def test_success_bar(self):
        p = ROOT / "prototype" / "out" / "metrics.json"
        if not p.exists():
            self.skipTest(f"{p} not found; run `python run.py` first")
        self.assertGreater(json.loads(p.read_text())["bootstrap"]["ci_low"], 0)


if __name__ == "__main__":
    unittest.main()
