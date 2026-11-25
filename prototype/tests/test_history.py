import unittest

import numpy as np
import pandas as pd

import math

from atoms.history import FEATURES, FEATURE_GROUPS, HistoryState, Target, featurize

COLS = ["user_id", "session_id", "position", "ts", "day", "game_id", "watch_ms", "play_now",
        "play_seconds", "completed", "score_percentile", "replays", "shared", "challenged"]
NAN = np.nan


def imp(user, sess, pos, ts, game, watch=5000, tap=0, ps=0.0, comp=0, rep=0):
    return dict(user_id=user, session_id=sess, position=pos, ts=float(ts), day=int(ts // 86400),
                game_id=game, watch_ms=watch, play_now=tap, play_seconds=ps, completed=comp,
                score_percentile=NAN, replays=rep, shared=0, challenged=0)


def frame(rows):
    return pd.DataFrame(rows, columns=COLS)


GAMES = pd.DataFrame(
    {"mechanic": ["m1", "m1", "m2", "m1"], "family": ["puzzle"] * 3 + ["arcade"],
     "difficulty": [3, 5, 2, 3], "editorial_stars": [3, 1, 4, 3], "added_day": [0, 0, 0, 0]},
    index=pd.Index(["g1", "g2", "g3", "g4"], name="game_id"))
FRIENDS = {"u1": {"u2"}, "u2": {"u1"}}
DAY = 86400


def history():
    rows = []
    # u2 finished g1 at ts=1000; u3 plays g2 briefly
    rows += [imp("u2", "h1", 1, 1000, "g1", tap=1, ps=60, comp=1),
             imp("u2", "h1", 2, 1000, "g2")]
    rows += [imp("u3", "h2", 1, 2000, "g1", tap=1, ps=30), imp("u3", "h2", 2, 2000, "g3", tap=1, ps=5)]
    return frame(rows)


def target(ts=5000, user="u1", slate=("g1", "g2"), prefix=None, sid="t1"):
    if prefix is None:
        prefix = frame([imp(user, sid, i + 1, ts, "g3") for i in range(3)])
    return Target(sid, user, float(ts), int(ts // DAY), prefix, list(slate))


class TestHistory(unittest.TestCase):
    def test_groups_cover_features(self):
        self.assertEqual(list(FEATURE_GROUPS),
                         ["quality", "prior", "affinity", "repeat", "skill", "friends", "prefix"])
        self.assertEqual([f for g in FEATURE_GROUPS.values() for f in g], FEATURES)

    def test_shapes_and_finite(self):
        t = [target(), target(ts=6000, user="u3", slate=("g2", "g3", "g1"), sid="t2")]
        out = featurize(history(), t, GAMES, FRIENDS)
        self.assertEqual(len(out), 5)
        self.assertTrue(set(FEATURES) <= set(out.columns))
        self.assertTrue(np.isfinite(out[FEATURES].to_numpy(float)).all())

    def test_zero_history_user_and_game_finite(self):
        # u9 has no history; g4 (arcade family has no history at all, so use g3's family via a new
        # game would be ideal) -> g4 is unseen with an unseen family
        t = target(user="u9", slate=("g4",))
        out = featurize(history(), [t], GAMES, FRIENDS)
        row = out.iloc[0]
        self.assertTrue(np.isfinite(row[FEATURES].to_numpy(float)).all())
        self.assertEqual(row["rp_best"], -1)
        self.assertEqual(row["fr_played"], 0)
        self.assertAlmostEqual(row["q_r1"], row["pr_r1"], places=9)
        self.assertAlmostEqual(row["q_r2"], row["pr_r2"], places=9)

    def test_unseen_game_in_known_family_uses_family_prior(self):
        games = GAMES.copy()
        games.loc["g5"] = ["m1", "puzzle", 3, 3, 0]   # puzzle family has history, g5 has none
        hist = history()
        st = HistoryState(games, FRIENDS)
        st.apply(hist)
        t = target(user="u9", slate=("g5",))
        out = featurize(hist, [t], games, FRIENDS).iloc[0]
        fam_prior = st._attr_prior("puzzle", "m1")
        glob = st._attr_prior("zzz", "zzz")
        self.assertAlmostEqual(out["pr_r1"], fam_prior[0], places=9)
        self.assertNotAlmostEqual(out["pr_r1"], glob[0], places=6)
        self.assertAlmostEqual(out["q_r1"], out["pr_r1"], places=9)
        fam_mean = st.elo.difficulty("g5", "puzzle")
        self.assertNotEqual(fam_mean, 0.0)
        self.assertAlmostEqual(out["sk_p"], 1 / (1 + math.exp(-(0.0 - fam_mean))), places=9)
        self.assertAlmostEqual(out["sk_p"], st.elo.p_success("u9", "g5", "puzzle"), places=9)

    def test_elo_eligibility_via_apply(self):
        st = HistoryState(GAMES, FRIENDS)
        st.apply(frame([imp("ua", "e1", 1, 10, "g1", tap=0),
                        imp("ub", "e1", 2, 10, "g1", tap=1, ps=3.0)]))
        self.assertEqual(st.elo._ns, {})
        self.assertEqual(st.elo.skill("ua"), 0.0)
        self.assertEqual(st.elo.skill("ub"), 0.0)
        st.apply(frame([imp("uc", "e2", 1, 20, "g1", tap=1, ps=NAN, comp=1)]))
        self.assertEqual(st.elo._ns, {"uc": 1})
        self.assertGreater(st.elo.skill("uc"), 0)        # a win
        self.assertLess(st.elo.difficulty("g1", "puzzle"), 0)

    def test_player_and_game_k_tracked_separately(self):
        st = HistoryState(GAMES, FRIENDS)
        st.apply(frame([imp(f"p{i}", f"k{i}", 1, 10 + i, "g1", tap=1, ps=30, comp=1)
                        for i in range(30)]))
        self.assertEqual(st.elo._nd["g1"], 30)
        d_before = st.elo.difficulty("g1", "puzzle")
        p0 = st.elo.p_success("newbie", "g1", "puzzle")
        st.apply(frame([imp("newbie", "kn", 1, 100, "g1", tap=1, ps=30, comp=1)]))
        # player K uses the player's own count (0 -> K=1); game K uses the game's count (30)
        self.assertAlmostEqual(st.elo.skill("newbie"), 1.0 * (1 - p0), places=9)
        k_game = 1 / (1 + 0.05 * 30)
        self.assertAlmostEqual(st.elo.difficulty("g1", "puzzle"), d_before - k_game * (1 - p0),
                               places=9)
        self.assertLess(k_game, 1.0)

    def test_no_leakage_future(self):
        base = featurize(history(), [target()], GAMES, FRIENDS)[FEATURES].to_numpy()
        extra = frame([
            imp("u2", "h9", 1, 5000, "g1", tap=1, ps=60, comp=1),         # ts == target ts
            imp("u1", "t1", 4, 5000, "g1", tap=1, ps=60, comp=1, rep=2),  # own slate outcome
            imp("u1", "t1", 5, 5000, "g2", tap=1, ps=60, comp=1),
            imp("u2", "h10", 1, 9000, "g1", tap=1, ps=60, comp=1),        # later
            imp("u1", "h11", 1, 9500, "g2", tap=1, ps=60, comp=1),
        ])
        with_extra = featurize(pd.concat([history(), extra], ignore_index=True), [target()],
                               GAMES, FRIENDS)[FEATURES].to_numpy()
        self.assertTrue(np.allclose(base, with_extra))

    def test_friend_counts(self):
        out = featurize(history(), [target(slate=("g1",))], GAMES, FRIENDS)
        self.assertEqual(out.iloc[0]["fr_finished"], 1)
        self.assertEqual(out.iloc[0]["fr_played"], 1)
        late = history().copy()
        late["ts"] = late["ts"] + 10000
        late["day"] = (late["ts"] // DAY).astype(float)
        out = featurize(late, [target(slate=("g1",))], GAMES, FRIENDS)
        self.assertEqual(out.iloc[0]["fr_finished"], 0)

    def test_prefix_flags_and_blank_fields(self):
        prefix = frame([
            imp("u1", "t1", 1, 5000, "g3", watch=NAN, tap=1, ps=NAN, comp=1),
            imp("u1", "t1", 2, 5000, "g4"),
            imp("u1", "t1", 3, 5000, "g4"),
        ])
        out = featurize(history(), [target(prefix=prefix, slate=("g1", "g4"))], GAMES, FRIENDS)
        r = out.iloc[0]  # g1: puzzle family, m1; prefix g3 is puzzle, m2
        self.assertEqual(r["px_fam_play"], 1)
        self.assertEqual(r["px_fam_finish"], 1)
        self.assertEqual(r["px_mech_play"], 0)
        r4 = out.iloc[1]  # g4: arcade family, m1; prefix g4 skips share family
        self.assertEqual(r4["px_fam_skip"], 1)
        self.assertEqual(r4["px_fam_play"], 0)
        self.assertTrue(np.isfinite(out[FEATURES].to_numpy(float)).all())


if __name__ == "__main__":
    unittest.main()
