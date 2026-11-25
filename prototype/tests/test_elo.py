import unittest

from atoms.elo import Elo


class TestElo(unittest.TestCase):
    def test_new_pair_is_half(self):
        self.assertEqual(Elo().p_success("u", "g", "f"), 0.5)

    def test_win_raises_skill_lowers_difficulty(self):
        e = Elo()
        e.update("u", "g", "f", 1)
        self.assertGreater(e.skill("u"), 0)
        self.assertLess(e.difficulty("g", "f"), 0)

    def test_k_decays(self):
        e = Elo()
        e.update("u", "g", "f", 1)
        first = e.skill("u")
        for _ in range(18):
            e.update("u", "g", "f", 1)
        before = e.skill("u")
        e.update("u", "g", "f", 1)
        self.assertGreater(first, e.skill("u") - before)

    def test_unseen_game_uses_family_mean(self):
        e = Elo()
        e.update("u", "g1", "f", 1)
        e.update("u2", "g2", "f", 0)
        mean = (e.difficulty("g1", "f") + e.difficulty("g2", "f")) / 2
        self.assertAlmostEqual(e.difficulty("new", "f"), mean, places=9)
        self.assertEqual(e.difficulty("new", "other"), 0.0)


if __name__ == "__main__":
    unittest.main()
