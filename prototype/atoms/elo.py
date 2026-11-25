"""Elo-style player skill / game difficulty ratings.

P(success) = 1 / (1 + exp(-(skill - difficulty))). After an outcome y in {0, 1}:
skill += K_s * (y - P), difficulty -= K_d * (y - P), with K = a / (1 + b * n)
where n is the number of earlier rated plays of that player (K_s) or game (K_d).
"""
from __future__ import annotations

import math


class Elo:
    def __init__(self, a: float = 1.0, b: float = 0.05):
        self.a, self.b = a, b
        self._s: dict[str, float] = {}
        self._ns: dict[str, int] = {}
        self._d: dict[str, float] = {}
        self._nd: dict[str, int] = {}
        # running per-family sum / count of rated games' difficulties
        self._fam_sum: dict[str, float] = {}
        self._fam_cnt: dict[str, int] = {}

    def skill(self, user: str) -> float:
        return self._s.get(user, 0.0)

    def difficulty(self, game: str, family: str) -> float:
        """Rated difficulty, or the mean over rated games of the family (0.0 if none)."""
        d = self._d.get(game)
        if d is not None:
            return d
        c = self._fam_cnt.get(family, 0)
        return self._fam_sum[family] / c if c else 0.0

    def p_success(self, user: str, game: str, family: str) -> float:
        z = self.skill(user) - self.difficulty(game, family)
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, user: str, game: str, family: str, y: int) -> None:
        d_old = self.difficulty(game, family)
        p = 1.0 / (1.0 + math.exp(-(self.skill(user) - d_old)))
        err = y - p
        ks = self.a / (1.0 + self.b * self._ns.get(user, 0))
        kd = self.a / (1.0 + self.b * self._nd.get(game, 0))
        self._s[user] = self.skill(user) + ks * err
        self._ns[user] = self._ns.get(user, 0) + 1
        d_new = d_old - kd * err
        if game in self._d:
            self._fam_sum[family] += d_new - self._d[game]
        else:
            self._fam_sum[family] = self._fam_sum.get(family, 0.0) + d_new
            self._fam_cnt[family] = self._fam_cnt.get(family, 0) + 1
        self._d[game] = d_new
        self._nd[game] = self._nd.get(game, 0) + 1
