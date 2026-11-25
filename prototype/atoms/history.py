"""Point-in-time history state and per-(session, game) features.

HistoryState is fed impression rows in time order (``apply``) and answers feature queries for a
target session using only what it has been fed so far. ``featurize`` drives it so that every
target sees exactly the history events with ts strictly earlier than its own ts.

Feature groups (used for ablation):
  quality   the game's own outcome rates / watch time, shrunk toward its attribute prior
  prior     the attribute prior itself (family+mechanic), editorial one-star flag, game age
  affinity  how much better/worse than the population this user does on the game's family/mechanic
  repeat    how often this user was shown the game and the best result they got on it
  skill     Elo success probability for this user on this game and simple skill/difficulty flags
  friends   how many friends have played / finished this game
  prefix    what happened in the session's three observed prefix impressions on the same family
            or mechanic
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from .elo import Elo
from .io import hist_outcome

FEATURE_GROUPS: dict[str, list[str]] = {
    "quality": ["q_r1", "q_r2", "q_r3", "q_watch", "q_logn"],
    "prior": ["pr_r1", "pr_r2", "pr_onestar", "pr_age"],
    "affinity": ["af_fam_r1", "af_fam_r2", "af_mech_r1", "af_mech_r2", "af_depth_x_fam"],
    "repeat": ["rp_seen", "rp_best"],
    "skill": ["sk_p", "sk_gap", "sk_lowskill_hard"],
    "friends": ["fr_played", "fr_finished"],
    "prefix": ["px_fam_play", "px_fam_finish", "px_fam_quit", "px_fam_skip",
               "px_mech_play", "px_mech_quit"],
}
FEATURES: list[str] = [f for g in FEATURE_GROUPS.values() for f in g]

K_QUALITY = 20.0     # shrink strength of a game's own rates toward its attribute prior
K_PRIOR = 50.0       # shrink strength of the family+mechanic prior toward family, then global
K_AFFINITY = 5.0     # shrink strength of a user's family/mechanic rate toward the population
WATCH_SCALE = 12000.0
DAY_SECONDS = 86400


class Target(NamedTuple):
    session_id: str
    user_id: str
    ts: float
    day: int
    prefix: pd.DataFrame
    slate_games: list


class _Counts:
    """Impressions and outcome counts (r>=1, r>=2, r==3) for one key."""
    __slots__ = ("n", "r1", "r2", "r3")

    def __init__(self):
        self.n = self.r1 = self.r2 = self.r3 = 0

    def add(self, r: int) -> None:
        self.n += 1
        self.r1 += r >= 1
        self.r2 += r >= 2
        self.r3 += r == 3


def _prepare(events: pd.DataFrame) -> list[tuple]:
    """Convert impression rows to plain tuples (done once, outside the hot loop).

    tuple = (ts, session_id, user, game, r, tapped, watch_ms, elo_y)
    elo_y is None when the row does not count for Elo (not tapped, or play_seconds < 5);
    a blank play_seconds on a tap still counts, with y = completed.
    """
    if len(events) == 0:
        return []
    r = hist_outcome(events).to_numpy()
    tapped = (events["play_now"].fillna(0) >= 1).to_numpy()
    ps = events["play_seconds"].to_numpy(float)
    comp = (events["completed"].fillna(0) > 0).to_numpy()
    elo_ok = tapped & ~(ps < 5)
    watch = events["watch_ms"].to_numpy(float)
    return list(zip(events["ts"].to_numpy(float).tolist(), events["session_id"].tolist(),
                    events["user_id"].tolist(), events["game_id"].tolist(), r.tolist(),
                    tapped.tolist(), watch.tolist(),
                    [int(c) if ok else None for c, ok in zip(comp, elo_ok)]))


class HistoryState:
    """Running counters over all history events applied so far."""

    def __init__(self, games: pd.DataFrame, friends: dict[str, set[str]]):
        self.friends = friends
        self.fam = games["family"].to_dict()
        self.mech = games["mechanic"].to_dict()
        self.diff = games["difficulty"].to_dict()
        self.stars = games["editorial_stars"].to_dict()
        self.added = games["added_day"].to_dict()
        self.elo = Elo()
        self.g_cnt: dict[str, _Counts] = {}
        self.g_watch: dict[str, list] = {}          # game -> [sum, n] of usable watch_ms
        self.watch_tot = [0.0, 0]
        self.glob = _Counts()
        self.fam_cnt: dict[str, _Counts] = {}
        self.mech_cnt: dict[str, _Counts] = {}
        self.fm_cnt: dict[tuple, _Counts] = {}
        self.u_sessions: dict[str, set] = {}
        self.u_fam: dict[tuple, _Counts] = {}
        self.u_mech: dict[tuple, _Counts] = {}
        self.u_seen: dict[tuple, int] = {}
        self.u_best: dict[tuple, int] = {}          # (user, game) -> best outcome over taps

    # ---- updating -------------------------------------------------------------------------
    def apply(self, events: pd.DataFrame) -> None:
        """Fold impression rows into the state; the caller guarantees time order."""
        self._apply_rows(_prepare(events))

    def _apply_rows(self, rows: list[tuple]) -> None:
        for ts, sid, user, game, r, tapped, watch, y in rows:
            fam = self.fam.get(game)
            if fam is None:
                continue
            mech = self.mech[game]
            self.glob.add(r)
            self.g_cnt.setdefault(game, _Counts()).add(r)
            self.fam_cnt.setdefault(fam, _Counts()).add(r)
            self.mech_cnt.setdefault(mech, _Counts()).add(r)
            self.fm_cnt.setdefault((fam, mech), _Counts()).add(r)
            if watch == watch and watch > 0:          # blank / 0 watch_ms is missing
                w = self.g_watch.setdefault(game, [0.0, 0])
                w[0] += watch
                w[1] += 1
                self.watch_tot[0] += watch
                self.watch_tot[1] += 1
            self.u_sessions.setdefault(user, set()).add(sid)
            self.u_fam.setdefault((user, fam), _Counts()).add(r)
            self.u_mech.setdefault((user, mech), _Counts()).add(r)
            self.u_seen[(user, game)] = self.u_seen.get((user, game), 0) + 1
            if tapped and r > self.u_best.get((user, game), -1):
                self.u_best[(user, game)] = r
            if y is not None:
                self.elo.update(user, game, fam, y)

    # ---- helpers --------------------------------------------------------------------------
    @staticmethod
    def _shrunk(c: _Counts | None, prior: tuple, k: float) -> tuple:
        """(P(r>=1), P(r>=2), P(r=3)) of counts c shrunk toward prior with strength k."""
        if c is None:
            return prior
        n = c.n
        return ((c.r1 + k * prior[0]) / (n + k), (c.r2 + k * prior[1]) / (n + k),
                (c.r3 + k * prior[2]) / (n + k))

    def _attr_prior(self, fam: str, mech: str) -> tuple:
        """Attribute prior: global rate -> family (shrunk) -> family+mechanic (shrunk)."""
        g = self.glob
        gp = (g.r1 / g.n, g.r2 / g.n, g.r3 / g.n) if g.n else (0.0, 0.0, 0.0)
        fp = self._shrunk(self.fam_cnt.get(fam), gp, K_PRIOR)
        return self._shrunk(self.fm_cnt.get((fam, mech)), fp, K_PRIOR)

    def _pop_rate(self, table: dict, key: str) -> tuple:
        c = table.get(key)
        if c and c.n:
            return c.r1 / c.n, c.r2 / c.n
        g = self.glob
        return (g.r1 / g.n, g.r2 / g.n) if g.n else (0.0, 0.0)

    @staticmethod
    def _user_rate(c: _Counts | None, pop: tuple) -> tuple:
        """User's shrunk rate minus the population rate (0 when the user has no data)."""
        n = c.n if c else 0
        r1, r2 = (c.r1, c.r2) if c else (0, 0)
        k = K_AFFINITY
        return (r1 + k * pop[0]) / (n + k) - pop[0], (r2 + k * pop[1]) / (n + k) - pop[1]

    # ---- feature extraction ---------------------------------------------------------------
    def features(self, user: str, session_ts: float, prefix: pd.DataFrame,
                 slate_games: list[str], day: int | None = None) -> np.ndarray:
        """Feature matrix (len(slate_games), len(FEATURES)); every value finite.

        Reads only the state (events already applied) and the session's own prefix.
        `day` defaults to the day containing session_ts.
        """
        if day is None:
            day = int(session_ts // DAY_SECONDS)
        px_play_f, px_fin_f, px_quit_f, px_skip_f = set(), set(), set(), set()
        px_play_m, px_quit_m = set(), set()
        if len(prefix):
            pr = hist_outcome(prefix).tolist()
            tp = (prefix["play_now"].fillna(0) >= 1).tolist()
            for g, r, t in zip(prefix["game_id"], pr, tp):
                f, m = self.fam.get(g), self.mech.get(g)
                if f is None:
                    continue
                if r >= 1:
                    px_play_f.add(f)
                    px_play_m.add(m)
                if r >= 2:
                    px_fin_f.add(f)
                if t and r == 0:
                    px_quit_f.add(f)
                    px_quit_m.add(m)
                if not t:
                    px_skip_f.add(f)

        depth = math.log1p(len(self.u_sessions.get(user, ())))
        skill = self.elo.skill(user)
        friends = self.friends.get(user, ())
        wmean = self.watch_tot[0] / self.watch_tot[1] if self.watch_tot[1] else 0.0
        out = np.zeros((len(slate_games), len(FEATURES)))
        for i, g in enumerate(slate_games):
            fam, mech = self.fam.get(g, ""), self.mech.get(g, "")
            prior = self._attr_prior(fam, mech)
            gc = self.g_cnt.get(g)
            q1, q2, q3 = self._shrunk(gc, prior, K_QUALITY)
            w = self.g_watch.get(g)
            ws, wn = (w[0], w[1]) if w else (0.0, 0)
            q_watch = (ws + K_QUALITY * wmean) / (wn + K_QUALITY) / WATCH_SCALE
            a_f1, a_f2 = self._user_rate(self.u_fam.get((user, fam)), self._pop_rate(self.fam_cnt, fam))
            a_m1, a_m2 = self._user_rate(self.u_mech.get((user, mech)), self._pop_rate(self.mech_cnt, mech))
            p = self.elo.p_success(user, g, fam)
            fr_p = fr_f = 0
            for fr in friends:
                b = self.u_best.get((fr, g), -1)
                fr_p += b >= 1
                fr_f += b >= 2
            out[i] = (
                q1, q2, q3, q_watch, math.log1p(gc.n if gc else 0),
                prior[0], prior[1], float(self.stars.get(g, 0) == 1),
                math.log1p(max(0, day - self.added.get(g, day))),
                a_f1, a_f2, a_m1, a_m2, depth * a_f1,
                math.log1p(self.u_seen.get((user, g), 0)), self.u_best.get((user, g), -1),
                p, abs(p - 0.75), float(skill < -0.5 and self.diff.get(g, 0) >= 4),
                fr_p, fr_f,
                float(fam in px_play_f), float(fam in px_fin_f), float(fam in px_quit_f),
                float(fam in px_skip_f), float(mech in px_play_m), float(mech in px_quit_m),
            )
        return out


def featurize(history: pd.DataFrame, targets: list[Target], games: pd.DataFrame,
              friends: dict[str, set[str]]) -> pd.DataFrame:
    """One row per (session_id, game_id) of the targets' slates, with the FEATURES columns.

    History and targets are walked in ts order. A target sees exactly the history events with
    ts strictly less than its own; events of its own session (same ts) are applied only after
    it has been featurized, as are all later events.
    """
    state = HistoryState(games, friends)
    hist = history.sort_values(["ts", "session_id", "position"], kind="stable")
    rows = _prepare(hist)
    ts_arr = [r[0] for r in rows]
    pos, n = 0, len(rows)
    ids, gids, mats = [], [], []
    for t in sorted(targets, key=lambda t: t.ts):
        end = pos
        while end < n and ts_arr[end] < t.ts:
            end += 1
        if end > pos:
            state._apply_rows(rows[pos:end])
            pos = end
        if not len(t.slate_games):
            continue
        mats.append(state.features(t.user_id, t.ts, t.prefix, list(t.slate_games), t.day))
        ids.extend([t.session_id] * len(t.slate_games))
        gids.extend(t.slate_games)
    if not mats:
        return pd.DataFrame(columns=["session_id", "game_id"] + FEATURES)
    out = pd.DataFrame(np.vstack(mats), columns=FEATURES)
    out.insert(0, "game_id", gids)
    out.insert(0, "session_id", ids)
    return out
