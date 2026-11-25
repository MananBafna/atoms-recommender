"""Metric, bootstrap and popularity baseline utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import label_relevance

K = 5


def ndcg5_per_session(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Per-session NDCG@5 (gain 2^rel-1), skipping sessions without gain, like data/evaluate.py.

    Games are ranked by descending score, ties keeping the input order of `scores`;
    games missing from `scores` rank last.
    """
    sc = scores[["session_id", "game_id", "score"]].copy()
    sc["score"] = pd.to_numeric(sc["score"], errors="coerce")
    sc = sc.dropna(subset=["score"]).drop_duplicates(["session_id", "game_id"], keep="last")
    df = labels[["session_id", "game_id", "rel"]].merge(sc, on=["session_id", "game_id"], how="left")
    df["score"] = df["score"].fillna(-np.inf)
    df["gain"] = 2.0 ** df["rel"].astype(float) - 1.0
    disc = 1.0 / np.log2(np.arange(K) + 2)
    out = {}
    for sid, g in df.groupby("session_id", sort=False):
        gains = g["gain"].to_numpy()
        ranked = gains[np.argsort(-g["score"].to_numpy(), kind="stable")]
        ideal = np.sort(gains)[::-1]
        idcg = float((ideal[:K] * disc[:len(ideal[:K])]).sum())
        if idcg <= 0:
            continue
        out[sid] = float((ranked[:K] * disc[:len(ranked[:K])]).sum()) / idcg
    return pd.Series(out, dtype=float)


def paired_bootstrap(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0) -> dict:
    """Bootstrap of the mean of a-b over sessions; 95% percentile interval."""
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(d), size=(n, len(d)))
    means = d[draws].mean(axis=1)
    return {"mean_diff": float(d.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5))}


def popularity_scores(history: pd.DataFrame, pairs: pd.DataFrame, prior_strength: float = 10) -> pd.DataFrame:
    """Smoothed share of impressions with relevance >= 1 per game (data/baseline_popularity.py).

    `history` needs the raw outcome columns; labels use the evaluation rule.
    """
    good = (label_relevance(history) >= 1).astype(float)
    g = pd.DataFrame({"game_id": history["game_id"].to_numpy(), "good": good.to_numpy()})
    agg = g.groupby("game_id")["good"].agg(["sum", "count"])
    rate = g["good"].sum() / max(1, len(g))
    out = pairs[["session_id", "game_id"]].copy()
    s = out["game_id"].map(agg["sum"]).fillna(0.0)
    c = out["game_id"].map(agg["count"]).fillna(0.0)
    out["score"] = ((s + prior_strength * rate) / (c + prior_strength)).to_numpy()
    return out
