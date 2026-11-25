"""Builders for pseudo-sessions (training), evaluation targets and per-split history."""
from __future__ import annotations

import pandas as pd

from .history import DAY_SECONDS, Target
from .io import label_relevance, load_impressions

PREFIX_LEN = 3


def _day_of(ts: float, prefix: pd.DataFrame) -> int:
    """Session day from ts; falls back to the prefix rows' day column if they disagree."""
    d = int(ts // DAY_SECONDS)
    pd_days = prefix["day"].dropna()
    if len(pd_days) and int(pd_days.iloc[0]) != d:
        return int(pd_days.iloc[0])
    return d


def pseudo_sessions(train: pd.DataFrame, days: tuple[int, int]) -> tuple[list[Target], pd.DataFrame]:
    """Turn train sessions (>= 7 impressions, day in `days`) into ranking targets.

    Positions 1-3 are the prefix, later positions the slate. Returns the targets and a frame of
    labels (session_id, game_id, rel) using the official relevance rule.
    """
    lo, hi = days
    df = train[(train["day"] >= lo) & (train["day"] <= hi)]
    df = df.sort_values(["session_id", "position"], kind="stable")
    df = df.assign(rel=label_relevance(df).to_numpy())
    targets: list[Target] = []
    lab: list[pd.DataFrame] = []
    for sid, g in df.groupby("session_id", sort=False):
        if len(g) < PREFIX_LEN + 4:
            continue
        prefix = g.iloc[:PREFIX_LEN]
        slate = g.iloc[PREFIX_LEN:]
        ts = float(g["ts"].iloc[0])
        targets.append(Target(sid, g["user_id"].iloc[0], ts, _day_of(ts, prefix),
                              prefix.reset_index(drop=True), slate["game_id"].tolist()))
        lab.append(slate[["session_id", "game_id", "rel"]])
    labels = pd.concat(lab, ignore_index=True) if lab else pd.DataFrame(columns=["session_id", "game_id", "rel"])
    return targets, labels


def eval_targets(split: str) -> list[Target]:
    """Targets for "val" or "test" from <split>_prefix and <split>_slates."""
    prefix = load_impressions(f"{split}_prefix").sort_values(["session_id", "position"], kind="stable")
    slates = load_impressions(f"{split}_slates")
    slate_by = {sid: g for sid, g in slates.groupby("session_id", sort=False)}
    targets = []
    for sid, p in prefix.groupby("session_id", sort=False):
        s = slate_by.get(sid)
        if s is None:
            continue
        ts = float(s["ts"].iloc[0])
        p = p.reset_index(drop=True)
        targets.append(Target(sid, p["user_id"].iloc[0], ts, _day_of(ts, p), p,
                              s["game_id"].tolist()))
    return targets


def history_for(split: str) -> pd.DataFrame:
    """History events usable for a split (the caller still cuts at each target's ts)."""
    train = load_impressions("train_impressions")
    if split == "holdout":
        return train
    if split == "val":
        return pd.concat([train, load_impressions("val_prefix")], ignore_index=True)
    if split == "test":
        return pd.concat([train, load_impressions("val_events"),
                          load_impressions("test_prefix")], ignore_index=True)
    raise ValueError(f"unknown split {split!r}")
