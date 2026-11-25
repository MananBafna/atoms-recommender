"""Data loading and the history outcome rule."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"

NUMERIC_COLS = ["position", "ts", "day", "watch_ms", "play_now", "play_seconds",
                "completed", "score_percentile", "replays", "shared", "challenged"]


def load_games() -> pd.DataFrame:
    """Game attributes indexed by game_id."""
    return pd.read_csv(DATA_DIR / "games.csv").set_index("game_id")


def load_impressions(name: str) -> pd.DataFrame:
    """Read data/<name>.csv, drop exact duplicate rows, numeric columns as float (blank -> NaN)."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv", dtype=str, keep_default_na=False)
    df = df.drop_duplicates().reset_index(drop=True)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce").astype(float)
    return df


def load_friends() -> dict[str, set[str]]:
    """Undirected friend graph as user -> set of friends."""
    df = pd.read_csv(DATA_DIR / "friends.csv", dtype=str)
    out: dict[str, set[str]] = {}
    for u, v in zip(df["user_id"], df["friend_id"]):
        out.setdefault(u, set()).add(v)
        out.setdefault(v, set()).add(u)
    return out


def hist_outcome(df: pd.DataFrame) -> pd.Series:
    """Relevance of each impression as used for *history* features.

    Differs from the evaluation label only for taps with blank play_seconds
    (logging gap): those count as played, r=2 if completed else r=1.
    Not tapped -> 0; tapped and played < 20 s -> 0; replay/share/challenge -> 3.
    """
    tapped = df["play_now"].fillna(0) >= 1
    ps = df["play_seconds"]
    blank = ps.isna()
    played = tapped & (blank | (ps >= 20))
    lift = ((df["replays"].fillna(0) > 0) | (df["shared"].fillna(0) > 0)
            | (df["challenged"].fillna(0) > 0))
    completed = df["completed"].fillna(0) > 0
    r = np.where(~played, 0, np.where(lift, 3, np.where(completed, 2, 1)))
    return pd.Series(r.astype(int), index=df.index)


def label_relevance(df: pd.DataFrame) -> pd.Series:
    """Official evaluation label (data/common.py::relevance), vectorised.

    Blank play_seconds counts as 0, so a tap with a logging gap is relevance 0.
    """
    tapped = df["play_now"].fillna(0) >= 1
    played = tapped & (df["play_seconds"].fillna(0) >= 20)
    lift = ((df["replays"].fillna(0) > 0) | (df["shared"].fillna(0) > 0)
            | (df["challenged"].fillna(0) > 0))
    completed = df["completed"].fillna(0) > 0
    r = np.where(~played, 0, np.where(lift, 3, np.where(completed, 2, 1)))
    return pd.Series(r.astype(int), index=df.index)
