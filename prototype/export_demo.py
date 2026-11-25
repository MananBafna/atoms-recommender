"""Export a sample of val sessions for the static demo page (demo/data.json).

    python prototype/export_demo.py

Reads prototype/out/demo_rows.pkl (every val slate pair with both scores, ranks, raw features and
gain-weighted contributions g_<feature>) and prototype/out/metrics.json. Picks 40 sessions with a
seeded rule (15 biggest wins, 10 biggest losses, 8 random cold-user sessions, 7 random sessions
with a zero-history game, de-duplicated), and turns the largest net contributions into plain
reason labels.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
from atoms.io import label_relevance, load_games, load_impressions  # noqa: E402

OUT = ROOT / "prototype" / "out"
DEMO = ROOT / "demo"
SEED = 0
K = 5
OUTCOME = {0: "Skipped / quit", 1: "Played 20s+", 2: "Finished", 3: "Replayed / shared"}
MIN_CHIP = 0.03  # nets smaller than this are noise around the population mean
NEW_GAME_N = 30
Q_R2_MEAN = json.loads((pathlib.Path(__file__).resolve().parent / "out" / "coefs.json").read_text())["standardizer_mean"]["q_r2"]  # fewer impressions than this in history -> "new game" (evaluate.py uses 30)


def ndcg_from_ranks(df: pd.DataFrame, rank_col: str) -> pd.Series:
    """NDCG@5 per session, same formula as data/evaluate.py; sessions with IDCG=0 dropped."""
    out = {}
    for sid, g in df.groupby("session_id", sort=False):
        gains = (2 ** g["rel"].to_numpy()) - 1
        order = np.argsort(g[rank_col].to_numpy(), kind="stable")
        disc = 1 / np.log2(np.arange(2, K + 2))
        dcg = float((gains[order][:K] * disc[: min(K, len(gains))]).sum())
        ideal = np.sort(gains)[::-1][:K]
        idcg = float((ideal * disc[: len(ideal)]).sum())
        if idcg > 0:
            out[sid] = dcg / idcg
    return pd.Series(out)


# ---- reason groups: (name, g_ columns) ----------------------------------------------------------
GROUPS = {
    "quality": ["q_r1", "q_r2", "q_r3", "q_watch", "q_logn"],
    "prior": ["pr_r1", "pr_r2"],
    "onestar": ["pr_onestar"],
    "age": ["pr_age"],
    "fam": ["af_fam_r1", "af_fam_r2", "af_depth_x_fam"],
    "mech": ["af_mech_r1", "af_mech_r2"],
    "repeat": ["rp_seen", "rp_best"],
    "skill": ["sk_p", "sk_gap"],
    "lowskill": ["sk_lowskill_hard"],
    "friends": ["fr_played", "fr_finished"],
    "px_fam": ["px_fam_play", "px_fam_finish", "px_fam_quit", "px_fam_skip"],
    "px_mech": ["px_mech_play", "px_mech_quit"],
}


def a_an(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def label_for(group: str, net: float, r: pd.Series, game: pd.Series, is_new: bool) -> str | None:
    """Plain wording for one group's net contribution, or None when no truthful wording exists."""
    pos = net > 0
    fam, mech = game["family"], game["mechanic"]
    fam = "word" if fam == "words" else fam  # "likes word games", not "likes words games"
    if group == "quality":
        if is_new:
            return "new game: judged by similar games"
        above = r["q_r2"] > Q_R2_MEAN
        if pos != above:
            return None  # net sign and the game's own finish rate disagree: no truthful wording
        return "other players often finish it" if pos else "other players rarely stick with it"
    if group == "prior":
        if is_new:
            return "new game: judged by similar games"
        return "games like it usually get played" if pos else "games like it rarely get played"
    if group == "onestar":
        return "low editor rating" if r["pr_onestar"] >= 1 and not pos else None
    if group == "age":
        return None  # game age has no wording a player would recognise
    if group == "fam":
        if r["af_fam_r1"] == 0 and r["af_fam_r2"] == 0:
            return None
        return f"likes {fam} games" if pos else f"rarely sticks with {fam} games"
    if group == "mech":
        if r["af_mech_r1"] == 0 and r["af_mech_r2"] == 0:
            return None
        return f"likes {mech} games" if pos else f"rarely sticks with {mech} games"
    if group == "repeat":
        if r["rp_seen"] <= 0:
            return None
        b = r["rp_best"]
        if b >= 2:
            return "finished it before"
        if b >= 1:
            return "played it before"
        return "passed on it before"
    if group == "skill":
        # sk_p has a negative weight: the ranker favours games the player is less sure to beat.
        if pos and r["sk_p"] < 0.5:
            return "a stretch for their skill"
        if not pos and r["sk_p"] >= 0.75:
            return "may be too easy for them"
        return None
    if group == "lowskill":
        return "may be too hard" if r["sk_lowskill_hard"] >= 1 and not pos else None
    if group == "friends":
        if r["fr_finished"] >= 1:
            n = int(r["fr_finished"])
            return f"{n} friend{'s' if n > 1 else ''} finished it"
        if r["fr_played"] >= 1:
            n = int(r["fr_played"])
            return f"{n} friend{'s' if n > 1 else ''} played it"
        return None
    if group == "px_fam":
        if r["px_fam_finish"] >= 1:
            return f"just finished {a_an(fam)} game"
        if r["px_fam_play"] >= 1:
            return f"just played {a_an(fam)} game"
        if r["px_fam_quit"] >= 1:
            return f"just quit {a_an(fam)} game"
        if r["px_fam_skip"] >= 1:
            return f"just skipped {a_an(fam)} game"
        return None
    if group == "px_mech":
        if r["px_mech_play"] >= 1:
            return f"just played {a_an(mech)} game"
        if r["px_mech_quit"] >= 1:
            return f"just quit {a_an(mech)} game"
        return None
    return None


def reasons(r: pd.Series, game: pd.Series, is_new: bool) -> list[dict]:
    cands = []
    for grp, feats in GROUPS.items():
        net = float(sum(r[f"g_{f}"] for f in feats))
        lab = label_for(grp, net, r, game, is_new)
        if lab is None:
            continue
        cands.append((abs(net), lab, net))
    cands.sort(reverse=True)
    out, seen = [], set()
    for _, lab, net in cands:
        if lab in seen or abs(net) < MIN_CHIP:
            continue
        seen.add(lab)
        out.append({"label": lab, "value": round(net, 4)})
        if len(out) == 3:
            break
    return out


def main() -> None:
    rows = pd.read_pickle(OUT / "demo_rows.pkl")
    metrics = json.loads((OUT / "metrics.json").read_text())
    games = load_games()
    train = load_impressions("train_impressions")
    prefix = load_impressions("val_prefix")
    prefix["rel"] = label_relevance(prefix)

    ours = ndcg_from_ranks(rows, "rank_ours")
    base = ndcg_from_ranks(rows, "rank_base")
    assert len(ours) == metrics["val"]["sessions_scored"] == 604, len(ours)
    assert round(ours.mean(), 4) == metrics["val"]["ndcg@5"], ours.mean()
    assert round(base.mean(), 4) == metrics["val_baseline"]["ndcg@5"], base.mean()

    n_sess = train.groupby("user_id")["session_id"].nunique()
    g_imps = train["game_id"].value_counts()
    sess_user = rows.drop_duplicates("session_id").set_index("session_id")["user_id"]
    cold = sess_user.map(lambda u: n_sess.get(u, 0) <= 2)
    zero = rows.assign(z=~rows["game_id"].isin(g_imps.index)).groupby("session_id")["z"].any()
    scored = ours.index
    assert int(cold.loc[scored].sum()) == metrics["slices"]["cold_users"]["n_sessions"]
    assert int(zero.loc[scored].sum()) == metrics["slices"]["sessions_with_zero_history_game"]["n_sessions"]

    diff = (ours - base).sort_values(kind="stable")
    rng = np.random.default_rng(SEED)
    picked: list[tuple[str, str]] = []

    def take(cands, n, bucket):
        got = 0
        for sid in cands:
            if got == n:
                break
            if sid in {p for p, _ in picked}:
                continue
            picked.append((sid, bucket))
            got += 1

    take(list(diff.index[::-1]), 15, "win")
    take(list(diff.index), 10, "loss")
    cold_ids = [s for s in scored if cold[s]]
    take(list(rng.permutation(cold_ids)), 8, "cold")
    zero_ids = [s for s in scored if zero[s]]
    take(list(rng.permutation(zero_ids)), 7, "new_game")
    assert len(picked) == 40

    def gattr(gid):
        g = games.loc[gid]
        return {"game_id": gid, "title": g["title"], "family": g["family"],
                "mechanic": g["mechanic"], "difficulty": int(g["difficulty"])}

    sessions = []
    for sid, bucket in picked:
        s = rows[rows["session_id"] == sid].sort_values("rank_ours")
        pfx = prefix[prefix["session_id"] == sid].sort_values("position")
        slate = []
        for _, r in s.iterrows():
            gid = r["game_id"]
            is_new = int(g_imps.get(gid, 0)) < NEW_GAME_N
            slate.append({**gattr(gid), "rank_ours": int(r["rank_ours"]), "rank_base": int(r["rank_base"]),
                          "rel": int(r["rel"]), "outcome_label": OUTCOME[int(r["rel"])],
                          "zero_history": bool(g_imps.get(gid, 0) == 0), "new_game": is_new,
                          "top_reasons": reasons(r, games.loc[gid], is_new)})
        sessions.append({
            "session_id": sid, "user_id": s["user_id"].iloc[0], "bucket": bucket,
            "is_cold_user": bool(cold[sid]), "has_zero_history_game": bool(zero[sid]),
            "user_history_sessions": int(n_sess.get(s["user_id"].iloc[0], 0)),
            "ndcg_ours": round(float(ours[sid]), 4), "ndcg_base": round(float(base[sid]), 4),
            "prefix": [{**gattr(g), "rel": int(rel), "outcome_label": OUTCOME[int(rel)]}
                       for g, rel in zip(pfx["game_id"], pfx["rel"])],
            "slate": slate,
        })

    b = metrics["bootstrap"]
    cs = metrics["slices"]["cold_users"]
    data = {
        "headline": {"ours": b["ours"], "baseline": b["baseline"], "mean_diff": b["mean_diff"],
                     "ci_low": b["ci_low"], "ci_high": b["ci_high"], "sessions": b["n_sessions"],
                     "cold": {"sessions": cs["n_sessions"], "ours": cs["ours"], "baseline": cs["baseline"],
                              "diff": cs["diff"], "ci_low": cs["ci_low"], "ci_high": cs["ci_high"]}},
        "sample": {"win": 15, "loss": 10, "cold": 8, "new_game": 7, "seed": SEED},
        "sessions": sessions,
    }
    DEMO.mkdir(exist_ok=True)
    (DEMO / "data.json").write_text(json.dumps(data, indent=1, allow_nan=False), encoding="utf-8")
    for s in sessions:
        assert all("rel" in g and g["rank_ours"] and g["rank_base"] for g in s["slate"])
        assert all(math.isfinite(c["value"]) for g in s["slate"] for c in g["top_reasons"])
    print(f"wrote {DEMO / 'data.json'}: {len(sessions)} sessions")


if __name__ == "__main__":
    main()
