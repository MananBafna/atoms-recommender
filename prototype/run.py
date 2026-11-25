"""End-to-end run: model selection, val/test rankings, analysis outputs.

    cd prototype && python run.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from atoms.evaluation import ndcg5_per_session, paired_bootstrap  # noqa: E402
from atoms.history import FEATURE_GROUPS, FEATURES, HistoryState, featurize  # noqa: E402
from atoms.io import DATA_DIR, hist_outcome, label_relevance, load_friends, load_games, load_impressions  # noqa: E402
from atoms.model import PairwiseRanker, PointwiseRanker  # noqa: E402
from atoms.sessions import eval_targets, history_for, pseudo_sessions  # noqa: E402

OUT = ROOT / "out"
SUB = ROOT.parent / "submission"
L2_GRID = [0.1, 1.0, 10.0]
T0 = time.perf_counter()


def stage(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


def fit_model(variant: str, l2: float, X: np.ndarray, rel: np.ndarray, sid: np.ndarray):
    if variant == "pointwise":
        return PointwiseRanker(l2).fit(X, rel)
    return PairwiseRanker(l2).fit(X, rel, sid)


def with_rel(feat: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach labels to the feature rows; fails if the join is not one-to-one."""
    m = feat.merge(labels, on=["session_id", "game_id"], how="left", validate="one_to_one")
    assert m["rel"].notna().all()
    return m


def mean_ndcg(scored: pd.DataFrame, labels: pd.DataFrame) -> float:
    return float(ndcg5_per_session(scored[["session_id", "game_id", "score"]], labels).mean())


def auc(y: np.ndarray, s: np.ndarray) -> float:
    r = pd.Series(s).rank().to_numpy()
    n1 = float(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank_pct(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Within-session position as a share of the slate ranked below: 1 = top, 0 = bottom."""
    n = df.groupby("session_id")[score_col].transform("size")
    rk = df.groupby("session_id")[score_col].rank(ascending=False, method="average")
    return (n - rk) / (n - 1).clip(lower=1)


def split_pct(df: pd.DataFrame, cols: list) -> dict:
    out = {}
    for name, m in (("rel_ge1", df["rel"] >= 1), ("rel_0", df["rel"] == 0)):
        out[name] = {"n": int(m.sum()), **{c: float(df.loc[m, c].mean()) for c in cols}}
    return out


def run_official(split: str, path: pathlib.Path) -> dict:
    r = subprocess.run([sys.executable, str(DATA_DIR / "evaluate.py"), "--split", split,
                        "--rankings", str(path)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    SUB.mkdir(exist_ok=True)
    games, friends = load_games(), load_friends()
    train = load_impressions("train_impressions")

    # ---- 1. pseudo-session features (history = train) ---------------------------------------
    stage("1. featurize pseudo-sessions, days 7-19 (train) and 20-23 (holdout)")
    tg_tr, lab_tr = pseudo_sessions(train, (7, 19))
    tg_ho, lab_ho = pseudo_sessions(train, (20, 23))
    feat = featurize(train, tg_tr + tg_ho, games, friends)
    tr_ids = {t.session_id for t in tg_tr}
    F_all = with_rel(feat, pd.concat([lab_tr, lab_ho], ignore_index=True))
    is_tr = F_all["session_id"].isin(tr_ids).to_numpy()
    F_tr, F_ho = F_all[is_tr].reset_index(drop=True), F_all[~is_tr].reset_index(drop=True)
    stage(f"   train sessions {len(tg_tr)} rows {len(F_tr)}; holdout sessions {len(tg_ho)} rows {len(F_ho)}")

    # ---- 2. model selection on the holdout --------------------------------------------------
    stage("2. fit variants x l2 on days 7-19, score holdout (days 20-23)")
    Xtr, ytr, str_ = F_tr[FEATURES].to_numpy(), F_tr["rel"].to_numpy(), F_tr["session_id"].to_numpy()
    Xho = F_ho[FEATURES].to_numpy()
    holdout = []
    for variant in ("pointwise", "pairwise"):
        res: dict[float, float] = {}

        def try_l2(l2: float) -> None:
            m = fit_model(variant, l2, Xtr, ytr, str_)
            sc = F_ho[["session_id", "game_id"]].assign(score=m.score(Xho))
            res[l2] = mean_ndcg(sc, lab_ho)
            holdout.append({"variant": variant, "l2": l2, "ndcg5": res[l2]})
            stage(f"   {variant:9s} l2={l2:<5g} holdout NDCG@5 {res[l2]:.4f}")

        for l2 in L2_GRID:
            try_l2(l2)
        best = max(res, key=res.get)
        if best == min(L2_GRID):
            stage(f"   {variant}: winner at the low edge, extending grid to 0.01")
            try_l2(0.01)
        elif best == max(L2_GRID):
            stage(f"   {variant}: winner at the high edge, extending grid to 100")
            try_l2(100.0)
    top = max(holdout, key=lambda r: r["ndcg5"])
    variant, l2 = top["variant"], top["l2"]
    stage(f"   chosen: {variant}, l2={l2:g} (holdout {top['ndcg5']:.4f})")

    # ---- 3. refit on days 7-23, rank val ----------------------------------------------------
    stage("3. refit on days 7-23; featurize and rank val")
    Xfit = F_all[FEATURES].to_numpy()
    model = fit_model(variant, l2, Xfit, F_all["rel"].to_numpy(), F_all["session_id"].to_numpy())
    val_t = eval_targets("val")
    vfeat = featurize(history_for("val"), val_t, games, friends)
    vfeat["score"] = model.score(vfeat[FEATURES].to_numpy())
    val_slates = load_impressions("val_slates")[["session_id", "user_id", "game_id"]]
    labels_val = pd.read_csv(DATA_DIR / "val_labels.csv").rename(columns={"relevance": "rel"})
    vdf = val_slates.merge(vfeat, on=["session_id", "game_id"], how="left", validate="one_to_one")
    assert len(vdf) == 9576 and np.isfinite(vdf["score"]).all()
    vdf = vdf.merge(labels_val, on=["session_id", "game_id"], validate="one_to_one")
    vdf[["session_id", "game_id", "score"]].to_csv(OUT / "rankings_val.csv", index=False)

    # ---- 4. official evaluation, bootstrap, slices ------------------------------------------
    stage("4. official evaluation, bootstrap, slices")
    ours_official = run_official("val", OUT / "rankings_val.csv")
    base_path = DATA_DIR / "baseline_val_rankings.csv"
    base_official = run_official("val", base_path)
    stage(f"   ours {ours_official['ndcg@5']}  baseline {base_official['ndcg@5']}")
    base = pd.read_csv(base_path)
    base_sc = base.rename(columns={"score": "score_base"})
    ours_ps = ndcg5_per_session(vdf[["session_id", "game_id", "score"]], labels_val)
    base_ps = ndcg5_per_session(base, labels_val)
    boot = paired_bootstrap(ours_ps, base_ps, n=2000, seed=0)
    boot.update(ours=float(ours_ps.mean()), baseline=float(base_ps.mean()), n_sessions=int(len(ours_ps)))
    stage(f"   bootstrap diff {boot['mean_diff']:+.4f}  CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")

    sess_user = val_slates.drop_duplicates("session_id").set_index("session_id")["user_id"]
    n_sess = train.groupby("user_id")["session_id"].nunique()
    cold_user = sess_user.map(lambda u: n_sess.get(u, 0) <= 2)
    g_imps = train["game_id"].value_counts()
    vd_g = val_slates.assign(zero=~val_slates["game_id"].isin(g_imps.index))
    zero_games = vd_g.groupby("session_id")["zero"].any()
    pfx = load_impressions("val_prefix")
    pfx_play = (label_relevance(pfx) >= 1).groupby(pfx["session_id"]).any()
    pfx_play = pfx_play.reindex(sess_user.index).fillna(False)

    def slice_stats(mask: pd.Series) -> dict:
        ids = ours_ps.index.intersection(mask[mask].index)
        out = {"n_sessions": int(len(ids)),
               "ours": float(ours_ps.loc[ids].mean()) if len(ids) else None,
               "baseline": float(base_ps.loc[ids].mean()) if len(ids) else None}
        if len(ids):
            b = paired_bootstrap(ours_ps.loc[ids], base_ps.loc[ids], n=2000, seed=0)
            out.update({"diff": b["mean_diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
        return out

    slices = {"cold_users": slice_stats(cold_user),
              "sessions_with_zero_history_game": slice_stats(zero_games.reindex(sess_user.index).fillna(False)),
              "prefix_had_play": slice_stats(pfx_play),
              "prefix_no_play": slice_stats(~pfx_play)}
    for k, v in slices.items():
        stage(f"   slice {k}: n={v['n_sessions']} ours={v['ours']} base={v['baseline']} "
              f"diff {v['diff']:+.4f} CI [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")

    # ---- 5. ablation (reporting only) -------------------------------------------------------
    stage("5. cumulative feature-group ablation on val")
    ablation, cols = [], []
    for grp, fs in FEATURE_GROUPS.items():
        cols = cols + fs
        m = fit_model(variant, l2, F_all[cols].to_numpy(), F_all["rel"].to_numpy(),
                      F_all["session_id"].to_numpy())
        sc = vdf[["session_id", "game_id"]].assign(score=m.score(vdf[cols].to_numpy()))
        nd = mean_ndcg(sc, labels_val)
        ablation.append({"added_group": grp, "n_features": len(cols), "val_ndcg5": nd})
        stage(f"   +{grp:9s} ({len(cols):2d} features) val NDCG@5 {nd:.4f}")
    pd.DataFrame(ablation).to_csv(OUT / "ablation.csv", index=False)

    # ---- 6. pseudo-cold check ---------------------------------------------------------------
    stage("6. pseudo-cold check on the holdout")
    m_tr = fit_model(variant, l2, Xtr, ytr, str_)
    rng = np.random.default_rng(0)
    all_games = np.array(sorted(games.index))
    masked = set(rng.choice(all_games, size=int(round(0.15 * len(all_games))), replace=False).tolist())
    feat_m = featurize(train[~train["game_id"].isin(masked)], tg_ho, games, friends)
    F_m = with_rel(feat_m, lab_ho)
    contains = F_m.groupby("session_id")["game_id"].apply(lambda s: s.isin(masked).any())
    ids = contains[contains].index
    prior_cols = FEATURE_GROUPS["prior"]
    Xm = F_m[FEATURES].to_numpy()
    sc_prior = F_m[["session_id", "game_id"]].assign(score=m_tr.score(Xm))
    F_nop = F_m.copy()
    gmean = F_tr[prior_cols].mean()
    for c in prior_cols:
        F_nop[c] = gmean[c]
    sc_noprior = F_m[["session_id", "game_id"]].assign(score=m_tr.score(F_nop[FEATURES].to_numpy()))
    sc_full = F_ho[["session_id", "game_id"]].assign(score=m_tr.score(Xho))
    lab_sub = lab_ho[lab_ho["session_id"].isin(ids)]
    nd_p = ndcg5_per_session(sc_prior, lab_sub)
    nd_n = ndcg5_per_session(sc_noprior, lab_sub)
    nd_u = ndcg5_per_session(sc_full, lab_sub)
    pseudo_cold = {"masked_games": len(masked), "sessions_with_masked_game": int(len(ids)),
                   "sessions_scored": int(len(nd_p)),
                   "with_attribute_prior": float(nd_p.mean()),
                   "prior_columns_at_global_mean": float(nd_n.mean()),
                   "unmasked_history_reference": float(nd_u.mean()),
                   "note": "model fitted on days 7-19; masked games lose all history rows; the quality "
                           "features still fall back to the attribute prior inside the history state"}
    # variant B: masked games' quality and prior rates replaced by the global rate (no attribute prior)
    th = train[~train["game_id"].isin(masked)]
    ro = hist_outcome(th)
    glob_rate = {"r1": float((ro >= 1).mean()), "r2": float((ro >= 2).mean()), "r3": float((ro == 3).mean())}
    F_g = F_m.copy()
    mrow = F_g["game_id"].isin(masked)
    for c, k in (("q_r1", "r1"), ("q_r2", "r2"), ("q_r3", "r3"), ("pr_r1", "r1"), ("pr_r2", "r2")):
        F_g.loc[mrow, c] = glob_rate[k]
    sc_glob = F_m[["session_id", "game_id"]].assign(score=m_tr.score(F_g[FEATURES].to_numpy()))
    nd_g = ndcg5_per_session(sc_glob, lab_sub)
    rows = F_m[["session_id", "game_id", "rel"]].copy()
    rows["s_prior"] = sc_prior["score"].to_numpy()
    rows["s_global"] = sc_glob["score"].to_numpy()
    rows["s_pr_mean"] = sc_noprior["score"].to_numpy()
    rows = rows[rows["session_id"].isin(ids)].copy()
    for c in ("s_prior", "s_global", "s_pr_mean"):
        rows["p_" + c[2:]] = rank_pct(rows, c)
    mrows = rows[rows["game_id"].isin(masked)]
    pseudo_cold.update({
        "global_rate_variant": float(nd_g.mean()),
        "global_rates_used": glob_rate,
        "masked_row_rank_pct": split_pct(mrows, ["p_prior", "p_global", "p_pr_mean"]),
        "rank_pct_note": "share of the slate ranked below the game (1 = top); masked games only; "
                         "global rates are computed on all of train, a constant, so not strictly point-in-time"})
    stage(f"   global-rate variant {pseudo_cold['global_rate_variant']:.4f}; masked-row rank pct "
          f"{pseudo_cold['masked_row_rank_pct']}")
    stage(f"   masked {len(masked)} games, {len(nd_p)} sessions: prior {pseudo_cold['with_attribute_prior']:.4f}"
          f" vs global-mean {pseudo_cold['prior_columns_at_global_mean']:.4f}"
          f" (unmasked {pseudo_cold['unmasked_history_reference']:.4f})")

    # ---- 7. latency -------------------------------------------------------------------------
    stage("7. latency per val slate (features + score, warm state)")
    state = HistoryState(games, friends)
    state.apply(history_for("val").sort_values(["ts", "session_id", "position"], kind="stable"))
    times = []
    for t in val_t:
        a = time.perf_counter()
        Xs = state.features(t.user_id, t.ts, t.prefix, list(t.slate_games), t.day)
        model.score(Xs)
        times.append((time.perf_counter() - a) * 1000.0)
    lat = {"p50": float(np.percentile(times, 50)), "p99": float(np.percentile(times, 99)),
           "n_slates": len(times)}
    stage(f"   p50 {lat['p50']:.3f} ms  p99 {lat['p99']:.3f} ms over {len(times)} slates")
    cat = sorted(games.index)
    ftimes = []
    for t in val_t:
        a = time.perf_counter()
        Xs = state.features(t.user_id, t.ts, t.prefix, cat, t.day)
        model.score(Xs)
        ftimes.append((time.perf_counter() - a) * 1000.0)
    lat["full_catalogue_200"] = {"p50": float(np.percentile(ftimes, 50)), "p99": float(np.percentile(ftimes, 99)),
                                 "n_targets": len(ftimes), "n_games": len(cat)}
    stage(f"   full catalogue ({len(cat)} games) p50 {lat['full_catalogue_200']['p50']:.2f} ms "
          f"p99 {lat['full_catalogue_200']['p99']:.2f} ms")

    # ---- 8. test ----------------------------------------------------------------------------
    stage("8. refit on pseudo-sessions from train + val_events (days 7-26); rank test")
    full = pd.concat([train, load_impressions("val_events")], ignore_index=True)
    tg_f, lab_f = pseudo_sessions(full, (7, 26))
    ffeat = with_rel(featurize(full, tg_f, games, friends), lab_f)
    model_t = fit_model(variant, l2, ffeat[FEATURES].to_numpy(), ffeat["rel"].to_numpy(),
                        ffeat["session_id"].to_numpy())
    test_t = eval_targets("test")
    tfeat = featurize(history_for("test"), test_t, games, friends)
    tfeat["score"] = model_t.score(tfeat[FEATURES].to_numpy())
    test_slates = load_impressions("test_slates")[["session_id", "game_id"]]
    tout = test_slates.merge(tfeat[["session_id", "game_id", "score"]], on=["session_id", "game_id"],
                             how="left", validate="one_to_one")
    assert len(tout) == 9724 and np.isfinite(tout["score"]).all()
    tout.to_csv(SUB / "rankings_test.csv", index=False)
    base_t = pd.read_csv(DATA_DIR / "baseline_test_rankings.csv")
    ov = []
    bt = base_t.set_index(["session_id", "game_id"])["score"]
    tout["b"] = bt.reindex(pd.MultiIndex.from_frame(tout[["session_id", "game_id"]])).to_numpy()
    for sid, g in tout.groupby("session_id"):
        a5 = set(g.nlargest(5, "score")["game_id"])
        b5 = set(g.nlargest(5, "b")["game_id"])
        ov.append(len(a5 & b5) / 5.0)
    test_summary = {"rows": int(len(tout)), "sessions": int(tout["session_id"].nunique()),
                    "train_pseudo_sessions": len(tg_f),
                    "top5_overlap_with_baseline": float(np.mean(ov))}
    stage(f"   test rows {test_summary['rows']}, sessions {test_summary['sessions']}, "
          f"top-5 overlap with baseline {test_summary['top5_overlap_with_baseline']:.3f}")

    # ---- 9. worst sessions, demo rows, outputs -----------------------------------------------
    stage("9. worst sessions, demo rows, metrics")
    contrib = model.contributions(vdf[FEATURES].to_numpy())
    cdf = pd.DataFrame(contrib, columns=["c_" + f for f in FEATURES], index=vdf.index)
    demo = pd.concat([vdf[["session_id", "user_id", "game_id", "rel"]].reset_index(drop=True)], axis=1)
    demo["score_ours"] = vdf["score"].to_numpy()
    demo = demo.merge(base_sc, on=["session_id", "game_id"], how="left", validate="one_to_one")
    demo["rank_ours"] = demo.groupby("session_id")["score_ours"].rank(ascending=False, method="first").astype(int)
    demo["rank_base"] = demo.groupby("session_id")["score_base"].rank(ascending=False, method="first").astype(int)
    demo = pd.concat([demo, vdf[FEATURES].reset_index(drop=True), cdf.reset_index(drop=True)], axis=1)
    demo.to_pickle(OUT / "demo_rows.pkl")

    diff = (ours_ps - base_ps.reindex(ours_ps.index)).sort_values()
    worst = []
    for sid in diff.index[:10]:
        g = demo[demo["session_id"] == sid]
        top1 = g.loc[g["rank_ours"] == 1].iloc[0]
        cs = top1[["c_" + f for f in FEATURES]]
        feat_name = cs.idxmax()[2:]
        worst.append({"session_id": sid, "ndcg_ours": float(ours_ps[sid]), "ndcg_base": float(base_ps[sid]),
                      "note": f"top-ranked {top1['game_id']} (rel {int(top1['rel'])}); main contributing "
                              f"feature {feat_name} ({cs.max():+.2f}, value {top1[feat_name]:.3f})"})

    # ---- round 2: reporting-only analyses ------------------------------------------------------
    stage("10. round 2 analyses (falsification, leave-one-group-out, skill, zero-history rows)")
    # 1. user swap
    users = np.array([t.user_id for t in val_t])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(val_t))
    for i in range(len(perm)):
        if users[perm[i]] == users[i]:
            for j in rng.permutation(len(perm)):
                if users[perm[j]] != users[i] and users[perm[i]] != users[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                    break
    assert (users[perm] != users).all()
    swapped = [t._replace(user_id=users[perm[i]]) for i, t in enumerate(val_t)]
    sfeat = featurize(history_for("val"), swapped, games, friends)
    sfeat["score"] = model.score(sfeat[FEATURES].to_numpy())
    sd = ndcg5_per_session(sfeat[["session_id", "game_id", "score"]], labels_val)
    sb = paired_bootstrap(sd, ours_ps, n=2000, seed=0)
    falsification = {"ndcg5": float(sd.mean()), "ours": float(ours_ps.mean()),
                     "diff_vs_ours": sb["mean_diff"], "ci_low": sb["ci_low"], "ci_high": sb["ci_high"],
                     "reference_quality_plus_prior": ablation[1]["val_ndcg5"],
                     "note": "user_id of each val session replaced by another session's user (permutation, "
                             "seed 0, no fixed points); prefix and slate unchanged"}
    stage(f"   user swap NDCG@5 {falsification['ndcg5']:.4f} (ours {falsification['ours']:.4f}, "
          f"quality+prior {falsification['reference_quality_plus_prior']:.4f})")

    # 2. leave one group out on the holdout
    nd_full = ndcg5_per_session(F_ho[["session_id", "game_id"]].assign(score=m_tr.score(Xho)), lab_ho)
    logo = {"full": float(nd_full.mean()), "n_sessions": int(len(nd_full)), "drop": {}}
    for grp, fs in FEATURE_GROUPS.items():
        keep = [f for f in FEATURES if f not in fs]
        m = fit_model(variant, l2, F_tr[keep].to_numpy(), ytr, str_)
        nd = ndcg5_per_session(F_ho[["session_id", "game_id"]].assign(score=m.score(F_ho[keep].to_numpy())),
                               lab_ho)
        b = paired_bootstrap(nd_full, nd, n=2000, seed=0)
        logo["drop"][grp] = {"ndcg5": float(nd.mean()), "full_minus_drop": b["mean_diff"],
                             "ci_low": b["ci_low"], "ci_high": b["ci_high"]}
        stage(f"   drop {grp:9s} holdout {nd.mean():.4f}  full-drop {b['mean_diff']:+.4f} "
              f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")

    # 3. skill analysis
    from atoms.model import Standardizer, fit_logistic
    raw = train[["session_id", "game_id", "play_now", "completed"]].drop_duplicates(["session_id", "game_id"])

    def tapped_rows(F):
        m = F.merge(raw, on=["session_id", "game_id"], how="left", validate="many_to_one")
        m = m[m["play_now"].fillna(0) >= 1]
        return m, (m["completed"].fillna(0) > 0).astype(float).to_numpy()

    Ttr, ytr_c = tapped_rows(F_tr)
    Tho, yho_c = tapped_rows(F_ho)
    aucs = {}
    for name, cs in (("af_fam_r2", ["af_fam_r2"]), ("af_fam_r2+sk_p", ["af_fam_r2", "sk_p"])):
        st = Standardizer().fit(Ttr[cs].to_numpy())
        w = fit_logistic(st.transform(Ttr[cs].to_numpy()), ytr_c, 1.0)
        z = st.transform(Tho[cs].to_numpy()) @ w[:-1] + w[-1]
        aucs[name] = auc(yho_c, z)
    # decomposition of the completion signal (same fit/score split as above)
    dec_sets = {"a_af_fam_r2": ["af_fam_r2"], "b_af_fam_r2+sk_p": ["af_fam_r2", "sk_p"],
                "c_af_fam_r2+q_r2": ["af_fam_r2", "q_r2"], "d_af_fam_r2+q_r2+sk_p": ["af_fam_r2", "q_r2", "sk_p"]}
    dec_scores, dec_auc, skp_w = {}, {}, None
    for name, cs in dec_sets.items():
        st = Standardizer().fit(Ttr[cs].to_numpy())
        w = fit_logistic(st.transform(Ttr[cs].to_numpy()), ytr_c, 1.0)
        dec_scores[name] = st.transform(Tho[cs].to_numpy()) @ w[:-1] + w[-1]
        dec_auc[name] = auc(yho_c, dec_scores[name])
        if name.startswith("d_"):
            skp_w = float(w[cs.index("sk_p")])
    rng_d = np.random.default_rng(0)
    n_h = len(yho_c)
    dboot = {k: [] for k in ("b_minus_a", "d_minus_c")}
    for _ in range(2000):
        ix = rng_d.integers(0, n_h, n_h)
        yb = yho_c[ix]
        if yb.sum() == 0 or yb.sum() == n_h:
            continue
        a1, a2, a3, a4 = (auc(yb, dec_scores[k][ix]) for k in dec_sets)
        dboot["b_minus_a"].append(a2 - a1)
        dboot["d_minus_c"].append(a4 - a3)
    dec_ci = {k: {"diff": float(dec_auc[kb] - dec_auc[ka]),
                  "ci_low": float(np.percentile(dboot[k], 2.5)), "ci_high": float(np.percentile(dboot[k], 97.5)),
                  "resamples": len(dboot[k])}
              for k, ka, kb in (("b_minus_a", "a_af_fam_r2", "b_af_fam_r2+sk_p"),
                                ("d_minus_c", "c_af_fam_r2+q_r2", "d_af_fam_r2+q_r2+sk_p"))}
    completion_decomp = {"auc": dec_auc, "paired_bootstrap_95": dec_ci,
                         "sk_p_weight_model_d_standardised": skp_w,
                         "sk_p_weight_sign_model_d": "positive" if skp_w > 0 else "negative",
                         "n_tapped_holdout_rows": int(n_h),
                         "note": "logistic fit (l2 1) on days 7-19 tapped rows, scored on days 20-23 tapped rows; "
                                 "bootstrap resamples rows, 2000 draws, seed 0"}
    stage(f"   completion AUC decomposition {dec_auc}; CIs {dec_ci}; sk_p weight {skp_w:+.3f}")
    low = F_ho.groupby("session_id")["sk_lowskill_hard"].max() > 0
    low_ids = nd_full.index.intersection(low[low].index)
    keep = [f for f in FEATURES if f not in FEATURE_GROUPS["skill"]]
    m_ns = fit_model(variant, l2, F_tr[keep].to_numpy(), ytr, str_)
    nd_ns = ndcg5_per_session(F_ho[["session_id", "game_id"]].assign(score=m_ns.score(F_ho[keep].to_numpy())),
                              lab_ho)
    bl = paired_bootstrap(nd_full.loc[low_ids], nd_ns.loc[low_ids], n=2000, seed=0)
    skill_analysis = {
        "completion_auc_holdout": aucs, "tapped_rows_train": int(len(Ttr)), "tapped_rows_holdout": int(len(Tho)),
        "low_skill_hard_sessions": {"n": int(len(low_ids)), "full": float(nd_full.loc[low_ids].mean()),
                                    "no_skill": float(nd_ns.loc[low_ids].mean()),
                                    "diff": bl["mean_diff"], "ci_low": bl["ci_low"], "ci_high": bl["ci_high"]},
        "completion_auc_decomposition": completion_decomp,
        "all_holdout_no_skill": float(nd_ns.mean()),
        "note": "low-skill = Elo skill < -0.5 with a difficulty >= 4 game in the slate (sk_lowskill_hard)"}
    stage(f"   skill: completion AUC {aucs}; low-skill sessions n={len(low_ids)} "
          f"full {skill_analysis['low_skill_hard_sessions']['full']:.4f} vs no-skill "
          f"{skill_analysis['low_skill_hard_sessions']['no_skill']:.4f}")

    # 5. zero-history games, row level
    dz = demo.copy()
    dz["zero"] = ~dz["game_id"].isin(g_imps.index)
    n_sl = dz.groupby("session_id")["game_id"].transform("size")
    dz["p_ours"] = (n_sl - dz["rank_ours"]) / (n_sl - 1).clip(lower=1)
    dz["p_base"] = (n_sl - dz["rank_base"]) / (n_sl - 1).clip(lower=1)
    zero_rows = split_pct(dz[dz["zero"]], ["p_ours", "p_base"])
    zero_rows["rank_pct_note"] = ("share of the slate ranked below the game (1 = top); val slate rows of "
                                  "games with 0 train impressions")
    stage(f"   zero-history game rows: {zero_rows}")

    # 7. first-order gain attributions (pointwise only)
    if variant == "pointwise":
        Z = model.std_.transform(vdf[FEATURES].to_numpy())
        gattr = np.zeros_like(Z)
        for wt, h in zip((1.0, 2.0, 4.0), model.heads_):
            p = 1.0 / (1.0 + np.exp(-np.clip(Z @ h[:-1] + h[-1], -30, 30)))
            gattr += wt * (p * (1 - p))[:, None] * h[:-1][None, :] * Z
        gdf = pd.DataFrame(gattr, columns=["g_" + f for f in FEATURES])
        demo = pd.concat([demo, gdf], axis=1)
        demo.to_pickle(OUT / "demo_rows.pkl")

    coefs = {"variant": variant, "l2": l2, "features": FEATURES,
             "standardizer_mean": dict(zip(FEATURES, model.std_.mean_.tolist())),
             "standardizer_scale": dict(zip(FEATURES, model.std_.scale_.tolist())),
             "coefs": {k: dict(zip(FEATURES, np.asarray(v).tolist())) for k, v in model.coefs().items()}}
    (OUT / "coefs.json").write_text(json.dumps(coefs, indent=2))

    metrics = {"holdout": holdout, "chosen": {"variant": variant, "l2": l2},
               "val": ours_official, "val_baseline": base_official, "bootstrap": boot,
               "slices": slices, "pseudo_cold": pseudo_cold,
               "latency_ms": lat, "worst_sessions": worst,
               "new_game_share": {"ours": ours_official.get("new_game_share@5"),
                                  "baseline": base_official.get("new_game_share@5")},
               "falsification_user_swap": falsification,
               "holdout_leave_one_group_out": logo, "skill_analysis": skill_analysis,
               "zero_history_game_rows": zero_rows,
               "ablation": ablation, "coefs": coefs["coefs"], "test": test_summary}
    metrics["runtime_seconds"] = None
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    metrics["runtime_seconds"] = round(time.perf_counter() - T0, 1)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    stage(f"done. val NDCG@5 {ours_official['ndcg@5']} vs baseline {base_official['ndcg@5']}; "
          f"CI low {boot['ci_low']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
