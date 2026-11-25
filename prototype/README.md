# Game-slate ranker prototype

Ranks the games of a session slate from the user's history and the session's first three
impressions. Python 3 with numpy and pandas only.

## Run

    cd prototype
    python run.py                                   # about 30 s end to end, prints stage timings
    python -m unittest discover -s tests -v
    python ../data/evaluate.py --split val --rankings out/rankings_val.csv

Outputs are `out/rankings_val.csv`, `out/metrics.json`, `out/ablation.csv`, `out/coefs.json`,
`out/demo_rows.pkl` (val features, contributions, labels, baseline scores), and
`../submission/rankings_test.csv`.

## Headline numbers (from `out/metrics.json`)

| | NDCG@5 (val, 604 scored sessions) |
|---|---|
| Baseline (`data/baseline_val_rankings.csv`) | 0.4032 |
| This model (pointwise expected gain, l2 = 0.1) | 0.5209 |
| Paired bootstrap difference (2,000 resamples, seed 0) | +0.1178, 95% CI [+0.0921, +0.1446] |

Scoring latency per validation slate (features plus score, warm state, 814 slates): p50 0.85 ms, p99 2.25 ms in the latest run. Scoring one validation target against all 200 games of the catalogue takes p50 2.28 ms and p99 4.71 ms over the same 814 targets, which is about 0.011 ms per game in this implementation. The full `python run.py` takes about 35 s on this machine (`runtime_seconds` in `out/metrics.json`). Timings vary a little from run to run.

The holdout (pseudo-sessions on days 20 to 23, model fitted on days 7 to 19) chose the variant:

| variant | l2 0.01 | l2 0.1 | l2 1 | l2 10 |
|---|---|---|---|---|
| pointwise | 0.5252 | **0.5258** | 0.5255 | 0.5248 |
| pairwise | 0.5208 | 0.5208 | 0.5208 | 0.5208 |

The 0.01 column was added because both variants peaked at the 0.1 edge of the grid.

The pairwise variant is effectively unregularised. Its NDCG is identical at every l2 because the pair
weights (gain differences) are unnormalised, so the penalty is negligible next to the weighted loss.
The 0.005 gap between the variants is within noise.

## Leakage rules

- Val history is the deduplicated `train_impressions.csv` plus `val_prefix.csv` rows of sessions with
  ts < the target ts. `val_events.csv` is never used for val.
- Test history is the deduplicated train plus `val_events.csv` plus `test_prefix.csv` rows of sessions
  with ts < the target ts.
- Features for a target use only events with ts strictly less than the session ts, plus that session's
  own prefix. All impressions of a session share one ts.
- Train is deduplicated with exact-row `drop_duplicates()` before use.
- In history, a tap with blank `play_seconds` counts as a play (r=2 if completed, else r=1); replay,
  share or challenge lift it to r=3. Evaluation labels use `common.relevance`. Blank or 0 `watch_ms`
  is missing, not zero.
- Training rows are pseudo-sessions: train sessions with at least 7 impressions, first three as the
  prefix, the rest as the slate. There is no position feature.

## Features

| group | feature | meaning |
|---|---|---|
| quality | q_r1, q_r2, q_r3 | game's rate of r>=1, r>=2, r=3, shrunk toward its attribute prior |
| | q_watch | shrunk mean watch time |
| | q_logn | log of the game's impression count |
| prior | pr_r1, pr_r2 | family+mechanic prior rates |
| | pr_onestar | editorial one-star flag |
| | pr_age | log days since the game was added |
| affinity | af_fam_r1/r2, af_mech_r1/r2 | user's shrunk rate on the game's family / mechanic minus the population rate |
| | af_depth_x_fam | affinity times log of the user's session count |
| repeat | rp_seen, rp_best | times the user saw the game, best outcome on it |
| skill | sk_p | Elo success probability for this user and game |
| | sk_gap | distance of that probability from 0.75 |
| | sk_lowskill_hard | low-skill user and a difficulty >= 4 game |
| friends | fr_played, fr_finished | friends who played / finished the game |
| prefix | px_fam_play/finish/quit/skip, px_mech_play/quit | what happened in the three prefix impressions on the same family / mechanic |

## Ablation (groups added cumulatively, chosen variant refitted each time, val NDCG@5)

Reporting only; nothing was selected from it.

| + group | features | val NDCG@5 |
|---|---|---|
| quality | 5 | 0.4106 |
| prior | 9 | 0.4137 |
| affinity | 14 | 0.5296 |
| repeat | 16 | 0.5255 |
| skill | 19 | 0.5250 |
| friends | 21 | 0.5229 |
| prefix | 27 | 0.5209 |

Nearly all of the gain comes from the per-user family and mechanic affinity. Groups added after
affinity do not improve val: each is within about 0.004 of the previous row, and the total drifts down by
0.009.

## Slices (val, ours vs baseline)

| slice | sessions | ours | baseline | diff, 95% CI (paired bootstrap) |
|---|---|---|---|---|
| cold users (<= 2 sessions in train) | 110 | 0.4199 | 0.3800 | +0.0400 [-0.0084, +0.0893] |
| slate has a zero-history game | 152 | 0.5187 | 0.3824 | +0.1363 [+0.0791, +0.1940] |
| prefix had a play | 240 | 0.5058 | 0.3676 | +0.1382 [+0.0988, +0.1781] |
| prefix had no play | 364 | 0.5309 | 0.4266 | +0.1043 [+0.0687, +0.1403] |

New-game share of top-5 slots: 0.0215 (baseline 0.0197).

## What does not work

- Cold users: the lift over baseline is +0.04 against +0.12 overall, since the dominant signal
  (user affinity) is absent.
- Pseudo-cold check (holdout, history removed for 30 games, 15% of the catalogue, 575 sessions): attribute prior
  0.4986 vs prior columns at the global mean 0.4975, against 0.5112 with full history. The prior
  columns add almost nothing here, because quality features already fall back to the prior internally.
- Worst sessions: in 7 of the 10 largest losses against the baseline the top-ranked game was driven
  mainly by a family or mechanic affinity feature (`af_mech_r2`, `af_fam_r2`) and had no relevant
  outcome (2 by `q_r2`, 1 by `fr_played`). Four of them score 0.0 where the baseline scored 0.63 to 1.0.
- Distribution skew: during val days the history contains only the 3-row prefixes of val sessions,
  while training pseudo-sessions see earlier sessions in full. History-based features for val users are
  therefore thinner than the ones the model was trained on.

## Further checks (reporting only; the model and the test rankings are unchanged)

- User-swap falsification (val): replacing each session's user with another session's user (seed 0)
  gives NDCG@5 0.3721 (ours 0.5209, quality+prior only 0.4137; difference -0.1488, CI [-0.1780, -0.1212]).
  The score does not stay near 0.52, so the lift depends on the session's own user. It falls below the
  quality+prior level because the model weights affinity heavily and a wrong user's affinity misleads it.
- Leave-one-group-out on the holdout (full model 0.5258, 684 scored sessions): dropping affinity gives 0.4372 (full minus drop
  +0.0885, CI [+0.0666, +0.1094]). Every other group is within noise: quality +0.0085, prior +0.0020,
  repeat +0.0004, skill -0.0015, friends +0.0040, prefix -0.0006 (all CIs include 0).
- Skill: completion AUC on tapped holdout rows (fit on days 7 to 19, scored on days 20 to 23) is 0.665 with af_fam_r2 alone and 0.684 with sk_p added, a gain of +0.020 (paired bootstrap 95% CI [+0.004, +0.034]). The game's own shrunk finish rate q_r2 added to af_fam_r2 gives 0.685, which is as high as the sk_p version. Adding sk_p on top of af_fam_r2 and q_r2 gives 0.692, a gain of +0.007 (CI [+0.000, +0.015]). The sk_p weight is positive in that model, so the direction holds, but most of what the skill rating predicts about finishing is game difficulty, which the game's own finish rate already captures. The player-specific part is small. In the three ranker heads sk_p carries a negative weight.
  On 134 holdout sessions of low-skill users with a difficulty 4 or 5 game in the slate, full vs no-skill
  model is 0.5191 vs 0.5162 (+0.0029, CI [-0.0124, +0.0177]): no detectable ranking benefit.
- Pseudo-cold, global-rate variant: with masked games' quality and prior rates set to the global rate (no
  attribute prior; the global rates come from all of train, a constant, so this is not strictly
  point-in-time), NDCG on the 575 affected sessions is 0.4990, against 0.4986 with the attribute prior.
  Row level (share of the slate ranked below the game, 1 = top), masked games with rel>=1 (n=208) /
  rel=0 (n=1477): attribute prior 0.557 / 0.400, global rate 0.529 / 0.382. The prior ranks played
  games a little higher, and unplayed ones too, so the NDCG effect is nil.
- Zero-history games in val slates (rows, same percentile): rel>=1 (n=25) ours 0.619 vs baseline 0.489;
  rel=0 (n=185) ours 0.488 vs baseline 0.545.
- `out/demo_rows.pkl` also holds `g_<feature>` columns: first-order contributions to expected gain,
  the sum over the three heads of weight (1, 2, 4) x p(1-p) x coefficient x standardised value.
