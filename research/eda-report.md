# Exploratory analysis of the feed dataset (Part 5)

Scope: read-only analysis of `data/`. Scripts are in `research/eda/` (`e1.py` data quality, `e2.py`/`e2b.py` leakage, `e3.py` distributions, `e4.py` logging policy, `e5.py` signals, `e6.py` watch/quits/time, `e7.py` extra checks, `e8.py` test-side counts, `feats.py` point-in-time feature builder). Outputs of the longer runs: `e5_out.txt`, `e7_out.txt`.

Conventions. "Relevance" = `common.relevance` (blank `play_seconds` counts as 0). `P(r>=1)` = share with relevance 1+; `P(r>=2)` = share with 2+. "Train signal tables" use the eval-like subset of train: sessions with 7+ impressions, position > 3, deduplicated, tapped-with-blank-seconds rows removed (49,748 rows). "Val" = the 9,576 val slate rows with `val_labels`, with features built from **train only** (strictly earlier `ts`) plus the session's own prefix. All "history" features are point-in-time (a session's rows share one `ts`, so only earlier sessions are used).

"Within-session concordance" (shortened to "conc.") = among pairs of slate games in the same session with different relevance, the share where the more relevant one has the higher signal (0.5 = no information). It is the quantity NDCG actually rewards, because ranking only happens inside a session.

Reference: on val, 814 sessions, 604 have at least one relevant slate game and are scored. Random NDCG@5 = 0.310 (my replication), popularity = 0.401 (baseline reports 0.4032).

---

## 1. Data quality

| Check | train (68,137 rows) | val_events (12,778) |
|---|---|---|
| Exact duplicate rows | **685 (1.0%)**, all appended at the end of the file (rows 67452+), 630 sessions, 516 of them at position > 3 | 0 |
| Dups by (session_id, game_id) / (session_id, position) | 685 / 685 (same rows as exact dups, no conflicting rows) | 0 / 0 |
| Blank `watch_ms` | 3,358 (4.93%), uniform across platform, position, day, and play_now (4.9% vs 5.2%) | 664 (5.20%) |
| `watch_ms` == 0 exactly | 2,628 (3.9%); play_now rate there is 0.150 = overall average, so 0 is a logging gap, not "did not watch" | 522 |
| Blank `play_seconds` | 303 (0.44%), **all on play_now=1 rows** (2.9% of taps); 166 of them have completed=1, 71 have replays>0 | 56 |
| play_seconds on non-taps | 0 rows with play_now=0 and play_seconds>0; 0 with completed=1 and play_now=0 | same |
| Impossible combos | replays/shared/challenged>0 with completed=0: 66 rows (45 shared, 20 challenged, 0 replays); score_percentile without completed: 0; completed=1 with ps<20: 0 | 8 |
| completed but no `score_percentile` | 1,672: all have_leaderboard=0 games (1,653) plus the 3% blank rows; every completed leaderboard game with known seconds has a score | 315 |
| Impossible values | none: watch_ms in [0,12000], play_seconds in [0,280], no negatives, no unknown users/games, no impressions before signup or before `added_day`, `ts`//86400 == `day` always | none |

Tapped games (10,401 after dedup):
- play_seconds distribution is bimodal: 1,205 (11.6%) are **<= 10 s** (6 below 2.5 s, 472 in 2.5-5, 738 in 5-10, only 16 in 10-20); the other 8,971 are >= 20 s (median ~85 s, max 280 s). No outliers: nothing above 280 s, so no clipping is needed. Play time is close to the game's `typical_session_sec` (mean ratio 0.99).
- Accidental/quick quits (tap then <20 s) = 11.7% of taps (1,220). They are not uniform noise: quit rate rises with difficulty (7.8% at d=1, 6.7% d=2, 10.0% d=3, 16.9% d=4, 22.9% d=5), is higher after very short previews (24% when watch<=1.5 s, 9-10% when watch>6 s), and is lowest in `words` (8.2%). It does not depend on position or `typical_session_sec`. So quits carry real signal (hard game for this player) in addition to mis-taps.
- **Blank play_seconds are not missing at random with respect to outcome.** In val, labels resolve the blanks: of the 44 blank-tapped slate rows, 44/44 are relevance >= 1 (12 rel1, 17 rel2, 15 rel3), while `common.relevance` on `val_events` scores them 0 (the label file was computed from unblanked data; `val_events` still carries the blanks). So in train/history, blank-with-tap should be treated as "unknown, probably positive" (completed=1 in 55% of them), not 0. Impact is small (0.44% of rows) but it biases history features and training labels toward 0.
- The 685 train dups are verbatim copies and change nothing in the label mean (0.2617 with, 0.2616 without), but they would double-count in user/game counts and could leak a row into both sides of a random split.
- Blank watch_ms and ps in the prefix files: val_prefix has 128 blank watch and 14 blank taps' seconds out of 2,442 rows; test_prefix 135 and 13 out of 2,511. The prefix blanks do **not** coincide with the same row's blanks in `val_events`: 230 of 2,442 prefix watch values differ from `val_events` (one side blank, the other populated), and 21 play_seconds differ. Never assume prefix and events are the same logged row.

## 2. Leakage check

**Val.**
- `val_events.csv` has 965 sessions; **all 814 evaluated sessions (prefix/slates/labels) are inside it**, with all of their rows: 12,018 rows = 2,442 prefix rows + 9,576 slate rows. 151 other val_events sessions are short (3-6 impressions, never evaluated).
- Every `(session_id, game_id)` in `val_slates` matches a `val_events` row (9,576 of 9,576) with full outcome columns. **The val labels are fully derivable from val_events**: relevance recomputed with `common.relevance` agrees on 9,532 of 9,576 rows; the 44 disagreements are exactly the blank-play_seconds rows (events say 0, labels say 1-3), i.e. val_events leaks even the harder cases up to the blanks.
- So for **val**, `val_events.csv` must not be used at all as history, except (at most) the 151 non-evaluated short sessions, and then only strictly earlier than the session's `ts`. Legitimate val history = `train_impressions.csv` (days 0-23, deduplicated) + the session's own 3 prefix rows (`val_prefix`) for in-session features. Using other evaluated sessions' outcomes from `val_events` (even earlier sessions, or other users: e.g. friends' outcomes on day 25 for a day 26 session) is not what the README describes for val and is not what happens at test time unless the earlier day is in val; I recommend not doing it.
- Session ids: train s000001-s005246, val s005247-s006211, test s006212-s007206, contiguous and non-overlapping; val sessions are not in train; `val_slates.ts` equals the prefix `ts`.

**Test.**
- None of the 837 test sessions appear in `val_events` or train (0 overlaps). No test slate outcomes exist in any provided file.
- Legitimate test history = train (days 0-23, deduplicated) + **all** of `val_events` (days 24-26, all precede the test days 27-29; README allows it) + the test session's own prefix (`test_prefix`). Val **labels** add nothing beyond `val_events` (they are the same outcomes). `val_events` includes blank play_seconds that val_labels resolve; if you want the resolved version of those 44 rows, you may use `val_labels` for them (it is the same val-period information), but it is a minor point.
- Users: 367 val users and 397 test users; 576 of 650 users have train history; 74 have none.

## 3. Distributions

**Relevance** (dedup train, all positions / eval-like subset / val labels):

| rel | train all (67,452) | train eval-like (n>=7, pos>3) | val labels (9,576) | val prefix (2,442) | test prefix (2,511) |
|---|---|---|---|---|---|
| 0 | 86.8% (58,570) | 87.2% | 87.0% (8,333) | 85.1% | 86.1% |
| 1 | 4.7% (3,153) | 4.6% | 4.7% (451) | 4.1% | 4.8% |
| 2 | 4.0% (2,693) | 4.0% | 3.8% (368) | 4.5% | 3.7% |
| 3 | 4.5% (3,036) | 4.2% | 4.4% (424) | 6.3% | 5.4% |

- Train and val label distributions match (P(r>=1) = 13.2% vs 13.0%). Positives are rare: a slate of ~12 games holds ~1.5 relevant games on average. In val, 210 of 814 sessions (26%) have no relevant slate game (not scored); max relevance per session: 0:210, 1:136, 2:169, 3:299.
- Rel 3 is as common as rel 2 and rel 1; rel 3 is nearly always a subset of completers (replays are logged mostly with completed=1).
- Prefix rel 3 rate is higher than the slate rate (6.3% val prefix vs 4.4% slate): see section 4 (position effect).

**Sessions.** Train: 5,246 sessions (4,363 with 7+ impressions = 83%); mean session length 13.0, median 11, max 48. Eval session lengths (val): slate size mean 11.8, median 10, min 4, max 42 (test: mean 11.6, median 9, max 42). Sessions per day grow from 42 (day 0) to ~320 (day 23): the feed is growing; val/test have ~270-296 sessions/day.

**Users.** 650 users, 342 android / 308 ios. Sessions in train history per user (576 users with any): mean 9.1, median 7, p10=2, p90=19, max 31; 74 users have 0 train sessions; 78 have 1-2. signup_day>=20: 123 users, >=24: 66. Friends: 2,185 pairs; median 6 friends per user, 5 users with none.

**Cold users** (<=2 sessions in the history window, the evaluator's definition):

| split | evaluated sessions | of them cold sessions | cold users | zero-history sessions |
|---|---|---|---|---|
| val (history = train) | 814 | 141 (78 zero-history + 63 with 1-2 sessions) | 67 | 78 |
| test (history = train + val_events) | 837 | 105 | 64 | 47 |

(The evaluator's `cold_user_sessions` counts only sessions that have a relevant slate item: 110 in val.) Late signups are common among evaluated sessions: 152 val and 181 test sessions belong to users who signed up on or after day 20 (73 / 83 on or after day 24).

**Games.** 200 games: 170 added on day 0, 18 added days 8-23, 12 added days 25-29 (g033 d26, g054/g064/g069 d25, g096/g099/g134/g135 d27, g196 d28, g004/g007/g137 d29). `editorial_stars` is null for 38 games (32 of them day-0 games, 6 later). Impressions per game in train: mean 337, median 331, min 0 (12 games), max 780.
- Games with < 30 impressions in the history window: **val 12 in train (all with 0 impressions)**; the 4 that show up in val slates (g033, g054, g064, g069, all added day 25-26) account for **210 slate rows (2.2%)**. They are rows with zero history, not just thin history.
- **Test** (train + val_events): 8 games with zero history (g004, g007, g096, g099, g134, g135, g137, g196: added days 27-29) = **457 slate rows (4.7%)**; the four earlier new games have 45-98 impressions from val_events and are no longer "new" by the evaluator's <30 rule. Counting all games with `added_day >= 25`, test slates have 686 rows (7.1%) vs 210 (2.2%) in val: **the cold-game problem is twice as large at test as at val**, and val under-tests it.
- Rows from new games in val: P(r>=1) = 11.9%, mean rel 0.233 (vs 13.0% / 0.257 average): close to the mean, i.e. new games are not inherently worse, just unknown.

## 4. Logging policy / selection bias

**Position.** (train, all rows) Rate of relevance falls with feed position and is not flat: mean relevance 0.296 / 0.292 / 0.291 at positions 1/2/3, 0.279 at 4, 0.247 at 5, then ~0.250 for 6-20, 0.239 beyond 20. P(r>=1) 14.5% (pos 1-3) vs 12.7% (pos 5+), P(r>=2) 9.6% vs 8.0%. Session length itself has no effect. So the prefix (positions 1-3) is rated about 17% better than the slate, which is where the "prefix rel 3 = 6.3% vs slate 4.4%" comes from. It is not a within-session fatigue effect from earlier plays: mean rel after 0 / 1 / 2 / 3 earlier plays in the session is 0.271 / 0.259 / 0.256 / 0.251 (a slight drop, then flat). Possible readings: position bias in the old feed's UI, or the old feed putting better items first; the data cannot separate these. **Slate order is shuffled, so position cannot be used at ranking time, but training rows at positions 1-3 are drawn from a different (better) distribution than slate rows**, so use only position > 3 rows (or position as a nuisance feature) if training on train.
- Mean position of a game does not track its popularity (Spearman between game mean position and play rate: -0.015), so the old feed did not systematically place popular games higher.

**Exposure follows popularity and editorial stars.** Spearman(train exposure count, game P(r>=1)) = +0.17; early-half game success predicts second-half exposure (+0.17). Exposure ratio (share of impressions / share of games) by `editorial_stars`: 1★ 0.47, 2★ 0.59, 3★ 1.08, 4★ 1.38, 5★ 1.58, null 0.95. Relevance by stars: P(r>=2) 4.3% / 6.8% / 8.0% / 10.2% / 9.8% (null 8.4%). The old feed shows highly-starred games more, and they are also a bit better, so exposure and quality are confounded. A game's historical rate is a biased estimate of its quality for games that were hardly shown.

**Personalisation by the old feed.** Weak. Across 175 users with 150+ impressions, the correlation between a user's exposure share in a mechanic (deviation from the population share) and their play rate in that mechanic is +0.11 (family: +0.15). So the old feed leans a little toward what users like, which inflates the apparent value of "user's history in X" slightly through exposure and not just taste. Slate game frequency vs train rate across games: Spearman +0.14.
- Repeats: 34.8% of (user, game) pairs seen again in train (mean 1.54 exposures per pair). A game is shown again with probability 34.5% after a rel-0/1 exposure, 37.6% after rel 2, 39.2% after rel 3: the old feed barely favours repeating games the user loved. In val, 573 slate rows (6%) are games the user previously completed or replayed.
- Other exposure ratios: `fast`/`calm`/`mixed` pace ~1.0, `3d` 0.90; `physics-launch` 0.75 and `spatial-puzzle` 0.79 are under-exposed (relevance of physics-launch is average), `match-three` 1.28 and `aim-shoot`/`merge` 1.12-1.13 over-exposed. Difficulty 5 games are under-exposed (0.85) and genuinely worse.

**New games over time.** Per game per day, new games (added after day 0) get ~20-35 impressions/day, vs ~16-20 per day for old games in the second half of train: new games are over-exposed by the old feed (a boost). Their rates: P(r>=1) 11.2% on the day they appear, 12.4% in days 1-5, 9.5-9.7% in days 5-10, vs 13.0% for old games; mean rel 0.215 (age 0) vs 0.257 (age > 10). So new games are slightly worse on average and heterogeneous (per-game P(r>=1) ranges 6.4% to 20.9% among the 18 mid-period additions). The share of impressions that are new games is the same in the prefix and the rest of the session (4.9% vs 5.3%): the old feed did not put them first.
- Game age (`day - added_day`): ~0.5 pp lower P(r>=2) in the first two days; tiny effect (conc. 0.504-0.506).

**Time drift.** Session volumes grow, but relevance rates by day are flat: P(r>=1) fluctuates between 11.2% and 14.1% with no trend; val days 24/25/26 = 13.4 / 12.5 / 13.1%. Game popularity is stable over time: Spearman of game P(r>=1) between days 0-11 and 12-23 = 0.63 (175 games), and between train and val labels = 0.52 (183 games with n>=20 in val; in-sample noise on small n limits this).

## 5. Signals vs relevance

All numbers: train (eval-like subset) / val. Counts in parentheses where useful. For the compact "strength" view see the concordance table at the end of this section.

### 5.1 Item attributes
| Signal | Finding |
|---|---|
| **Game popularity** (leave-one-out train P(r>=1), smoothed) | Quintile 1 -> 5 mean rel 0.167 / 0.224 / 0.237 / 0.297 / 0.345 (train), 0.175 / 0.215 / 0.241 / 0.288 / 0.374 (val); P(r>=1) 9.0% -> 17.5% (val 9.1% -> 19.1%). Conc. 0.575 / 0.574. The bulk of the baseline's performance. |
| **Game mean preview watch** (mean of `watch_ms`, excl. blank and zero) | Spearman with game P(r>=1) = 0.87 across 200 games; conc. 0.583 / 0.576 vs 0.575/0.574 for popularity; hand-ranking by it alone on val gives NDCG@5 0.416 vs 0.401 for popularity. A smoother, higher-volume game-quality estimate (every impression has a watch value, only 15% have a play outcome), useful for games with little history. |
| `difficulty` | mean rel by d=1..5: 0.263 / 0.268 / 0.273 / 0.234 / 0.169 (train), 0.275 / 0.264 / 0.275 / 0.250 / 0.157 (val); d=5 is clearly worse (P(r>=2) 4.7%); d=1-3 flat. Conc. 0.515 (inverted sign) train, 0.519 val. Mostly via quits (quit|play 7% for d<=2, 10% d=3, 17% d=4, 24% d=5) and lower completion (69% d=1 -> 46% d=5). |
| `typical_session_sec` | Flat/noisy: mean rel by bucket (<=60, 60-80, 80-100, 100-120, >120) 0.263 / 0.240 / 0.266 / 0.254 / 0.246 train; 0.298 / 0.237 / 0.269 / 0.248 / 0.243 val; no monotone pattern (conc. 0.502 / 0.517). Not a signal on its own. |
| `editorial_stars` | 1★ is bad (mean rel 0.175 / 0.161, n = 818 / 168), 4★ best in train (0.279), 5★ similar to 3-4★ (n=1,487; val 0.281); stars 2-3 equal (0.245). Conc. 0.507 / 0.508: weak, mostly the 1★ tail. Null stars (38 games): rel 0.252, i.e. average. |
| `has_leaderboard` | none: 0.255 vs 0.253 (train), 0.253 vs 0.258 (val). Affects only whether a `score_percentile` exists (which is the skill source, see 5.3). |
| `mechanic` | Strong, 14 levels: build-manage 0.165, card-strategy 0.175, rhythm 0.195 (worst); aim-shoot 0.320, runner-dodge 0.313, merge 0.310 (best) in train; the same low group is low in val (build-manage 0.166, rhythm 0.161, card-strategy 0.203), the top group is less stable (val: physics-launch 0.322, number-logic 0.304, aim-shoot 0.303). |
| `family` | Same story (5 levels): strategy 0.171 / 0.188, music 0.195 / 0.161, words 0.251 / 0.261, puzzle 0.281 / 0.277, action 0.287 / 0.293. Note `music` = {rhythm} and each of the 5 families is a union of mechanics, so family is a coarser version of mechanic (nested). |
| `pace` | fast 0.274, calm 0.258, mixed 0.230 (train); 0.281 / 0.259 / 0.232 (val). Consistent across splits, effect ~ -0.03 on mixed. |
| `theme`, `art_style` | Small and unstable: theme range 0.237-0.270 train, 0.220-0.283 val, with order changes (retro +0.018 train, -0.022 val; nature/space low in both). art_style: 3d 0.269 train but 0.254 val; cartoon 0.252 train but 0.277 val: not replicated. |
| game age (`day - added_day`) | age 0: 0.215 / 0.241 (val n=137), >10 days: 0.257 / 0.260. A weak penalty for brand-new games (mid-period additions are slightly worse and more variable). Conc. 0.504 / 0.506. |

### 5.2 Watching the preview (`watch_ms`) and what it does and doesn't tell you
Train (dedup, all rows):

| watch_ms bucket | n | play_now | quit given play | P(r>=1) | P(r>=2) |
|---|---|---|---|---|---|
| exactly 0 | 2,606 | 0.150 | 0.092 | 0.134 | 0.091 |
| 0-1 s | 800 | 0.029 | 0.304 | 0.020 | 0.001 |
| 1-2 s | 6,125 | 0.038 | 0.146 | 0.030 | 0.004 |
| 2-3 s | 12,239 | 0.059 | 0.161 | 0.048 | 0.013 |
| 3-4 s | 11,955 | 0.100 | 0.149 | 0.083 | 0.036 |
| 4-6 s | 17,125 | 0.182 | 0.110 | 0.156 | 0.103 |
| 6-8 s | 8,876 | 0.285 | 0.102 | 0.248 | 0.190 |
| 8-10 s | 3,733 | 0.362 | 0.104 | 0.316 | 0.257 |
| 10-12 s | 658 | 0.46-0.69 | 0.09-0.11 | 0.41-0.62 | 0.31-0.38 |
| blank | 3,335 | 0.161 | | 0.130 | |

- Long watching is **strongly predictive of tapping Play** (corr 0.26; play_now rises from ~4% at 1-2 s to 36-46% at 8+ s) and of completion (P(r>=2) 0.4% -> 26-31%), and mildly of play quality (corr with relevance among played = 0.22; quit rate drops from 15% to 10%). Within user (watch relative to the user's mean): P(r>=1) from 5.6% (bottom quintile) to 27.3% (top).
- **But `watch_ms` exists for slate games only in history and the 3 prefix rows, never for the slate rows being ranked** (slate files have just user, session, game, ts). Its usable forms are: (a) game-level mean watch (see 5.1), (b) the user's own watch history as a per-user/mechanic/family "interest" signal (user-level mean watch Spearman with play rate 0.60), (c) the prefix watch times in-session. (c) is weak for the slate: mean prefix watch buckets <2 s .. >7 s give val slate P(r>=1) 12.7% / 12.8% / 12.9% / 13.1% / 13.8% (n = 440..2,621), i.e. almost nothing at the session level.
- Trap: watch == 0 (3.9%) and blank (4.9%) are both logging gaps (play_now 0.150 and 0.161 = average), not "no interest". Treat both as missing, not as 0 s.
- A feature built from *post-exposure* watch of the same impression (e.g. when training with the target row's own watch_ms) leaks the outcome in a way unavailable at ranking time.

### 5.3 User taste and skill (point-in-time user history, val shown alongside train)

| Signal | Finding |
|---|---|
| **User's own affinity by mechanic** (user's smoothed past P(r>=1) in this mechanic, minus the population rate of the mechanic, so the global mechanic effect is removed) | Buckets (-1,-.05] / (-.05,-.02] / (-.02,.02] / (.02,.05] / (.05,1]: mean rel train 0.139 / 0.192 / 0.238 / 0.303 / **0.487**; val 0.120 / 0.197 / 0.244 / 0.313 / **0.533** (n = 3,211 / 1,610 / 2,266 / 550 / 1,939). P(r>=2) 4.0% -> 17.0% train, 3.2% -> 18.7% val. Conc. 0.600 / 0.632. Raw past rate in the mechanic (>=5 past impressions): (<=2%) 0.122 / 0.120 to (>25%) 0.603 / 0.591 mean rel, P(r>=1) 7.4% -> 26.4%. Replicates on val. Among games the user has never seen (seen_n=0): conc. 0.564 / 0.585 (val), so it is not just repeat-play of favourites. |
| **Affinity by family** | Same shape, a little cleaner: mean rel 0.120 -> 0.445 train, 0.103 -> 0.509 val across the same 5 buckets; conc. 0.597 / **0.636** (0.568 / 0.600 on unseen games). Family and mechanic are nested, so use one as the main and the other as a smoothing back-off. |
| Affinity by pace | Weaker: conc. 0.548 / 0.568; mean rel 0.160 -> 0.355 train, 0.149 -> 0.369 val. |
| Affinity by theme / art_style | Near noise: conc. 0.511/0.505 (theme) and 0.504/0.506 (art), though raw buckets move by ~0.07 in train (mixture with the true mechanic effect). Not worth modelling. |
| User history volume in the mechanic | the 345 train / 326 val rows where the user has > 30 past impressions in the mechanic show the highest relevance (0.533 / 0.383 mean rel); with 0-3 past impressions the affinity is basically equal to the population rate. Shrinkage matters. |
| User overall rate / sessions / platform / hour | Raising the whole slate, **not the ranking**: user past P(r>=1) buckets give session mean rel 0.207 -> 0.313 (train); the between-session variance is only 9.3% of total relevance variance, and a user-only score gives NDCG@5 0.3096 on val = random. Zero-history users: mean rel 0.247 / 0.281. Platform: P(r>=1) 12.9% vs 12.9% (train), 12.8% vs 13.2% (val). Hour of day / day of week: flat in train (P(r>=1) 12.5-13.2% across hour bins, 12.9-13.3% by weekday); val hour buckets differ (0-4h 0.321, 4-8h 0.200) but that is 3-day, session-level noise (not seen in train), and it is irrelevant for within-session ranking anyway. Interactions hour x family, hour x difficulty, platform x family, platform x difficulty were all within +/-2 pp of the cell averages. |
| **Skill x difficulty** | User skill = mean `score_percentile` of past completed leaderboard games (known for 79% of train, 86% of val rows). Terciles (<53.5 / 53.5-68.5 / >68.5 in train). Train, given a tap: **quit rate** by tercile (d1-2 / d3 / d4-5): low skill 9.2% / 14.2% / 26.3%; mid 5.5% / 8.8% / 17.9%; high 6.9% / 7.3% / 9.7%. **Completion given play >= 20 s** (d1-2 / d3 / d4-5): low 61% / 51% / 36%; mid 69% / 60% / 45%; high 64% / 67% / 60%. Relevance by cell (mean rel): low-skill hard 0.196 vs low-skill easy 0.272; high-skill hard 0.267 vs high-skill easy 0.282 (val: low-skill hard 0.197 vs 0.296 easy; high-skill hard 0.284 vs 0.274 easy). So: **low-skill users quit and fail hard games, high-skill users do not**; the mismatch is the real signal, not difficulty itself. Concordance of "easier is better" within sessions by tercile: low 0.524 / 0.540, mid 0.513 / 0.497, high 0.506 / 0.507 (train/val); 0.518 / 0.532 for users with no skill estimate. Skill alone lifts levels (low 0.242, mid 0.266, high 0.285 mean rel) but that is a user-level shift. Percentile scores themselves vary by difficulty (d1 -> d5 mean score_percentile: q1 47.5 -> 33.7, q4 78.0 -> 74.6). |

### 5.4 Friends
Friend signal = number of the user's friends who played this game with relevance >= 1 (or >= 2) strictly before the target session (any time earlier).
- Train: P(r>=1) 12.2% (0 friends, n=38,663) -> 14.1% (1) -> 16.8% (2) -> 19.1% (3+, n=1,020); using friends who completed (>=2): 12.4% / 15.1% / 16.2% / 20.7%. Mean rel 0.237 -> 0.424 (good) and 0.239 -> 0.478 (completed).
- Val: 12.0% / 13.0% / 17.7% / 17.2% and mean rel 0.225 / 0.280 / 0.367 / 0.382; completed: 12.2% / 13.7% / 19.4% / 17.6% (n=7,125 / 1,638 / 495 / 318). Replicates.
- Not just popularity: friends' games are somewhat more popular (mean pop 0.146 vs 0.129), but the lift holds inside popularity terciles: val (friend completed vs none): low-pop P(r>=1) 12.2% vs 9.7%, mean rel 0.248 vs 0.184; mid-pop 11.7% vs 12.0% (flat in P(r>=1)), mean rel 0.250 vs 0.231; high-pop 19.3% vs 15.7%, mean rel 0.426 vs 0.294. Train terciles: P(r>=2) 6.7% vs 5.3% / 11.7% vs 8.5% / 13.3% vs 9.5%. Real but modest: conc. 0.524 / 0.531 (good), 0.519 / 0.529 (completed).
- Coverage: 35% of val slate rows have a friend who played it, 26% a friend who completed it (train 22% / 16%). The friend signal works for cold users as well as warm ones (val conc. cold 0.541, warm 0.530).

### 5.5 Repeat exposure
- User saw this game before: mean rel 0.243 (never seen, n=32,648 train / 4,937 val), 0.268 / 0.257 (1 prior), 0.271 / 0.270 (2), 0.282 / 0.228 (3), 0.403 / 0.351 (4+, n=657 / 396). Modest and noisy until the user is a repeat visitor.
- **What the user did last time is the signal**, not the count. Best previous relevance on this game: none 0.243 / 0.249; seen but rel 0 (n=13,950 / 3,775): 0.226 / 0.221 (P(r>=1) 11.6% / 11.4%); rel 1: 0.277 / 0.217; **rel 2: 0.532 / 0.478 (P(r>=1) 23% / 22%); rel 3: 0.638 / 0.657 (P(r>=1) 27% / 28%)** (n = 911 + 1,228 train, 249 + 324 val; 573 val slate rows = 6%). Concordance 0.576 / 0.570 — on par with popularity. A game previously completed/replayed comes back ~2.5x more relevant than a rel-0 exposure; a game seen before with no play is slightly worse than never seen (-0.015).
- Recency: seen <= 1 day ago 0.288 (train) / 0.342 (val, n=82), 1-3 days 0.266 / 0.308, 3-7 0.269 / 0.254, >7 days 0.281 / 0.260: weak, slight recency boost in val.

### 5.6 Within-session (prefix) signals
Prefix = positions 1-3 with outcomes.
- Number of good outcomes in prefix 0 / 1 / 2 / 3: slate mean rel 0.255 / 0.257 / 0.274 / 0.277 (val, n = 5,853 / 3,125 / 533 / 65) and 0.249 / 0.260 / 0.264 / 0.376 (train) — a session-level mood effect that does not change within-session order (prefix quits: 0.256 vs 0.275, none).
- **Same family as a prefix game the user enjoyed (rel >= 1)**: slate rows of such games have mean rel 0.398 train / 0.415 val (P(r>=1) 18.7% / 19.9%, n = 5,196 / 953) vs 0.227 / 0.208 for families not present in the prefix (n = 22,977 / 4,590) and 0.246 / 0.274 for families seen but not played. Completed prefix game in the family: 0.442 / 0.433 (n = 3,563 / 697). **Same mechanic** with a good prefix game: 0.548 / 0.452 (n = 1,600 / 290), P(r>=1) 24.3% / 19.7%.
- **Same family after a quit** (tap <20 s): 0.296 / 0.348 mean rel; n = 599 / 89; **same mechanic after a quit: 0.323 / 0.533, n = 189 / 30**. Quits in the prefix **do not** make that family worse; if anything they look like interest (the player engaged enough to tap) — a trap for "negative feedback" logic. Sample sizes are small, so the claim is only "no evidence of a penalty".
- Within-session concordance of "number of good prefix games in the same family": 0.525 / 0.530; of mechanic: 0.518 / 0.509. Smaller than the user-history affinity (and partly redundant with it, since a game of the family just played also raises the user's smoothed affinity estimates).
- The non-prefix part of the same session is not available, so there is no way to adapt after the first 3.

### 5.7 Concordance summary (within-session pairwise, 0.5 = nothing)
| Signal | train | val |
|---|---|---|
| user affinity: family | 0.597 | **0.636** |
| user affinity: mechanic | 0.600 | **0.632** |
| game mean preview watch | 0.583 | 0.576 |
| best previous rel on this game (repeat) | 0.576 | 0.570 |
| game popularity (LOO train rate) | 0.575 | 0.574 |
| user affinity: pace | 0.548 | 0.568 |
| population rate of the mechanic / family | 0.555 / 0.546 | 0.529 / 0.534 |
| friends played (>=1) | 0.524 | 0.531 |
| friends completed | 0.519 | 0.529 |
| prefix: family played well | 0.525 | 0.530 |
| difficulty (easier better) | 0.515 | 0.519 |
| times seen | 0.510 | 0.515 |
| prefix: mechanic played well | 0.518 | 0.509 |
| editorial stars | 0.507 | 0.508 |
| game age | 0.504 | 0.506 |
| user affinity: theme / art_style | 0.511 / 0.504 | 0.505 / 0.506 |
| typical_session_sec, has_leaderboard | ~0.50 | ~0.48-0.49 |

Hand-built composites on val (no fitting; z-scores added; NDCG@5 over the 604 scored sessions): random 0.310; popularity 0.401; game watch alone 0.416; popularity + family affinity + mechanic affinity 0.514; + friends + repeat 0.494; + watch 0.497; + prefix-family + difficulty 0.503. Take these as "affinity adds ~0.11 NDCG on top of popularity", not as a tuned model: the extra signals were added without calibration and the later terms made the equal-weight blend worse, which says weights matter.

For cold users (val, <=2 sessions): conc. popularity 0.545 (warm 0.579), affinity-family 0.553 (warm 0.651), affinity-mechanic 0.536 (warm 0.650), friends 0.541 (warm 0.530), game watch 0.562 (warm 0.579), difficulty (easier is better) 0.526 (warm 0.517). The baseline NDCG@5 for cold sessions is 0.378 vs 0.406 for warm: the signals available to a cold user are popularity, game watch, friends, difficulty, and the 3 prefix rows.

---

## Most promising signals for the ranker (ranked)
1. **User x mechanic/family affinity**, with shrinkage toward the population mechanic/family rate and tracked on rel >= 1 and rel >= 2 outcomes. Strongest single signal (conc. 0.60-0.64 vs 0.575 for popularity), replicates on val, and survives restricting to games the user never saw. Top bucket has mean rel ~0.5 (2x average).
2. **Game quality, smoothed**: popularity (P(r>=1)/P(r>=2) with shrinkage) combined with the game's mean preview watch, which is far more plentiful per game (and for games with few plays). The watch-based estimate is as good as the baseline popularity rate and more stable for newer games.
3. **Previous outcome on this exact game** (best/last relevance; replay-worthy games return): rel 2-3 previously gives 2.5x lift; rel 0 previously gives a slight penalty. A repeat-aware score is cheap to compute.
4. **Skill x difficulty mismatch** (user skill from past `score_percentile`, difficulty 4-5 for low-skill users): predicts quits and non-completion strongly (quit 26% for low-skill hard vs 9% easy; completion 36% vs 61%), helping the rel>=2 end of the gain. Difficulty alone is a weak signal (conc. 0.515).
5. **Friends** who played/completed the game (monotone with number of friends; modest within popularity strata): ~+0.05 absolute on P(r>=1), consistent across cold and warm users.
6. **In-session**: same family/mechanic as a prefix game the user played and did not quit, and a cold-start fallback for sessions with little history. Weaker than 1-3 but the only personalised signal for the 78 zero-history val / 47 test sessions.
7. Population mechanic/family/pace effects (strategy, music/rhythm and "mixed" pace low; action/puzzle/fast high) and 1★ editorial stars as a negative: small additions or priors for new games.

## Traps / gotchas
- **val_events contains val slate outcomes** (labels fully derivable, 9,576 of 9,576 rows). Never use it as history for val; use it only for test. Train on train; evaluate val with features from train + the session's own prefix.
- **Train contains 685 exact duplicate rows** (appended at the end). Deduplicate before computing counts, rates or splits.
- **Blank `play_seconds` on taps are mostly positives** (44/44 in val labels where visible; 55% have completed=1). `common.relevance` scores them 0; for history/training treat them as unknown/likely positive or drop them. Blank / zero `watch_ms` are logging gaps, not zero interest.
- **Position effect and slate selection**: training on positions 1-3 mixes in a better-than-slate distribution (+17% mean rel). Slate order is shuffled, so no position feature at inference. Eval-like training rows = sessions with 7+ impressions, position > 3.
- **User-level and session-level features cannot change within-session ranking** (user history volume, overall rate, platform, hour, prefix engagement, session length): they shift every slate item together. A user-only score has NDCG@5 = 0.3096 = random. Only user x item interactions and item features matter. Also 26% of val sessions have no relevant slate item at all and are not scored.
- **The cold-game problem is bigger at test than val**: val slates have 2.2% rows from zero-history games (4 games); test has 4.7% from zero-history games (8 games) and 7.1% from games added on day 25+. The evaluator's `new_game_share@5` counts top-5 slots with <30 impressions; ranking all unknown games in the middle (as the baseline does) is neutral, but they are not worse than average (val: 11.9% P(r>=1) vs 13.0%). For test, history = train + val_events, so g033/g054/g064/g069 will no longer be new.
- **Exposure is not random**: popular and 4-5-star games were shown more, new games were boosted, and a mild personalisation exists. Popularity and "user history in X" are partly products of what the old feed showed (exposure bias), and an offline score only measures agreement with outcomes on games the old feed showed. Counts for rarely-shown games are noisy and inflate/deflate with few exposures: shrink.
- **Per-session positives are scarce**: ~1.5 relevant games among ~12; NDCG@5 on 604 sessions is noisy (a val run to run difference of 0.01 is within noise). Use paired comparisons and cold/warm splits.
- **Friend signal timing**: friends' outcomes must be strictly earlier than the target `ts`; friends' same-day sessions after the target would leak. Also friend outcomes from val-day sessions are fine for test but not for val.
- **Prefix and `val_events` rows are separately blanked**: 230/2,442 prefix watch values disagree with `val_events` for the same (session, position), so do not merge them naively.
- **Quits are not negative evidence for the family** (same-family slate games after a prefix quit are above average), and quit rate correlates with difficulty; the metric treats a quit as 0 even though it signals interest. Do not build a "dislike" feature from a quit without the skill/difficulty context.
- **Weak or non-replicating signals** (don't over-invest): theme, art_style, typical_session_sec, has_leaderboard, platform, hour of day, day of week, editorial stars above 1.
- Val is only 3 days, 814 sessions, and the day-24-26 window is the only evidence of how train-derived rates transfer (train-vs-val per-game Spearman 0.52). Tuning many parameters on it will overfit; the test period is another 3 days with more new games.
