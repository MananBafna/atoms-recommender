# Skill and difficulty models for Atoms: what Duolingo and rating systems offer

Scope: Duolingo's published work, game and test rating systems (Elo, Glicko-2, TrueSkill, Rasch/IRT), and flow / target-success-rate research.

Method note: every fact below comes from a page or PDF I fetched and read in this session. Where a page failed or could not be read, it is listed under "Gaps" at the end. Duolingo's reported numbers come from a product with a very large user base; they are context, not forecasts for Atoms.

---

## A. Duolingo

### A1. "Learning how to help you learn: Introducing Birdbrain" (Duolingo blog, 2020)
URL: https://blog.duolingo.com/learning-how-to-help-you-learn-introducing-birdbrain/
- Birdbrain tracks how much each learner knows and how hard each piece of material is, and predicts whether a learner will answer a given exercise correctly.
- Duolingo's Session Generator uses those predictions to pick exercises at each learner's level.
- Reported results: A/B tests showed Birdbrain-built lessons helped learners learn more and raised engagement and daily return. No figures are given on this page.
- Rollout: personalized lessons began in March 2020; over 20% of lessons were personalized by October 2020.
- The page gives no equations, no target success rate, and does not mention Elo or IRT.

### A2. IEEE Spectrum article on Duolingo's lesson personalization
URL: https://spectrum.ieee.org/duolingo (year not shown in the text I retrieved; secondary source, journalism)
- First version of Birdbrain: logistic regression inspired by item response theory. Probability of a correct answer is a function of exercise difficulty and learner ability.
- Exercise difficulty is estimated as the sum of the difficulties of its components (exercise type, vocabulary words).
- After each exercise the system takes one stochastic gradient step on the relevant parameters. The article calls this a generalization of the Elo rating system. A wrong answer lowers the learner's ability estimate and raises the exercise's difficulty estimate. If the model already expected a correct answer, little changes.
- No numeric target success rate. The stated aim is to keep learners "engaged but still learning at the edge of their abilities".
- In controlled tests Birdbrain raised both engagement (time on task) and learning measures. Version 2 replaced the single ability number with a 40-dimensional vector from a recurrent network, updated in near real time.

### A3. Settles & Meeder, "A Trainable Spaced Repetition Model for Language Learning" (ACL 2016)
URL: https://aclanthology.org/P16-1174.pdf (text extracted from the PDF)
- Half-life regression (HLR): probability of recalling an item is p = 2^(-lag / h), where lag is time since last practice and h is the "half-life". Half-life is estimated as h = 2^(theta . x), with x a vector of practice-history features.
- Features: per-learner-per-item counters (times seen, correct, wrong) plus a sparse indicator per word type (about 20k) meant to capture inherent word difficulty.
- Trained on 12.9 million learner-word traces. Beat the Leitner box method and plain logistic regression on error and ranking metrics.
- Live test I (just under 1M students, 6 weeks), HLR vs Leitner: +0.3% retention in any activity and in lessons, -7.3% in practice sessions (significant). Shipped anyway on other grounds.
- Live test II (3.3M students, 2 weeks): dropping the per-word indicator features (which had overfit and made some words decay unfairly fast) raised retention: +12.0% any activity, +1.7% lessons, +9.5% practice (all significant). The simpler feature set also avoids cold start for new courses.
- Relevance: skill estimates should depend on time since last practice, and a model using only the learner's own interaction counters can be more robust than one with many item-specific parameters.

### A4. Yancey & Settles, "A Sleeping, Recovering Bandit Algorithm for Optimizing Recurring Notifications" (KDD 2020)
URL: https://research.duolingo.com/papers/yancey.kdd20.pdf (text extracted from the PDF)
- Setting: pick one template per daily reminder. Reward r = 1 if the user completes a lesson within 2 hours of the reminder, else 0. The short window limits credit for lessons users would have done anyway.
- "Sleeping" arms: some templates are only eligible for some users (e.g. a streak-based one). Each arm is scored only against rounds where it was eligible, as a relative difference: s_hat(a) = (mu_plus - mu_minus) / mu_minus, where mu_plus is the reward rate when the arm was chosen and mu_minus is the rate when it was eligible but not chosen. Weighted importance sampling corrects for the earlier policy's unequal choice probabilities.
- "Recovering" arms: fresh templates beat stale ones (novelty effect). Score for arm a at round t: s(a,t) = s_hat(a) - delta * 0.5^(d / h), where d is days since this user last got that arm, delta is a base penalty, h a half-life. The paper's offline sweep used h = 15 days. A never-seen arm gets no penalty.
- Selection: softmax over scores, P(a) = exp(s(a,t)/tau) / sum over eligible arms of exp(s/tau). tau controls exploration.
- Arm histories are truncated to recent data, just enough to get the 95% interval of the score within +/-0.15 percentage points. For most arms this took roughly 1 to 5 weeks of data at Duolingo's volume.
- Offline gain over uniform random: +1.2% (one arm per template) and +1.8% (template by UI language). Online test: +0.5% daily active users, +0.4% lessons completed, +2.2% new-user day-1 retention and +2.0% day-7 (marked significant in the paper's table), with 5% of rounds held out on uniform random.

### A5. Duolingo on streaks
URL: https://blog.duolingo.com/how-duolingo-streak-builds-habit/ (year not shown in the retrieved text)
- A milestone animation after extending a streak: +1.7% in new learners still active 7 days later.
- Letting learners equip up to two Streak Freezes: +0.38% relative daily active learners, without more days off.
- Learners at a 7-day streak are 3.6 times more likely to finish their course (the page does not say how this was measured; treat as correlation).
- Stated mechanisms: habit through repetition in a stable context, a visible number as accountability, later loss aversion. Stated downside: a broken streak can discourage people.

### A6. Duolingo on leaderboards and experimentation
URLs: https://blog.duolingo.com/duolingo-leagues-leaderboards/ and https://blog.duolingo.com/improving-duolingo-one-experiment-at-a-time (years not shown)
- Leagues: a new league starts every Sunday in the learner's time zone; learners are grouped with others of similar study habits and time zone; 10 leagues now, up from 5; top Diamond learners can enter a Diamond Tournament. No quantitative results on that page. Leaderboards were first tested in 2018.
- Experimentation: a few hundred experiments often run at once; the internal service supported over 2,000 experiments in about three years; hypothesis written before launch; gradual rollout; nightly reports on curated plus universal metrics.
- The iOS leaderboard test raised lessons started and lessons completed, so it launched. A promotion for offline lessons raised revenue slightly but cut retention and daily actives, so it was stopped.

---

## B. Rating systems

### B1. Elo (standard formulas, via Wikipedia)
URL: https://en.wikipedia.org/wiki/Elo_rating_system
- Expected score of A against B: E_A = 1 / (1 + 10^((R_B - R_A)/400)). Update: R_A' = R_A + K * (S_A - E_A), where S_A is 1, 0.5 or 0.
- A 100-point advantage is about a 64% expected score; 200 points is about 76%.
- Original K values were 16 (masters) and 32 (weaker players). K caps movement per game.

### B2. Pelanek, "Applications of the Elo rating system in adaptive educational systems" (Computers & Education, 2016)
URL: https://www.fi.muni.cz/~xpelanek/publications/CAE-elo.pdf (author-hosted manuscript; text extracted)
- Player-vs-item form: P(correct) = 1 / (1 + e^-(s - d)). After each answer: s := s + K*(correct - P) and d := d + K*(P - correct). Both start at 0. The paper says this is closely related to the Rasch model.
- A fixed K = 0.4 was common in earlier student-modelling work. An alternative is an uncertainty function U(n) = a / (1 + b*n) in place of K, n being answers so far. Grid search gave a = 1, b = 0.05: fast coarse estimates early, stability later.
- Items and students can use different uncertainty functions: items are expected to have near-constant difficulty and many answers each; student skill is expected to change.
- The paper presents Elo's strength as simplicity and robustness, not expressive power. A 2025 follow-up found in my search (https://pmc.ncbi.nlm.nih.gov/articles/PMC12784335, snippet only, not fetched) reportedly finds Elo estimates biased in some settings, so treat estimates as ranking signals rather than exact measurements.

### B3. Glicko-2 (Glickman, worked-example document, 2022)
URL: http://www.glicko.net/glicko/glicko2.pdf (text extracted)
- Each player has rating r, rating deviation RD (uncertainty) and volatility sigma (expected fluctuation). Unrated players start at r = 1500, RD = 350, sigma = 0.06. A 95% interval is r +/- 2*RD.
- Results are processed in "rating periods", with all games in a period treated as simultaneous. It works best with an average of at least 10-15 games per player per period.
- If a player does not play in a period, only the RD grows; rating and volatility stay put. This is the built-in handling for returning players.
- System constant tau (reasonable range 0.3 to 1.2) limits volatility changes; smaller if very improbable results are expected.

### B4. Lichess puzzle ratings (source file PuzzleFinisher)
URL: https://github.com/lichess-org/lila/blob/master/modules/puzzle/src/main/PuzzleFinisher.scala (read via the fetch tool's summary of the file; the Lichess blog URL returned 403)
- Each puzzle attempt is computed as a Glicko game: the user is one side, the puzzle the other. A solve raises the user's rating and lowers the puzzle's.
- Updates are blended with a weight that depends on puzzle theme and on whether the puzzle's rating is still provisional; the puzzle rating change is clamped per update.
- Puzzle rating updates are skipped for users flagged as dubious and are rate limited (300 per day per the file).
- Relevance: a working precedent for rating the content and the player in the same loop.

### B5. TrueSkill (Microsoft Research project page)
URL: https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/ (the original Herbrich et al. 2006 paper was not fetched)
- Skill is a Gaussian: mean mu and uncertainty sigma. New player: mu = 25, sigma about 8.33. Leaderboards use the conservative estimate mu - 3*sigma.
- Sigma is slightly inflated before each game to allow for skill drift. Updates scale with relative uncertainty (like an adaptive K).
- Built for multiplayer and team rankings from finishing order. Convergence in a 2-player game: about 12 games per the page.
- Relevance: designed around head-to-head ranking outcomes. Atoms is solo scoring, so Elo/Glicko against a game "opponent" fits better. The conservative mu - 3*sigma idea is useful for ranked displays.

### B6. Rasch (1PL) model (via Wikipedia)
URL: https://en.wikipedia.org/wiki/Rasch_model
- P(correct) = e^(b_n - d_i) / (1 + e^(b_n - d_i)), with person ability b_n and item difficulty d_i. When ability equals difficulty, P = 0.5. The page did not give the 2PL formula; 2PL adds a per-item slope ("discrimination").

---

## C. Flow and target success rate

### C1. Wilson, Shenhav, Straccia, Cohen, "The Eighty Five Percent Rule for optimal learning" (Nature Communications, 2019)
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6831579
- For binary classification learned by gradient descent with Gaussian noise, learning speed is highest when training error is about 15.87%, i.e. about 85% accuracy.
- The optimum shifts with noise shape: 85% (Gaussian), 82% (Laplacian), 75% (Cauchy).
- Tested by simulation (perceptron, small neural network, a perceptual-learning model). No new human experiments. The link to flow and engagement is a proposal, not tested.
- A learner with perfect memory that learns the same regardless of trial order has no sweet spot. Multi-category tasks and batch learning are left for future work.

### C2. Flow (Csikszentmihalyi, 1975), as reviewed in Frontiers in Psychology, 2020
URL: https://www.frontiersin.org/articles/10.3389/fpsyg.2020.00158/full (title and author not captured from the page text; I read only the first part of it)
- The 1975 three-channel model places flow between boredom (too easy) and anxiety (too hard); a later quadrant version defines flow as challenge and skill both above the person's own average.
- Challenge-skill ratios explain little variance in reported experience across activities, and the paper says inferring flow from balance alone "is not a safe bet". The challenge-enjoyment relation varies by activity and person.
- Practical reading: challenge matching is a reasonable hypothesis for a ranking signal, not a proven cause of enjoyment. Test it.

---

## Ideas transferable to Atoms

Common frame: every play is a "match" between a player and a game. The outcome y is in [0,1]. Ability and difficulty are learned together, as in Birdbrain and Lichess puzzles. Nothing here needs more than a batch job and a few tables.

### Idea 1. Rate players and games together, per mechanic (Elo on a Rasch curve)

Mechanism. Rasch probability plus an Elo-style update (Pelanek's form):

    P = 1 / (1 + exp(-(theta - b)))        # theta: player skill, b: game difficulty
    theta <- theta + K_p * (y - P)
    b     <- b     + K_g * (P - y)

If the player beats the prediction, theta rises and b falls a little; if they underperform, the reverse. Use an uncertainty-shaped step K = a / (1 + 0.05*n), n = plays so far for that player or that game (separate a for players and games; games get many plays so their K shrinks fast). The classic Elo scale form, for comparison: E = 1 / (1 + 10^((R_g - R_p)/400)), R' = R + K*(S - E).

Atoms adaptation.
- Outcome y: score_percentile / 100 among prior plays of that game, a soft result (like a draw being 0.5 in chess). With a percentile outcome, P = 0.5 means "median player", which makes the scale easy to read and to check.
- Quits: do not set y = 0 automatically. An early quit with no sign of failure is a signal about interest, not skill (see Idea 2). A quit after visible failure (ran out of lives, repeated misses) counts as low y. Skip the skill update for ambiguous early exits.
- Per-player-per-mechanic skill: store theta[player, mechanic] plus a global theta[player]. A new (player, mechanic) pair starts at the player's global theta, and its large early K lets the data correct that. Mechanic = a category tag (memory, reaction, logic, numeric, spatial, word...) over the 500 games; if tags are thin, cluster games by play data first.
- Game difficulty b: initialise from the editorial label (map labels to starting b values) and let K_g shrink as plays accumulate. By roughly 20-30 plays per game b is mostly data-driven (my estimate; Lichess keeps a provisional flag and a damped weight on new puzzles, so copy that). Exclude provisional games from target-band matching.
- Time decay: when a player returns after a gap, inflate their uncertainty. Glicko-2 does this by growing RD while inactive; a one-line version is K_p *= 1 + days_inactive/30, capped. Settles & Meeder is the evidence that estimates should depend on lag.
- Self-selection bias: players choose games they expect to do well at, which drags b downward. Compute b mainly from plays the feed served, and compare with plays the player sought out. Gate or rate limit updates to resist gaming, as Lichess does.

What two engineers can ship in weeks.
- Week 1: an events table (player, game, mechanic, y, completed, timestamp) and a nightly job that recomputes theta and b with the update above. Back-test: does P predict whether a play lands above the player's own median? Report log-loss against "always predict 0.5".
- Week 2: serve theta and b from a table; keep the editorial label as fallback; log P with every impression so later experiments can use it.
- Not needed yet: Glicko-2's volatility step, TrueSkill, 2PL. Glicko-2 wants about 10-15 games per rating period, which many Atoms players will not reach on one mechanic, so plain Elo with an uncertainty function fits better at this scale. Move to 2PL (P = 1/(1+exp(-a_g*(theta-b))), with a per-game slope a_g; standard form, not read on a fetched page) once data shows some games separate players much more sharply than others.

### Idea 2. Separate interest from ability, and recommend in a target success band without trapping players

Mechanism. Two models per (player, mechanic):
- Ability (Idea 1): predicted P that the player does well on game g.
- Interest: engagement beyond what ability explains. For each impression record: played past the first 10 seconds, finished, replayed, came back next session. Compare with what P alone would predict. Interest = the smoothed residual (a Beta prior per player-mechanic).

Why: a quit on a low-P game is probably "too hard"; a quit on a mid-P game is more likely "not for me". Fitting interest on engagement residuals, not raw quits, stops difficulty masquerading as dislike.

Selection rule: score = interest + w * closeness_of_P_to_band - repeat_penalty, subject to quotas.
- Target band: start with predicted P between about 0.55 and 0.80, centred near 0.7. Wilson et al. give about 0.85 for learning speed in a simple two-choice task and say the optimum moves with noise; they prove nothing about enjoyment. Duolingo does not publish its number. So treat band edges as an A/B parameter, not a fact. Under the Rasch curve, P = 0.7 corresponds to the game being ln(0.7/0.3) = 0.85 logit units below the player's skill.
- Anti-trap terms, borrowed from the recovering-bandit idea in Idea 3: subtract delta * 0.5^(d/h) from a mechanic's or game's score if the player played it d days ago; reserve a fixed share of the feed (say 1 in 5) for stretch games at P around 0.4-0.55; cap any one mechanic's share of the feed; downrank a game once P > 0.9 for that player (mastered, little new information, though still available by search or "play again"). Quotas are easier to explain and debug than score tweaks alone.
- Cold start: for a new player, serve games with high population interest and mid difficulty, update theta after 3-5 plays, and trust P only once n for that player-mechanic passes a threshold.

What two engineers can ship in weeks.
- Week 3: the feed ranker as a plain scoring function over candidate games (interest, band closeness, recency penalty, quota), logging every feature.
- Week 4: A/B with three arms: current ranking, interest only, interest plus band. Metrics: sessions per week, plays per session, day-7 return, share of plays quit within 10 seconds. Sample size, my own rule of thumb and not from a source: n per arm is about 16*p*(1-p)/d^2, so detecting a 1 percentage point change on a 10% baseline needs about 14,000 players per arm. At 50k monthly players, test large differences and run for weeks.

### Idea 3. Recovering, sleeping bandit for daily challenge and notification choice

Mechanism (Yancey & Settles). Per arm a:

    s_hat(a)    = (mu_plus - mu_minus) / mu_minus
    s(a,t)      = s_hat(a) - delta * 0.5^(d / h)       # d = days since this user last got arm a
    P(choose a) = exp(s(a,t)/tau) / sum over eligible arms of exp(s/tau)

Arms are compared only on rounds where they were eligible (sleeping), and a recently shown arm pays a penalty that fades with half-life h (recovering).

Atoms adaptation.
- Arms: message templates for the daily push ("streak at risk", "someone passed you", "new game in your favourite mechanic") or daily-challenge mechanic categories. Aim for 10-20 arms; 500 games as arms would be too sparse for 50k players.
- Eligibility (sleeping): "streak at risk" only when a streak exists; "someone passed you" only when the player is on a leaderboard they have played; mechanic arms only where theta is known.
- Reward: reuse Duolingo's short attribution window. r = 1 if the player completes at least one game within 2 hours of the message. Log the choice probability with every send so scores can be reweighted later.
- Once the bandit picks a mechanic or message type, choose the specific daily-challenge game by the Idea 1/2 band rule.
- Keep a random holdout of about 5% of sends (Duolingo does) so the bandit keeps getting unbiased data.
- Start h near 15 days (the paper's sweep value) and tune.

What two engineers can ship in weeks. A table of arms, a daily job computing s_hat from logged sends, and a function applying the three formulas above (roughly 30 lines). The binding limit is data: Duolingo needed about 1-5 weeks per arm to reach +/-0.15 pp at its scale, and Atoms has far fewer sends per arm. So run fewer than 10 arms, group arms at template level, and keep tau high (more exploration) for the first month.

### Idea 4. Daily-return mechanics: streak with slack, cohort leagues, an experiment habit

Mechanism (Duolingo posts). A visible streak with a freeze, milestone moments, and weekly leaderboards in small groups of similar learners.

Atoms adaptation.
- Streak = at least one completed game per local day. Add one freeze that refills weekly. Duolingo's reported lifts (+0.38% for a second freeze, +1.7% day-7 for a milestone animation) are far too small to measure at Atoms' size, so ship the mechanic on prior evidence and measure only coarse outcomes.
- Weekly league: each Sunday group players into cohorts (for example about 30; Duolingo's page does not state its size) by recent activity and time zone, ranked by weekly points. Weight points by game difficulty b so farming easy games pays less. Cohorting by theta as well is an Atoms idea; Duolingo's page does not claim it.
- Process: write the hypothesis before launch, keep a 5-10% holdout for a month, track one primary metric per feature, and stop features that raise one metric while hurting another (the offline-lessons example).

What two engineers can ship in weeks. Streak plus freeze: days, one small table. Weekly cohort league: about a week using a scheduled job (assign cohort, sum weekly points, promote and relegate by rank).

---

## Where the analogy breaks

1. Duolingo has a curriculum and a learning goal. Success near the edge of ability serves learning, and learning aligns with return. Atoms is entertainment: players may enjoy replaying games they are already good at, and mastery loops may be a feature. The 70-85% band is a hypothesis to tune.
2. Wilson et al. derive 85% for learning speed in a simple two-choice task with Gaussian noise; they ran no human experiments and call the engagement link untested. The flow review finds challenge-skill balance explains little of reported experience. Judge the band on retention and session length, not on theory.
3. Duolingo outcomes are correct or incorrect per exercise. Atoms outcomes are scores, relative to others, with time pressure, luck and a different scoring shape per game. Percentile works as y, but it makes ability relative to a moving population, and games will differ in how sharply they separate players (the case for 2PL later).
4. Self-selection. Duolingo's Session Generator chooses the next exercise. In a swipe feed, players skip what they expect to dislike and linger on what they like, so plays are not a random sample, and a quit mixes too-hard, dislike and plain boredom.
5. Birdbrain v1 estimated item difficulty as a sum of component features. Atoms games are less decomposable, so a per-game b learned from plays is simpler, but long-tail games need enough plays first.
6. Effect sizes. Duolingo's reported lifts (0.4% to 2%) were measurable with very large populations. Atoms must test bolder changes for longer.
7. Leaderboards are part of the Atoms product. Keep the private rating used for recommendations separate from the public leaderboard score, or players may read the feed as a judgement of them.
8. Duolingo's streak evidence mixes small A/B lifts with an unexplained correlation (3.6 times completion), and the page itself notes broken streaks can discourage players. Do not copy streak pressure without a freeze or recovery path.

---

## Gaps and what I could not read
- Lichess's own blog post on puzzle ratings returned 403. The Lichess source file summary I received did not show the Glicko constants (tau, defaults), so those are not reported.
- The Pelanek 2016 ScienceDirect page returned 403; I used the author-hosted PDF.
- The original TrueSkill paper (Herbrich et al., 2006) and Csikszentmihalyi's 1975 book were not fetched. TrueSkill facts come from the Microsoft project page; flow facts from a 2020 review whose title and author I did not capture.
- The Birdbrain blog page names no model details; model details come from IEEE Spectrum (secondary). Several Duolingo pages did not show publication years.
- The Elo and Rasch pages were used for standard formulas only. The Elo page was cut off after the first 100k characters; the formulas appear early.
- Birdbrain's published equations and any Duolingo target success rate were not found on any page I read.
