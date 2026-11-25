# Research notes: Netflix, Spotify, Roblox

Scope: how these companies learn from implicit feedback, balance exploration and exploitation, handle cold start, optimise for long-term satisfaction, and track session intent. Read with Atoms in mind: a vertical feed of 12-second previews, about 500 games, many of them days old, attributes only, two part-time engineers, under 100 ms.

## How this was sourced

Each fact below comes from a page or PDF that was opened and read in this session. Where a primary page could not be opened, the source entry says so and names the copy that was read instead.

Pages that failed:
- `netflixtechblog.com` and `medium.com` posts (artwork, long-term satisfaction, interleaving): HTTP 403 on every attempt. Facts were taken from a verbatim syndicated mirror (long-term satisfaction), a Netflix conference slide deck (artwork), and a third-party summary (artwork). The interleaving post could not be read at all.
- `dl.acm.org` (Spotify BaRT paper): HTTP 403. The paper itself was not read. The Spotify patent covering the same system was read instead.
- Roblox pages (`about.roblox.com`, `create.roblox.com/docs/discovery`, `devforum.roblox.com`, `blog.roblox.com`): TLS certificate error in the fetch tool on every attempt. **No Roblox page was read.** Roblox is therefore reported as a gap, not as findings.

---

## Sources

### 1. Gomez-Uribe and Hunt, "The Netflix Recommender System: Algorithms, Business Value, and Innovation"
- Year: 2015 (ACM TMIS, vol. 6, 2016)
- URL read: https://gwern.net/doc/reinforcement-learning/exploration/2015-gomezuribe.pdf (copy of the paper)
- Facts:
  - Netflix moved from predicting 1-5 star ratings to treating viewing behaviour as the main signal. The paper says ratings mattered while the business shipped DVDs, because a rating was the main proof that someone had watched.
  - Short-term temporal trends (minutes to days) are described as strong predictors of what members will watch, especially when combined with personalization. They run a separate "Trending Now" ranker for this. Atoms analogue: a "hot this week" signal.
  - Offline experiments assume members would have behaved the same under a new algorithm. The authors say offline results are not as predictive of A/B outcomes as they would like, and that a new algorithm producing very different recommendations will look worse offline than it is.
  - Retention is the main A/B target, but it is hard to move: a 0.1% difference needs a very meaningful improvement. They also watch local engagement and overall engagement, and treat retention wins unsupported by engagement as suspect and repeat them.
  - They typically test 5 to 10 variants against one control in a single test, and run several compatible tests at once on the same members, to move faster.
  - New-member cold start: the paper says the system works well for members with large histories and poorly for new members. Their approach is a sign-up survey where new members pick videos from an algorithmically populated set, used as input to all algorithms.
  - Interleaving is named here as something under exploration for fast, local-metric tests.

### 2. Netflix, "Recommending for Long-Term Member Satisfaction at Netflix"
- Year: 2024 (Netflix Tech Blog, 29 Aug 2024)
- URL read (mirror of the Netflix post, original returned 403): https://noise.getoto.net/2024/08/29/recommending-for-long-term-member-satisfaction-at-netflix/
- Original: https://netflixtechblog.com/recommending-for-long-term-member-satisfaction-at-netflix-ac15cada49ef (not opened)
- Facts:
  - Framed as a contextual bandit. A member visit is the context, the recommendations shown are the action, and immediate or delayed feedback feeds a reward function.
  - Retention is the stated north star but is a poor training target: noisy, low-sensitivity, hard to attribute to one recommendation, and slow (one signal per account per month).
  - Click-only reward (1 for click, 0 otherwise) is called out as risking clickbait. This maps directly onto "attractive preview, dull game".
  - They use a proxy reward r(user, item) built from richer behaviour. Examples given: finishing a season quickly (positive), finishing then thumbs-down (negative), a 10-minute play (ambiguous), and exploring a new genre after a title (positive).
  - Delayed feedback is handled by predicting the final outcome from what has been observed so far, p(final | observed). Training rewards use observed plus predicted feedback, so the policy can update soon after a recommendation.
  - "Reward engineering" is an iterative loop: hypothesis, new proxy reward, train a policy, A/B test. When offline gains do not show up online, the fix is to revise the proxy, not to ship the model.
  - Open questions listed by Netflix include learning the proxy automatically from retention and choosing how long to wait for delayed feedback.

### 3. Netflix, "Artwork Personalization" (deck) and Reforge summary
- Year: 2017 blog, 2018 talks
- URL read, Netflix deck: https://www.slideshare.net/slideshow/artwork-personalization-at-netflix/121997305
- URL read, third-party summary: https://www.reforge.com/blog/brief-netflix-artwork-personalization-through-multi-armed-bandit-testing
- Original blog (not opened, 403): https://netflixtechblog.com/artwork-personalization-c589f074ad76
- Facts:
  - The artwork problem is a bandit: context is member, device and page; the action is which image to show for a title; the reward is the "take fraction", meaning the member played and engaged, with no engagement or abandonment counted as bad.
  - Algorithms listed in the deck: epsilon-greedy, UCB, Thompson sampling (a Beta distribution per image), per-image regression ("contextual greedy"), LinUCB, and contextual Thompson sampling (citing Chapelle and Li 2011).
  - Attribution is a stated difficulty: most viewing comes from recommendations, so it is unclear whether a play came from the title being recommended, the artwork, or both. The Reforge page also notes that the reward weighs engagement quality to avoid clickbait artwork.
  - Cold start: the deck gives no dedicated new-title method. It describes warm-starting the bandit from batch-learned data and a fallback chain (personalized pick, then unpersonalized pick, then a default image).
  - Offline evaluation is "replay": count matches between logged actions and the new policy's choice, which is unbiased when logged probabilities are kept but needs a lot of data and has high variance. The deck suggests doubly-robust estimation. Exploration probability and candidate pool are logged at serving time.
  - Serving: the bandit is precomputed per profile and title, stored in a cache, and read once per request; a fraction of traffic stays on epsilon-greedy as a sanity check. Peak load above 20 million requests per second.
  - Reported outcome: A/B lift, rolled out to over 130M members, with the largest gains on lesser-known titles.
  - Feedback-loop risk: the bandit only sees data from its own choices.

### 4. Chapelle and Li, "An Empirical Evaluation of Thompson Sampling"
- Year: 2011 (NIPS)
- URL read: https://proceedings.nips.cc/paper_files/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf
- Facts:
  - Thompson sampling for a Bernoulli bandit: keep a Beta posterior per arm, sample from each, play the largest sample, update. In simulations it had lower regret than UCB.
  - It has essentially no parameter to tune in its basic form. The authors recommend it as a standard baseline.
  - Delay: feedback usually arrives in batches. Their simulation uses a dynamic set of 10 items where items retire and new ones appear (probability 10^-3 per step). Thompson sampling degraded less than UCB as the delay grew, because randomisation spreads choices while the model is stale.
  - Contextual version: logistic regression with a Gaussian approximation of the posterior (batch updates, previous posterior used as prior), sampled for exploration. Tested on display ads and on news recommendation with 10, 30 and 60 minute update delays; randomised methods held up better than deterministic UCB at long delays.
  - Tweaking the posterior to explore less can help.

### 5. McInerney et al., "Explore, Exploit, and Explain" (BaRT), via the Spotify patent
- Year: paper 2018 (RecSys); patent US 11,301,513
- URL read: https://patents.google.com/patent/US11301513 (first 100,000 of about 122,000 characters)
- Paper link, not opened (403): https://dl.acm.org/doi/10.1145/3240323.3240354
- Facts (from the patent, which describes the system called "Bart"):
  - Each action is an (item, explanation) pair, for example a playlist plus a reason such as "more like X". The explanation is part of what is personalized.
  - Context includes user attributes, item attributes, time of day and platform. The reward model predicts engagement (click or consumption probability); the main version is a logistic model with one-hot and interaction terms, with factorization machines as an option.
  - Exploration is done in two steps: pick the item with epsilon-style exploration, then pick the explanation for it the same way. This avoids exploring the full product of items and explanations and keeps logged selection probabilities from becoming tiny.
  - Every impression is logged with context, action, feedback and propensity. Later feedback (plays, saves) is joined back.
  - Training re-weights examples by uniform-policy probability divided by logged propensity (counterfactual risk minimisation), with capped and normalised weights to control variance.
  - Serving uses a frozen model to score items; an offline evaluation service checks new models before deployment.

### 6. Zhang, Baldwin-McDonald, Ciosek, Maystre, Russo, "Impatient Bandits: Optimizing for the Long-Term Without Delay"
- Year: 2025 (arXiv 2501.07761; v2 forthcoming in JMLR)
- URL read: https://arxiv.org/abs/2501.07761 and https://arxiv.org/pdf/2501.07761
- Facts:
  - Content exploration is modelled as a bandit with delayed rewards. Waiting for the full outcome takes weeks; short-term proxies are only loosely tied to the goal.
  - The method trains a Bayesian filter that combines partial, progressively revealed engagement with the final outcome, and predicts the long-term result from early signals. The bandit uses that belief.
  - Authors are listed at Spotify; the paper states the motivating problem arose at Spotify, with a long-term outcome of average engagement over the 60 days after a recommendation (applied to podcasts, where the goal is shows users return to over two months).
  - The paper reports an A/B test over about two months on hundreds of millions of users and says the method is now used in production. It outperformed optimising a short-term proxy and optimising only the delayed reward.
  - Predictions of the long-term outcome were reasonably accurate after about 10 days of data in the fitted example.
  - A related Spotify approach referenced in the paper's text combines "clickiness" and "stickiness" models into one score. That detail is from one sentence and was not examined further.

### 7. Steck, "Calibrated Recommendations"
- Year: 2018 (RecSys; author at Netflix)
- URL read: https://salanghei.github.io/download/paper/CalibratedRecommendations.pdf (author-hosted copy)
- Facts:
  - Problem: a model trained for accuracy tends to fill a list with the user's single biggest interest and crowd out smaller ones.
  - Calibration is measured as KL divergence between the user's distribution over categories (taken from their history) and the distribution in the recommended list, with a small smoothing term (0.01 in the paper) so the divergence stays finite.
  - Fix is a greedy re-rank after scoring: at each step add the item that maximises (1 - lambda) x model score minus lambda x calibration cost. The objective is submodular, so greedy has a (1 - 1/e) guarantee, and it yields an ordered list.
  - Items can belong to several categories with weights, and several category types (genre, sub-genre) can be calibrated.
  - Small lambda improved calibration considerably with a small accuracy loss; large lambda hurts accuracy quickly.
  - A user whose history is 2% in one genre is considered badly calibrated if that genre is recommended only 1% of the time, since KL is very sensitive when the true share is small.

---

## Ideas transferable to Atoms

### 1. Treat the preview as "artwork": bandit on appeal, reward on play quality (Netflix artwork, Netflix long-term reward)
- What the source does: the bandit picks the image, but the reward is not the click. It is play plus engagement, specifically to avoid clickbait. Netflix's long-term post says a click-only reward promotes clickbait and uses a richer proxy.
- Adaptation: swipe-stops and "Play now" taps are the click. Define the reward as a staged signal, for example: tapped play (small), finished the round (medium), replayed or posted a score or challenged a friend within 24 hours (large), quit in the first 15 seconds of play or never returned to the game (negative). Weigh "played and replayed" far above "tapped". This is the only defence against attractive-preview, dull-game items.
- Two-person build, weeks: log impression, play start, round end, replay, share, challenge with a session id; compute one proxy score per game per day in a SQL job; compare the preview-to-play rate with the play-to-replay rate per game. A game with high first and low second is flagged for editors rather than demoted automatically. This is a day-level batch job plus one dashboard, not a model.

### 2. Thompson sampling for new games, run in the serving layer (Chapelle and Li, Netflix artwork)
- What the source does: a Beta posterior per arm, sample, show the best sample. It needs no tuning, handles delayed batch updates better than UCB, and Netflix used the same family for artwork with warm-starting.
- Adaptation: one Beta per game (or per game and user segment) on the staged reward from idea 1, with an informative prior set from attributes (mean of similar games by mechanic, difficulty, length) so a day-old game does not start at zero. New games get wide posteriors and are sampled more, which is exploration with no hand-written quota. Update hourly in batches; the paper suggests this is safe. A cap on share of impressions per unproven game limits the cost of a bad new game.
- Two-person build: a table of (alpha, beta) per game, a cron job to update from the log, and about 30 lines at request time that samples 500 draws and ranks. Well within 100 ms. Keep logging the selection probability for each impression so you can evaluate offline later (idea 5).

### 3. Early-signal prediction of long-term value for new games (Spotify Impatient Bandits, Netflix delayed feedback)
- What the source does: predict the long-term outcome (60-day engagement) from early partial signals, and update the belief as more signals arrive, so the bandit does not wait for the final number.
- Adaptation: the long-term outcome for Atoms could be "did this player come back to play this game on 3 or more days within 14 days". After day one you only have "finished a round" and "replayed in session". Fit a simple regression from those early signals to the later outcome using the games that are old enough, and use the prediction as the reward for new games. Spotify's own result is a hundreds-of-millions-user A/B test, so the scale does not transfer, but the method works with small tables.
- Two-person build: a weekly script that fits a logistic or linear model of 14-day return on first-day signals; its prediction replaces the raw reward in the bandit for games under about 7 days old. No Bayesian filter needed at first.

### 4. Calibrated re-ranking so the feed matches each player's mix of game types (Steck)
- What the source does: take the user's category distribution from history, then greedily re-rank so the list reflects it, including minor interests, with one lambda knob.
- Adaptation: categories are mechanic, theme, pace and length instead of genre. A player who plays 70% memory games and 30% reaction games should see a mix near that, not 100% memory. Use a recency-weighted count with a prior so that a new player starts from the global mix. Watching must not count the same as playing; weight categories by minutes played or rounds finished, not by video dwell time.
- Two-person build: a function that takes the ranked list and the player's category shares, and does the greedy loop for the first 20 slots. Roughly a day of work. Start with lambda around 0.3 and tune against replay rate.

### 5. Test cheaply: keep propensities, use replay, and interleave rankers (Netflix artwork replay, Spotify propensity weighting, Gomez-Uribe interleaving mention)
- What the source does: log the probability with which each item was shown; use replay or inverse-propensity weighting to estimate a new policy offline; keep a small slice on simple epsilon-greedy as a sanity check. Netflix notes that offline tests mislead when the new policy differs a lot from production.
- Adaptation: at 50k monthly players an A/B test on retention will not reach significance for small gains (the Netflix paper itself says a 0.1% retention move is hard to detect even at their size). Use local metrics (replay rate, finish rate) and compare rankers by interleaving them inside one feed, which shares the player's attention and needs far fewer players. This is a suggestion from reading the Netflix paper's mention of interleaving; the interleaving post itself could not be opened, so the claimed sensitivity gains are not verified here.
- Two-person build: write the impression log with a propensity column from day one (cheap now, hard to add later); implement team-draft interleaving of two rankers in the feed API (a few dozen lines).

---

## What does NOT transfer

- **Scale-based machinery.** Netflix precomputes 20M+ requests per second into caches, and trains with large feature pipelines. At 50k players and 500 games, a single table and request-time sampling is enough.
- **Retention as the target.** Netflix and Spotify train on proxies precisely because retention is noisy and slow; at Atoms' size retention per game is far noisier. Use the staged proxy and treat retention as a periodic check, not a training signal.
- **Fixed catalogue assumption.** Artwork bandits choose among a handful of images for a stable title. Atoms' arms are the games themselves, and the set grows 10x, so priors from attributes matter more than they do for Netflix.
- **Watching as the main signal.** All Netflix sources treat a play as a clear intent signal. In Atoms the preview is only a promise, and the preview-to-play gap is exactly what must be separated from play quality.
- **Skill and difficulty.** None of these sources models player skill or game difficulty matching; they recommend content that is consumed passively. Matching difficulty to ability, leaderboard rank and friend challenges have no counterpart in them and must be designed separately.
- **Explanations (BaRT).** Showing "why this" is useful on a playlist home page but adds little to a one-item-at-a-time swipe feed.
- **Roblox.** Nothing in this file is from Roblox, because its pages could not be read in this session. Roblox is the closest product analogue (many small games, new titles, social) and should be researched separately.
- **Session intent.** None of the sources read here describes in-session intent tracking in detail beyond the "Trending Now" short-term signal in the Netflix paper. For Atoms, within-session adaptation (skip streaks, a just-finished game) will need its own design.
