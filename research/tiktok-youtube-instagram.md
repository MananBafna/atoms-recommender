# Research: TikTok / Douyin, YouTube, Instagram recommenders (for Atoms)

Scope: how large video feeds learn from implicit behaviour, correct its biases, balance familiar vs new, track in-session interest, handle cold start, and combine several objectives.

Reading notes: every fact below was read on the page or PDF named. Where a primary page failed to load, that is stated. "Not stated" means the page I read did not cover it, not that the platform does not do it. Years are from the paper/page where shown; blog posts whose fetched text had no date are marked.

---

## 1. Sources

### 1.1 Covington, Adams, Sargin - "Deep Neural Networks for YouTube Recommendations" (RecSys 2016)
URL: https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/ (abstract). Full paper PDF read: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf

- **Watch time as the label, not clicks.** The ranker is trained with weighted logistic regression: clicked impressions are weighted by watch time, unclicked impressions get weight 1. The learned odds approximate expected watch time. The paper says ranking by click-through promotes "clickbait" the user does not finish, so watch time is used.
- **"Example age" feature (bias toward the past / freshness).** A model trained on a several-week window reflects average past popularity. Fix: feed the age of each training example as a feature in training, set it to zero at serving so the model predicts "as of now". The paper shows this restores the upload-time popularity pattern for fresh videos.
- **Train on all watches, not only watches the recommender produced.** Otherwise new content would struggle to surface and the system would over-exploit.
- **A fixed number of training examples per user**, so heavy users do not dominate the loss. Relevant to a small feed with a few very active players.
- Demographics, region and device are used as priors so new users get reasonable output; user history is the average of embedded watched-video IDs.
- Funnel: candidate generation (hundreds) then ranking (dozens). The paper says live A/B results are not always correlated with offline metrics.

### 1.2 Zhao et al. - "Recommending What Video to Watch Next: A Multitask Ranking System" (RecSys 2019)
URL read: https://daiwk.github.io/assets/youtube-multitask.pdf (mirror of the paper). The ACM page https://dl.acm.org/doi/10.1145/3298689.3346997 returned 403.

- **Two objective families.** Engagement (clicks, time spent) and satisfaction (like, rating). Binary behaviours use cross-entropy, time/rating behaviours use squared loss.
- **Combining objectives:** the final score is a weighted multiplication of the predictions. The weights are tuned by hand against both engagement and satisfaction metrics.
- **MMoE (multi-gate mixture of experts)** replaces a shared-bottom network because shared bottoms hurt when tasks are weakly correlated. Few experts, one gate per task. At equal model cost, live tests improved both time spent and survey-rating metrics. 10% gate dropout fixed "gate polarisation", which hit about 20% of distributed training runs.
- **Shallow tower for position / selection bias.** A side network takes position (crossed with device); its output is added to the main model's logit during training. The position feature is dropped 10% of the time to stop over-reliance, and at serving time it is treated as missing.
- Satisfaction is measured live by survey responses and dismissal rate, not only time spent.
- The authors note offline metrics often disagree with online ones when learning from implicit feedback, and recommend a simpler model for better generalisation.

### 1.3 Chen et al. - "Top-K Off-Policy Correction for a REINFORCE Recommender System" (WSDM 2019)
URL: https://arxiv.org/abs/1812.02353 (abstract). Full text read from https://arxiv.org/pdf/1812.02353

- Problem: logged feedback only covers items an earlier model chose (selection bias). Fix: importance-weight updates by (new policy probability / logging policy probability), with the weight **capped** (cap 3 in production; cap 5 let the model overfit a few lucky logged actions). Capping gave the biggest online gain among the variance-reduction methods they tried.
- **Top-K correction** for pages showing several items: avoids a policy that bets everything on the top-1 item. Production K=16; K=1 lost 0.66% view time against K=16.
- **Exploration is cautious.** Epsilon-greedy is called not viable in production. They use Boltzmann sampling (softmax with temperature) over the top candidates, returning the top few deterministically and sampling the rest. Only 5% of users saw the stochastic policy and training on that data gave +0.07% view time.
- Notes that new items arrive constantly, so the action space grows and new items have sparse feedback.

### 1.4 Zhan et al. (Kuaishou) - "Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation" (KDD 2022)
URL: https://arxiv.org/abs/2206.06003 (abstract). Full text read from https://arxiv.org/pdf/2206.06003

- **The bias:** longer videos accumulate more watch time even at equal fit, so a watch-time model drifts toward long videos and the feedback loop amplifies it. Duration affects both exposure and watch time; the exposure effect is the bias, the watch-time effect is legitimate.
- **D2Q method:** split training data into equal-sized groups by duration quantile, and replace the label with the watch-time *quantile within its duration group*. One shared model across groups; at inference the predicted quantile is mapped back to seconds using the video's group.
- The number of groups is a trade-off: more groups remove more bias until per-group samples get too thin and error rises.
- Deployed on Kuaishou; reported better watch-time prediction and more real-time consumption in live tests (the pages I read give no single headline number).
- The paper says its baselines include the weighted regression approach from Covington et al., used on streaming surfaces such as TikTok and Kuaishou.

### 1.5 ByteDance - "Monolith: Real Time Recommendation System With Collisionless Embedding Table" (2022)
URL: https://arxiv.org/abs/2209.07663 (abstract). Full text read from https://arxiv.org/pdf/2209.07663

- **Why online training:** user interest drifts ("could shift their zeal every next minute"), so serving parameters should be updated from new feedback as close to real time as possible.
- Training workers consume a live stream and push only touched sparse parameters to serving at **minute-level intervals**; dense parameters sync less often.
- On a public ad-click benchmark (Criteo), shorter sync intervals (5 h, 1 h, 30 min) gave consistently better AUC than batch training.
- **Collisionless embedding table** (cuckoo hash): every ID gets its own embedding instead of sharing hashed slots. Memory is controlled by (a) an admission filter: IDs seen fewer than a threshold number of times get no embedding, and (b) expiry of inactive IDs.
- Deliberately trades some reliability for freshness (tolerates losing a few parameter-server shards).
- Relevance: it is infrastructure for huge sparse ID spaces. The idea (update quickly, keep rare IDs out) is useful; the system is not needed at 500 items.

### 1.6 YouTube - "On YouTube's recommendation system" (blog; year not shown in the fetched text)
URL: https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/

- Clicks alone were found misleading (2011); watch time was added in 2012 and caused an immediate 20% drop in views, which the team treated as the correct outcome.
- **Valued watch time:** surveys ask viewers to rate videos 1-5 stars. A model predicts ratings for videos that got no survey, with some real responses held back to test accuracy. Only time on videos rated 4-5 stars counts as valued. Low ratings trigger a "why" follow-up.
- Signals are weighted **per user**: if you share every video, shares count less for you.
- Shares, likes and dislikes are used as satisfaction hints.
- Diversity is described as personalised: topic-relevant videos mixed with other interests; "similar viewers" matching can surface things you never searched for.
- Not stated on this page: how new users or new uploads are handled.

### 1.7 YouTube Help - "How YouTube recommendations work" (current help page, undated)
URL: https://support.google.com/youtube/answer/16089387?hl=en

- Surveys on videos you watched are used to measure satisfaction, not just watch time.
- "Not interested" is a negative signal to avoid similar content; "Don't recommend channel" is a stronger one.
- Surfaces weight inputs differently: Up Next relies mainly on the video you are on; the homepage mainly on watch history. This is the only session-context statement on the page.
- Without history, the homepage shows little; users fall back to subscriptions and trending.

### 1.8 TikTok - how For You works (2020 newsroom post, read second-hand via TechCrunch)
URL read: https://techcrunch.com/2020/06/18/tiktok-explains-how-the-recommendation-system-behind-its-for-you-feed-works
The primary page https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you/ returned HTTP 503 (twice, also with ?lang=en). The TikTok transparency URLs I tried served an unrelated India notice. So these TikTok claims are second-hand.

- Signals: likes, shares, comments, follows, captions/sounds/hashtags of videos engaged with. Language, country and device settings count for less.
- **Completion is called a strong interest signal**, weighted above weak ones such as same-country.
- Creator follower count and past performance are not direct factors.
- **New users** pick interest categories; if skipped, they get a general feed of popular videos until early likes, comments and replays arrive.
- **Diversity rules:** removes duplicates and already-seen content; avoids two videos in a row from the same creator or with the same sound; occasionally inserts videos outside apparent interests to widen exposure.
- Explicit feedback: long-press "Not interested", add to favourites.

### 1.9 Meta - Instagram Explore recommender system (2019 post; the title begins "Powered by ...")
URL: https://ai.meta.com/blog/powered-by-ai-instagrams-explore-recommender-system/

- **Seeds and retrieval:** seeds are accounts the person liked or saved; account embeddings (trained on interaction sequences) find similar accounts; media from them gives tens of thousands of candidates, 500 sampled per request.
- **Funnel:** a distilled lightweight model 500 -> 150, a mid network 150 -> 50, a deep network 50 -> 25 (first page).
- **Multi-task predictions** (like, save, "see fewer posts like this") are combined by a **value model**: weighted sum of positive-action probabilities minus a weighted penalty for negative actions; weights tuned by offline replay and Bayesian optimisation.
- **Diversity penalty:** a post's score is reduced if its author or seed was already shown, with the penalty growing further down the batch.

### 1.10 Meta - "Scaling the Instagram Explore recommendations system" (Aug 2023)
URL: https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/

- Two-tower retrieval: item embeddings are precomputed daily and served via nearest-neighbour index; the **user embedding is computed at request time from fresh features**. User-item interaction features are excluded at this stage because they would break caching.
- The second stage is a heavier multi-task, multi-label model scoring the top 100, which can use interaction features.
- The value model is still a tunable weighted combination (positives added, "see less" subtracted).
- Ranking models are fine-tuned about hourly; some retrieval sources use the user's most recent interactions, others long-term interests.
- Some recommendations are precomputed off-peak to protect latency.
- Not stated: an exploration mechanism, or new-user / new-media handling.

### 1.11 Instagram Explore system card (Meta transparency centre; undated in fetch)
URL: https://transparency.meta.com/features/explaining-ranking/ig-explore/

- Retrieval up to 1,500 items; early ranking (two-tower) keeps 100; late ranking is multi-task/multi-label.
- Predicted outcomes include likes, shares, **watch time**, **completion (over 95% of the video)**, follows, saves.
- **Negative signals:** skips under three seconds, quick dismissals, "Not interested"; similar content is filtered afterwards.
- Many inputs are explicitly "recent past" (time spent in Explore, time on an author's posts, swipe rate in reel chains).

### 1.12 Instagram - "Shedding more light on how Instagram works" (Mosseri; year not shown in fetch)
URL: https://about.instagram.com/blog/announcements/shedding-more-light-on-how-instagram-works

- Explore and Reels draw mostly from accounts you do not follow; **how quickly engagement is building matters more in Explore** than in Feed.
- Reels predicts: watch the whole reel, like, rate as entertaining/funny, visit the audio page. Surveys tune these predictions, with attention to smaller creators.
- Poster-level popularity is used so a wide range of creators can reach an audience.
- Feed avoids many posts from one person in a row.
- Later Mosseri statements ("watch time, likes per reach, sends per reach") appeared only in search-result summaries and trade blogs, which I did not read as primary text. Not relied on here.

### 1.13 Meta - "Designing a constrained exploration system" (Dec 2020)
URL: https://engineering.fb.com/2020/12/10/web/how-instagram-suggests-new-content/

- Exploration is **bounded**: suggestions come from seeds the user already engaged with, merged in priority order with round-robin across seeds so none dominates; for sparse users, 1-2 hop graph neighbours; brand-new users get popular media.
- The source mix and media-type mix are kept close to the normal feed; suggestions appear only at the end of the feed.
- Scored by a log-linear value model combining predicted positive actions (like, comment, save) with predicted negative feedback (not interested, see fewer); weights tuned by offline session replay plus online Bayesian optimisation.
- The authors say design was guided by product principles ("feel like home") rather than by AUC/NDCG alone.

---

## 2. Ideas transferable to Atoms

### Idea 1 - Separate "watched the preview", "played" and "enjoyed"; combine per-stage predictions with explicit negatives
- **What the sources do:** YouTube moved from clicks to watch time, then to survey-weighted valued watch time, because a raw signal can mislead (1.6, 1.7). Instagram predicts several actions and combines them in a value model with negative terms (1.9, 1.13). YouTube's multitask ranker multiplies weighted predictions (1.2).
- **Adapting to games:** Atoms has a funnel videos do not: preview watch -> "Play now" -> finish a 1-3 minute play -> replay / share / challenge / leaderboard attempt. A preview swipe-away is a weak signal (the preview may be poor, not the game). A play abandoned early may mean "too hard". So train separate predictors: P(tap Play | preview shown), P(finish | played), P(replay | finished), P(share or challenge | finished). Combine them as a weighted product, and subtract negatives (quit within N seconds, "not interested"). Skips under 3 seconds are an explicit negative at Instagram (1.11); for Atoms, a preview skipped in under about 2 seconds is a similar cheap negative.
- **Two-person build (about 1-2 weeks):** log every funnel event with game ID, player ID, position and timestamp. Start with one small model per stage (logistic regression or gradient-boosted trees on attributes plus per-player aggregates). Keep weights in a config file and check changes by replaying logged sessions, as Meta does. A one-tap "Was this fun?" after about 1 in 10 plays can act as a mini satisfaction survey (1.6), used only to calibrate the weights, not as a main feature.

### Idea 2 - Remove difficulty and length confounding as D2Q removes duration bias; handle position with a bias term
- **What the sources do:** D2Q turns watch time into a quantile within the video's duration bucket so long videos are not rewarded for length (1.4). YouTube's shallow tower learns a position effect in training and drops it at serving (1.2). Covington weights users equally (1.1).
- **Adapting to games:** (a) *Skill vs difficulty:* judge outcomes relative to what players of similar skill usually get on that game. Keep a per-game, per-skill-band baseline for finish rate and score percentile, and use "result vs the band's expected result" as the label. A low-skill player quitting a hard game is then not read as dislike. (b) *Length:* 1-minute and 3-minute games have different natural play time; normalise time-on-game within length buckets. (c) *Position:* log feed position, include it as a training feature, set it to a fixed value at serving, with 10% dropout (1.2). In a vertical swipe feed, position 1 would otherwise dominate.
- **Two-person build:** a nightly SQL job for per-game-per-band baselines and a residual label. D2Q shows too many groups hurts because samples get thin, so at 500 games use 3-5 skill bands, not per-player buckets. Position-as-feature is a small change.

### Idea 3 - Bounded exploration with logged propensities for new games
- **What the sources do:** YouTube's top-K paper samples from a softmax over top candidates rather than epsilon-greedy, exposes a small share of users, and caps importance weights (1.3). Covington trains on watches from all sources so new content can surface (1.1). Instagram explores only near existing seeds, round-robin so no seed dominates, and keeps the mix close to the normal feed (1.13).
- **Adapting to games:** a game a few days old has almost no plays, and a model trained on past exposure will keep hiding it (selection bias from the previous model). Reserve a fixed number of slots per session (for example 2 of every 10 cards) for new or under-sampled games, picked by sampling with a temperature from games whose attributes are close to what the player enjoyed (the "near the seed" constraint). **Log the probability with which each card was chosen.** Later, weight training examples by 1/probability, capped (3 was the production cap at YouTube) so the exploration data corrects bias instead of adding noise. Stop exploring a title once it passes an impression threshold (similar in spirit to Monolith's admission threshold, 1.5).
- **Two-person build (about a week):** an exposure log with propensity per impression, a "needs data" flag per game, and a slot-filling rule in the feed service. No new infrastructure.

### Idea 4 - Session-level interest through fast features, not fast training
- **What the sources do:** Monolith pushes updates every minute (1.5). Instagram computes the user embedding at request time from fresh features and uses "recent past" counters (1.10, 1.11). YouTube's Up Next leans on the current video (1.7). Feeds penalise repeats: Instagram's author penalty (1.9) and TikTok's no-two-in-a-row rule (1.8).
- **Adapting to games:** a 1-3 minute play is long compared with a video swipe, so a session holds few observations (a handful of previews and 2-4 plays). Keep small per-session state in memory or a key-value store: last N games viewed/played, their mechanic, theme and pace, and outcomes (finished, quit early, replayed). Feed features such as "share of the last 5 plays in one mechanic" and "recent finish rate" to the ranker at request time. Add a fatigue term: the same mechanic or theme repeated back to back gets a penalty that grows with the repeat. A game-specific rule: after several quick losses in a row, offer an easier game or a replay of a game the player did well on.
- **Two-person build:** no retraining needed. Session state in the feed service plus a rerank step. Retrain the main model daily; that is plenty for this data volume.

### Idea 5 - Cold start from attributes, with a skill estimate and shrinkage
- **What the sources do:** Covington uses demographics and context as priors for new users (1.1). TikTok asks new users to pick interests, else shows popular items until early likes, comments and replays arrive (1.8). Instagram sends brand-new users popular media and uses 1-2 hop neighbours for sparse users (1.13). Instagram's early ranking stage uses user and item features only, so it can score unseen items (1.10).
- **Adapting to games:** since games have attributes only, make attributes the main representation. A game is a vector of mechanic, difficulty, length, theme, art style, pace, and editorial stars when present. A player's taste vector is the average of attribute vectors of games they finished or replayed (Covington averages watched-video embeddings; here the average is over attributes). A new game is scored from attributes on day one. A new player starts from two or three onboarding picks (like TikTok's interest picker) plus popularity and editorial stars, then a **skill estimate** from early plays (for example an Elo-style update of finish/score against game difficulty). Blend in each game's observed rates as plays accrue, shrunk toward the attribute-based prior (a Bayesian average) so a game with 3 plays does not look great or terrible by luck.
- **Two-person build (about 1-2 weeks):** hand-built one-hot attribute features, a simple rating update for skill, and shrinkage in a few lines of code.

---

## 3. What does NOT transfer

- **Two-tower retrieval with nearest-neighbour search, multi-stage funnels (500 -> 150 -> 50 -> 25) and distillation (1.9-1.11).** Built for billions of items. With about 500 games (a few thousand later), scoring every game per request with one ranker fits well inside 100 ms.
- **Monolith's collisionless embedding tables, parameter-server sync and minute-level online training (1.5).** It solves memory and freshness for huge sparse ID spaces. At 50k monthly players, a daily batch job plus in-session features is enough.
- **REINFORCE with off-policy gradient over millions of actions (1.3).** The principles (log propensities, cap weights, sample instead of epsilon-greedy) transfer; the training machinery is not worth it for two engineers.
- **MMoE and large multi-task networks (1.2).** Gains were shown at YouTube scale. With limited data, several small separate models or a gradient-boosted model per objective are easier to debug. The paper itself warns that simpler models generalise better when offline and online metrics disagree.
- **Survey-trained "valued watch time" at platform scale (1.6).** Millions of survey responses train that predictor. Atoms can only run a small sampled survey and use it to calibrate weights.
- **Creator-side rules** (small-creator boosts, creator reach tests, content-quality filters in 1.12). They assume a creator marketplace; Atoms has editorial control and attributes.
- **Social-graph signals** (sends, follows, seeds from followed accounts). Challenges and leaderboards may become similar signals later, but there is no data now.
- **Popularity-velocity signals** (how fast engagement is building, 1.12) are noisy at 50k monthly players; use long windows and shrinkage if at all.
- **Effect sizes.** Gains quoted in the papers (0.07%, 0.66%, 0.85% view time) came from huge traffic. Atoms cannot detect effects that small, so prefer bigger, simpler interventions and judge them with log replay plus a few large A/B tests.
