# Atoms recommender: summary

## The problem

Atoms players watch a 12 s preview, tap Play now, then play a 1 to 3 minute skill game. Two players can tap the same preview and have opposite experiences because one is outmatched, so a quit can mean "too hard" as easily as "not for me".

## The insight

<!-- card -->
Taste decides what gets started; fit decides what gets finished, and an offline score built on the old feed's slates mostly sees the first.

Family and mechanic taste is the only feature group whose removal hurts ranking beyond noise. Without it NDCG@5 (how well the top five slots match what players did) drops 0.0885 (95% range 0.0666 to 0.1094). Finishing depends mostly on game difficulty. My player and game rating predicts finishing slightly better than the game's finish rate alone (+0.0074 AUC, the chance of ranking a finisher above a non-finisher, lower bound 0.0000) but adds nothing detectable to ranking, where the game rate is already a feature and any play of 20 s or more counts. I keep it because it is cheap and worth testing online.

## Three ideas I adapted

1. Rate players and games together (Duolingo's Birdbrain, Lichess puzzles), so a quit on a too-hard game reads as a mismatch.
2. Predict each step from preview to finish and judge finishes against players of similar skill (YouTube's multitask ranker, Kuaishou's duration debiasing).
3. Try new games by Thompson sampling (each shown in proportion to its chance of being best), starting from similar games and logging each card's chance of being shown (Chapelle and Li, Netflix's artwork bandits).

## Result

<!-- card -->
A logistic ranker predicting play, finish, and replay, share or challenge scores 0.5209 NDCG@5 on 604 validation sessions against 0.4032 for popularity: +0.1178, 95% range +0.0921 to +0.1446. `rankings_test.csv` comes from the same model refit on train and validation events, with nothing tuned on test.

## The biggest caveat

Cold users (2 sessions or fewer) gain +0.0400 with a range that includes 0, the starting guess for new games gives no gain, and test has about twice validation's share of slate rows from brand-new games. An online test of my primary metric, satisfying plays per 10 impressions, can show whether fit matters.
