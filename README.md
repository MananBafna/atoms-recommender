# Game recommendations for Atoms

My take-home submission for a recommender for Atoms, a swipe feed of short competitive mental games.

**Live demo:** https://atoms-ranking-replay.vercel.app

On validation the ranker scores 0.521 NDCG@5 against 0.403 for popularity (+0.118, 95% range +0.092 to +0.145, 604 sessions).

## What is here

- `submission/`: summary and report (PDF and Markdown), architecture diagram, `rankings_test.csv`.
- `prototype/`: the ranker in Python (numpy and pandas), its tests and output numbers. Its README explains the method.
- `demo/`: the page deployed above.
- `research/`: notes on other recommenders and the data analysis the report cites.

## Rerun

The Atoms dataset is not included. Copy the provided `data/` folder into the repository root, then:

    cd prototype
    python run.py
    python -m unittest discover -s tests
