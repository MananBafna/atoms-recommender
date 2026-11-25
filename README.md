# Game recommendations for Atoms

This is my take-home submission for a recommender for Atoms, a swipe feed of short competitive mental games. The repository holds the written submission, the prototype ranker that produces every number in it, a small demo and the analysis the report cites.

On the validation sessions the ranker scores 0.521 NDCG@5, against 0.403 for the popularity baseline (a gain of +0.118, with a 95% range of +0.092 to +0.145 over 604 sessions).

## What is here

- `submission/` holds the summary and the report as PDF and Markdown, the architecture diagram, and `rankings_test.csv` with the ranked test slates.
- `prototype/` holds the Python code, which uses numpy and pandas only. `run.py` builds the features, picks and fits the model, writes the validation and test rankings, and writes `out/metrics.json`, which holds every number the report quotes. `tests/` holds the unit tests. `prototype/README.md` explains the method and the rules that keep validation and test data out of training.
- `demo/` holds a static page that replays 40 validation sessions and shows the ranker's order next to the popularity baseline, with the reasons behind each placement.
- `research/` holds my notes on other recommenders and the data analysis the report cites.

## Data

The dataset and the brief belong to Atoms and are not included here. To rerun anything, copy the provided `data/` folder into the repository root, so that `data/train_impressions.csv` and the rest sit next to `prototype/`.

## How to reproduce

    cd prototype
    python run.py
    python -m unittest discover -s tests

`run.py` takes about 35 seconds and rewrites `prototype/out/` and `submission/rankings_test.csv`. The run is deterministic, so the files come out the same each time. To score the validation rankings with the provided script:

    python ../data/evaluate.py --split val --rankings out/rankings_val.csv

## How to open the demo

The demo needs no data folder. From the repository root, start the small static server and open the page:

    node serve.mjs

Then go to http://localhost:3000/demo/
