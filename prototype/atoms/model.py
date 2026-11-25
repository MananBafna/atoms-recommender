"""Standardisation, a deterministic logistic fit and the two rankers."""
from __future__ import annotations

import numpy as np


class Standardizer:
    """Column-wise z-score; zero-variance columns map to 0."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        self.zero_ = sd <= 1e-12
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        Z[:, self.zero_] = 0.0
        return Z


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float, w: np.ndarray | None = None,
                 max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Weighted L2 logistic regression by Newton/IRLS. Returns weights with the bias last.

    The penalty applies to the feature weights only, not the bias.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    sw = np.ones(n) if w is None else np.asarray(w, dtype=float)
    A = np.hstack([X, np.ones((n, 1))])
    pen = np.full(d + 1, float(l2))
    pen[-1] = 0.0
    beta = np.zeros(d + 1)
    for _ in range(max_iter):
        z = np.clip(A @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = A.T @ (sw * (p - y)) + pen * beta
        H = (A * (sw * p * (1 - p))[:, None]).T @ A + np.diag(pen) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        beta = beta - step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class PointwiseRanker:
    """Three cumulative heads (r>=1, r>=2, r=3); score = expected gain."""

    def __init__(self, l2: float):
        self.l2 = l2

    def fit(self, X, rel) -> "PointwiseRanker":
        rel = np.asarray(rel)
        self.std_ = Standardizer().fit(X)
        Z = self.std_.transform(X)
        self.heads_ = [fit_logistic(Z, (rel >= 1).astype(float), self.l2),
                       fit_logistic(Z, (rel >= 2).astype(float), self.l2),
                       fit_logistic(Z, (rel >= 3).astype(float), self.l2)]
        return self

    def score(self, X) -> np.ndarray:
        Z = self.std_.transform(X)
        p = [_sigmoid(Z @ h[:-1] + h[-1]) for h in self.heads_]
        return 1.0 * p[0] + 2.0 * p[1] + 4.0 * p[2]

    def contributions(self, X) -> np.ndarray:
        """Per-feature linear terms of the P(r>=2) head on standardised features."""
        return self.std_.transform(X) * self.heads_[1][:-1]

    def coefs(self) -> dict[str, np.ndarray]:
        return {"r>=1": self.heads_[0][:-1], "r>=2": self.heads_[1][:-1], "r=3": self.heads_[2][:-1]}


class PairwiseRanker:
    """Logistic model on within-session feature differences, weighted by gain difference."""

    def __init__(self, l2: float):
        self.l2 = l2

    def fit(self, X, rel, session_ids) -> "PairwiseRanker":
        X = np.asarray(X, dtype=float)
        rel = np.asarray(rel)
        sid = np.asarray(session_ids)
        self.std_ = Standardizer().fit(X)
        Z = self.std_.transform(X)
        gain = 2.0 ** rel - 1.0
        order = np.argsort(sid, kind="stable")
        s_sorted = sid[order]
        cuts = np.flatnonzero(s_sorted[1:] != s_sorted[:-1]) + 1
        di, dj = [], []
        for ix in np.split(order, cuts):
            r = rel[ix]
            a, b = np.nonzero(r[:, None] > r[None, :])
            di.append(ix[a])
            dj.append(ix[b])
        di, dj = np.concatenate(di), np.concatenate(dj)
        if len(di) == 0:
            self.w_ = np.zeros(X.shape[1])
            return self
        D = Z[di] - Z[dj]
        wt = np.abs(gain[di] - gain[dj])
        # mirrored pairs make the fit symmetric, so the bias is ~0 and dropped
        Xp = np.vstack([D, -D])
        yp = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
        self.w_ = fit_logistic(Xp, yp, self.l2, np.concatenate([wt, wt]))[:-1]
        return self

    def score(self, X) -> np.ndarray:
        return self.std_.transform(X) @ self.w_

    def contributions(self, X) -> np.ndarray:
        return self.std_.transform(X) * self.w_

    def coefs(self) -> dict[str, np.ndarray]:
        return {"pairwise": self.w_}
