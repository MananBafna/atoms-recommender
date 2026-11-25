import unittest

import numpy as np

from atoms.model import PairwiseRanker, PointwiseRanker, Standardizer, fit_logistic


def synth(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    latent = X[:, 0] + 0.5 * rng.normal(size=n)
    rel = np.digitize(latent, [0.0, 0.8, 1.6])
    return X, rel, latent


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


class TestModel(unittest.TestCase):
    def test_logistic_recovers_sign(self):
        X, rel, _ = synth()
        w = fit_logistic(Standardizer().fit(X).transform(X), (rel >= 1).astype(float), l2=1.0)
        self.assertGreater(w[0], 0)
        self.assertGreater(w[0], abs(w[1]))

    def test_pointwise_orders(self):
        X, rel, _ = synth()
        m = PointwiseRanker(l2=1.0).fit(X, rel)
        self.assertGreater(spearman(m.score(X), X[:, 0]), 0.8)
        self.assertEqual(m.contributions(X).shape, X.shape)

    def test_pairwise_ignores_session_offsets(self):
        X, rel, _ = synth(600)
        sid = np.repeat(np.arange(50), 12)
        m = PairwiseRanker(l2=1.0).fit(X, rel, sid)
        off = np.random.default_rng(1).normal(size=(50, 1)) * 1000
        Xo = X + off[sid]
        for s in range(50):
            ix = sid == s
            self.assertTrue(np.array_equal(np.argsort(m.score(X[ix]), kind="stable"),
                                           np.argsort(m.score(Xo[ix]), kind="stable")))
        self.assertGreater(spearman(m.score(X), X[:, 0]), 0.8)
        self.assertEqual(m.contributions(X).shape, X.shape)

    def test_l2_shrinks(self):
        X, rel, _ = synth()
        Z = Standardizer().fit(X).transform(X)
        y = (rel >= 1).astype(float)
        self.assertLess(np.linalg.norm(fit_logistic(Z, y, 100.0)[:-1]),
                        np.linalg.norm(fit_logistic(Z, y, 0.01)[:-1]))

    def test_standardizer_zero_variance(self):
        X = np.column_stack([np.ones(5), np.arange(5.0)])
        Z = Standardizer().fit(X).transform(X)
        self.assertTrue(np.all(Z[:, 0] == 0))
        self.assertTrue(np.isfinite(Z).all())


if __name__ == "__main__":
    unittest.main()
