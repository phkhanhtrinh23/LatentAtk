from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .utils import sigmoid


class Recommender:
    n_users: int
    n_items: int

    def fit(self, dataset) -> "Recommender":
        raise NotImplementedError

    def full_scores(self, u: int) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, u: int, k: int, exclude_seen=None) -> np.ndarray:
        s = self.full_scores(u)
        if exclude_seen is not None:
            s = s.copy()
            s[exclude_seen] = -np.inf
        return np.argpartition(-s, k)[:k][np.argsort(-s[np.argpartition(-s, k)[:k]])]


class MF(Recommender):
    def __init__(self, dim=64, lr=0.05, reg=0.02, epochs=60, seed=0, noise=0.0):
        self.dim, self.lr, self.reg, self.epochs, self.seed = dim, lr, reg, epochs, seed
        self.noise = noise

    def fit(self, dataset):
        rng = np.random.default_rng(self.seed)
        self.n_users, self.n_items = dataset.n_users, dataset.n_items
        self.P = rng.normal(0, 0.1, (self.n_users, self.dim)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (self.n_items, self.dim)).astype(np.float32)
        R = dataset.R.tocoo()
        users, items = R.row, R.col
        pos_sets = [set(dataset.user_items(u).tolist()) for u in range(self.n_users)]
        n = len(users)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            U, I = users[order], items[order]
            J = rng.integers(0, self.n_items, size=n)
            # Resample sampled negatives that are actually positives.
            for idx in range(n):
                while J[idx] in pos_sets[U[idx]]:
                    J[idx] = rng.integers(0, self.n_items)
            x = np.sum(self.P[U] * (self.Q[I] - self.Q[J]), axis=1)
            g = sigmoid(-x)[:, None]
            dP = g * (self.Q[I] - self.Q[J]) - self.reg * self.P[U]
            dQI = g * self.P[U] - self.reg * self.Q[I]
            dQJ = -g * self.P[U] - self.reg * self.Q[J]
            np.add.at(self.P, U, self.lr * dP)
            np.add.at(self.Q, I, self.lr * dQI)
            np.add.at(self.Q, J, self.lr * dQJ)
            if self.noise > 0:
                self.P += rng.normal(0, self.noise, self.P.shape).astype(np.float32)
                self.Q += rng.normal(0, self.noise, self.Q.shape).astype(np.float32)
        return self

    def full_scores(self, u: int) -> np.ndarray:
        return self.Q @ self.P[u]


class FedMF(MF):
    def __init__(self, dim=64, lr=0.05, reg=0.02, epochs=60, seed=0, noise=0.015):
        super().__init__(dim, lr, reg, epochs, seed, noise=noise)


class GMF(Recommender):
    def __init__(self, dim=32, lr=0.05, reg=0.01, epochs=45, seed=0):
        self.dim, self.lr, self.reg, self.epochs, self.seed = dim, lr, reg, epochs, seed

    def fit(self, dataset):
        rng = np.random.default_rng(self.seed)
        self.n_users, self.n_items = dataset.n_users, dataset.n_items
        self.P = rng.normal(0, 0.1, (self.n_users, self.dim)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (self.n_items, self.dim)).astype(np.float32)
        self.w = np.full(self.dim, 0.3, np.float32)
        R = dataset.R.tocoo()
        users, items = R.row, R.col
        pos_sets = [set(dataset.user_items(u).tolist()) for u in range(self.n_users)]
        n = len(users)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            U, I = users[order], items[order]
            J = rng.integers(0, self.n_items, size=n)
            for idx in range(n):
                while J[idx] in pos_sets[U[idx]]:
                    J[idx] = rng.integers(0, self.n_items)
            phi_i = self.P[U] * self.Q[I]
            phi_j = self.P[U] * self.Q[J]
            x = (phi_i - phi_j) @ self.w
            g = sigmoid(-x)[:, None]
            self.w += self.lr * ((g * (phi_i - phi_j)).mean(0) - self.reg * self.w)
            dP = g * self.w * (self.Q[I] - self.Q[J]) - self.reg * self.P[U]
            dQI = g * self.w * self.P[U] - self.reg * self.Q[I]
            dQJ = -g * self.w * self.P[U] - self.reg * self.Q[J]
            np.add.at(self.P, U, self.lr * dP)
            np.add.at(self.Q, I, self.lr * dQI)
            np.add.at(self.Q, J, self.lr * dQJ)
        return self

    def full_scores(self, u: int) -> np.ndarray:
        return (self.P[u] * self.Q) @ self.w


class ItemKNN(Recommender):
    def __init__(self, shrink=10.0, seed=0):
        self.shrink, self.seed = shrink, seed

    def fit(self, dataset):
        self.n_users, self.n_items = dataset.n_users, dataset.n_items
        self.Rcsr = dataset.R.tocsr().astype(np.float32)
        G = (self.Rcsr.T @ self.Rcsr).toarray().astype(np.float32)
        norm = np.sqrt(np.diag(G)) + 1e-8
        sim = G / (norm[:, None] * norm[None, :] + self.shrink)
        np.fill_diagonal(sim, 0.0)
        self.sim = sim
        return self

    def full_scores(self, u: int) -> np.ndarray:
        items = self.Rcsr[u].indices
        if len(items) == 0:
            return np.zeros(self.n_items, np.float32)
        return self.sim[items].sum(0)


class LightGCN(Recommender):
    def __init__(self, dim=32, lr=0.05, reg=0.01, epochs=40, layers=2, seed=0):
        self.dim, self.lr, self.reg = dim, lr, reg
        self.epochs, self.layers, self.seed = epochs, layers, seed

    def _build_norm_adj(self, R):
        n_u, n_i = R.shape
        top = sp.hstack([sp.csr_matrix((n_u, n_u)), R])
        bot = sp.hstack([R.T, sp.csr_matrix((n_i, n_i))])
        A = sp.vstack([top, bot]).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()
        dinv = np.zeros_like(deg, dtype=float)
        np.power(deg, -0.5, where=deg > 0, out=dinv)
        D = sp.diags(dinv)
        return (D @ A @ D).tocsr()

    def _propagate(self, X):
        acc, cur = X.copy(), X
        for _ in range(self.layers):
            cur = self.Ahat @ cur
            acc = acc + cur
        return acc / (self.layers + 1)

    def fit(self, dataset):
        rng = np.random.default_rng(self.seed)
        self.n_users, self.n_items = dataset.n_users, dataset.n_items
        self.Ahat = self._build_norm_adj(dataset.R)
        E = rng.normal(0, 0.1, (self.n_users + self.n_items, self.dim)).astype(np.float32)
        R = dataset.R.tocoo()
        users, items = R.row, R.col
        pos_sets = [set(dataset.user_items(u).tolist()) for u in range(self.n_users)]
        n = len(users)
        for _ in range(self.epochs):
            Z = self._propagate(E)
            gZ = np.zeros_like(Z)
            order = rng.permutation(n)
            U, I = users[order], items[order]
            J = rng.integers(0, self.n_items, size=n)
            for idx in range(n):
                while J[idx] in pos_sets[U[idx]]:
                    J[idx] = rng.integers(0, self.n_items)
            ui, ii, ij = U, self.n_users + I, self.n_users + J
            x = np.sum(Z[ui] * (Z[ii] - Z[ij]), axis=1)
            g = sigmoid(-x)[:, None]
            np.add.at(gZ, ui, g * (Z[ii] - Z[ij]))
            np.add.at(gZ, ii, g * Z[ui])
            np.add.at(gZ, ij, -g * Z[ui])
            gE = self._propagate(gZ) - self.reg * E
            E = E + self.lr / max(1, n // 1000) * gE
        self.Z = self._propagate(E)
        return self

    def full_scores(self, u: int) -> np.ndarray:
        return self.Z[self.n_users:] @ self.Z[u]


VICTIMS = {"bpr-mf": MF, "mf": MF, "lightgcn": LightGCN, "gmf": GMF,
           "itemknn": ItemKNN, "fedmf": FedMF}

FAMILIES = {"mf": "factorization", "lightgcn": "graph", "gmf": "neural",
            "itemknn": "neighborhood", "fedmf": "federated"}


def build_victim(name: str, **kw) -> Recommender:
    name = name.lower()
    if name not in VICTIMS:
        raise ValueError(f"unknown victim '{name}', choose from {list(VICTIMS)}")
    return VICTIMS[name](**kw)
