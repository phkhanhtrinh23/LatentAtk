from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .utils import sigmoid


@dataclass
class Dataset:
    R: sp.csr_matrix
    test: dict
    item_genre: np.ndarray | None = None
    n_genuine_users: int | None = None

    def __post_init__(self):
        self.R = self.R.tocsr()
        if self.n_genuine_users is None:
            self.n_genuine_users = self.R.shape[0]
        self._pop = np.asarray(self.R.sum(axis=0)).ravel()

    @property
    def n_users(self) -> int:
        return self.R.shape[0]

    @property
    def n_items(self) -> int:
        return self.R.shape[1]

    @property
    def popularity(self) -> np.ndarray:
        return self._pop

    def user_items(self, u: int) -> np.ndarray:
        return self.R.indices[self.R.indptr[u]:self.R.indptr[u + 1]]

    def add_fake_users(self, profiles: list[list[int]]) -> "Dataset":
        rows, cols = [], []
        for r, items in enumerate(profiles):
            for it in items:
                rows.append(self.n_users + r)
                cols.append(int(it))
        fake = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(self.n_users + len(profiles), self.n_items),
        )
        full = sp.vstack([self.R, sp.csr_matrix((len(profiles), self.n_items))]).tolil()
        full[self.n_users:, :] = fake[self.n_users:, :]
        return Dataset(full.tocsr(), dict(self.test), self.item_genre, self.n_genuine_users)


def make_synthetic(
    n_users: int = 300,
    n_items: int = 600,
    n_genres: int = 8,
    density: float = 0.03,
    seed: int = 0,
    likes_per_user: int = 2,
) -> Dataset:
    rng = np.random.default_rng(seed)
    item_genre = rng.integers(0, n_genres, size=n_items)
    item_pop = rng.normal(0.0, 1.0, size=n_items)

    aff = np.full((n_users, n_genres), -5.0)
    for u in range(n_users):
        liked = rng.choice(n_genres, size=likes_per_user, replace=False)
        aff[u, liked] = rng.uniform(2.0, 3.5, size=likes_per_user)

    # Solve a global bias so the realized density matches the requested value.
    base_logit = aff[np.arange(n_users)[:, None], item_genre[None, :]] + 0.8 * item_pop[None, :]
    bias = _solve_bias(base_logit, density, rng)

    rows, cols = [], []
    for u in range(n_users):
        p = sigmoid(base_logit[u] + bias + rng.normal(0, 0.25, n_items))
        drawn = np.where(rng.random(n_items) < p)[0]
        if len(drawn) < 8:
            drawn = np.argsort(-p)[:8]
        rows.extend([u] * len(drawn))
        cols.extend(drawn.tolist())

    R = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_users, n_items))
    R = (R > 0).astype(np.float32)
    R, test = _leave_one_out(R, rng)
    return Dataset(R, test, item_genre, n_users)


def _solve_bias(base_logit, density, rng, iters=25):
    lo, hi = -12.0, 6.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        d = float(sigmoid(base_logit + mid).mean())
        if d > density:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _leave_one_out(R: sp.csr_matrix, rng) -> tuple[sp.csr_matrix, dict]:
    train = R.tolil(copy=True)
    test = {}
    for u in range(train.shape[0]):
        items = list(train.rows[u])
        if len(items) >= 2:
            held = int(rng.choice(items))
            test[u] = held
            train[u, held] = 0
    train = train.tocsr()
    train.eliminate_zeros()
    return train, test


def load_movielens(path: str, min_rating: float = 4.0, seed: int = 0) -> Dataset:
    import csv

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MovieLens ratings file not found: {path}")
    u_map, i_map, pairs = {}, {}, []
    with open(p, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if float(row[2]) < min_rating:
                continue
            u = u_map.setdefault(row[0], len(u_map))
            it = i_map.setdefault(row[1], len(i_map))
            pairs.append((u, it))
    n_u, n_i = len(u_map), len(i_map)
    rows = np.array([p[0] for p in pairs])
    cols = np.array([p[1] for p in pairs])
    R = sp.csr_matrix((np.ones(len(pairs), np.float32), (rows, cols)), shape=(n_u, n_i))
    R = (R > 0).astype(np.float32)
    R, test = _leave_one_out(R, np.random.default_rng(seed))
    return Dataset(R, test, None, n_u)
