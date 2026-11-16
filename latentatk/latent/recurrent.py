"""Budget-conditioned recurrent bridge policy (elastic-depth latent deliberation).

A small shared block is applied ``M`` times to refine a hidden state ``h`` that reads
out the bridge-scoring weights ``w``; the per-victim budget ``M`` is set by a difficulty
heuristic (no extra RL). Two variants let us test the LoopFormer collapse claim:

* ``elastic`` -- residual, **step-conditioned** recurrence ``h_{k+1}=h_k+tanh(W_h h_k +
  W_t e_k)`` trained with a **shortcut-consistency** term tying a short-budget readout to
  the full-depth one. Keeps refining and conditions on the victim.
* ``naive``   -- plain contraction ``h_{k+1}=tanh(W_h h_k)`` with no step conditioning and
  no consistency. Collapses to a victim-agnostic fixed point.

Both are trained by REINFORCE on the **real** post-retraining HR (no surrogate); only the
*depth mechanism* differs. Backprop through the recurrence is plain BPTT (manual).
"""
from __future__ import annotations

import numpy as np

from ..attacks.base import Attacker, AttackResult
from ..attacks.group import GroupContext, discover_receptive, _dedup
from ..latent.amortized import item_features, _build_plan
from ..latent.policy import LatentRolePolicy
from ..utils import CostMeter

N_FEAT = 5        # item-feature weights produced by the readout
S_DIM = 3         # step-embedding dim
DEFAULT_MIX = {"PR": 0.15, "CA": 0.10, "SN": 0.70, "IN": 0.05}


def step_emb(k, Lmax):
    t = k / max(1, Lmax)
    return np.array([t, np.sin(np.pi * t), np.cos(np.pi * t)], float)


class RecurrentBridgePolicy:
    def __init__(self, H=8, Lmax=6, elastic=True, seed=0):
        rng = np.random.default_rng(seed)
        self.H, self.Lmax, self.elastic = H, Lmax, elastic
        sc = 0.3
        self.We = rng.normal(0, sc, (H, N_FEAT))      # encode victim summary s (dim N_FEAT)
        self.Wh = rng.normal(0, sc, (H, H))
        self.Wt = rng.normal(0, sc, (H, S_DIM))
        # readout init near the 'support' strategy (warm start): w = U h
        self.U = rng.normal(0, 0.05, (N_FEAT, H))
        self.params = ["We", "Wh", "Wt", "U"]

    def forward(self, s, M):
        h = np.tanh(self.We @ s)
        hs = [h]
        zs = []
        for k in range(M):
            if self.elastic:
                z = self.Wh @ h + self.Wt @ step_emb(k, self.Lmax)
                h = h + np.tanh(z)
            else:
                z = self.Wh @ h
                h = np.tanh(z)
            zs.append(z); hs.append(h)
        w = self.U @ h
        return w, {"s": s, "hs": hs, "zs": zs, "M": M}

    def backward(self, cache, gw):
        """Backprop dL/dw -> param grads (BPTT). gw = dL/dw (shape N_FEAT)."""
        s, hs, zs, M = cache["s"], cache["hs"], cache["zs"], cache["M"]
        g = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        g["U"] += np.outer(gw, hs[-1])
        gh = self.U.T @ gw
        for k in reversed(range(M)):
            dtanh = 1.0 - np.tanh(zs[k]) ** 2
            if self.elastic:
                gt = dtanh * gh
                g["Wh"] += np.outer(gt, hs[k])
                g["Wt"] += np.outer(gt, step_emb(k, self.Lmax))
                gh = gh + self.Wh.T @ gt        # residual identity + through block
            else:
                gt = dtanh * gh
                g["Wh"] += np.outer(gt, hs[k])
                gh = self.Wh.T @ gt
        dt0 = 1.0 - np.tanh(self.We @ s) ** 2
        g["We"] += np.outer(dt0 * gh, s)
        return g

    def step(self, grads, lr):
        flat = np.concatenate([grads[p].ravel() for p in self.params])
        norm = np.linalg.norm(flat) + 1e-9
        for p in self.params:
            setattr(self, p, getattr(self, p) + lr * grads[p] / norm)  # fixed-norm ascent

    # --- sampling / scoring over the candidate pool ---
    @staticmethod
    def sample_bridges(w, F, pool, n_bridges, rng, tau=1.0):
        s = (F[pool] @ w) / tau
        p = np.exp(s - s.max()); p /= p.sum()
        k = min(n_bridges, len(pool))
        draws = rng.choice(len(pool), size=k, replace=True, p=p)
        Ephi = (p[:, None] * F[pool]).sum(0)
        glogp_w = (F[pool][draws] - Ephi).sum(0)      # dlogp/dw
        return pool[np.unique(draws)], glogp_w

    @staticmethod
    def top_bridges(w, F, pool, n_bridges):
        order = pool[np.argsort(-(F[pool] @ w))]
        return order[:min(n_bridges, len(pool))]

    def weights_trace(self, s, M):
        """Readout weights at every loop step (for the collapse diagnostic)."""
        _, cache = self.forward(s, M)
        return [self.U @ h for h in cache["hs"]]


def victim_summary(dataset, target, probe, counts, receptive, broad, cost=None):
    """Black-box difficulty summary s (dim N_FEAT) from queries only."""
    ranks = []
    for u in receptive[:40]:
        sc = probe.full_scores(int(u))
        ranks.append((sc > sc[target]).mean())     # fraction of items above target
        if cost is not None:
            cost.add_query(1)
    difficulty = float(np.mean(ranks)) if ranks else 0.5
    density = dataset.R.nnz / (dataset.n_users * dataset.n_items)
    recfreq = float(np.mean(np.maximum(counts, 0)) / (np.max(counts) + 1e-9))
    return np.array([difficulty, density * 50, recfreq, 0.0, 1.0], float), difficulty


def budget_from_difficulty(difficulty, Lmax=6):
    return int(np.clip(round(1 + difficulty * (Lmax - 1)), 1, Lmax))


class ElasticDepthAtk(Attacker):
    name = "ElasticDepthAtk"
    TOK_PLAN, TOK_REFINE = 8400, 1200

    def __init__(self, budget=50, m=30, seed=0, policy=None, mode="elastic",
                 fixed_M=None, n_bridges=60, role_mix=None):
        super().__init__(budget, m, seed)
        self.policy = policy or RecurrentBridgePolicy(elastic=(mode != "naive"), seed=seed)
        self.mode = mode                 # 'elastic' | 'naive' | 'noloop'
        self.fixed_M = fixed_M
        self.n_bridges = n_bridges
        self.MIX = role_mix or dict(DEFAULT_MIX)

    def _budget(self, difficulty):
        if self.mode == "noloop":
            return 1
        if self.fixed_M is not None:
            return self.fixed_M
        return budget_from_difficulty(difficulty, self.policy.Lmax)

    def _plan_shape(self, M):
        """Deeper deliberation refines the plan: broader bridges + more camouflage,
        which lowers cross-fake overlap (harder to filter). Naive/no-loop do not refine."""
        if self.mode == "elastic":
            n_bridges = 40 + 8 * (M - 1)                 # broaden with depth
            sn = max(0.45, 0.70 - 0.04 * (M - 1))
            ca = min(0.35, 0.10 + 0.045 * (M - 1))
            inact = min(0.18, 0.05 + 0.02 * (M - 1))
            mix = {"PR": max(0.0, 1 - sn - ca - inact), "CA": ca, "SN": sn, "IN": inact}
            return n_bridges, mix
        return 44, dict(DEFAULT_MIX)                     # collapsed / shallow: fixed shape

    def attack(self, dataset, target, probe=None):
        rng = np.random.default_rng(self.seed)
        cost = CostMeter()
        receptive, counts = discover_receptive(dataset, target, probe, cost, rng, n_receptive=80)
        broad = rng.choice(dataset.n_genuine_users, size=min(120, dataset.n_genuine_users), replace=False)
        s, diff = victim_summary(dataset, target, probe, counts, receptive, broad, cost=cost)
        F = item_features(dataset, target, probe, counts, receptive, broad, cost)
        pool = np.where(counts > 0)[0]
        M = self._budget(diff)
        w, _ = self.policy.forward(s, M)
        n_bridges, mix = self._plan_shape(M)
        bridges = self.policy.top_bridges(w, F, pool, n_bridges)
        pol = LatentRolePolicy(mix)
        profiles = pol.plan(GroupContext(bridges=bridges, fillers=_fillers(dataset, target),
                                          target=int(target)), self.budget, self.m, rng)
        cost.add_call(self.TOK_PLAN, note=f"elastic-depth M={M}")
        out = [_dedup(p, self.m) for p in profiles]
        return AttackResult(profiles=out, cost=cost, meta={"M": M, "difficulty": round(diff, 3)})


def _fillers(dataset, target, n=80):
    pop = dataset.popularity.copy().astype(float); pop[target] = -1
    return np.argsort(-pop)[:n]
