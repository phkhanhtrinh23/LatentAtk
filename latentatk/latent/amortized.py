"""SGD-trained amortized attack policy (the paper's RL-internalized latent agent).

Instead of a fixed hand-set bridge rule, this module *learns* the bridge-selection
policy by **REINFORCE** (policy-gradient SGD) across many (victim, target) instances,
using the **real** post-retraining HR@10 as the reward -- no surrogate. The learned
weight vector ``w`` is shared across victims, so it *amortizes* the per-victim bridge
search into parameters: at deployment on a held-out victim, one forward pass scores the
candidate items and emits the plan, with no online search.

Policy. For a (victim, target) instance we score each candidate item ``i`` (items the
receptive neighborhood likes) by ``s_i = w . phi(i)`` over interpretable features
``phi`` = [receptive-support, log-popularity, embedding-similarity-to-target,
broad-affinity, bias]. Bridges are sampled from ``softmax(s/tau)`` during training
(for exploration) and taken greedily (top-n) at deployment.

Gradient. With draws ``D ~ Categorical(softmax(s))`` and reward ``R`` (real HR@10):
``grad_w log p(D) = sum_{d in D} (phi_d - E_p[phi])`` and ``w <- w + lr*(R-b)*grad`` with a
moving-average baseline ``b`` -- standard REINFORCE.
"""
from __future__ import annotations

import numpy as np

from ..attacks.base import Attacker, AttackResult
from ..attacks.group import GroupContext, discover_receptive, _dedup
from ..latent.policy import LatentRolePolicy
from ..utils import CostMeter

FEATURES = ("support", "logpop", "recfreq_recep", "recfreq_broad", "target_cooccurrence")
DEFAULT_MIX = {"PR": 0.15, "CA": 0.10, "SN": 0.70, "IN": 0.05}


def _z(x):
    s = x.std()
    return (x - x.mean()) / (s + 1e-9)


def _rec_frequency(probe, dataset, users, cost=None, k=50):
    """How often each item appears in the victim's top-k recommendations for ``users``.

    A purely **black-box** signal (only ``recommend`` queries) of how the victim
    associates items with a user group -- defined for *any* victim family, with or
    without embeddings.
    """
    freq = np.zeros(dataset.n_items)
    for u in users:
        seen = dataset.user_items(int(u))
        rec = probe.recommend(int(u), k, exclude_seen=seen)
        freq[rec] += 1.0
        if cost is not None:
            cost.add_query(1)
    return freq


def item_features(dataset, target, probe, counts, receptive_users, broad_users, cost=None):
    """Per-item **black-box** feature matrix ``[n_items, len(FEATURES)]`` (z-scored).

    All features come from public item statistics or victim *queries* only -- no access
    to the victim's parameters -- so the same policy transfers across victim families:

    * ``support``        -- receptive-neighborhood support (from target-rank queries),
    * ``logpop``         -- item interaction count (public metadata),
    * ``recfreq_recep``  -- item frequency in the victim's recommendations to receptive users,
    * ``recfreq_broad``  -- item frequency in the victim's recommendations to a broad sample,
    * ``target_cooccurrence`` -- genuine users who interacted with both the item and the
      target, normalized by the number of genuine users.
    """
    n = dataset.n_items
    support = np.maximum(counts, 0.0)
    logpop = np.log1p(dataset.popularity.astype(float))
    rf_recep = _rec_frequency(probe, dataset, receptive_users, cost)
    rf_broad = _rec_frequency(probe, dataset, broad_users, cost)
    users_with_target = dataset.R.getcol(int(target)).indices
    users_with_target = users_with_target[users_with_target < dataset.n_genuine_users]
    cooccurrence = np.zeros(n)
    if len(users_with_target):
        cooccurrence = np.asarray(dataset.R[users_with_target].sum(axis=0)).ravel() \
            / dataset.n_genuine_users
    return np.stack([_z(support), _z(logpop), _z(rf_recep), _z(rf_broad),
                     _z(cooccurrence)], axis=1)


class AmortizedBridgePolicy:
    """Learnable linear bridge-scoring policy, shared across victims."""

    def __init__(self, w=None):
        # init at the 'support' strategy (the strong hand baseline) for a warm start
        self.w = np.array([1.0, 0.0, 0.0, 0.0, 0.0]) if w is None else np.asarray(w, float)

    def _pool(self, counts):
        return np.where(counts > 0)[0]

    def scores(self, F, pool):
        return F[pool] @ self.w

    def sample_bridges(self, F, pool, n_bridges, rng, tau=1.0):
        """Sample a bridge set; return (bridges, grad_logp_accumulator over w)."""
        s = self.scores(F, pool) / tau
        p = np.exp(s - s.max()); p /= p.sum()
        k = min(n_bridges, len(pool))
        draws = rng.choice(len(pool), size=k, replace=True, p=p)
        Ephi = (p[:, None] * F[pool]).sum(0)
        grad = (F[pool][draws] - Ephi).sum(0)         # sum_d (phi_d - E_p[phi])
        bridges = pool[np.unique(draws)]
        return bridges, grad

    def top_bridges(self, F, pool, n_bridges):
        s = self.scores(F, pool)
        order = pool[np.argsort(-s)]
        return order[:min(n_bridges, len(pool))]


def _ctx(dataset, target, bridges, n_fillers=80):
    pop = dataset.popularity.copy().astype(float); pop[target] = -1
    fillers = np.argsort(-pop)[:n_fillers]
    return GroupContext(bridges=np.asarray(bridges), fillers=fillers, target=int(target))


def _build_plan(dataset, target, bridges, budget, m, rng, mix=None):
    pol = LatentRolePolicy(dict(mix or DEFAULT_MIX))
    profs = pol.plan(_ctx(dataset, target, bridges), budget, m, rng)
    return [_dedup(p, m) for p in profs]


def train_amortized(instances, reward_fn, steps=40, lr=0.4, batch=4,
                    n_bridges=60, tau=1.0, budget=50, m=30, seed=0, verbose=True):
    """REINFORCE training of the bridge policy across many victims.

    ``instances``  : list of dicts with keys dataset, target, probe, counts, eval_users.
    ``reward_fn``  : callable(instance, profiles) -> real HR@10 (retrains the victim).
    Returns the trained :class:`AmortizedBridgePolicy`.
    """
    rng = np.random.default_rng(seed)
    policy = AmortizedBridgePolicy()
    feats = [item_features(it["dataset"], it["target"], it["probe"], it["counts"],
                           it["receptive_users"], it["broad_users"]) for it in instances]
    pools = [np.where(it["counts"] > 0)[0] for it in instances]
    base = np.full(len(instances), np.nan)       # per-victim reward baseline (EMA)
    for step in range(steps):
        glogps, advs, rewards = [], [], []
        for _ in range(batch):
            j = rng.integers(len(instances))
            it, F, pool = instances[j], feats[j], pools[j]
            bridges, glogp = policy.sample_bridges(F, pool, n_bridges, rng, tau)
            profs = _build_plan(it["dataset"], it["target"], bridges, budget, m, rng)
            R = reward_fn(it, profs)
            if np.isnan(base[j]):
                base[j] = R
            # per-victim centered advantage (baseline) so heterogeneous families
            # (graph HR~900 vs MF HR~300) contribute comparably; magnitude handled
            # by the fixed-norm gradient step below.
            advs.append(R - base[j])
            base[j] = 0.85 * base[j] + 0.15 * R
            glogps.append(glogp); rewards.append(R)
        grad = sum(a * g for a, g in zip(advs, glogps))
        policy.w += lr * grad / (np.linalg.norm(grad) + 1e-9)   # fixed-norm step
        if verbose and (step % 5 == 0 or step == steps - 1):
            print(f"  step {step:>3}  mean HR@10={np.mean(rewards):7.1f}  "
                  f"w=[" + ", ".join(f"{x:+.2f}" for x in policy.w) + "]")
    return policy


class AmortizedLatentAtk(Attacker):
    """Deploy a trained :class:`AmortizedBridgePolicy` one-shot (amortized inference)."""

    name = "AmortizedLatentAtk (trained)"
    TOK_PLAN = 8400
    TOK_REFINE = 1200

    def __init__(self, budget=50, m=30, seed=0, k_refine=2, policy=None,
                 n_bridges=60, role_mix=None):
        super().__init__(budget, m, seed)
        self.k_refine = k_refine
        self.policy = policy or AmortizedBridgePolicy()
        self.n_bridges = n_bridges
        self.MIX = role_mix or dict(DEFAULT_MIX)

    def _score_ctx(self, dataset, target, probe, cost, rng):
        receptive, counts = discover_receptive(dataset, target, probe, cost, rng, n_receptive=80)
        broad = rng.choice(dataset.n_genuine_users,
                           size=min(120, dataset.n_genuine_users), replace=False)
        F = item_features(dataset, target, probe, counts, receptive, broad, cost)
        pool = np.where(counts > 0)[0]
        bridges = self.policy.top_bridges(F, pool, self.n_bridges)
        return _ctx(dataset, target, bridges)

    def attack(self, dataset, target, probe=None):
        rng = np.random.default_rng(self.seed)
        cost = CostMeter()
        assert probe is not None, "AmortizedLatentAtk needs black-box probe access"
        ctx = self._score_ctx(dataset, target, probe, cost, rng)
        pol = LatentRolePolicy(self.MIX)
        profiles = pol.plan(ctx, self.budget, self.m, rng)
        cost.add_call(self.TOK_PLAN, note="amortized planning pass")
        for j in range(self.k_refine):
            ctx = self._score_ctx(dataset, target, probe, cost, rng)
            profiles = pol.refine(profiles, ctx, self.m, rng)
            cost.add_call(self.TOK_REFINE, note=f"refinement pass {j}")
        out = [_dedup(p, self.m) for p in profiles]
        return AttackResult(profiles=out, cost=cost,
                            meta={"roles": pol.last_role_counts, "k_refine": self.k_refine,
                                  "w": [float(x) for x in self.policy.w]})
