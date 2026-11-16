"""Baseline attacks: NoneAttack, RandomAttack, BandwagonAttack, and a PGA-style
optimization attack.

These mirror the non-agentic baselines in the paper. They inject *independent* fake
users with no coordination among them.
"""
from __future__ import annotations

import numpy as np

from .base import Attacker, AttackResult
from ..utils import CostMeter


class NoneAttack(Attacker):
    """No injection (clean reference)."""

    name = "NoneAttack"

    def attack(self, dataset, target, probe=None):
        return AttackResult(profiles=[], cost=CostMeter())


class RandomAttack(Attacker):
    """Each fake user rates the target plus uniformly random filler items."""

    name = "RandomAttack"

    def attack(self, dataset, target, probe=None):
        rng = np.random.default_rng(self.seed)
        profiles = []
        for _ in range(self.budget):
            fillers = rng.choice(dataset.n_items, size=self.m - 1, replace=False)
            profiles.append([int(target)] + [int(x) for x in fillers if x != target])
        return AttackResult(profiles=profiles, cost=CostMeter())


class BandwagonAttack(Attacker):
    """Each fake user rates the target plus the globally most popular items."""

    name = "BandwagonAttack"

    def attack(self, dataset, target, probe=None):
        pop = dataset.popularity.copy()
        pop[target] = -1
        top = np.argsort(-pop)[: self.m - 1]
        profiles = [[int(target)] + [int(x) for x in top] for _ in range(self.budget)]
        return AttackResult(profiles=profiles, cost=CostMeter())


class PGAAttack(Attacker):
    """Projected-gradient-style poisoning (surrogate, simplified).

    Uses the clean probe victim's item embeddings as a surrogate to pick filler
    items whose latent factors are most aligned with the target, so that co-rating
    them with the target maximizes the surrogate's target score. Independent fake
    users (no coordination).
    """

    name = "PGA"

    def attack(self, dataset, target, probe=None):
        rng = np.random.default_rng(self.seed)
        cost = CostMeter()
        if probe is not None and hasattr(probe, "Q"):
            q = probe.Q
            sim = q @ q[target]
            sim[target] = -np.inf
            cand = np.argsort(-sim)[: 4 * self.m]
            cost.add_query(1)
        else:
            pop = dataset.popularity.copy(); pop[target] = -1
            cand = np.argsort(-pop)[: 4 * self.m]
        profiles = []
        for _ in range(self.budget):
            fillers = rng.choice(cand, size=min(self.m - 1, len(cand)), replace=False)
            profiles.append([int(target)] + [int(x) for x in fillers])
        return AttackResult(profiles=profiles, cost=cost)
