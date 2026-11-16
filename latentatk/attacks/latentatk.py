"""Role-based profile construction with optional latent-plan refinement."""
from __future__ import annotations

import numpy as np

from .base import Attacker, AttackResult
from .group import discover_context, _dedup
from ..latent.policy import LatentRolePolicy
from ..utils import CostMeter

TOK_PLAN = 8400       # output tokens for the single latent planning pass
TOK_REFINE = 1200     # output tokens per refinement pass


class LatentAtk(Attacker):
    name = "LatentAtk (ours)"

    # internalized role mixture (the "latent agent subspaces" exercised in one pass)
    DEFAULT_MIX = {"PR": 0.15, "CA": 0.10, "SN": 0.70, "IN": 0.05}

    def __init__(self, budget=50, m=30, seed=0, k_refine=2, role_mix=None,
                 refine_eps=0.05, stop_mode="none"):
        super().__init__(budget, m, seed)
        self.k_refine = k_refine
        self.refine_eps = refine_eps      # magnitude threshold (fixedpoint mode)
        self.stop_mode = stop_mode        # none (default) | fixedpoint | cos | erank
        self.MIX = role_mix or dict(self.DEFAULT_MIX)

    def _should_stop(self, shift, cos_theta, eranks):
        """Return True when the current pass looks non-productive under ``stop_mode``."""
        if self.stop_mode == "none":
            return False
        if self.stop_mode == "fixedpoint":
            return shift < self.refine_eps                    # plan stopped moving
        if self.stop_mode == "cos":                           # direction reversal
            return cos_theta is not None and cos_theta < 0.0
        if self.stop_mode == "erank":                         # diversity narrowing
            return len(eranks) >= 2 and eranks[-1] < eranks[-2]
        raise ValueError(f"unknown stop_mode {self.stop_mode!r}")

    def attack(self, dataset, target, probe=None):
        rng = np.random.default_rng(self.seed)
        cost = CostMeter()
        assert probe is not None, "LatentAtk needs black-box probe access"
        # one context discovery (fewer queries than the iterative teacher)
        ctx = discover_context(dataset, target, probe, cost, rng,
                               n_receptive=80, n_bridges=60)
        policy = LatentRolePolicy(self.MIX)

        # single latent planning pass: assign roles + fill all profiles at once
        profiles = policy.plan(ctx, self.budget, self.m, rng)
        cost.add_call(TOK_PLAN, note="latent planning pass")

        # k refinement passes with a productivity stop: re-read feedback and re-snipe,
        # but halt once the pass stops being productive under the selected signal.
        n_items = dataset.n_items
        hists = [_plan_hist(profiles, n_items)]                     # v0: initial plan
        eranks = [_effective_rank(_plan_matrix(profiles, n_items))]
        plan_shifts, cos_thetas, k_used = [], [], 0
        for j in range(self.k_refine):
            ctx = discover_context(dataset, target, probe, cost, rng,
                                   n_receptive=80, n_bridges=60)
            prev = profiles
            profiles = policy.refine(prev, ctx, self.m, rng)
            cost.add_call(TOK_REFINE, note=f"refinement pass {j}")
            k_used += 1
            plan_shifts.append(_plan_shift(prev, profiles))        # magnitude lens
            hists.append(_plan_hist(profiles, n_items))
            eranks.append(_effective_rank(_plan_matrix(profiles, n_items)))
            ct = _cos_theta(hists) if len(hists) >= 3 else None    # direction lens
            cos_thetas.append(ct)
            if self._should_stop(plan_shifts[-1], ct, eranks):
                break

        out = [_dedup(p, self.m) for p in profiles]
        return AttackResult(profiles=out, cost=cost,
                            meta={"roles": policy.last_role_counts,
                                  "k_refine": self.k_refine, "k_used": k_used,
                                  "stop_mode": self.stop_mode,
                                  "plan_shifts": plan_shifts,
                                  "cos_thetas": cos_thetas,
                                  "eranks": eranks})


def _plan_shift(prev, cur):
    """Fixed-point gap (magnitude lens): mean per-worker Jaccard distance between the
    profiles before and after a refinement pass -- the plan analogue of ``||h - f(h)||``.
    0 means the pass changed nothing (a fixed point); ~1 means the plan moved a lot."""
    dists = []
    for a, b in zip(prev, cur):
        sa, sb = set(a), set(b)
        union = sa | sb
        if not union:
            continue
        dists.append(1.0 - len(sa & sb) / len(union))
    return float(np.mean(dists)) if dists else 0.0


def _plan_hist(profiles, n_items):
    """Injected-item frequency histogram: the plan's state vector for the direction
    lens (how many fake users push each item)."""
    v = np.zeros(n_items, dtype=float)
    for p in profiles:
        for it in p:
            if 0 <= it < n_items:
                v[it] += 1.0
    return v


def _plan_matrix(profiles, n_items):
    """Binary budget x item injection matrix, for the effective-rank (diversity) lens."""
    mat = np.zeros((len(profiles), n_items), dtype=float)
    for i, p in enumerate(profiles):
        for it in p:
            if 0 <= it < n_items:
                mat[i, it] = 1.0
    return mat


def _effective_rank(mat):
    """Scale-free effective rank (paper Eq. 4): entropy of the normalized singular-value
    spectrum. Measures how diverse the injected profiles are; a falling value across
    passes signals representational collapse toward redundant fakes."""
    s = np.linalg.svd(mat, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def _cos_theta(hists):
    """Angular change (paper Eq. 3): alignment of the two most recent plan-update
    vectors. cos < 0 => the latest pass reverses the previous one (oscillatory drift)."""
    d1 = hists[-2] - hists[-3]
    d2 = hists[-1] - hists[-2]
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(d1, d2) / (n1 * n2))
