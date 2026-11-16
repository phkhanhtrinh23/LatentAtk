"""The role-switching group teacher (explicit multi-round debate).

A Coordinator drives a pool of ``B`` fake-user workers over ``R`` rounds. Each round
it reads black-box feedback, assigns a role to every worker (warm up with PR/CA,
then escalate to SN, always keeping some CA/IN for stealth), and the workers
incrementally build their profiles. Cost grows as ``O(A*R)`` LLM calls plus per-round
victim queries -- this is the expensive teacher that LatentAtk distills.
"""
from __future__ import annotations

import numpy as np

from .base import Attacker, AttackResult
from .group import discover_context, ROLE_FN, _dedup
from ..utils import CostMeter

TOK_WORKER = 700      # output tokens per worker (role) call
TOK_COORD = 1600      # output tokens per Coordinator call


class GroupTeacher(Attacker):
    name = "Multi-role teacher"

    def __init__(self, budget=50, m=30, seed=0, rounds=8, refine_every=3):
        super().__init__(budget, m, seed)
        self.rounds = rounds
        self.refine_every = refine_every

    def _role_mix(self, phase):
        """Coordinator strategy: fractions of (PR, CA, SN, IN) at campaign ``phase``."""
        if phase < 0.25:          # warm-up / victim probe
            sn = 0.20
        elif phase < 0.5:         # first push
            sn = 0.50
        else:                     # main attack
            sn = 0.78
        ca, inact = 0.12, 0.05
        pr = max(0.0, 1.0 - sn - ca - inact)
        return {"PR": pr, "CA": ca, "SN": sn, "IN": inact}

    def attack(self, dataset, target, probe=None):
        rng = np.random.default_rng(self.seed)
        cost = CostMeter()
        assert probe is not None, "GroupTeacher needs black-box probe access"
        ctx = discover_context(dataset, target, probe, cost, rng,
                               n_receptive=100, n_bridges=70)
        # ordered accumulation per worker, plus whether it ever played Sniper
        acc = [[] for _ in range(self.budget)]
        sniped = [False] * self.budget
        chunk = max(2, (2 * self.m) // self.rounds)
        role_counts = {"PR": 0, "CA": 0, "SN": 0, "IN": 0}

        for t in range(self.rounds):
            phase = t / max(1, self.rounds - 1)
            mix = self._role_mix(phase)
            cost.add_call(TOK_COORD, note=f"coordinator round {t}")    # Coordinator
            if t % self.refine_every == 0 and t > 0:                   # refine bridges
                ctx = discover_context(dataset, target, probe, cost, rng,
                                       n_receptive=100, n_bridges=70)
            roles = _assign(self.budget, mix, rng)
            for w, role in enumerate(roles):
                role_counts[role] += 1
                items = ROLE_FN[role](ctx, chunk, rng)
                if role == "SN":
                    sniped[w] = True
                acc[w].extend(int(x) for x in items)
                cost.add_call(TOK_WORKER, note=f"worker {w} role {role}")

        # finalize: snipers keep the target first, then the bridge items they
        # accumulated across rounds (the payload), then other filler. This prevents
        # warm-up filler from crowding out the payload when truncating to m.
        bridge_set = set(int(x) for x in ctx.bridges)
        out = []
        for w in range(self.budget):
            items = acc[w]
            if sniped[w]:
                bridges = [x for x in items if x in bridge_set and x != target]
                other = [x for x in items if x not in bridge_set and x != target]
                items = [target] + bridges + other
            out.append(_dedup(items, self.m))
        return AttackResult(profiles=out, cost=cost,
                            meta={"roles": role_counts, "rounds": self.rounds})


def _assign(n, mix, rng):
    roles = []
    for r, frac in mix.items():
        roles += [r] * int(round(frac * n))
    while len(roles) < n:
        roles.append("SN")
    roles = roles[:n]
    rng.shuffle(roles)
    return roles
