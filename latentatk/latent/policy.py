from __future__ import annotations

import numpy as np

from ..attacks.group import ROLE_FN


class LatentRolePolicy:
    def __init__(self, mix: dict):
        self.mix = mix
        self.roles_per_worker: list[str] = []
        self.last_role_counts: dict = {}

    def _assign_roles(self, budget, rng):
        roles = []
        for r, frac in self.mix.items():
            roles += [r] * int(round(frac * budget))
        while len(roles) < budget:
            roles.append("SN")
        roles = roles[:budget]
        rng.shuffle(roles)
        self.roles_per_worker = roles
        self.last_role_counts = {r: roles.count(r) for r in ("PR", "CA", "SN", "IN")}
        return roles

    def plan(self, ctx, budget, m, rng):
        roles = self._assign_roles(budget, rng)
        return [ROLE_FN[role](ctx, m, rng) for role in roles]

    def refine(self, profiles, ctx, m, rng):
        target = ctx.target
        out = []
        for role, prof in zip(self.roles_per_worker, profiles):
            if role == "SN":
                fresh = ROLE_FN["SN"](ctx, m, rng)
                merged = [target] + [x for x in fresh if x != target] \
                    + [x for x in prof if x != target]
                seen, dedup = set(), []
                for it in merged:
                    if it not in seen:
                        seen.add(it); dedup.append(it)
                    if len(dedup) >= m:
                        break
                out.append(dedup)
            else:
                out.append(prof)
        return out
