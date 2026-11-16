"""Single-Agent (ReAct) baseline: one LLM agent, no roles, no latent post-training.

A single ReAct-style agent reasons over black-box victim feedback, discovers bridge
items in a *shallow* one-shot pass, and writes *homogeneous* fake profiles (target +
a few bridges + popular fillers). Unlike the role-switching teacher it has no role
specialization or Coordinator, and unlike LatentAtk it has not amortized a refined
role schedule during training. It therefore sits between the optimization baselines
and the multi-agent attackers on effectiveness, is highly detectable (every fake user
looks alike and rates the target), and costs a moderate amount of autoregressive text
reasoning.

This is the ``Single-Agent (ReAct)`` row of the paper's main table.
"""
from __future__ import annotations

import numpy as np

from .base import Attacker, AttackResult
from .group import discover_context, _take, _dedup
from ..utils import CostMeter


class SingleAgent(Attacker):
    name = "Single-Agent (ReAct)"

    def __init__(self, budget=50, m=30, seed=0, n_steps=14, tok_step=850):
        super().__init__(budget, m, seed)
        self.n_steps = n_steps      # ReAct reasoning iterations (text)
        self.tok_step = tok_step    # output tokens per reasoning step

    def attack(self, dataset, target, probe=None):
        rng = np.random.default_rng(self.seed)
        cost = CostMeter()
        assert probe is not None, "SingleAgent needs black-box probe access"

        # Shallow, single-pass bridge discovery: no role specialization, no
        # refinement -- one agent commits to a small, greedily chosen bridge set.
        ctx = discover_context(dataset, target, probe, cost, rng,
                               n_receptive=30, n_bridges=18, n_fillers=80)

        # ReAct reasoning loop, all in text -- the autoregressive cost a single
        # agent still pays even without inter-agent messages.
        for s in range(self.n_steps):
            cost.add_call(self.tok_step, note=f"react step {s}")

        # Homogeneous profiles: target + a few bridges + popular fillers. With no
        # role mix and a shallow bridge set, the target/bridge signal is weaker than
        # LatentAtk's refined, role-structured plan.
        n_bridge = max(1, int(round(0.30 * (self.m - 1))))
        profiles = []
        for _ in range(self.budget):
            bridges = [int(x) for x in _take(ctx.bridges, n_bridge, rng, weighted=True)]
            fillers = [int(x) for x in _take(ctx.fillers, self.m - 1 - len(bridges), rng)]
            profiles.append(_dedup([int(target)] + bridges + fillers, self.m))
        return AttackResult(profiles=profiles, cost=cost,
                            meta={"n_steps": self.n_steps})
