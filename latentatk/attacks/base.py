"""Attacker interface and result container."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils import CostMeter


@dataclass
class AttackResult:
    """Output of an attack: the injected fake profiles and the cost of producing them.

    profiles : list[list[int]]
        One item-id list per injected fake user (length <= ``m``).
    cost : CostMeter
        Calls / tokens / victim queries spent to produce the plan.
    meta : dict
        Optional diagnostics (role assignment counts, discovered bridges, ...).
    """

    profiles: list
    cost: CostMeter = field(default_factory=CostMeter)
    meta: dict = field(default_factory=dict)


class Attacker:
    """Black-box promotion attacker.

    Parameters
    ----------
    budget : int
        Number of fake users to inject (``B`` in the paper).
    m : int
        Per-user interaction budget.
    seed : int
    """

    name = "attacker"

    def __init__(self, budget: int = 50, m: int = 30, seed: int = 0):
        self.budget, self.m, self.seed = budget, m, seed

    def attack(self, dataset, target: int, probe=None) -> AttackResult:
        """Produce fake profiles that promote ``target``.

        ``probe`` is a victim trained on clean data that the attacker may query
        (black-box). Returns an :class:`AttackResult`.
        """
        raise NotImplementedError
