"""Seeding, cost accounting, and small helpers."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class CostMeter:
    """Tracks the inference cost of producing an attack plan.

    Mirrors the paper's cost model (Sec. Threat Model / RQ4 Efficiency):
      - ``llm_calls``      : number of agent (LLM) forward passes,
      - ``tokens``         : generated (output) tokens,
      - ``victim_queries`` : black-box queries to the victim recommender.

    The role-switching teacher pays ``O(A*R)`` calls; LatentAtk pays ``O(1+k)``.
    """

    llm_calls: int = 0
    tokens: int = 0
    victim_queries: int = 0
    _log: list = field(default_factory=list)

    def add_call(self, tokens: int = 0, note: str = "") -> None:
        self.llm_calls += 1
        self.tokens += int(tokens)
        if note:
            self._log.append(note)

    def add_query(self, n: int = 1) -> None:
        self.victim_queries += int(n)

    def as_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "victim_queries": self.victim_queries,
        }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
