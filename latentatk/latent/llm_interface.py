"""Interface for backbones that produce and refine interaction profiles."""
from __future__ import annotations

from typing import Protocol


class Observation(Protocol):
    """What the latent agent sees: item metadata/statistics and probe feedback."""
    bridges: object
    fillers: object
    target: int


class LLMBackbone:
    """Interface a trained latent agent must satisfy.

    A conforming implementation (e.g. a LoRA-tuned 7B model decoded with Coconut-style
    continuous thoughts) should:
      * accept an :class:`Observation`,
      * internally roll out the PR/CA/SN/IN roles in latent space,
      * return ``budget`` item-id lists (one per fake user), each of length <= ``m``.
    """

    def plan(self, obs: Observation, budget: int, m: int) -> list[list[int]]:
        raise NotImplementedError

    def refine(self, profiles, obs: Observation, m: int) -> list[list[int]]:
        raise NotImplementedError
