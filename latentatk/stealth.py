"""Stealth as an explicit, measurable objective for the attacker (Camouflageur).

The anytime-detection paper's key law is that expected detection time scales like
``1 / KL(injected-signal || benign)``: the more the injected interactions diverge from
benign behaviour, the faster any monitor catches them. That turns stealth -- previously
a heuristic served by the Camouflageur / Inactive roles -- into a *quantity to
minimise*: make the injected item distribution ``q`` close to the benign distribution
``p`` (small KL), **subject to still moving HR@K**.

This module provides:

* :func:`injected_item_distribution` / :func:`benign_item_distribution` / :func:`kl` --
  the divergence objective itself, plus its dominant single-coordinate term (the item
  that most drives detection, usually the target).
* :func:`stealth_mix` -- a one-parameter knob that trades payload for camouflage by
  shifting the PR/CA/SN/IN role mixture from Sniper-heavy (effective, detectable) to
  Camouflageur/Inactive-heavy (stealthy, weak). Feed it to ``LatentAtk(role_mix=...)``
  to trace the effectiveness/stealth Pareto frontier.
"""
from __future__ import annotations

import numpy as np

from .seqdetect import benign_item_distribution

benign_item_distribution = benign_item_distribution  # re-export


def injected_item_distribution(profiles, n_items: int, smoothing: float = 0.5) -> np.ndarray:
    """Empirical item distribution ``q`` over all injected interactions (smoothed)."""
    counts = np.zeros(n_items, dtype=float)
    for prof in profiles:
        for it in prof:
            counts[int(it)] += 1.0
    counts += smoothing
    return counts / counts.sum()


def kl(q: np.ndarray, p: np.ndarray) -> float:
    """KL(q || p) in nats. Both must be strictly positive, summing to 1."""
    return float(np.sum(q * np.log(q / p)))


def max_coordinate_kl(q: np.ndarray, p: np.ndarray):
    """Largest single-item contribution ``q_j log(q_j/p_j)`` and its item id.

    The sequential monitor's power is dominated by the most over-injected coordinate
    (typically the target), so this term predicts detection time better than the full
    KL does. Returns ``(value, item_id)``.
    """
    contrib = q * np.log(q / p)
    j = int(np.argmax(contrib))
    return float(contrib[j]), j


def stealth_mix(level: float, sn_max: float = 0.85, sn_min: float = 0.02) -> dict:
    """Role mixture for a given stealth ``level`` in ``[0, 1]``.

    ``level = 0``  -> Sniper-dominant (max effectiveness, max divergence, fast to detect).
    ``level = 1``  -> Camouflageur/Inactive-dominant (min divergence, hard to detect,
                      little promotion).

    A small floor ``sn_min`` keeps a minimal payload so the target is still injected
    (you cannot promote an item without ever rating it -- the irreducible signal).
    """
    level = float(np.clip(level, 0.0, 1.0))
    sn = sn_min + (1.0 - level) * (sn_max - sn_min)
    rest = 1.0 - sn
    return {"PR": 0.10 * rest, "CA": 0.55 * rest, "SN": sn, "IN": 0.35 * rest}
