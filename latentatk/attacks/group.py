from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GroupContext:
    bridges: np.ndarray
    fillers: np.ndarray
    target: int


def discover_context(dataset, target, probe, cost, rng,
                     n_receptive=80, n_bridges=60, n_fillers=80) -> GroupContext:
    n_genuine = dataset.n_genuine_users
    sample = rng.choice(n_genuine, size=min(n_receptive * 3, n_genuine), replace=False)
    target_score = []
    for u in sample:
        s = probe.full_scores(int(u))
        target_score.append(s[target])
        cost.add_query(1)
    sample = sample[np.argsort(-np.asarray(target_score))]
    receptive = sample[:n_receptive]

    counts = np.zeros(dataset.n_items)
    for u in receptive:
        items = dataset.user_items(int(u))
        counts[items] += 1.0
    counts[target] = -1
    bridges = np.argsort(-counts)[:n_bridges]
    bridges = bridges[counts[bridges] > 0]

    pop = dataset.popularity.copy().astype(float)
    pop[target] = -1
    fillers = np.argsort(-pop)[:n_fillers]
    return GroupContext(bridges=bridges, fillers=fillers, target=int(target))


def discover_receptive(dataset, target, probe, cost, rng, n_receptive=100):
    n_genuine = dataset.n_genuine_users
    sample = rng.choice(n_genuine, size=min(n_receptive * 3, n_genuine), replace=False)
    ts = []
    for u in sample:
        ts.append(probe.full_scores(int(u))[target])
        cost.add_query(1)
    receptive = sample[np.argsort(-np.asarray(ts))][:n_receptive]
    counts = np.zeros(dataset.n_items)
    for u in receptive:
        counts[dataset.user_items(int(u))] += 1.0
    counts[target] = -1.0
    return receptive, counts


def _take(arr, k, rng, weighted=False):
    arr = np.asarray(arr)
    if len(arr) == 0:
        return np.array([], dtype=int)
    k = min(k, len(arr))
    if weighted:
        w = np.linspace(1.0, 0.3, len(arr))
        w = w / w.sum()
        return rng.choice(arr, size=k, replace=False, p=w)
    return rng.choice(arr, size=k, replace=False)


def role_sniper(ctx, m, rng):
    n_bridge = max(1, int(round(0.75 * (m - 1))))
    bridges = [int(x) for x in _take(ctx.bridges, n_bridge, rng, weighted=True)]
    remaining = max(0, m - 1 - len(bridges))
    fillers = [int(x) for x in _take(ctx.fillers, remaining, rng)]
    out = [ctx.target] + bridges + fillers
    return _dedup(out, m)


def role_profiler(ctx, m, rng):
    items = list(_take(ctx.fillers, int(0.6 * m), rng))
    items += list(_take(ctx.bridges, m - len(items), rng))
    out = [int(x) for x in items]
    if rng.random() < 0.3:
        out = [ctx.target] + out
    return _dedup(out, m)


def role_camouflageur(ctx, m, rng):
    half = int(0.7 * m)
    items = list(_take(ctx.fillers, half, rng))
    extra = rng.choice(ctx.fillers if len(ctx.fillers) else [0], size=m - half, replace=True)
    items += [int(x) for x in extra]
    return _dedup(items, m)


def role_inactive(ctx, m, rng):
    items = list(_take(ctx.fillers, max(1, m // 3), rng))
    return _dedup(items, m)


ROLE_FN = {
    "PR": role_profiler,
    "CA": role_camouflageur,
    "SN": role_sniper,
    "IN": role_inactive,
}


def _dedup(items, m):
    seen, out = set(), []
    for it in items:
        it = int(it)
        if it not in seen:
            seen.add(it)
            out.append(it)
        if len(out) >= m:
            break
    return out
