from __future__ import annotations

import time
import numpy as np

from .base import Attacker, AttackResult
from .group import _dedup
from ..neural.records import HybridRecord, TOPOLOGY_WIDTHS, scripted_teacher
from ..utils import CostMeter


def victim_family(probe):
    name = type(probe).__name__.lower()
    return {
        "mf": "factorization", "gmf": "neural", "lightgcn": "graph",
        "itemknn": "neighborhood", "fedmf": "federated",
    }.get(name, "factorization")


def _genuine_matrix(dataset):
    return dataset.R[:dataset.n_genuine_users]


def _cooccurrence(dataset, target):
    genuine = _genuine_matrix(dataset)
    users = genuine.getcol(int(target)).indices
    if len(users) == 0:
        cooc = np.zeros(dataset.n_items)
    else:
        cooc = np.asarray(genuine[users].sum(axis=0)).ravel().astype(float)
    cooc[int(target)] = 0.0
    return cooc


def _receptive_support(dataset, target, cooc, n_receptive):
    genuine = _genuine_matrix(dataset)
    overlap = np.asarray(genuine @ cooc).ravel()
    order = np.argsort(-overlap)
    receptive = order[overlap[order] > 0][:n_receptive]
    if len(receptive) == 0:
        receptive = order[:max(1, n_receptive)]
    support = np.asarray(genuine[receptive].sum(axis=0)).ravel().astype(float)
    support[int(target)] = 0.0
    reach = float((overlap > 0).mean())
    return receptive, support, reach


def _log_features(dataset, target, item_ids, cooc, support, n_receptive):
    item_ids = np.asarray(item_ids, int)
    pop = dataset.popularity.astype(float)
    ng = max(1, int(dataset.n_genuine_users))
    log_scale = np.log1p(float(pop.max())) + 1e-9
    support_i = support[item_ids]
    return np.stack([
        support_i / max(1, int(n_receptive)),
        np.log1p(pop[item_ids]) / log_scale,
        support_i / (pop[item_ids] + 1.0),
        pop[item_ids] / ng,
        cooc[item_ids] / ng,
    ], axis=1)


def log_candidate_features(dataset, target, n_candidates=50, n_receptive=80):
    n_receptive = min(n_receptive, dataset.n_genuine_users)
    cooc = _cooccurrence(dataset, target)
    receptive, support, reach = _receptive_support(dataset, target, cooc, n_receptive)
    difficulty = float(np.clip(1.0 - reach, 0.0, 1.0))
    order = np.argsort(-support)
    order = order[(support[order] > 0) & (order != int(target))][:n_candidates]
    if len(order) == 0:
        pop = dataset.popularity.astype(float)
        order = np.argsort(-pop)
        order = order[order != int(target)][:n_candidates]
    features = _log_features(dataset, target, order, cooc, support, len(receptive))
    return order.astype(int), features, difficulty


def log_observation_record(dataset, target, family, budget, m, detector_pressure=0.0,
                           n_candidates=50, qid="deployment", seed=0):
    order, features, difficulty = log_candidate_features(dataset, target, n_candidates)
    teacher = scripted_teacher(family, difficulty, detector_pressure)
    return HybridRecord(
        qid=qid, family=family, target=int(target), budget=budget, profile_size=m,
        difficulty=difficulty,
        density=float(dataset.R.nnz / (dataset.n_users * dataset.n_items)),
        detector_pressure=float(detector_pressure), candidate_ids=order.tolist(),
        candidate_features=features.astype(float).round(4).tolist(),
        role_mix=teacher.role_mix, topology=teacher.topology, depth=teacher.depth,
        bridge_weights=teacher.bridge_weights, role_traces=teacher.role_spans,
        topology_probs=teacher.topology_probs, seed=seed,
    )


def observation_record(dataset, target, probe, cost, rng, budget, m,
                       detector_pressure=0.0, n_candidates=50):
    record = log_observation_record(dataset, target, victim_family(probe), budget, m,
                                    detector_pressure, n_candidates)
    order = np.asarray(record.candidate_ids, int)
    features = np.asarray(record.candidate_features, float)
    return record, features, order


def _role_counts(role_mix, budget):
    mix = np.asarray(role_mix, float).clip(0)
    mix = np.ones(4) if mix.sum() == 0 else mix / mix.sum()
    counts = np.round(mix * budget).astype(int)
    order = np.argsort(-mix)
    diff = int(budget) - int(counts.sum())
    i = 0
    while diff != 0:
        j = int(order[i % len(order)])
        if diff > 0:
            counts[j] += 1
            diff -= 1
        elif counts[j] > 0:
            counts[j] -= 1
            diff += 1
        i += 1
    return tuple(int(c) for c in counts)


def _profiler_subpools(topology, pool, n_pr, rng):
    size = len(pool)
    if n_pr <= 0 or size == 0:
        return []
    if topology == 0:
        return [pool.copy() for _ in range(n_pr)]
    if topology == 1:
        window = min(int(np.ceil(2 * size / n_pr)), size)
        starts = [int(round(i * (size - window) / max(n_pr - 1, 1))) for i in range(n_pr)]
        return [pool[s:s + window] for s in starts]
    if topology == 2:
        base, remainder = divmod(size, n_pr)
        blocks, idx = [], 0
        for i in range(n_pr):
            step = base + (1 if i < remainder else 0)
            blocks.append(pool[idx:idx + step])
            idx += step
        return blocks
    subset = int(np.ceil(size / 2))
    return [pool[rng.choice(size, size=min(subset, size), replace=False)]
            for _ in range(n_pr)]


def _popularity_fill(dataset, target, k, rng, exclude=()):
    if k <= 0:
        return []
    pop = dataset.popularity.astype(float).copy()
    pop[int(target)] = 0.0
    for item in exclude:
        if 0 <= int(item) < len(pop):
            pop[int(item)] = 0.0
    available = int((pop > 0).sum())
    k = min(int(k), available)
    if k <= 0:
        return []
    return rng.choice(len(pop), size=k, replace=False, p=pop / pop.sum()).tolist()


def profiles_from_action(dataset, target, record, role_mix, topology, bridge_weights,
                         budget, m, rng, routing_width=None):
    weights = np.asarray(bridge_weights, float)
    cooc = _cooccurrence(dataset, target)
    n_receptive = min(80, dataset.n_genuine_users)
    _, support, _ = _receptive_support(dataset, target, cooc, n_receptive)
    width = TOPOLOGY_WIDTHS[int(topology)] if routing_width is None else routing_width
    pool_size = max(1, int(round(float(width))))
    # Score a log catalogue at least as wide as the maximum route width.
    rank = np.lexsort((dataset.popularity.astype(float), support))[::-1]
    rank = rank[rank != int(target)]
    catalogue = rank[:min(len(rank), max(128, pool_size))]
    catalogue_features = _log_features(dataset, target, catalogue, cooc, support, n_receptive)
    scores = catalogue_features @ weights
    order = np.argsort(-scores)
    pool_size = min(pool_size, len(catalogue))
    pool = catalogue[order[:pool_size]]
    pool_scores = scores[order[:pool_size]]

    n_pr, n_ca, n_sn, n_in = _role_counts(role_mix, budget)
    cooccurrence = cooc
    sub_pools = _profiler_subpools(int(topology), pool, n_pr, rng)
    top3 = [int(x) for x in pool[np.argsort(-pool_scores)[:3]]]

    profiles = []
    for sub_pool in sub_pools:
        sub_pool = np.asarray(sub_pool, int)
        nearest = sub_pool[np.argsort(-cooccurrence[sub_pool])]
        bridges = [int(x) for x in nearest[:min(len(nearest), m)]]
        fill = _popularity_fill(dataset, target, m - len(bridges), rng, exclude=bridges)
        profiles.append(_dedup(bridges + fill, m))
    for _ in range(n_ca):
        profiles.append(_dedup(_popularity_fill(dataset, target, m, rng), m))
    for _ in range(n_sn):
        head = [int(target)] + top3
        fill = _popularity_fill(dataset, target, m - len(head), rng, exclude=head)
        profiles.append(_dedup(head + fill, m))
    return profiles, {"PR": n_pr, "CA": n_ca, "SN": n_sn, "IN": n_in}


class HybridLatentAtk(Attacker):
    name = "HybridLatentAtk (neural)"

    def __init__(self, planner, budget=30, m=20, seed=0, detector_pressure=0.0):
        super().__init__(budget, m, seed)
        self.planner = planner
        self.detector_pressure = float(detector_pressure)

    def attack(self, dataset, target, probe=None):
        assert probe is not None
        rng = np.random.default_rng(self.seed)
        cost = CostMeter()
        rec, _, _ = observation_record(dataset, target, probe, cost, rng, self.budget, self.m,
                                       self.detector_pressure)
        t0 = time.perf_counter()
        action = self.planner.predict(rec.prompt())
        latency = time.perf_counter() - t0
        profiles, counts = profiles_from_action(
            dataset, target, rec, action["role_mix"], action["topology"],
            action["bridge_weights"], self.budget, self.m, rng,
            routing_width=action.get("route_width"))
        cost.add_call(0, note="hybrid latent plan")
        return AttackResult(profiles, cost, {
            **action, "roles": counts, "latency_s": latency,
            "measured_output_tokens": 0,
        })
