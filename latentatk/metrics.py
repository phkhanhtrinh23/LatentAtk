from __future__ import annotations

import numpy as np


def _target_rank(scores: np.ndarray, target: int, seen: np.ndarray) -> int:
    s = scores.copy()
    s[seen] = -np.inf
    better = np.sum(s > s[target])
    return int(better) + 1


def promotion_metrics(victim, test_users, target, dataset, ks=(5, 10, 20)):
    ks = tuple(ks)
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    n = 0
    for u in test_users:
        seen = dataset.user_items(u)
        if target in seen:
            continue
        scores = victim.full_scores(u)
        rank = _target_rank(scores, target, seen)
        n += 1
        for k in ks:
            if rank <= k:
                hits[k] += 1
                ndcg[k] += 1.0 / np.log2(rank + 1)
    out = {}
    n = max(n, 1)
    for k in ks:
        out[f"HR@{k}"] = hits[k] / n
        out[f"NDCG@{k}"] = ndcg[k] / n
    return out


def benign_recall(victim, dataset, ks=(10,)):
    ks = tuple(ks)
    hit = {k: 0 for k in ks}
    n = 0
    for u, item in dataset.test.items():
        if u >= victim.n_users:
            continue
        seen = dataset.user_items(u)
        scores = victim.full_scores(u)
        s = scores.copy()
        s[seen] = -np.inf
        rank = int(np.sum(s > s[item])) + 1
        n += 1
        for k in ks:
            if rank <= k:
                hit[k] += 1
    n = max(n, 1)
    return {f"Rec@{k}": hit[k] / n for k in ks}
