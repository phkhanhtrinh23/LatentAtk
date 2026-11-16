from __future__ import annotations

import numpy as np

from .victims import build_victim
from .attacks import build_attacker
from .metrics import promotion_metrics, benign_recall
from .detect import detector_f1
from .utils import set_seed


def pick_tail_target(dataset, seed=0):
    pop = dataset.popularity
    rng = np.random.default_rng(seed)
    lo, hi = np.quantile(pop, 0.10), np.quantile(pop, 0.35)
    cand = np.where((pop >= max(1, lo)) & (pop <= max(2, hi)))[0]
    if len(cand) == 0:
        cand = np.where(pop >= 1)[0]
    if dataset.item_genre is not None:
        genre_pop = np.zeros(int(dataset.item_genre.max()) + 1)
        for g in range(len(genre_pop)):
            genre_pop[g] = pop[dataset.item_genre == g].sum()
        best_genre = int(np.argmax(genre_pop))
        g_cand = cand[dataset.item_genre[cand] == best_genre]
        if len(g_cand) > 0:
            cand = g_cand
    return int(rng.choice(cand))


def evaluate_all(dataset, victim_name="bpr-mf", attacker_names=None,
                 budget=50, m=30, ks=(5, 10, 20), n_test=250, seed=0,
                 target=None, victim_kwargs=None, attacker_kwargs=None):
    set_seed(seed)
    victim_kwargs = victim_kwargs or {}
    attacker_kwargs = attacker_kwargs or {}
    attacker_names = attacker_names or ["none", "random", "bandwagon", "pga",
                                        "single", "teacher", "latentatk"]
    if target is None:
        target = pick_tail_target(dataset, seed)

    clean = build_victim(victim_name, seed=seed, **victim_kwargs).fit(dataset)

    genuine = np.arange(dataset.n_genuine_users)
    rng = np.random.default_rng(seed)
    test_users = rng.choice(genuine, size=min(n_test, len(genuine)), replace=False)

    rows = []
    for name in attacker_names:
        akw = dict(attacker_kwargs.get(name, {}))
        attacker = build_attacker(name, budget=budget, m=m, seed=seed, **akw)
        res = attacker.attack(dataset, target, probe=clean)

        if res.profiles:
            poisoned = dataset.add_fake_users(res.profiles)
            victim = build_victim(victim_name, seed=seed, **victim_kwargs).fit(poisoned)
        else:
            poisoned, victim = dataset, clean

        promo = promotion_metrics(victim, test_users, target, poisoned, ks)
        rec = benign_recall(victim, poisoned, ks=(10,))
        det = detector_f1(res.profiles, dataset, target, seed=seed)
        row = {
            "attacker": attacker.name,
            **{k: round(v * 1000, 1) for k, v in promo.items()},
            "Rec@10": round(rec["Rec@10"], 4),
            "DetF1": round(det["f1"], 3),
            "Calls": res.cost.llm_calls,
            "Tokens": res.cost.tokens,
            "Queries": res.cost.victim_queries,
            "Fake": len(res.profiles),
        }
        row.update({f"_meta_{k}": v for k, v in res.meta.items()})
        rows.append(row)
    return rows, int(target)


def format_table(rows, ks=(5, 10, 20)):
    cols = ["attacker"] + [f"HR@{k}" for k in ks] + [f"NDCG@{k}" for k in ks] + \
           ["Rec@10", "DetF1", "Calls", "Tokens"]
    widths = {c: max(len(c), 18 if c == "attacker" else 7) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    out = [line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(out)
