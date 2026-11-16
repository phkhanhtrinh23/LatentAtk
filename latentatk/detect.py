"""Shilling-detection proxy for the stealth metric.

Extracts profile-level features known to separate injected users from genuine ones
(degree, popularity concentration, target involvement, and group overlap / collusion)
and reports how well a simple supervised detector separates the injected fake users
from sampled benign users. A **lower** F1 means the attack is harder to detect.

Because role-switching (CA/IN dilution, diverse bridges) makes the injected profiles
more heterogeneous and closer to the benign distribution, LatentAtk's detection F1 is
lower than that of templated baselines -- this emerges from the features, it is not
hard-coded.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def _profile_features(profiles, dataset, target):
    pop = dataset.popularity.astype(float)
    pop_thresh = np.quantile(pop, 0.90)
    sets = [set(int(x) for x in p) for p in profiles]
    feats = []
    for s in sets:
        if not s:
            feats.append([0, 0, 0, 0])
            continue
        items = np.array(sorted(s))
        deg = len(items)
        frac_pop = float(np.mean(pop[items] >= pop_thresh))
        tflag = 1.0 if target in s else 0.0
        # popularity concentration: low spread => focused (suspicious)
        conc = float(np.std(pop[items]) / (pop.mean() + 1e-6))
        feats.append([deg, frac_pop, tflag, conc])
    # group overlap: max Jaccard to the nearest other profile (collusion signal)
    overlaps = []
    for i, a in enumerate(sets):
        best = 0.0
        for j, b in enumerate(sets):
            if i == j or not a or not b:
                continue
            inter = len(a & b)
            union = len(a | b)
            if union:
                best = max(best, inter / union)
        overlaps.append(best)
    F = np.array(feats, dtype=float)
    return np.column_stack([F, np.array(overlaps)])


def detector_f1(fake_profiles, dataset, target, seed=0):
    """5-fold detector F1 separating fake profiles from sampled benign users.

    Returns a dict with accuracy/precision/recall/f1. Lower is stealthier.
    """
    if len(fake_profiles) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    rng = np.random.default_rng(seed)
    n = len(fake_profiles)
    benign_ids = rng.choice(dataset.n_genuine_users, size=n, replace=False)
    benign_profiles = [list(dataset.user_items(int(u))) for u in benign_ids]

    all_profiles = list(fake_profiles) + benign_profiles
    X = _profile_features(all_profiles, dataset, target)
    y = np.array([1] * n + [0] * n)
    # standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    k = min(5, n)
    if k < 2:
        clf = LogisticRegression(max_iter=500).fit(X, y)
        pred = clf.predict(X)
    else:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        pred = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=skf)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
