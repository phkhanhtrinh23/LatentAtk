import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latentatk.attacks.hybrid import profiles_from_action
from latentatk.data import make_synthetic
from latentatk.neural.records import HybridRecord


def _record():
    return HybridRecord(
        qid="x", family="factorization", target=9, budget=12, profile_size=6,
        difficulty=0.5, density=0.1, detector_pressure=0,
        candidate_ids=list(range(9)),
        candidate_features=[[float(i), 0, 0, 0, 1] for i in range(9)],
        role_mix=[0.1, 0.1, 0.7, 0.1], topology=0, depth=2,
        bridge_weights=[1, 0, 0, 0, 0], role_traces=["a", "b", "c", "d"],
    )


def test_hybrid_action_renders_valid_profiles():
    ds = make_synthetic(n_users=30, n_items=20, density=0.15, seed=0)
    rec = _record()
    profiles, counts = profiles_from_action(
        ds, rec.target, rec, rec.role_mix, rec.topology, rec.bridge_weights,
        rec.budget, rec.profile_size, np.random.default_rng(0))
    injected = counts["PR"] + counts["CA"] + counts["SN"]
    assert len(profiles) == injected
    assert all(len(p) <= rec.profile_size and len(p) == len(set(p)) for p in profiles)
    assert sum(counts.values()) == rec.budget
