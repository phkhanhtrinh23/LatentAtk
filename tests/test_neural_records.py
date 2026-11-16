import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latentatk.neural.records import (
    HybridRecord, TOPOLOGY_WIDTHS, load_records, save_records,
    scripted_teacher, soft_topology_target,
)


def test_hybrid_records_roundtrip(tmp_path):
    teacher = scripted_teacher("graph", 0.8, 0.6)
    rec = HybridRecord(
        qid="x", family="graph", target=2, budget=10, profile_size=5,
        difficulty=0.8, density=0.1, detector_pressure=0.6,
        candidate_ids=[1, 3], candidate_features=[[1, 0, 0, 0, 1], [0, 1, 0, 0, 1]],
        role_mix=teacher.role_mix, topology=teacher.topology, depth=teacher.depth,
        bridge_weights=teacher.bridge_weights, role_traces=teacher.role_spans,
        topology_probs=teacher.topology_probs,
    )
    path = tmp_path / "records.jsonl"
    save_records([rec], path)
    got = load_records(path)[0]
    assert got == rec
    assert "family=graph" in got.prompt()
    assert got.plan_text().endswith("</PLAN>")


def test_scripted_teacher_matches_worked_example():
    teacher = scripted_teacher("graph", 0.6, 0.4)
    assert np.allclose(teacher.role_mix, [0.30, 0.42, 0.13, 0.15], atol=1e-2)
    assert np.allclose(teacher.topology_probs, [0.149, 0.382, 0.294, 0.175], atol=1e-2)
    assert teacher.topology == 1
    assert teacher.depth == 4
    assert teacher.bridge_weights == [0.6, 0.2, 1.0, 0.5, 0.1]
    assert teacher.role_spans[3] == "CO topology=chain width=38 depth=4 inactive=0.15"


def test_scripted_teacher_is_conditioned_on_family_and_pressure():
    low = scripted_teacher("factorization", 0.5, 0.0)
    high = scripted_teacher("factorization", 0.5, 1.0)
    graph = scripted_teacher("graph", 0.5, 0.0)
    assert high.role_mix[2] < low.role_mix[2]
    assert graph.bridge_weights != low.bridge_weights


def test_soft_topology_target_tracks_pressure():
    rec = HybridRecord(
        qid="x", family="graph", target=2, budget=10, profile_size=5,
        difficulty=0.5, density=0.1, detector_pressure=0.5,
        candidate_ids=[1], candidate_features=[[1, 0, 0, 0, 0]],
        role_mix=[0.1, 0.2, 0.6, 0.1], topology=0, depth=2,
        bridge_weights=[1, 1, 1, 1, 0], role_traces=["a", "b", "c", "d"])
    middle = soft_topology_target(rec)
    rec.detector_pressure = 1.0
    high = soft_topology_target(rec)
    assert abs(sum(middle) - 1.0) < 1e-7
    assert all(value > 0 for value in middle)
    assert sum(p * w for p, w in zip(high, TOPOLOGY_WIDTHS)) < sum(
        p * w for p, w in zip(middle, TOPOLOGY_WIDTHS))
