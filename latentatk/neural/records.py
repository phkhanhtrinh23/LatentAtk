from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


FAMILIES = ("factorization", "neural", "graph", "neighborhood", "federated")
TOPOLOGIES = ("star", "chain", "forest", "broadcast")
TOPOLOGY_WIDTHS = np.asarray([18.0, 38.0, 50.0, 72.0])
ROLES = ("PR", "CA", "SN", "IN")

BASE_ROLE_SPLIT = {
    "factorization": (0.40, 0.25, 0.25, 0.10),
    "neural": (0.38, 0.27, 0.25, 0.10),
    "graph": (0.30, 0.30, 0.25, 0.15),
    "neighborhood": (0.45, 0.20, 0.25, 0.10),
    "federated": (0.40, 0.28, 0.22, 0.10),
}
BRIDGE_WEIGHTS = {
    "factorization": (0.9, 0.3, 0.6, 0.2, 0.1),
    "neural": (0.9, 0.3, 0.6, 0.2, 0.1),
    "graph": (0.6, 0.2, 1.0, 0.5, 0.1),
    "neighborhood": (1.0, 0.2, 0.5, 0.2, 0.1),
    "federated": (0.8, 0.3, 0.6, 0.3, 0.1),
}
TOPOLOGY_LOGIT_A = np.asarray([0.2, 0.6, 0.4, 0.0])
TOPOLOGY_LOGIT_B = np.asarray([-1.2, 0.3, 0.8, 1.0])
TOPOLOGY_LOGIT_C = np.asarray([1.4, 0.5, -0.4, -1.0])


@dataclass
class HybridRecord:
    qid: str
    family: str
    target: int
    budget: int
    profile_size: int
    difficulty: float
    density: float
    detector_pressure: float
    candidate_ids: list[int]
    candidate_features: list[list[float]]
    role_mix: list[float]
    topology: int
    depth: int
    bridge_weights: list[float]
    role_traces: list[str]
    topology_probs: list[float] | None = None
    seed: int = 0
    environment_id: str | None = None
    victim_name: str | None = None
    n_users: int = 120
    n_items: int = 240
    generation_density: float = 0.04
    teacher_metadata: dict | None = None

    def validate(self) -> None:
        if not isinstance(self.qid, str) or not self.qid:
            raise ValueError("qid must be a nonempty string")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        for name in ("target", "budget", "profile_size", "depth", "n_users", "n_items", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.target < self.n_items or min(self.budget, self.profile_size, self.n_users) < 1:
            raise ValueError("invalid target, budget, profile size or environment dimensions")
        if not self.candidate_ids or len(self.candidate_ids) != len(self.candidate_features):
            raise ValueError("candidate ids/features length mismatch")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate ids must be unique")
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < self.n_items
               for i in self.candidate_ids):
            raise ValueError("invalid candidate id")
        features = np.asarray(self.candidate_features, dtype=float)
        if features.shape != (len(self.candidate_ids), 5) or not np.isfinite(features).all():
            raise ValueError("candidate features must be finite rows of five values")
        for name in ("role_mix", "topology_probs"):
            values = getattr(self, name)
            if values is None and name == "topology_probs":
                if self.teacher_metadata and self.teacher_metadata.get("provider") != "scripted":
                    raise ValueError("API/import records require explicit topology_probs")
                continue
            values = np.asarray(values, dtype=float)
            if (values.shape != (4,) or not np.isfinite(values).all() or
                    (values < 0).any() or not np.isclose(values.sum(), 1.0, atol=1e-4)):
                raise ValueError(f"{name} must contain four probabilities summing to one")
        if isinstance(self.topology, bool) or not isinstance(self.topology, int) or not 0 <= self.topology < 4:
            raise ValueError("invalid topology")
        if not 1 <= self.depth <= 6:
            raise ValueError("depth must be in [1, 6]")
        weights = np.asarray(self.bridge_weights, dtype=float)
        if weights.shape != (5,) or not np.isfinite(weights).all():
            raise ValueError("bridge_weights must contain five finite values")
        if len(self.role_traces) != 4 or any(not isinstance(t, str) or not t.strip() for t in self.role_traces):
            raise ValueError("four nonempty role traces are required")
        for name in ("difficulty", "density", "detector_pressure", "generation_density"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    def prompt(self) -> str:
        rows = []
        for item, feat in zip(self.candidate_ids, self.candidate_features):
            vals = ",".join(f"{float(x):+.2f}" for x in feat)
            rows.append(f"{int(item)}:[{vals}]")
        candidates = " ".join(rows)
        return (
            "Design a black-box recommender robustness-test campaign. Think in latent "
            "space, then return a compact machine-readable plan.\n"
            f"family={self.family} target={self.target} budget={self.budget} "
            f"profile_size={self.profile_size} difficulty={self.difficulty:.3f} "
            f"density={self.density:.4f} detector_pressure={self.detector_pressure:.3f}\n"
            "candidate features are [support,logpop,receptive_frequency,broad_frequency,"
            "target_cooccurrence].\n"
            f"candidates: {candidates}\n<PLAN>"
        )

    def plan_text(self) -> str:
        mix = ",".join(f"{r}:{p:.3f}" for r, p in zip(ROLES, self.role_mix))
        weights = ",".join(f"{x:+.3f}" for x in self.bridge_weights)
        return (f"roles={mix};topology={TOPOLOGIES[self.topology]};depth={self.depth};"
                f"weights=[{weights}]</PLAN>")


def save_records(records: Iterable[HybridRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            rec.validate()
            fh.write(json.dumps(asdict(rec), separators=(",", ":")) + "\n")


def load_records(path: str | Path) -> list[HybridRecord]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = HybridRecord(**json.loads(line))
                rec.validate()
                out.append(rec)
    return out


@dataclass
class TeacherTarget:
    role_mix: list[float]
    topology_probs: list[float]
    topology: int
    depth: int
    bridge_weights: list[float]
    role_spans: list[str]


def _softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - logits.max()
    weights = np.exp(stable)
    return weights / weights.sum()


def scripted_teacher(family: str, difficulty: float, pressure: float,
                     max_depth: int = 6) -> TeacherTarget:
    difficulty = float(np.clip(difficulty, 0, 1))
    pressure = float(np.clip(pressure, 0, 1))

    rho = np.asarray(BASE_ROLE_SPLIT[family], float)
    shift = min(0.3 * pressure, rho[2])
    rho[1] += shift
    rho[2] -= shift
    rho = rho / rho.sum()

    logits = TOPOLOGY_LOGIT_A + TOPOLOGY_LOGIT_B * difficulty + TOPOLOGY_LOGIT_C * pressure
    topology_probs = _softmax(logits)
    topology = int(np.argmax(topology_probs))
    kappa = int(TOPOLOGY_WIDTHS[topology])

    depth = int(np.clip(round(1 + 5 * difficulty), 1, max_depth))
    weights = list(BRIDGE_WEIGHTS[family])

    role_spans = [
        f"PR rho={rho[0]:.2f} bridges=top-{kappa} near target",
        f"CA rho={rho[1]:.2f} fill=popularity",
        f"SN rho={rho[2]:.2f} payload=target+top3",
        f"CO topology={TOPOLOGIES[topology]} width={kappa} depth={depth} "
        f"inactive={rho[3]:.2f}",
    ]
    return TeacherTarget(rho.tolist(), topology_probs.tolist(), topology, depth,
                         weights, role_spans)


def soft_topology_target(record: HybridRecord) -> list[float]:
    if record.topology_probs is not None:
        return list(record.topology_probs)
    if record.teacher_metadata and record.teacher_metadata.get("provider") != "scripted":
        raise ValueError("missing teacher topology probabilities")
    return scripted_teacher(record.family, record.difficulty,
                            record.detector_pressure).topology_probs


LABEL_FIELDS = ("role_mix", "topology", "depth", "bridge_weights", "role_traces", "topology_probs")


def observation(record: HybridRecord) -> dict:
    excluded = {*LABEL_FIELDS, "teacher_metadata"}
    return {key: value for key, value in asdict(record).items() if key not in excluded}


def content_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def observation_hash(records: Iterable[HybridRecord]) -> str:
    return content_hash([observation(record) for record in records])
