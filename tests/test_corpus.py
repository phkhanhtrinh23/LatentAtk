from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latentatk.neural import corpus as data
from latentatk.neural.records import LABEL_FIELDS, HybridRecord, content_hash, observation_hash, save_records
from latentatk.neural.teachers import OpenRouterTeacher, resolve_teacher


def record(seed=0, victim="mf"):
    return HybridRecord(
        qid=f"{victim}-{seed}", family=data.VICTIM_FAMILIES[victim], target=1,
        budget=30, profile_size=20, difficulty=0.5, density=0.04, detector_pressure=0,
        candidate_ids=[2], candidate_features=[[0, 0, 0, 0, 0]],
        role_mix=[0.25] * 4, topology=1, depth=3, bridge_weights=[1] * 5,
        role_traces=["PR", "CA", "SN", "CO"], topology_probs=[0.25] * 4,
        seed=seed, environment_id=content_hash({"seed": seed}), victim_name=victim,
        teacher_metadata={"provider": "scripted", "source_id": "scripted"})


def test_strict_labels_reject_invalid_probability_and_depth():
    for values in ({"role_mix": [-1, 1, 1, 0]}, {"depth": 7},
                   {"bridge_weights": [float("nan")] * 5}, {"role_traces": ["PR"]},
                   {"teacher_metadata": {"provider": "openrouter"}, "topology_probs": None}):
        with pytest.raises(ValueError):
            replace(record(), **values).validate()


def test_split_identity_and_cached_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "make_record", lambda seed, victim, pressure, **kwargs: record(seed, victim))
    config = {"data": {"train": 5, "valid": 5, "test": 5, "rl": 5}}
    paths = data.prepare_splits(config, tmp_path)
    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    identities = [identity for split in manifest["splits"].values() for identity in split["environment_ids"]]
    assert len(identities) == len(set(identities)) == 20
    assert data.prepare_splits(config, tmp_path) == paths
    with pytest.raises(ValueError, match="configuration"):
        data.prepare_splits({"data": {**config["data"], "seed": 1}}, tmp_path)


@pytest.mark.parametrize("counts", [
    {"train": 5, "valid": 5, "test": 5},
    {"train": 5, "valid": 5, "test": 5, "rl": 7},
    {"train": 5, "valid": 5, "test": True, "rl": 5},
])
def test_split_counts_are_validated_before_writing(monkeypatch, tmp_path, counts):
    def unexpected_generation(*args, **kwargs):
        raise AssertionError("invalid split sizes must fail before data generation")

    monkeypatch.setattr(data, "make_record", unexpected_generation)
    out = tmp_path / "splits"
    with pytest.raises(ValueError, match="count must be supplied"):
        data.prepare_splits({"data": counts}, out)
    assert not out.exists()


def test_scripted_corpus_shared_across_runs_and_tamper_rejected(tmp_path):
    train = tmp_path / "observations.jsonl"
    save_records([record()], train)
    spec = {"name": "scripted", "provider": "scripted"}
    first = data.prepare_corpus({}, spec, 0, {"train": train}, tmp_path / "corpora")
    second = data.prepare_corpus({}, spec, 9, {"train": train}, tmp_path / "corpora")
    assert first == second
    save_records([replace(record(), depth=2)], first)
    with pytest.raises(ValueError, match="modified"):
        data.prepare_corpus({}, spec, 0, {"train": train}, tmp_path / "corpora")


def test_openrouter_missing_labels_never_falls_back(tmp_path):
    client = object.__new__(OpenRouterTeacher)
    client.spec = {"name": "source-alias", "max_retries": 0}
    client.model = "provider/pinned-model"
    client.catalogue_entry = {"id": client.model, "supported_parameters": ["seed"]}
    client._request = lambda *args: {
        "id": "request-1", "model": client.model,
        "choices": [{"finish_reason": "stop", "message": {"content": '{"depth": 2}'}}]}
    audit = tmp_path / "requests.jsonl"
    with pytest.raises(RuntimeError, match="six label fields"):
        client.label(record(), 42, audit)
    request = json.loads(audit.read_text())
    assert request["response"]["id"] == "request-1"
    assert request["request"]["provider"]["allow_fallbacks"] is False
    assert request["request"]["seed"] == 42


@pytest.mark.parametrize("supports_seed", [True, False])
def test_openrouter_sends_only_supported_parameters_and_records_actual_seed(tmp_path, supports_seed):
    client = object.__new__(OpenRouterTeacher)
    client.spec = {"name": "source-alias", "max_retries": 0}
    client.model = "provider/pinned-model"
    client.catalogue_entry = {
        "id": client.model, "supported_parameters": ["seed", "max_tokens"] if supports_seed else ["max_tokens"]}
    labels = {name: asdict(record())[name] for name in LABEL_FIELDS}
    client._request = lambda *args: {
        "id": "request-1", "model": client.model,
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(labels)}}]}
    audit = tmp_path / "requests.jsonl"
    result = client.label(record(), 42, audit)
    payload = json.loads(audit.read_text())["request"]
    assert ("seed" in payload) == supports_seed
    assert "temperature" not in payload
    assert payload["max_tokens"] == 2048
    assert result.teacher_metadata["seed"] == (42 if supports_seed else None)
    assert result.teacher_metadata["requested_seed"] == 42
    if not supports_seed:
        client._request = lambda *args: {"model": client.model}
        with pytest.raises(RuntimeError, match="requires a request ID"):
            client.label(record(), 42, tmp_path / "missing-id.jsonl")
    client.spec["temperature"] = 0.5
    with pytest.raises(ValueError, match="requested parameter temperature"):
        client.label(record(), 42, tmp_path / "unsupported.jsonl")


def test_unconfigured_alias_and_imported_identity_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="explicit versioned"):
        resolve_teacher({"name": "claude-fable-5-2", "provider": "openrouter"})
    path = tmp_path / "import.jsonl"
    save_records([record(1)], path)
    with pytest.raises(ValueError, match="observations"):
        data._read_import(path, [record(0)], {"name": "api"})
    assert observation_hash([record()]) == observation_hash([replace(record(), depth=1)])
