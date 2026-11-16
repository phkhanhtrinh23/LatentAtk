from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from ..data import make_synthetic
from ..victims import FAMILIES as VICTIM_FAMILIES
from ..eval import pick_tail_target
from ..attacks.hybrid import log_observation_record
from .records import (
    content_hash, load_records, observation_hash, save_records,
)
from .teachers import (
    OpenRouterTeacher, PROMPT_VERSION, append_json, resolve_teacher,
)
from .provenance import write_json


VICTIMS = ("mf", "gmf", "lightgcn", "itemknn", "fedmf")
DATA_DEFAULTS = {
    "n_users": 120, "n_items": 240, "density": 0.04, "budget": 30,
    "profile_size": 20, "candidates": 50, "seed": 10000,
}


def source_hash():
    root = Path(__file__).resolve().parents[1]
    paths = ("neural/corpus.py", "data.py", "eval.py", "attacks/hybrid.py",
             "neural/records.py", "neural/teachers.py", "neural/provenance.py")
    return content_hash({path: (root / path).read_text() for path in paths})


def make_record(seed, victim_name, pressure, n_users=120, n_items=240, candidates=50,
                density=0.04, budget=30, profile_size=20):
    ds = make_synthetic(n_users=n_users, n_items=n_items, density=density, seed=seed)
    target = pick_tail_target(ds, seed)
    identity = content_hash({"seed": seed, "n_users": n_users, "n_items": n_items,
                             "density": density})
    record = log_observation_record(
        ds, target, VICTIM_FAMILIES[victim_name], budget=budget, m=profile_size,
        detector_pressure=pressure, n_candidates=candidates,
        qid=f"{victim_name}-{seed}-p{int(pressure * 100)}", seed=seed)
    return replace(record, environment_id=identity, victim_name=victim_name,
                   n_users=n_users, n_items=n_items, generation_density=density,
                   teacher_metadata={"provider": "scripted", "source_id": "scripted",
                                     "version": "scripted-v1"})


def prepare_splits(config, out):
    settings = {**DATA_DEFAULTS, **config.get("data", config)}
    split_names = ("train", "valid", "test", "rl")
    for name in split_names:
        count = settings.get(name)
        if (isinstance(count, bool) or not isinstance(count, int) or
                count < len(VICTIMS) or count % len(VICTIMS)):
            raise ValueError(f"{name} count must be supplied as a positive multiple of {len(VICTIMS)}")
    settings_hash = content_hash({"settings": settings, "source_hash": source_hash()})
    out = Path(out)
    manifest_path = out / "split_manifest.json"
    paths = {name: out / f"{name}.jsonl" for name in split_names}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings_hash"] != settings_hash:
            raise ValueError("split configuration differs from cached manifest")
        identities = set()
        for name, path in paths.items():
            records = load_records(path)
            entry = manifest["splits"][name]
            if len(records) != settings[name] or content_hash([asdict(r) for r in records]) != entry["content_hash"]:
                raise ValueError(f"{name}: cached split changed")
            current = {record.environment_id for record in records}
            if None in current or len(current) != len(records) or identities & current:
                raise ValueError("environment identity overlap in splits")
            identities.update(current)
        return paths
    if any(path.exists() for path in paths.values()):
        raise ValueError("split files exist without a complete manifest; choose a clean output directory")
    manifest = {"version": 1, "settings": settings, "settings_hash": settings_hash,
                "source_hash": source_hash(), "splits": {}}
    offset = 0
    for name, path in paths.items():
        count = settings[name]
        records = []
        for index in range(count):
            victim = VICTIMS[index % len(VICTIMS)]
            seed = settings["seed"] + offset + index
            pressure = ((index // len(VICTIMS)) % 4) / 3.0
            records.append(make_record(
                seed, victim, pressure, **{key: settings[key] for key in (
                    "n_users", "n_items", "candidates", "density", "budget", "profile_size")}))
        save_records(records, path)
        manifest["splits"][name] = {
            "count": len(records), "observation_hash": observation_hash(records),
            "content_hash": content_hash([asdict(record) for record in records]),
            "environment_ids": [record.environment_id for record in records],
        }
        offset += count
    write_json(manifest_path, manifest)
    return paths


def _read_import(path, observations, teacher):
    records = load_records(path)
    if observation_hash(records) != observation_hash(observations):
        raise ValueError("imported corpus observations/order differ from the shared train split")
    for record in records:
        metadata = record.teacher_metadata or {}
        if metadata.get("source_id") != teacher["name"]:
            raise ValueError(f"{record.qid}: imported teacher source_id differs")
        if not metadata.get("provider") or metadata["provider"] == "scripted" or not metadata.get("model"):
            raise ValueError(f"{record.qid}: imported API corpus lacks provider/model provenance")
        if teacher.get("model") and metadata["model"] != teacher["model"]:
            raise ValueError(f"{record.qid}: imported model differs from configured teacher")
        required = ("prompt_version", "prompt", "response", "retry_count", "usage", "cost_usd")
        if any(key not in metadata for key in required):
            raise ValueError(f"{record.qid}: incomplete imported teacher provenance")
        if not metadata["prompt"] or not metadata["response"]:
            raise ValueError(f"{record.qid}: imported corpus requires full prompt and raw response")
        if metadata.get("usage") is None and not metadata.get("usage_reason"):
            raise ValueError(f"{record.qid}: missing usage requires a reason")
        if metadata.get("cost_usd") is None and not metadata.get("cost_reason"):
            raise ValueError(f"{record.qid}: missing cost requires a reason")
        if not metadata.get("request_id") and metadata.get("seed") is None:
            raise ValueError(f"{record.qid}: imported teacher requires request_id or seed")
        if record.topology_probs is None:
            raise ValueError(f"{record.qid}: imported corpus lacks topology labels")
    return records


def prepare_corpus(config, teacher, run, splits, out):
    teacher = resolve_teacher(teacher)
    name = teacher["name"]
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in name):
        raise ValueError("teacher name must be a simple directory identifier")
    provider = teacher.get("provider", "scripted")
    out = Path(out) / name / ("shared" if provider == "scripted" else f"run-{run:02d}")
    path, manifest_path = out / "train.jsonl", out / "corpus_manifest.json"
    observations = load_records(splits["train"])
    import_path = teacher.get("import_path")
    if import_path:
        import_path = Path(str(import_path).format(run=run, teacher=name))
    signature = {
        "teacher": teacher, "run": None if provider == "scripted" else run,
        "observation_hash": observation_hash(observations), "prompt_version": PROMPT_VERSION,
        "seed": config.get("seed", 0),
        "source_hash": source_hash(),
        "import_hash": content_hash(import_path.read_text()) if import_path else None,
    }
    signature_hash = content_hash(signature)
    previous_elapsed, accepted_count, accepted_hash = 0.0, 0, None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        previous_elapsed = manifest.get("generation_wall_s", 0.0)
        accepted_count = manifest.get("accepted", 0)
        accepted_hash = manifest.get("accepted_hash")
        if manifest["signature_hash"] != signature_hash:
            raise ValueError("corpus configuration/provenance differs from cached manifest")
        if manifest.get("complete"):
            records = load_records(path)
            if (len(records) != len(observations) or observation_hash(records) != observation_hash(observations) or
                    content_hash([asdict(record) for record in records]) != manifest["content_hash"]):
                raise ValueError("cached corpus was modified")
            return path
    elif path.exists() or (out / "accepted.jsonl").exists():
        raise ValueError("corpus files exist without a manifest")
    started = time.perf_counter()
    manifest = {"version": 1, "signature": signature, "signature_hash": signature_hash,
                "complete": False, "generation_wall_s": previous_elapsed,
                "accepted": accepted_count, "accepted_hash": accepted_hash}
    write_json(manifest_path, manifest)
    if provider == "scripted":
        records = observations
    elif provider == "import":
        if not import_path:
            raise ValueError("import teacher requires import_path")
        records = _read_import(import_path, observations, teacher)
    elif provider == "openrouter":
        partial = out / "accepted.jsonl"
        records = load_records(partial) if partial.exists() else []
        if observation_hash(records) != observation_hash(observations[:len(records)]) or len(records) > len(observations):
            raise ValueError("partial corpus differs from shared train observations")
        if accepted_hash and (len(records) < accepted_count or
                              content_hash([asdict(record) for record in records[:accepted_count]]) != accepted_hash):
            raise ValueError("accepted partial corpus was modified")
        client = OpenRouterTeacher(teacher)
        write_json(out / "model_catalogue.json", client.catalogue_entry)
        seed = int(config.get("seed", 0)) + int(run) * 100000
        for index in range(len(records), len(observations)):
            record = client.label(observations[index], seed + index, out / "requests.jsonl")
            append_json(partial, asdict(record))
            records.append(record)
            manifest["accepted"] = len(records)
            manifest["accepted_hash"] = content_hash([asdict(record) for record in records])
            manifest["generation_wall_s"] = previous_elapsed + time.perf_counter() - started
            write_json(manifest_path, manifest)
    else:
        raise ValueError(f"unsupported teacher provider: {provider}")
    save_records(records, path)
    costs = [(record.teacher_metadata or {}).get("cost_usd") for record in records]
    usage = None
    if provider == "openrouter":
        requests = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines() if line.strip()]
        usages = [(request.get("response") or {}).get("usage") for request in requests]
        costs = [(item or {}).get("cost") for item in usages]
        fields = ("prompt_tokens", "completion_tokens", "total_tokens")
        usage = {field: sum(item[field] for item in usages)
                 if all(item and item.get(field) is not None for item in usages) else None
                 for field in fields}
    cost = 0.0 if provider == "scripted" else sum(costs) if all(c is not None for c in costs) else None
    cost_reason = None if cost is not None else "provider did not report all request costs"
    if provider == "import":
        cost, cost_reason = None, "imported accepted records do not include a complete retry request audit"
    manifest.update({
        "complete": True, "accepted": len(records),
        "content_hash": content_hash([asdict(record) for record in records]),
        "cost_usd": cost, "cost_reason": cost_reason,
        "request_log": "requests.jsonl" if provider == "openrouter" else None,
        "usage": usage,
        "usage_reason": "not applicable" if provider == "scripted" else
                        "see request audit; missing provider usage stays null",
        "generation_wall_s": previous_elapsed + time.perf_counter() - started,
    })
    write_json(manifest_path, manifest)
    return path
