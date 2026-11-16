from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np


def stable_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hash(path):
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and (
        p.suffix in {".pt", ".bin", ".safetensors", ".model"}
        or p.name in {"hybrid_meta.json", "adapter_config.json", "config.json",
                      "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                      "vocab.json", "merges.txt", "added_tokens.json", "generation_config.json"}
    ))
    if not files:
        raise ValueError(f"No checkpoint files in {root}")
    return stable_hash({str(p.relative_to(root)): file_hash(p) for p in files})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cuda_devices(planner=None):
    import torch

    if not torch.cuda.is_available():
        return []
    if planner is None:
        return list(range(torch.cuda.device_count()))
    return sorted({p.device.index for p in planner.model.parameters() if p.is_cuda})


def synchronize(planner=None):
    import torch

    for device in _cuda_devices(planner):
        torch.cuda.synchronize(device)


def hardware_info(planner=None):
    import torch

    devices = _cuda_devices(planner)
    info = {"gpu_count": len(devices), "gpus": [torch.cuda.get_device_name(i) for i in devices],
            "dtype": None, "model_revision": None}
    if planner is not None:
        info["dtype"] = str(next(planner.model.parameters()).dtype)
        info["model_revision"] = getattr(planner.model.config, "_commit_hash", None)
        if info["model_revision"] is None:
            info["model_revision_reason"] = "local checkpoint or loader did not expose a commit"
    return info


def verify_checkpoint(path, required_stage=None):
    digest = checkpoint_hash(path)
    manifest_path = Path(path) / "training_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("checkpoint_hash") != digest:
            raise ValueError(f"checkpoint content hash mismatch: {path}")
        if required_stage and manifest.get("config", {}).get("stage") != required_stage:
            raise ValueError(f"expected a {required_stage} checkpoint: {path}")
    elif required_stage:
        raise ValueError(f"a verified {required_stage} training manifest is required: {path}")
    return digest


def save_training_manifest(out, config, records_path, source_hash, result, hardware):
    manifest = {"config": config, "config_hash": stable_hash(config),
                "records_hash": file_hash(records_path), "source_checkpoint_hash": source_hash,
                "result": result, "hardware": hardware,
                "checkpoint_hash": checkpoint_hash(out)}
    write_json(Path(out) / "training_manifest.json", manifest)
    return manifest
