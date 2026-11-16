from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

from .hybrid import _assemble_prerope, _prerope_kv_hooks
from .records import HybridRecord, soft_topology_target
from .provenance import stable_hash, file_hash, write_json, read_json, synchronize, hardware_info


def compressed_teacher_kv(planner, record: HybridRecord):
    torch = planner.torch
    tok, model = planner.tokenizer, planner.model
    trace_ids = [tok(f" {trace}", add_special_tokens=False)["input_ids"]
                 for trace in record.role_traces]
    if len(trace_ids) != 4 or any(not part for part in trace_ids):
        raise ValueError("KV distillation requires four nonempty role traces")
    limit = getattr(planner, "max_input_tokens", 1536)
    prompt_limit = limit - sum(map(len, trace_ids))
    if prompt_limit < 1:
        raise ValueError(f"teacher traces exceed input limit for {record.qid}")
    ids = list(tok(record.prompt(), add_special_tokens=False)["input_ids"])[-prompt_limit:]
    spans = []
    for part in trace_ids:
        start = len(ids)
        ids.extend(part)
        spans.append(list(range(start, len(ids))))
    model.eval()
    kv_heads, head_dim = planner._kv_shape()
    store, handles = _prerope_kv_hooks(model)
    with torch.no_grad():
        try:
            model(input_ids=torch.tensor([ids], device=planner.input_device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
    summaries = [_assemble_prerope(torch, store, span, kv_heads, head_dim).mean(dim=1)
                 for span in spans]
    return torch.stack(summaries, dim=1).cpu().to(torch.float16)


def precompute_teacher_cache(planner, records, out_dir, reference_hash=None):
    if not reference_hash:
        raise ValueError("a Stage-1 checkpoint hash is required for KV targets")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "reference_hash": reference_hash,
        "corpus_hash": stable_hash([asdict(record) for record in records]),
        "max_input_tokens": getattr(planner, "max_input_tokens", 1536),
        "representation": "local-role-kv-v1",
        "source_code_hash": stable_hash({
            name: file_hash(Path(__file__).with_name(name))
            for name in ("distill.py", "hybrid.py", "records.py")
        }),
        "reference_config_hash": stable_hash({
            "dtype": str(next(planner.model.parameters()).dtype),
            "model_revision": getattr(planner.model.config, "_commit_hash", None),
            "max_depth": planner.max_depth,
            "n_features": planner.n_features,
        }),
    }
    manifest_path = out / "manifest.json"
    files = {}
    if manifest_path.exists():
        existing = read_json(manifest_path)
        files = existing.pop("files", {})
        if existing != manifest:
            raise ValueError(f"KV cache provenance mismatch: {out}")
    elif any(out.iterdir()):
        raise ValueError(f"KV cache has no provenance manifest: {out}")
    manifest["files"] = files
    write_json(manifest_path, manifest)
    planner.eval()
    for parameter in planner.parameters():
        parameter.requires_grad_(False)
    for i, rec in enumerate(records, 1):
        path = out / f"{rec.qid}.pt"
        expected = files.get(path.name)
        if path.exists():
            if expected is None or file_hash(path) != expected:
                raise ValueError(f"KV tensor hash missing or mismatched: {path}")
        else:
            temporary = path.with_suffix(".pt.tmp")
            planner.torch.save(compressed_teacher_kv(planner, rec), temporary)
            temporary.replace(path)
            files[path.name] = file_hash(path)
            write_json(manifest_path, manifest)
        if i % 10 == 0 or i == len(records):
            print(f"[teacher-kv] {i}/{len(records)}", flush=True)


def _target_ids(planner, records):
    torch = planner.torch
    texts = [r.plan_text() for r in records]
    batch = planner.tokenizer(texts, return_tensors="pt", padding=True,
                              add_special_tokens=False)
    ids = batch.input_ids
    ids[batch.attention_mask == 0] = -100
    return ids.to(torch.long)


def distillation_loss(planner, records, kv_dir=None, kv_weight=0.0):
    torch = planner.torch
    prompts = [r.prompt() for r in records]
    depths = torch.tensor([r.depth for r in records], device=planner.device)
    target_ids = _target_ids(planner, records)
    out = planner.forward(prompts, target_depth=depths, plan_target_ids=target_ids,
                          output_cache=bool(kv_dir and kv_weight > 0))

    role = torch.tensor([r.role_mix for r in records], device=planner.device)
    topology = torch.tensor([soft_topology_target(r) for r in records],
                            device=planner.device)
    weights = torch.tensor([r.bridge_weights for r in records], device=planner.device)
    role_loss = -(role * out.role_probs.clamp_min(1e-8).log()).sum(-1).mean()
    topology_loss = -(topology * torch.nn.functional.log_softmax(
        out.topology_logits, dim=-1)).sum(-1).mean()
    weight_loss = torch.nn.functional.smooth_l1_loss(out.bridge_weights, weights)

    stop_target = torch.zeros_like(out.stop_logits)
    for b, depth in enumerate(depths.tolist()):
        stop_target[b, depth - 1:] = 1.0
    stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(out.stop_logits, stop_target)
    losses = {
        "lm": out.lm_loss, "role": role_loss, "topology": topology_loss,
        "weights": weight_loss, "stop": stop_loss,
    }
    total = out.lm_loss + role_loss + topology_loss + weight_loss + 0.5 * stop_loss

    if kv_dir and kv_weight > 0:
        targets = [torch.load(Path(kv_dir) / f"{r.qid}.pt", map_location=planner.device,
                              weights_only=True) for r in records]
        teacher = torch.cat(targets, dim=0).float()
        n_roles = min(teacher.shape[1], out.student_kv.shape[1])
        student = out.student_kv[:, :n_roles].float()
        student = torch.nn.functional.normalize(student, dim=-1)
        teacher = torch.nn.functional.normalize(teacher[:, :n_roles], dim=-1)
        squared = (student - teacher).pow(2).sum(dim=-1).sum(dim=3)
        losses["kv"] = squared.mean()
        total = total + float(kv_weight) * losses["kv"]
    losses["total"] = total
    return losses


def train_distillation(planner, records, out_dir, base_model, epochs=1, lr=2e-4,
                       batch_size=1, kv_dir=None, kv_weight=0.0, seed=0,
                       valid_records=None, valid_kv_dir=None, max_updates=None,
                       optimizer="adamw", weight_decay=0.01, betas=(0.9, 0.999),
                       max_grad_norm=1.0):
    if not records or batch_size < 1:
        raise ValueError("training requires records and a positive batch size")
    updates = (int(max_updates) if max_updates is not None else
               int(epochs) * ((len(records) + batch_size - 1) // batch_size))
    if updates < 1 or optimizer.lower() not in {"adamw", "adam"}:
        raise ValueError("positive updates and optimizer adamw/adam are required")
    torch = planner.torch
    rng = random.Random(seed)
    planner.train()
    params = list(planner.parameters())
    optimizer_class = torch.optim.AdamW if optimizer.lower() == "adamw" else torch.optim.Adam
    optim = optimizer_class(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    synchronize(planner)
    t0, step, epoch = time.perf_counter(), 0, 0
    with (out / "train_log.jsonl").open("w", encoding="utf-8") as log:
        while step < updates:
            epoch += 1
            order = list(range(len(records)))
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                batch = [records[i] for i in order[start:start + batch_size]]
                optim.zero_grad(set_to_none=True)
                losses = distillation_loss(planner, batch, kv_dir=kv_dir, kv_weight=kv_weight)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optim.step()
                step += 1
                row = {"epoch": epoch, "step": step,
                       **{k: round(float(v.detach()), 6) for k, v in losses.items()},
                       "elapsed_s": round(time.perf_counter() - t0, 2)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"[distill] {row}", flush=True)
                if step == updates:
                    break
        synchronize(planner)
        elapsed = time.perf_counter() - t0
        validation, validation_elapsed = None, 0.0
        if valid_records:
            validation_start = time.perf_counter()
            validation = evaluate_distillation(planner, valid_records,
                                              kv_dir=valid_kv_dir, kv_weight=kv_weight,
                                              batch_size=batch_size)
            synchronize(planner)
            validation_elapsed = time.perf_counter() - validation_start
            log.write(json.dumps({"step": step, "validation": validation}) + "\n")
    planner.save(out, base_model)
    gpu_count = hardware_info(planner).get("gpu_count", 0)
    return {"steps": step, "elapsed_s": elapsed,
            "gpu_hours": elapsed * gpu_count / 3600, "validation": validation,
            "validation_elapsed_s": validation_elapsed,
            "validation_gpu_hours": validation_elapsed * gpu_count / 3600,
            "checkpoint_selection": "final"}


def evaluate_distillation(planner, records, kv_dir=None, kv_weight=0.0, batch_size=1):
    torch = planner.torch
    planner.eval()
    totals, count = {}, 0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            losses = distillation_loss(planner, batch, kv_dir=kv_dir,
                                       kv_weight=kv_weight)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value) * len(batch)
            count += len(batch)
    planner.train()
    return {key: value / count for key, value in totals.items()}
