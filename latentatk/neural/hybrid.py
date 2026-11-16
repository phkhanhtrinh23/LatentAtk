from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any


@dataclass
class PlannerOutput:
    role_probs: Any
    topology_logits: Any
    bridge_weights: Any
    stop_logits: Any
    latent_states: Any
    lm_loss: Any = None
    student_kv: Any = None
    topology_probs: Any = None
    route_width: Any = None


class HybridLatentPlanner:
    HEADS_FILE = "hybrid_heads.pt"
    META_FILE = "hybrid_meta.json"

    def __init__(self, model, tokenizer, max_depth=6, n_features=5, adapter_path=None,
                 max_input_tokens=1536, loader_config=None):
        import torch

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = int(max_depth)
        self.n_features = int(n_features)
        self.max_input_tokens = int(max_input_tokens)
        if not 1 <= self.max_depth <= 6 or self.max_input_tokens < 1:
            raise ValueError("Require depth 1–6 and positive max_input_tokens")
        self.loader_config = dict(loader_config or {})
        self.text_config = getattr(model.config, "text_config", model.config)
        hidden = int(self.text_config.hidden_size)
        output = model.get_output_embeddings().weight
        device, dtype = output.device, output.dtype
        self.hybrid_link = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(), torch.nn.LayerNorm(hidden)
        ).to(device=device, dtype=dtype)
        self.role_head = torch.nn.Linear(hidden, 4).to(device=device, dtype=dtype)
        self.topology_head = torch.nn.Linear(hidden, 4).to(device=device, dtype=dtype)
        self.topology_routes = torch.nn.ModuleList([
            torch.nn.Linear(hidden, hidden, bias=False) for _ in range(4)
        ]).to(device=device, dtype=dtype)
        for route in self.topology_routes:
            torch.nn.init.zeros_(route.weight)
        self.bridge_head = torch.nn.Linear(hidden, n_features).to(device=device, dtype=dtype)
        self.stop_head = torch.nn.Linear(hidden, 1).to(device=device, dtype=dtype)
        self.modules = [self.hybrid_link, self.role_head, self.topology_head,
                        self.topology_routes,
                        self.bridge_head, self.stop_head]
        if adapter_path and Path(adapter_path, self.HEADS_FILE).exists():
            self.load_heads(adapter_path)

    @property
    def device(self):
        return next(self.hybrid_link.parameters()).device

    @property
    def input_device(self):
        return self.model.get_input_embeddings().weight.device

    def _kv_shape(self):
        cfg = self.text_config
        if getattr(cfg, "qk_nope_head_dim", None):
            return int(cfg.num_attention_heads), int(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return int(kv_heads), int(head_dim)

    def train(self):
        self.model.train()
        for module in self.modules:
            module.train()

    def eval(self):
        self.model.eval()
        for module in self.modules:
            module.eval()

    def parameters(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p
        for module in self.modules:
            yield from module.parameters()

    def planning_parameters(self):
        for module in self.modules:
            yield from module.parameters()

    def _latent_prefix(self, prompts, max_steps=None, output_cache=False):
        torch = self.torch
        max_steps = self.max_depth if max_steps is None else int(max_steps)
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                               max_length=self.max_input_tokens).to(self.input_device)
        embed = self.model.get_input_embeddings()
        pieces = [embed(batch.input_ids)]
        mask = batch.attention_mask
        states, latent_positions = [], []
        for _ in range(max_steps):
            x = torch.cat(pieces, dim=1)
            out = self.model(inputs_embeds=x, attention_mask=mask, use_cache=False,
                             output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].to(self.device)
            states.append(h)
            pieces.append(self.hybrid_link(h).unsqueeze(1).to(self.input_device))
            mask = torch.cat([mask, torch.ones(mask.shape[0], 1, device=mask.device,
                                               dtype=mask.dtype)], dim=1)
            latent_positions.append(x.shape[1])
        student_kv = None
        if output_cache:
            x = torch.cat(pieces, dim=1)
            store, handles = _prerope_kv_hooks(self.model)
            try:
                self.model(inputs_embeds=x, attention_mask=mask, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            student_kv = _assemble_prerope(torch, store, latent_positions,
                                           *self._kv_shape()).to(self.device)
        return pieces, mask, torch.stack(states, dim=1), latent_positions, student_kv

    def forward(self, prompts, target_depth=None, plan_target_ids=None, output_cache=False):
        torch = self.torch
        pieces, mask, states, positions, student_kv = self._latent_prefix(
            prompts, self.max_depth, output_cache=output_cache)
        stop_logits = self.stop_head(states).squeeze(-1)
        if target_depth is None:
            stop = torch.sigmoid(stop_logits) >= 0.5
            chosen = torch.full((states.shape[0],), self.max_depth - 1,
                                device=self.device, dtype=torch.long)
            for b in range(states.shape[0]):
                idx = torch.where(stop[b])[0]
                if idx.numel():
                    chosen[b] = idx[0]
        else:
            chosen = torch.as_tensor(target_depth, device=self.device, dtype=torch.long) - 1
            chosen = chosen.clamp(0, self.max_depth - 1)
        h = states[torch.arange(states.shape[0], device=self.device), chosen]

        lm_loss = None
        if plan_target_ids is not None:
            target = plan_target_ids.to(self.input_device)
            target_mask = target.ne(-100)
            safe = target.masked_fill(~target_mask, self.tokenizer.pad_token_id)
            inp = torch.cat([*pieces, self.model.get_input_embeddings()(safe)], dim=1)
            limit = getattr(self.text_config, "max_position_embeddings", None)
            if limit and inp.shape[1] > limit:
                raise ValueError("Prompt, latents and plan targets exceed backbone context length")
            full_mask = torch.cat([mask, target_mask.to(mask.dtype)], dim=1)
            labels = torch.full(inp.shape[:2], -100, device=self.input_device, dtype=torch.long)
            labels[:, -target.shape[1]:] = target
            lm_loss = self.model(inputs_embeds=inp, attention_mask=full_mask,
                                 labels=labels, use_cache=False).loss.to(self.device)

        topology_logits = self.topology_head(h).float()
        topology_probs = torch.softmax(topology_logits, dim=-1)
        residuals = torch.stack([route(h) for route in self.topology_routes], dim=1)
        routed_h = h + (topology_probs.to(residuals.dtype).unsqueeze(-1) * residuals).sum(1)
        widths = torch.tensor([18.0, 38.0, 50.0, 72.0], device=self.device)
        return PlannerOutput(
            role_probs=torch.softmax(self.role_head(routed_h).float(), dim=-1),
            topology_logits=topology_logits,
            bridge_weights=self.bridge_head(routed_h).float(),
            stop_logits=stop_logits.float(), latent_states=states,
            lm_loss=lm_loss, student_kv=student_kv, topology_probs=topology_probs,
            route_width=(topology_probs * widths).sum(-1),
        )

    def predict(self, prompt):
        torch = self.torch
        self.eval()
        original_tokens = len(self.tokenizer(prompt, truncation=False)["input_ids"])
        with torch.no_grad():
            batch = self.tokenizer([prompt], return_tensors="pt", truncation=True,
                                   max_length=self.max_input_tokens).to(self.input_device)
            inputs = self.model.get_input_embeddings()(batch.input_ids)
            mask, cache = batch.attention_mask, None
            input_tokens = int(mask.sum().item())
            for step in range(self.max_depth):
                out = self.model(inputs_embeds=inputs, attention_mask=mask,
                                 past_key_values=cache, use_cache=True,
                                 output_hidden_states=True)
                cache = getattr(out, "past_key_values", None)
                if cache is None:
                    cache = getattr(out, "cache", None)
                if cache is None:
                    raise RuntimeError("Backbone did not return an inference KV cache")
                h = out.hidden_states[-1][:, -1, :].to(self.device)
                if torch.sigmoid(self.stop_head(h).float()).item() >= 0.5:
                    break
                inputs = self.hybrid_link(h).unsqueeze(1).to(self.input_device)
                mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
            topology_logits = self.topology_head(h).float()
            topology_probs = torch.softmax(topology_logits, -1)
            residuals = torch.stack([route(h) for route in self.topology_routes], dim=1)
            routed_h = h + (topology_probs.to(residuals.dtype).unsqueeze(-1) * residuals).sum(1)
            role_probs = torch.softmax(self.role_head(routed_h).float(), dim=-1)
            bridge_weights = self.bridge_head(routed_h).float()
            widths = torch.tensor([18.0, 38.0, 50.0, 72.0], device=self.device)
            route_width = (topology_probs * widths).sum(-1)
        return {
            "role_mix": role_probs[0].cpu().tolist(),
            "topology": int(topology_logits[0].argmax().item()),
            "topology_probs": topology_probs[0].cpu().tolist(),
            "route_width": float(route_width[0].item()),
            "bridge_weights": bridge_weights[0].cpu().tolist(),
            "depth": step + 1,
            "input_tokens": input_tokens,
            "original_input_tokens": original_tokens,
            "truncated_tokens": max(0, original_tokens - input_tokens),
            "truncated": original_tokens > input_tokens,
            "max_input_tokens": self.max_input_tokens,
            "decoded_tokens": 0,
            "inference_kv_cache": True,
        }

    def save(self, out_dir, base_model):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        state = {}
        for name in ("hybrid_link", "role_head", "topology_head", "topology_routes",
                     "bridge_head", "stop_head"):
            state[name] = getattr(self, name).state_dict()
        self.torch.save(state, out / self.HEADS_FILE)
        (out / self.META_FILE).write_text(json.dumps({
            "base_model": str(base_model), "max_depth": self.max_depth,
            "n_features": self.n_features, "max_input_tokens": self.max_input_tokens,
            "loader": self.loader_config,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
        }, indent=2))

    def load_heads(self, adapter_path):
        state = self.torch.load(Path(adapter_path) / self.HEADS_FILE,
                                map_location=self.device, weights_only=True)
        expected = {"hybrid_link", "role_head", "topology_head", "topology_routes", "bridge_head", "stop_head"}
        if set(state) != expected:
            raise ValueError("Checkpoint must contain every planning module exactly once")
        for name, values in state.items():
            getattr(self, name).load_state_dict(values)


def _prerope_kv_hooks(model):
    store, handles = {}, []
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if ".self_attn." not in name:
            continue
        layer, projection = name.rsplit(".self_attn.", 1)
        if projection not in {"k_proj", "v_proj", "kv_a_proj_with_mqa", "kv_b_proj"}:
            continue
        attention = modules[layer + ".self_attn"]
        if projection in {"kv_a_proj_with_mqa", "kv_b_proj"}:
            store.setdefault(layer, {})["mla"] = {
                "rope_dim": attention.qk_rope_head_dim,
                "nope_dim": attention.qk_nope_head_dim,
                "value_dim": attention.v_head_dim,
                "heads": attention.num_heads,
            }
        kind = {"k_proj": "k", "v_proj": "v",
                "kv_a_proj_with_mqa": "compressed", "kv_b_proj": "expanded"}[projection]

        def hook(_module, _inputs, output, layer=layer, kind=kind):
            if kind == "expanded" and output.ndim == 4 and output.shape[1] == 1:
                output = output.squeeze(1)
            store.setdefault(layer, {})[kind] = output

        handles.append(module.register_forward_hook(hook))
    if not handles:
        raise ValueError("Backbone has no supported attention KV projections")
    return store, handles


def _layer_index(name):
    match = re.search(r"\.layers\.(\d+)", name)
    return int(match.group(1)) if match else -1


def _assemble_prerope(torch, store, positions, kv_heads, head_dim):
    if not store:
        raise ValueError("No KV projections captured")
    layer_stack, destination = [], None
    for layer in sorted(store, key=_layer_index):
        parts = store[layer]
        if "mla" in parts:
            cfg = parts["mla"]
            if not {"compressed", "expanded"} <= parts.keys():
                raise ValueError(f"Missing MLA projections in {layer}")
            expanded = parts["expanded"][:, positions, :]
            expanded = expanded.reshape(*expanded.shape[:-1], cfg["heads"],
                                        cfg["nope_dim"] + cfg["value_dim"])
            nope, value = expanded.split([cfg["nope_dim"], cfg["value_dim"]], dim=-1)
            rope = parts["compressed"][:, positions, -cfg["rope_dim"]:]
            rope = rope.unsqueeze(-2).expand(*nope.shape[:-1], cfg["rope_dim"])
            key = torch.cat([nope, rope], dim=-1)
            value = torch.nn.functional.pad(value, (0, key.shape[-1] - value.shape[-1]))
        else:
            if not {"k", "v"} <= parts.keys():
                raise ValueError(f"Missing key/value projections in {layer}")
            key = parts["k"][:, positions, :]
            value = parts["v"][:, positions, :]
            shape = tuple(key.shape[:-1]) + (kv_heads, head_dim)
            key, value = key.reshape(shape), value.reshape(shape)
        if key.shape[-2:] != (kv_heads, head_dim):
            raise ValueError(f"Incompatible KV shape in {layer}: {key.shape}")
        destination = key.device if destination is None else destination
        layer_stack.append(torch.stack([key, value], dim=2).to(destination))
    return torch.stack(layer_stack, dim=2)


def load_planner(base_model, adapter_path=None, trainable=True, max_depth=6,
                 lora_rank=16, dtype="bfloat16", revision=None, max_input_tokens=1536,
                 device_map=None, local_files_only=False, trust_remote_code=False,
                 dequantize_fp8=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype not in dtypes:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose {tuple(dtypes)}")
    saved_meta = {}
    if adapter_path:
        meta_path = Path(adapter_path, HybridLatentPlanner.META_FILE)
        if not all(Path(adapter_path, name).exists() for name in (
                HybridLatentPlanner.META_FILE, HybridLatentPlanner.HEADS_FILE, "adapter_config.json")):
            raise ValueError(f"Incomplete planner checkpoint: {adapter_path}")
        saved_meta = json.loads(meta_path.read_text())
        if str(saved_meta["base_model"]) != str(base_model):
            raise ValueError("Checkpoint backbone does not match requested model")
        if int(saved_meta["max_depth"]) != max_depth:
            raise ValueError("Checkpoint latent depth does not match requested depth")
        saved_revision = saved_meta.get("loader", {}).get("revision")
        if revision not in {None, saved_revision, saved_meta.get("resolved_revision")}:
            raise ValueError("Checkpoint revision does not match requested revision")
    pinned_revision = saved_meta.get("resolved_revision") or revision
    options = {"revision": pinned_revision, "local_files_only": local_files_only,
               "trust_remote_code": trust_remote_code}
    raw_config, _ = PretrainedConfig.get_config_dict(base_model, **options)
    native_kimi = raw_config.get("model_type") == "kimi_k2"
    loader = AutoModelForCausalLM
    if native_kimi:
        from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
        native_config = dict(raw_config)
        native_config.pop("auto_map", None)
        native_config["model_type"] = "deepseek_v3"
        config = DeepseekV3Config.from_dict(native_config)
        loader = DeepseekV3ForCausalLM
    else:
        config = AutoConfig.from_pretrained(base_model, **options)
    resolved_revision = getattr(config, "_commit_hash", None)
    if resolved_revision:
        options["revision"] = resolved_revision
    text_config = getattr(config, "text_config", config)
    model_type = getattr(text_config, "model_type", "")
    if model_type not in {"qwen2", "llama", "gemma3_text", "deepseek_v3", "kimi_k2"}:
        raise ValueError(f"Unsupported planner architecture {model_type!r}")
    context_limit = getattr(text_config, "max_position_embeddings", None)
    if context_limit and max_input_tokens + max_depth >= context_limit:
        raise ValueError("Input limit and latent depth exceed backbone context length")
    model_options = {}
    quantization = getattr(config, "quantization_config", None)
    if quantization:
        if quantization.get("quant_method") != "fp8":
            raise ValueError("Only full precision or fine-grained FP8 checkpoints are supported")
        if not dequantize_fp8:
            raise ValueError("FP8 checkpoints require dequantize_fp8=True for planner training")
        from transformers import FineGrainedFP8Config
        fp8 = FineGrainedFP8Config(**{**quantization, "dequantize": True})
        if not getattr(fp8, "dequantize", False):
            raise RuntimeError("Upgrade transformers to a version supporting FP8 dequantize=True")
        model_options["quantization_config"] = fp8
    if native_kimi:
        model_options["attn_implementation"] = "eager"
    tok = AutoTokenizer.from_pretrained(base_model, **options)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if getattr(config, "model_type", "") == "gemma3":
        from transformers import Gemma3ForConditionalGeneration
        loader = Gemma3ForConditionalGeneration
    resolved_device = device_map or ("cuda" if torch.cuda.is_available() else "cpu")
    model, loading = loader.from_pretrained(
        base_model, config=config, torch_dtype=dtypes[dtype], device_map=resolved_device,
        output_loading_info=True, **options, **model_options,
    )
    mismatches = {key: loading.get(key, []) for key in
                  ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    if any(mismatches.values()):
        raise ValueError(f"Backbone checkpoint is incompatible with its loader: {mismatches}")
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    elif trainable:
        from peft import LoraConfig, get_peft_model
        projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                       "down_proj", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"}
        targets = [name for name, _ in model.named_modules()
                   if name.rsplit(".", 1)[-1] in projections
                   and (".self_attn." in name or ".mlp." in name)
                   and "vision" not in name]
        if not targets:
            raise ValueError("Backbone has no supported LoRA projection targets")
        model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=targets,
        ))
    _, handles = _prerope_kv_hooks(model)
    for handle in handles:
        handle.remove()
    loader_config = {"revision": revision, "dtype": dtype, "device_map": resolved_device,
                     "local_files_only": local_files_only, "trust_remote_code": trust_remote_code,
                     "max_input_tokens": max_input_tokens, "lora_rank": lora_rank,
                     "max_depth": max_depth, "architecture": model_type,
                     "native_kimi": native_kimi, "dequantize_fp8": dequantize_fp8}
    planner = HybridLatentPlanner(model, tok, max_depth=max_depth, adapter_path=adapter_path,
                                  max_input_tokens=max_input_tokens, loader_config=loader_config)
    if not trainable:
        planner.eval()
    return planner
