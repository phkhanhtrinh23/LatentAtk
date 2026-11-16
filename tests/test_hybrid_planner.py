from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latentatk.neural.hybrid import (
    HybridLatentPlanner, _assemble_prerope, _prerope_kv_hooks, load_planner,
)


class TokenBatch(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return TokenBatch({key: value.to(device) for key, value in self.items()})


class TinyTokenizer:
    def __call__(self, texts, return_tensors=None, truncation=False, max_length=None, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": list(range(len(texts)))}
        length = min(len(texts[0]), max_length) if truncation else len(texts[0])
        ids = torch.arange(length).unsqueeze(0)
        return TokenBatch(input_ids=ids, attention_mask=torch.ones_like(ids))


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, num_attention_heads=2)
        self.embed = torch.nn.Embedding(32, 8)
        self.output = torch.nn.Linear(8, 32)
        self.calls = []

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.output

    def forward(self, inputs_embeds, attention_mask, past_key_values=None,
                use_cache=False, output_hidden_states=False):
        self.calls.append((inputs_embeds.shape[1], attention_mask.shape[1], use_cache))
        states = inputs_embeds.cumsum(dim=1)
        if past_key_values is not None:
            states = states + past_key_values
        return SimpleNamespace(hidden_states=(states,), past_key_values=states[:, -1:])


def test_inference_prefills_once_then_feeds_single_latents():
    model = TinyBackbone()
    planner = HybridLatentPlanner(model, TinyTokenizer(), max_depth=3, max_input_tokens=4)
    torch.nn.init.zeros_(planner.stop_head.weight)
    torch.nn.init.constant_(planner.stop_head.bias, -10)
    result = planner.predict("abcdef")
    assert model.calls == [(4, 4, True), (1, 5, True), (1, 6, True)]
    assert result["input_tokens"] == 4
    assert result["original_input_tokens"] == 6
    assert result["truncated_tokens"] == 2
    assert result["inference_kv_cache"] is True
    assert result["decoded_tokens"] == 0
    assert result["depth"] == 3


def test_mla_key_reconstruction_and_value_padding_preserve_gradients():
    compressed = torch.tensor([[[8., 9., 10.], [1., 2., 3.]]], requires_grad=True)
    expanded = torch.arange(8., requires_grad=True).reshape(1, 2, 4)
    store = {"model.layers.0": {
        "mla": {"heads": 2, "nope_dim": 1, "rope_dim": 2, "value_dim": 1},
        "compressed": compressed, "expanded": expanded,
    }}
    values = _assemble_prerope(torch, store, [1], 2, 3)
    assert values.shape == (1, 1, 1, 2, 2, 3)
    assert torch.equal(values[0, 0, 0, 0], torch.tensor([[4., 2., 3.], [6., 2., 3.]]))
    assert torch.equal(values[0, 0, 0, 1], torch.tensor([[5., 0., 0.], [7., 0., 0.]]))
    values.sum().backward()
    assert compressed.grad is not None


def test_unsupported_kv_architecture_fails_explicitly():
    with pytest.raises(ValueError, match="no supported attention KV"):
        _prerope_kv_hooks(torch.nn.Linear(4, 4))


def test_incomplete_layer_capture_is_rejected():
    with pytest.raises(ValueError, match="Missing key/value"):
        _assemble_prerope(torch, {"model.layers.0": {"k": torch.zeros(1, 2, 4)}}, [0], 2, 2)


def test_resume_pins_saved_commit_before_fetching_base_config(tmp_path, monkeypatch):
    saved = {"base_model": "model", "max_depth": 6, "resolved_revision": "saved-commit",
             "loader": {"revision": None}}
    (tmp_path / "hybrid_meta.json").write_text(json.dumps(saved))
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "hybrid_heads.pt").touch()
    revisions = []

    def fetch_config(model, **options):
        revisions.append(options["revision"])
        raise RuntimeError("stop before loading")

    fake_transformers = SimpleNamespace(
        AutoConfig=object(), AutoModelForCausalLM=object(), AutoTokenizer=object(),
        PretrainedConfig=SimpleNamespace(get_config_dict=fetch_config),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    with pytest.raises(RuntimeError, match="stop before loading"):
        load_planner("model", tmp_path)
    assert revisions == ["saved-commit"]
    with pytest.raises(ValueError, match="revision does not match"):
        load_planner("model", tmp_path, revision="different-commit")
    assert revisions == ["saved-commit"]
