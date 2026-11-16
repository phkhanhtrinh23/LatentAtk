from dataclasses import make_dataclass
import json
from types import SimpleNamespace

import pytest

from latentatk.neural import distill, rl


class TinyPlanner:
    def __init__(self, torch):
        self.torch = torch
        self.model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(0.5))
        self.head = torch.nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self.max_depth = 6
        self.n_features = 5
        self.max_input_tokens = 64
        self.input_device = self.device
        self.model.config = SimpleNamespace(_commit_hash="revision")
        self.saved = 0

    def train(self):
        self.model.train()
        self.head.train()

    def eval(self):
        self.model.eval()
        self.head.eval()

    def parameters(self):
        return [*self.model.parameters(), *self.head.parameters()]

    def planning_parameters(self):
        return self.head.parameters()

    def forward(self, prompts):
        return self.head(self.torch.ones(1, 1))

    def save(self, out, base_model):
        self.saved += 1


@pytest.fixture
def planner():
    return TinyPlanner(pytest.importorskip("torch"))


def test_distillation_exact_updates_and_final_checkpoint(planner, monkeypatch, tmp_path):
    seen = []

    def loss(current, records, **kwargs):
        seen.extend(records)
        return {"total": current.head.weight.square().sum()}

    monkeypatch.setattr(distill, "distillation_loss", loss)
    monkeypatch.setattr(distill, "synchronize", lambda planner: None)
    monkeypatch.setattr(distill, "hardware_info", lambda planner: {"gpu_count": 0})
    result = distill.train_distillation(planner, [1, 2, 3], tmp_path, "tiny",
                                       batch_size=2, max_updates=5)
    assert result["steps"] == 5
    assert result["checkpoint_selection"] == "final"
    assert planner.saved == 1
    assert len((tmp_path / "train_log.jsonl").read_text().splitlines()) == 5
    assert set(seen) == {1, 2, 3}


def test_rl_retrain_budget_and_backbone_freeze(planner, monkeypatch, tmp_path):
    torch = planner.torch
    original = [parameter.detach().clone() for parameter in planner.model.parameters()]
    retrains = []

    def action(current, output, budget):
        return {"logp": output.sum(), "entropy": output.sum(),
                "topology": torch.tensor(0), "depth": torch.tensor(1)}

    def reward(env, chosen, seed, **kwargs):
        retrains.append(seed)
        return 1.0, {"component_survival": [1.0, 0.5], "victim_retrain_s": 0.1,
                     "gated_min": 0.5, "hr10": 0.5, "survival": 0.75, "worst_survival": 0.5}

    monkeypatch.setattr(rl, "_sample_action", action)
    monkeypatch.setattr(rl, "real_reward", reward)
    monkeypatch.setattr(rl, "synchronize", lambda planner: None)
    monkeypatch.setattr(rl, "hardware_info", lambda planner: {"gpu_count": 0})
    record = SimpleNamespace(qid="env", family="graph", budget=30, prompt=lambda: "prompt")
    history = rl.train_detector_aware_rl(planner, [SimpleNamespace(record=record)],
                                         tmp_path, "tiny", steps=2, batch_size=3)
    assert len(history) == 2
    assert len(retrains) == len(set(retrains)) == 6
    assert not planner.model.training
    assert all(not parameter.requires_grad for parameter in planner.model.parameters())
    assert all(torch.equal(before, after) for before, after in zip(original, planner.model.parameters()))
    cost = json.loads((tmp_path / "stage_cost.json").read_text())
    assert cost["victim_retrains"] == 6
    assert planner.saved == 1


def test_kv_cache_requires_stage1_reference(tmp_path):
    with pytest.raises(ValueError, match="Stage-1"):
        distill.precompute_teacher_cache(None, [], tmp_path)


def test_sampled_depth_selects_action_head_state(monkeypatch):
    torch = pytest.importorskip("torch")
    planner = SimpleNamespace(
        torch=torch, device=torch.device("cpu"), topology_head=torch.nn.Linear(1, 4),
        role_head=torch.nn.Linear(1, 4), bridge_head=torch.nn.Linear(1, 5),
        topology_routes=torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False) for _ in range(4)]))
    for route in planner.topology_routes:
        torch.nn.init.zeros_(route.weight)
    seen = []
    planner.topology_head.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].clone()))
    monkeypatch.setattr(torch.distributions.Categorical, "sample", lambda self: torch.tensor(1))
    output = SimpleNamespace(stop_logits=torch.zeros(1, 2),
                             latent_states=torch.tensor([[[1.0], [7.0]]], requires_grad=True))
    action = rl._sample_action(planner, output, budget=30)
    assert action["depth"].item() == 2
    assert torch.equal(seen[0], torch.tensor([[7.0]]))


@pytest.mark.parametrize("damage", ["tensor", "missing_hash", "source_hash"])
def test_kv_cache_rejects_corruption(planner, monkeypatch, tmp_path, damage):
    record = make_dataclass("Record", [("qid", str)])("env")
    monkeypatch.setattr(distill, "compressed_teacher_kv", lambda planner, rec: planner.torch.ones(1))
    distill.precompute_teacher_cache(planner, [record], tmp_path, reference_hash="stage1")
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    if damage == "tensor":
        (tmp_path / "env.pt").write_bytes(b"corrupt")
    elif damage == "missing_hash":
        manifest["files"] = {}
        path.write_text(json.dumps(manifest))
    else:
        manifest["source_code_hash"] = "changed"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash|provenance"):
        distill.precompute_teacher_cache(planner, [record], tmp_path, reference_hash="stage1")
