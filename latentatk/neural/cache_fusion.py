"""Gated fusion for heterogeneous cache-to-cache planning specialists."""
from __future__ import annotations

from dataclasses import dataclass


class HeterogeneousCacheFusion:
    """Project differently sized cache summaries into a shared gated space."""

    def __init__(self, input_dims, hidden=128, n_features=5, device="cuda"):
        import torch

        self.torch = torch
        self.projections = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(),
                                torch.nn.LayerNorm(hidden)) for d in input_dims
        ]).to(device)
        self.gates = torch.nn.ModuleList([torch.nn.Linear(hidden, 1) for _ in input_dims]).to(device)
        self.role_head = torch.nn.Linear(hidden, 4).to(device)
        self.topology_head = torch.nn.Linear(hidden, 4).to(device)
        self.bridge_head = torch.nn.Linear(hidden, n_features).to(device)
        self.depth_head = torch.nn.Linear(hidden, 6).to(device)
        self.device = device

    def parameters(self):
        modules = (self.projections, self.gates, self.role_head, self.topology_head,
                   self.bridge_head, self.depth_head)
        for module in modules:
            yield from module.parameters()

    def train(self):
        for m in (self.projections, self.gates, self.role_head, self.topology_head,
                  self.bridge_head, self.depth_head):
            m.train()

    def eval(self):
        for m in (self.projections, self.gates, self.role_head, self.topology_head,
                  self.bridge_head, self.depth_head):
            m.eval()

    def forward(self, specialist_inputs, mask=None):
        torch = self.torch
        z = [proj(x.to(self.device).float()) for proj, x in zip(self.projections, specialist_inputs)]
        logits = torch.cat([gate(x) for gate, x in zip(self.gates, z)], dim=-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.to(self.device), -1e9)
        gate = torch.softmax(logits, dim=-1)
        fused = sum(gate[:, i:i + 1] * value for i, value in enumerate(z))
        return {
            "role_probs": torch.softmax(self.role_head(fused), -1),
            "topology_logits": self.topology_head(fused),
            "bridge_weights": self.bridge_head(fused),
            "depth_logits": self.depth_head(fused),
            "gates": gate,
        }


def fusion_loss(output, records):
    device = output["role_probs"].device
    import torch as pt

    roles = pt.tensor([r.role_mix for r in records], device=device)
    topology = pt.tensor([r.topology for r in records], device=device)
    weights = pt.tensor([r.bridge_weights for r in records], device=device)
    depth = pt.tensor([r.depth - 1 for r in records], device=device)
    return (-(roles * output["role_probs"].clamp_min(1e-8).log()).sum(-1).mean() +
            pt.nn.functional.cross_entropy(output["topology_logits"], topology) +
            pt.nn.functional.smooth_l1_loss(output["bridge_weights"], weights) +
            0.5 * pt.nn.functional.cross_entropy(output["depth_logits"], depth))


# ---------------------------------------------------------------------------
# Role/layer-preserving residual fusion (C2C-v2)


class _ExpertEncoder:
    """Encode ``[B, role, layer, cache]`` without erasing role/layer identity."""

    def __init__(self, input_dim, hidden, n_roles, n_layers, device):
        import torch

        self.torch = torch
        self.norm = torch.nn.LayerNorm(input_dim).to(device)
        self.project = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden), torch.nn.GELU(),
            torch.nn.LayerNorm(hidden),
        ).to(device)
        self.role_embedding = torch.nn.Embedding(n_roles, hidden).to(device)
        self.layer_embedding = torch.nn.Embedding(n_layers, hidden).to(device)
        self.layer_score = torch.nn.Linear(hidden, 1).to(device)
        self.modules = [self.norm, self.project, self.role_embedding,
                        self.layer_embedding, self.layer_score]

    def __call__(self, x):
        torch = self.torch
        b, n_roles, n_layers, _ = x.shape
        z = self.project(self.norm(x))
        roles = self.role_embedding(torch.arange(n_roles, device=x.device))[None, :, None]
        layers = self.layer_embedding(torch.arange(n_layers, device=x.device))[None, None]
        z = z + roles + layers
        attention = torch.softmax(self.layer_score(z).squeeze(-1), dim=2)
        return (attention.unsqueeze(-1) * z).sum(2), attention

    def parameters(self):
        for module in self.modules:
            yield from module.parameters()

    def state_dict(self):
        return {str(i): module.state_dict() for i, module in enumerate(self.modules)}

    def load_state_dict(self, state):
        for i, module in enumerate(self.modules):
            module.load_state_dict(state[str(i)])

    def train(self, mode=True):
        for module in self.modules:
            module.train(mode)


@dataclass
class FusionLoss:
    total: object
    supervised: object
    expert_aux: object
    dropout_aux: object
    router: object


class RoleLayerResidualFusion:
    """Output-specific small-expert residuals over a large-expert cache base.

    The two experts are deliberately asymmetric: the 1.5B cache supplies a safe base,
    while the 0.5B cache can add a task-specific correction.  Independent routers let
    role allocation, topology, bridge selection and stopping use different mixtures.
    """

    TASKS = ("role", "topology", "bridge", "depth")

    def __init__(self, input_dims, hidden=128, n_roles=4, n_layers=4,
                 n_features=5, device="cuda"):
        import torch

        if len(input_dims) != 2:
            raise ValueError("C2C-v2 expects [small, large] cache dimensions")
        self.torch = torch
        self.device = torch.device(device)
        self.small = _ExpertEncoder(input_dims[0], hidden, n_roles, n_layers, self.device)
        self.large = _ExpertEncoder(input_dims[1], hidden, n_roles, n_layers, self.device)
        self.role_queries = torch.nn.ParameterDict({
            task: torch.nn.Parameter(torch.randn(hidden, device=self.device) / hidden ** 0.5)
            for task in self.TASKS
        })
        self.small_align = torch.nn.ModuleDict({
            task: torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                                      torch.nn.LayerNorm(hidden))
            for task in self.TASKS
        }).to(self.device)
        self.corrections = torch.nn.ModuleDict({
            task: torch.nn.Sequential(
                torch.nn.Linear(3 * hidden, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, hidden), torch.nn.LayerNorm(hidden))
            for task in self.TASKS
        }).to(self.device)
        self.routers = torch.nn.ModuleDict({
            task: torch.nn.Sequential(
                torch.nn.Linear(3 * hidden, hidden // 2), torch.nn.GELU(),
                torch.nn.Linear(hidden // 2, 1))
            for task in self.TASKS
        }).to(self.device)
        self.heads = torch.nn.ModuleDict({
            "role": torch.nn.Linear(hidden, 4),
            "topology": torch.nn.Linear(hidden, 4),
            "bridge": torch.nn.Linear(hidden, n_features),
            "depth": torch.nn.Linear(hidden, 6),
        }).to(self.device)
        self._module_groups = [self.role_queries, self.small_align, self.corrections,
                               self.routers, self.heads]

    def parameters(self):
        yield from self.small.parameters()
        yield from self.large.parameters()
        for module in self._module_groups:
            yield from module.parameters()

    def train(self, mode=True):
        self.small.train(mode); self.large.train(mode)
        for module in self._module_groups:
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def _pool_roles(self, tokens, task):
        torch = self.torch
        query = self.role_queries[task]
        score = torch.einsum("brh,h->br", tokens, query) / tokens.shape[-1] ** 0.5
        return (torch.softmax(score, -1).unsqueeze(-1) * tokens).sum(1)

    def forward(self, small_x, large_x, availability=None):
        """Run with per-row availability ``[small, large]``.

        Both experts is the deployment path.  Single-expert rows are used for expert
        dropout and for matched counterfactual controls.
        """
        torch = self.torch
        small_tokens, small_layer_attention = self.small(small_x.to(self.device).float())
        large_tokens, large_layer_attention = self.large(large_x.to(self.device).float())
        batch = small_tokens.shape[0]
        if availability is None:
            availability = torch.ones(batch, 2, dtype=torch.bool, device=self.device)
        availability = availability.to(self.device).bool()
        if (~availability.any(-1)).any():
            raise ValueError("every row needs at least one available expert")

        representations, gates = {}, []
        for task in self.TASKS:
            small = self.small_align[task](self._pool_roles(small_tokens, task))
            large = self._pool_roles(large_tokens, task)
            features = torch.cat([large, small, torch.abs(small - large)], -1)
            gate = torch.sigmoid(self.routers[task](features))
            correction = self.corrections[task](
                torch.cat([small, large, small - large], -1))
            both = large + gate * correction
            small_only = availability[:, 0:1] & ~availability[:, 1:2]
            large_only = availability[:, 1:2] & ~availability[:, 0:1]
            fused = torch.where(small_only, small, both)
            fused = torch.where(large_only, large, fused)
            representations[task] = fused
            # Report effective contribution: zero if the small expert is unavailable,
            # one if it is the only expert, and the learned residual gate otherwise.
            effective = torch.where(small_only, torch.ones_like(gate), gate)
            effective = torch.where(large_only, torch.zeros_like(gate), effective)
            gates.append(effective)

        return {
            "role_probs": torch.softmax(self.heads["role"](representations["role"]), -1),
            "topology_logits": self.heads["topology"](representations["topology"]),
            "bridge_weights": self.heads["bridge"](representations["bridge"]),
            "depth_logits": self.heads["depth"](representations["depth"]),
            "gates": torch.cat(gates, -1),
            "small_layer_attention": small_layer_attention,
            "large_layer_attention": large_layer_attention,
        }

    def state_dict(self):
        return {
            "small": self.small.state_dict(), "large": self.large.state_dict(),
            **{f"group_{i}": module.state_dict()
               for i, module in enumerate(self._module_groups)},
        }

    def load_state_dict(self, state):
        self.small.load_state_dict(state["small"])
        self.large.load_state_dict(state["large"])
        for i, module in enumerate(self._module_groups):
            module.load_state_dict(state[f"group_{i}"])


def per_example_fusion_losses(output, targets):
    """Return total and taskwise supervised losses without batch reduction."""
    import torch

    roles, topology, weights, depth = targets
    task = {
        "role": -(roles * output["role_probs"].clamp_min(1e-8).log()).sum(-1),
        "topology": torch.nn.functional.cross_entropy(
            output["topology_logits"], topology, reduction="none"),
        "bridge": torch.nn.functional.smooth_l1_loss(
            output["bridge_weights"], weights, reduction="none").mean(-1),
        "depth": 0.5 * torch.nn.functional.cross_entropy(
            output["depth_logits"], depth, reduction="none"),
    }
    return sum(task.values()), task


def residual_fusion_loss(model, small_x, large_x, targets, dropout_availability=None,
                         expert_weight=0.20, dropout_weight=0.20, router_weight=0.10,
                         router_temperature=0.25):
    """Supervise fused and single-expert paths plus counterfactual routers."""
    torch = model.torch
    batch = small_x.shape[0]
    both_mask = torch.ones(batch, 2, dtype=torch.bool, device=model.device)
    small_mask = torch.tensor([True, False], device=model.device).repeat(batch, 1)
    large_mask = torch.tensor([False, True], device=model.device).repeat(batch, 1)
    both = model.forward(small_x, large_x, both_mask)
    small = model.forward(small_x, large_x, small_mask)
    large = model.forward(small_x, large_x, large_mask)
    both_total, _ = per_example_fusion_losses(both, targets)
    small_total, small_tasks = per_example_fusion_losses(small, targets)
    large_total, large_tasks = per_example_fusion_losses(large, targets)

    if dropout_availability is None:
        dropout_availability = both_mask
    dropped = model.forward(small_x, large_x, dropout_availability)
    dropout_total, _ = per_example_fusion_losses(dropped, targets)

    router_terms = []
    for i, task in enumerate(model.TASKS):
        # Soft counterfactual label: >.5 precisely when the small-only prediction has
        # lower task loss.  Detaching prevents the experts gaming their router labels.
        advantage = (large_tasks[task] - small_tasks[task]).detach()
        target = torch.sigmoid(advantage / router_temperature)
        router_terms.append(torch.nn.functional.binary_cross_entropy(
            both["gates"][:, i].clamp(1e-5, 1 - 1e-5), target))
    router = torch.stack(router_terms).mean()
    supervised = both_total.mean()
    expert_aux = 0.5 * (small_total.mean() + large_total.mean())
    dropout_aux = dropout_total.mean()
    total = (supervised + expert_weight * expert_aux + dropout_weight * dropout_aux +
             router_weight * router)
    return FusionLoss(total, supervised, expert_aux, dropout_aux, router), both
