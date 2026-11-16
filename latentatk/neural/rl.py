from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np

from ..attacks.hybrid import profiles_from_action
from ..data import make_synthetic
from ..metrics import promotion_metrics
from ..seqdetect import PaperMonitorEnsemble
from ..victims import build_victim
from .provenance import synchronize, hardware_info, write_json


@dataclass
class RLEnvironment:
    record: object
    dataset: object
    probe: object
    victim_name: str
    test_users: np.ndarray
    victim_kwargs: dict
    defender: object = None


def build_environment(record, settings=None, fit_probe=False, evaluation_seed=None):
    settings = settings or {}
    victim_settings = settings.get("victim", {})
    victim_name = record.victim_name or record.qid.split("-")[0]
    ds = make_synthetic(n_users=record.n_users, n_items=record.n_items,
                        density=record.generation_density, seed=record.seed)
    kwargs = {} if victim_name == "itemknn" else {"epochs": victim_settings.get("epochs", 15)}
    if victim_name == "lightgcn":
        kwargs["layers"] = victim_settings.get("lightgcn_layers", 2)
    probe = build_victim(victim_name, seed=record.seed, **kwargs).fit(ds) if fit_probe else None
    seed = record.seed + 99 if evaluation_seed is None else evaluation_seed
    users = np.random.default_rng(seed).choice(
        ds.n_genuine_users, size=min(victim_settings.get("test_users", 100), ds.n_genuine_users),
        replace=False)
    defender = PaperMonitorEnsemble(ds, settings.get("detector"))
    return RLEnvironment(record, ds, probe, victim_name, users, kwargs, defender)


def _sample_action(planner, out, budget):
    torch = planner.torch
    hazards = torch.sigmoid(out.stop_logits[0]).clamp(1e-4, 1 - 1e-4)
    probs, survive = [], torch.ones((), device=planner.device)
    for i, hazard in enumerate(hazards):
        p = survive if i == len(hazards) - 1 else survive * hazard
        probs.append(p)
        survive = survive * (1 - hazard)
    depth_probs = torch.stack(probs) / torch.stack(probs).sum()
    depth_probs = 0.9 * depth_probs + 0.1 / len(depth_probs)
    depth_dist = torch.distributions.Categorical(probs=depth_probs)
    depth_idx = depth_dist.sample()
    h = out.latent_states[:, depth_idx, :]
    topology_logits = planner.topology_head(h).float()
    route_probs = torch.softmax(topology_logits, dim=-1)
    residuals = torch.stack([route(h) for route in planner.topology_routes], dim=1)
    routed_h = h + (route_probs.to(residuals.dtype).unsqueeze(-1) * residuals).sum(1)
    role_probs = torch.softmax(planner.role_head(routed_h).float()[0], dim=-1)
    role_probs = 0.9 * role_probs + 0.1 / role_probs.shape[-1]
    role_dist = torch.distributions.Multinomial(total_count=budget, probs=role_probs)
    role_counts = role_dist.sample()
    role_mix = role_counts / budget
    topology_dist = torch.distributions.Categorical(probs=0.8 * route_probs[0] + 0.2 / 4)
    topology = topology_dist.sample()
    bridge_weights = planner.bridge_head(routed_h).float()[0]
    weight_dist = torch.distributions.Normal(bridge_weights, 0.25)
    weights = weight_dist.sample()
    logp = (role_dist.log_prob(role_counts) + topology_dist.log_prob(topology) +
            depth_dist.log_prob(depth_idx) + weight_dist.log_prob(weights).sum())
    return {
        "role_mix": role_mix, "topology": topology, "depth": depth_idx + 1,
        "weights": weights, "logp": logp,
        "entropy": (role_dist.entropy() + topology_dist.entropy() +
                    depth_dist.entropy() + weight_dist.entropy().sum()),
    }


def real_reward(env, action, seed=0, stealth_weight=0.7, compute_weight=0.08,
                lagrange_mu=None, omega=None, max_depth=6, promotion_gate=0.25):
    rng = np.random.default_rng(seed)
    profiles, _ = profiles_from_action(
        env.dataset, env.record.target, env.record,
        action["role_mix"].detach().cpu().numpy(), int(action["topology"].item()),
        action["weights"].detach().cpu().numpy(), env.record.budget,
        env.record.profile_size, rng)
    poisoned = env.dataset.add_fake_users(profiles)
    retrain_start = time.perf_counter()
    victim = build_victim(env.victim_name, seed=seed, **env.victim_kwargs).fit(poisoned)
    retrain_s = time.perf_counter() - retrain_start
    hr = promotion_metrics(victim, env.test_users, env.record.target, poisoned, (10,))["HR@10"]
    if env.defender is None:
        env.defender = PaperMonitorEnsemble(env.dataset)
    det = env.defender.monitor_profiles(profiles, seed=seed, update=False)
    component = np.asarray(det["component_survival"], float)
    if omega is None:
        s_omega = float(component.mean())
    else:
        s_omega = float(np.asarray(omega, float) @ component)
    worst = det["worst_survival"]
    depth_cost = float(action["depth"].item()) / float(max_depth)
    gate = min(1.0, float(hr) / promotion_gate)
    gated_survival = s_omega * gate
    gated_min = worst * gate
    if lagrange_mu is None:
        reward = float(hr + stealth_weight * gated_survival - compute_weight * depth_cost)
    else:
        reward = float(hr + float(lagrange_mu) * gated_min - compute_weight * depth_cost)
    return reward, {"victim_retrain_s": retrain_s, "hr10": float(hr), "survival": s_omega,
                    "worst_survival": worst,
                    "component_survival": component.tolist(),
                    "detector_survival": det["family_survival"],
                    "gated_survival": gated_survival, "gated_min": gated_min,
                    "depth_cost": depth_cost}


def train_detector_aware_rl(planner, envs, out_dir, base_model, steps=300, lr=5e-5,
                            stealth_weight=0.7, compute_weight=0.08, seed=0,
                            beta=1.0, constrained=False, tau=0.7, eta=0.05, mu_init=0.0,
                            freeze_backbone=True, batch_size=1, optimizer="adamw",
                            weight_decay=0.01, betas=(0.9, 0.999), max_grad_norm=1.0,
                            entropy_weight=0.003, promotion_gate=0.25):
    if not envs or steps < 1 or batch_size < 1 or promotion_gate <= 0:
        raise ValueError("RL requires environments, positive updates/batch size and reward gate")
    if optimizer.lower() not in {"adamw", "adam"}:
        raise ValueError("optimizer must be adamw or adam")
    torch = planner.torch
    rng = random.Random(seed)
    planner.train()
    if freeze_backbone:
        for parameter in planner.model.parameters():
            parameter.requires_grad_(False)
        planner.model.eval()
        params = list(planner.planning_parameters())
    else:
        params = list(planner.parameters())
    optimizer_class = torch.optim.AdamW if optimizer.lower() == "adamw" else torch.optim.Adam
    optim = optimizer_class(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)
    baselines, family_baselines, history = {}, {}, []
    mu, constraint_estimate, survival_sum, survival_count, omega = float(mu_init), None, None, 0, None
    synchronize(planner)
    t0 = time.perf_counter()
    retrain_s = 0.0
    for step in range(1, steps + 1):
        optim.zero_grad(set_to_none=True)
        batch_rows = []
        for sample in range(batch_size):
            env = envs[rng.randrange(len(envs))]
            out = planner.forward([env.record.prompt()])
            action = _sample_action(planner, out, env.record.budget)
            reward, parts = real_reward(
                env, action, seed=seed + (step - 1) * batch_size + sample + 1,
                stealth_weight=stealth_weight, compute_weight=compute_weight,
                lagrange_mu=mu if constrained else None, omega=omega,
                max_depth=planner.max_depth, promotion_gate=promotion_gate)
            retrain_s += parts["victim_retrain_s"]
            component = np.asarray(parts["component_survival"], float)
            survival_sum = component.copy() if survival_sum is None else survival_sum + component
            survival_count += 1
            weights = np.exp(-beta * survival_sum / survival_count)
            omega = weights / weights.sum()
            key, family = env.record.qid, env.record.family
            old_baseline = baselines.get(key, family_baselines.get(family, reward))
            advantage = reward - old_baseline
            baselines[key] = reward if key not in baselines else 0.85 * old_baseline + 0.15 * reward
            family_old = family_baselines.get(family, reward)
            family_baselines[family] = 0.95 * family_old + 0.05 * reward
            loss = -action["logp"] * advantage - entropy_weight * action["entropy"]
            (loss / batch_size).backward()
            batch_rows.append({"qid": key, "reward": reward, "advantage": advantage,
                               "loss": float(loss.detach()),
                               "topology": int(action["topology"].item()),
                               "depth": int(action["depth"].item()), **parts})
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optim.step()
        if constrained:
            sample_constraint = float(np.mean([row["gated_min"] for row in batch_rows]))
            constraint_estimate = (sample_constraint if constraint_estimate is None
                                   else 0.9 * constraint_estimate + 0.1 * sample_constraint)
            mu = max(0.0, mu + eta * (tau - constraint_estimate))
        numeric = ("reward", "advantage", "loss", "hr10", "survival", "worst_survival")
        row = {"step": step, "mu": mu, "omega": omega.tolist(),
               **{key: float(np.mean([value[key] for value in batch_rows])) for key in numeric},
               "samples": batch_rows, "elapsed_s": round(time.perf_counter() - t0, 2)}
        history.append(row)
        if step <= 5 or step % 10 == 0 or step == steps:
            print(f"[detector-rl] { {k: v for k, v in row.items() if k != 'samples'} }", flush=True)
    synchronize(planner)
    elapsed = time.perf_counter() - t0
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    planner.save(out, base_model)
    write_json(out / "rl_log.json", history)
    write_json(out / "stage_cost.json", {
        "steps": steps, "batch_size": batch_size, "victim_retrains": steps * batch_size,
        "elapsed_s": elapsed, "victim_retrain_s": retrain_s,
        "gpu_hours": elapsed * hardware_info(planner).get("gpu_count", 0) / 3600,
        "checkpoint_selection": "final", "backbone_frozen": freeze_backbone,
    })
    return history
