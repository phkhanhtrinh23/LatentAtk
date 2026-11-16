"""Smoke tests: the pipeline runs and LatentAtk promotes the target.

Run with:  pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from latentatk.data import make_synthetic
from latentatk.eval import evaluate_all, pick_tail_target
from latentatk.victims import build_victim
from latentatk.attacks.latentatk import LatentAtk
from latentatk.seqdetect import (
    SequentialShillDetector, AdaptiveSequentialDefender,
    UserConditionedShillDetector, SessionConditionedShillDetector,
    MultiFamilyAdaptiveDefender,
)
from latentatk.stealth import (injected_item_distribution, benign_item_distribution,
                               kl, stealth_mix)


def _rows_by_name():
    ds = make_synthetic(n_users=150, n_items=300, density=0.06, seed=0)
    rows, target = evaluate_all(
        ds, victim_name="bpr-mf",
        attacker_names=["none", "random", "teacher", "latentatk"],
        budget=30, m=20, n_test=120, seed=0,
    )
    return {r["attacker"]: r for r in rows}


def test_pipeline_runs():
    rows = _rows_by_name()
    assert "LatentAtk (ours)" in rows
    assert rows["NoneAttack"]["Fake"] == 0


def test_leave_one_out_is_really_held_out():
    """Every evaluation positive must be absent from the fitted interaction matrix."""
    ds = make_synthetic(n_users=80, n_items=120, density=0.08, seed=17)
    assert ds.test
    for user, item in ds.test.items():
        assert item not in set(ds.user_items(user))


def test_latentatk_promotes_target():
    rows = _rows_by_name()
    # LatentAtk should lift the target well above the no-attack baseline
    assert rows["LatentAtk (ours)"]["HR@10"] >= rows["NoneAttack"]["HR@10"]


def test_latentatk_cheaper_than_teacher():
    rows = _rows_by_name()
    assert rows["LatentAtk (ours)"]["Tokens"] < rows["Multi-role teacher"]["Tokens"]
    assert rows["LatentAtk (ours)"]["Calls"] < rows["Multi-role teacher"]["Calls"]


def test_sequential_detector_is_anytime_valid():
    """Family-wise false-alarm rate on a benign (null) stream must stay <= alpha."""
    ds = make_synthetic(n_users=150, n_items=300, density=0.06, seed=0)
    det = SequentialShillDetector(ds, alpha=0.1)
    fwer_rate, fdr_rate = det.benign_flag_rate(n_events=600, n_trials=100, seed=0)
    assert fwer_rate <= 0.1 + 1e-9
    assert fdr_rate <= 0.1 + 1e-9


def test_stealth_reduces_kl_and_delays_detection():
    """More camouflage -> lower injected-vs-benign KL -> later (or no) detection."""
    ds = make_synthetic(n_users=150, n_items=300, density=0.06, seed=0)
    target = pick_tail_target(ds, 0)
    clean = build_victim("bpr-mf", seed=0).fit(ds)
    det = SequentialShillDetector(ds, alpha=0.05)
    p = benign_item_distribution(ds)

    def kl_and_time(level):
        atk = LatentAtk(budget=30, m=20, seed=0, role_mix=stealth_mix(level))
        prof = atk.attack(ds, target, probe=clean).profiles
        q = injected_item_distribution(prof, ds.n_items)
        r = det.monitor_profiles(prof, seed=0)
        t = r.t_events_fdr if np.isfinite(r.t_events_fdr) else float("inf")
        return kl(q, p), t

    kl_lo, t_lo = kl_and_time(0.0)     # Sniper-heavy
    kl_hi, t_hi = kl_and_time(0.95)    # Camouflageur/Inactive-heavy
    assert kl_hi < kl_lo               # camouflage shrinks the divergence
    assert t_hi >= t_lo                # ... and delays (or prevents) detection


def test_adaptive_defender_updates_only_between_campaigns():
    ds = make_synthetic(n_users=80, n_items=120, density=0.08, seed=4)
    defender = AdaptiveSequentialDefender(ds, alphas=(0.05,), smoothings=(0.1, 2.0))
    profiles = [[0, 1, 2, 3] for _ in range(20)]
    before = defender.weights.copy()
    result = defender.monitor_profiles(profiles, seed=0, update=True)
    assert result["round"] == 1
    assert np.isclose(defender.weights.sum(), 1.0)
    assert not np.shares_memory(before, defender.weights)


def test_conditioned_detectors_are_valid_under_their_benchmarks():
    ds = make_synthetic(n_users=100, n_items=150, density=0.06, seed=9)
    user = UserConditionedShillDetector(ds, alpha=0.1)
    session = SessionConditionedShillDetector(ds, alpha=0.1)
    ufwer, ufdr = user.benign_flag_rate(n_users=20, n_trials=60, seed=9)
    sfwer, sfdr = session.benign_flag_rate(
        n_profiles=20, profile_size=12, n_trials=60, seed=9)
    assert ufwer <= 0.1 and ufdr <= 0.1
    assert sfwer <= 0.1 and sfdr <= 0.1


def test_multi_family_defender_reports_every_monitor_family():
    ds = make_synthetic(n_users=80, n_items=120, density=0.08, seed=10)
    defender = MultiFamilyAdaptiveDefender(ds)
    result = defender.monitor_profiles([[1, 2, 3, 4] for _ in range(15)], seed=2)
    assert any(name.startswith("global") for name in result["family_survival"])
    assert any(name.startswith("user") for name in result["family_survival"])
    assert any(name.startswith("session") for name in result["family_survival"])
    assert np.isclose(defender.weights.sum(), 1.0)
