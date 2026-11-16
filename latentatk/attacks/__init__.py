"""Attackers: baselines, a single-agent baseline, the role-switching group teacher,
and LatentAtk."""
from .base import Attacker, AttackResult
from .baselines import NoneAttack, RandomAttack, BandwagonAttack, PGAAttack
from .single_agent import SingleAgent
from .teacher import GroupTeacher
from .latentatk import LatentAtk

REGISTRY = {
    "none": NoneAttack,
    "random": RandomAttack,
    "bandwagon": BandwagonAttack,
    "pga": PGAAttack,
    "single": SingleAgent,
    "teacher": GroupTeacher,
    "latentatk": LatentAtk,
}


def build_attacker(name: str, **kw) -> Attacker:
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"unknown attacker '{name}', choose from {list(REGISTRY)}")
    return REGISTRY[key](**kw)
