from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LAMBDAS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def benign_item_distribution(dataset, smoothing: float = 0.5) -> np.ndarray:
    pop = dataset.popularity.astype(float)
    p = pop + smoothing
    return p / p.sum()


def events_from_profiles(profiles, rng):
    items, owners = [], []
    for uidx, prof in enumerate(profiles):
        p = [int(x) for x in prof]
        rng.shuffle(p)
        items.extend(p)
        owners.extend([uidx] * len(p))
    return np.asarray(items, dtype=int), np.asarray(owners, dtype=int)


@dataclass
class DetectionResult:
    detected: bool
    t_events_fwer: float
    t_events_fdr: float
    t_users_fwer: float
    t_users_fdr: float
    n_events: int
    log_evalue: float


class SequentialShillDetector:
    def __init__(self, dataset, alpha: float = 0.05, lambdas=DEFAULT_LAMBDAS,
                 smoothing: float = 0.5):
        self.alpha = float(alpha)
        self.n_items = int(dataset.n_items)
        self.p = benign_item_distribution(dataset, smoothing)
        self.lam = np.asarray(lambdas, dtype=float)[:, None]
        floor = 1e-12
        self.logmiss = np.log(np.clip(1.0 - self.lam * self.p[None, :], floor, None))
        self.loghit = np.log(np.clip(1.0 + self.lam * (1.0 - self.p[None, :]), floor, None))
        self.logL = np.log(self.lam.shape[0])
        self.logb_fwer = np.log(self.n_items / self.alpha)
        self.log_ranks = np.log(np.arange(1, self.n_items + 1))

    def monitor(self, items, owners=None) -> DetectionResult:
        items = np.asarray(items, dtype=int)
        n = len(items)
        if n == 0:
            return DetectionResult(False, np.inf, np.inf, np.inf, np.inf, 0, -np.inf)

        L = np.zeros_like(self.logmiss)
        t_fwer = t_fdr = np.inf
        logb = self.logb_fwer
        logb_ebh_base = np.log(self.n_items / self.alpha)
        max_logM = -np.inf

        for t in range(n):
            h = items[t]
            L += self.logmiss
            L[:, h] += self.loghit[:, h] - self.logmiss[:, h]
            M = _logsumexp_axis0(L) - self.logL
            cur_max = float(M.max())
            max_logM = max(max_logM, cur_max)

            if not np.isfinite(t_fwer) and cur_max >= logb:
                t_fwer = t + 1
            if not np.isfinite(t_fdr):
                Msort = np.sort(M)[::-1]
                if np.any(Msort >= (logb_ebh_base - self.log_ranks)):
                    t_fdr = t + 1
            if np.isfinite(t_fwer) and np.isfinite(t_fdr):
                break

        def users_at(te):
            if not np.isfinite(te) or owners is None:
                return np.inf if not np.isfinite(te) else float("nan")
            return int(owners[int(te) - 1]) + 1

        detected = np.isfinite(t_fwer) or np.isfinite(t_fdr)
        return DetectionResult(
            detected=detected,
            t_events_fwer=t_fwer, t_events_fdr=t_fdr,
            t_users_fwer=users_at(t_fwer), t_users_fdr=users_at(t_fdr),
            n_events=n, log_evalue=max_logM / np.log(10),
        )

    def monitor_profiles(self, profiles, seed: int = 0) -> DetectionResult:
        rng = np.random.default_rng(seed)
        items, owners = events_from_profiles(profiles, rng)
        return self.monitor(items, owners)

    def benign_flag_rate(self, n_events: int, n_trials: int = 200, seed: int = 0):
        rng = np.random.default_rng(seed)
        fwer = fdr = 0
        for _ in range(n_trials):
            stream = rng.choice(self.n_items, size=n_events, p=self.p)
            r = self.monitor(stream)
            fwer += int(np.isfinite(r.t_events_fwer))
            fdr += int(np.isfinite(r.t_events_fdr))
        return fwer / n_trials, fdr / n_trials


class AdaptiveSequentialDefender:
    def __init__(self, dataset, alphas=(0.01, 0.05, 0.10),
                 smoothings=(0.1, 0.5, 2.0), eta: float = 0.7):
        self.monitors = [SequentialShillDetector(dataset, alpha=a, smoothing=s)
                         for a in alphas for s in smoothings]
        self.weights = np.full(len(self.monitors), 1.0 / len(self.monitors))
        self.eta = float(eta)
        self.round = 0

    def monitor_profiles(self, profiles, seed: int = 0, update: bool = True):
        results = [m.monitor_profiles(profiles, seed=seed) for m in self.monitors]
        survival = np.asarray([
            1.0 if not np.isfinite(r.t_events_fdr) else r.t_events_fdr / max(1, r.n_events)
            for r in results
        ], dtype=float)
        weighted_survival = float(self.weights @ survival)
        worst_survival = float(survival.min())
        before = self.weights.copy()
        if update:
            self.weights *= np.exp(-self.eta * survival)
            self.weights /= self.weights.sum()
            self.round += 1
        return {
            "weighted_survival": weighted_survival,
            "worst_survival": worst_survival,
            "component_survival": survival.tolist(),
            "weights_before": before.tolist(),
            "weights_after": self.weights.tolist(),
            "round": self.round,
            "results": results,
        }


class UserConditionedShillDetector:
    family = "user_inclusion"

    def __init__(self, dataset, alpha: float = 0.05, lambdas=DEFAULT_LAMBDAS,
                 smoothing: float = 0.5):
        self.alpha = float(alpha)
        self.n_items = int(dataset.n_items)
        genuine = dataset.R[:dataset.n_genuine_users]
        included = np.asarray((genuine > 0).sum(axis=0)).ravel().astype(float)
        self.p = (included + smoothing) / (dataset.n_genuine_users + 2 * smoothing)
        self.lam = np.asarray(lambdas, dtype=float)[:, None]
        self.logmiss = np.log(np.clip(1.0 - self.lam * self.p[None, :], 1e-12, None))
        self.loghit = np.log(np.clip(
            1.0 + self.lam * (1.0 - self.p[None, :]), 1e-12, None))
        self.logL = np.log(len(lambdas))
        self.logb_fwer = np.log(self.n_items / self.alpha)
        self.log_ranks = np.log(np.arange(1, self.n_items + 1))

    def monitor_profiles(self, profiles, seed: int = 0):
        del seed
        L = np.zeros_like(self.logmiss)
        t_fwer = t_fdr = np.inf
        events = users = 0
        event_fwer = event_fdr = np.inf
        max_logM = -np.inf
        for uidx, profile in enumerate(profiles):
            events += len(profile); users += 1
            hits = np.unique(np.asarray(profile, dtype=int))
            L += self.logmiss
            L[:, hits] += self.loghit[:, hits] - self.logmiss[:, hits]
            M = _logsumexp_axis0(L) - self.logL
            cur_max = float(M.max()); max_logM = max(max_logM, cur_max)
            if not np.isfinite(t_fwer) and cur_max >= self.logb_fwer:
                t_fwer, event_fwer = users, events
            if not np.isfinite(t_fdr):
                threshold = self.logb_fwer - self.log_ranks
                if np.any(np.sort(M)[::-1] >= threshold):
                    t_fdr, event_fdr = users, events
            if np.isfinite(t_fwer) and np.isfinite(t_fdr):
                break
        n_events = sum(len(profile) for profile in profiles)
        return DetectionResult(
            np.isfinite(t_fwer) or np.isfinite(t_fdr), event_fwer, event_fdr,
            t_fwer, t_fdr, n_events, max_logM / np.log(10))

    def benign_flag_rate(self, n_users=30, n_trials=200, seed=0):
        rng = np.random.default_rng(seed)
        fwer = fdr = 0
        for _ in range(n_trials):
            profiles = [np.flatnonzero(rng.random(self.n_items) < self.p).tolist()
                        for _ in range(n_users)]
            result = self.monitor_profiles(profiles)
            fwer += int(np.isfinite(result.t_users_fwer))
            fdr += int(np.isfinite(result.t_users_fdr))
        return fwer / n_trials, fdr / n_trials


class SessionConditionedShillDetector:
    family = "session_phase"

    def __init__(self, dataset, alpha: float = 0.05, lambdas=DEFAULT_LAMBDAS,
                 smoothing: float = 0.5, n_phases: int = 3, seed: int = 1729):
        self.alpha = float(alpha)
        self.n_items = int(dataset.n_items)
        self.n_phases = int(n_phases)
        counts = np.full((self.n_phases, self.n_items), float(smoothing))
        rng = np.random.default_rng(seed)
        for user in range(dataset.n_genuine_users):
            items = np.asarray(dataset.user_items(user), dtype=int).copy()
            rng.shuffle(items)
            for position, item in enumerate(items):
                phase = min(self.n_phases - 1,
                            int(self.n_phases * position / max(1, len(items))))
                counts[phase, item] += 1
        self.p = counts / counts.sum(axis=1, keepdims=True)
        self.lam = np.asarray(lambdas, dtype=float)[:, None, None]
        p = self.p[None, :, :]
        self.logmiss = np.log(np.clip(1.0 - self.lam * p, 1e-12, None))
        self.loghit = np.log(np.clip(1.0 + self.lam * (1.0 - p), 1e-12, None))
        self.logL = np.log(len(lambdas))
        self.n_hypotheses = self.n_items
        self.logb_fwer = np.log(self.n_hypotheses / self.alpha)
        self.log_ranks = np.log(np.arange(1, self.n_hypotheses + 1))

    def _monitor_contextual(self, items, phases, owners=None):
        L = np.zeros((self.logmiss.shape[0], self.n_items), dtype=float)
        t_fwer = t_fdr = np.inf
        max_logM = -np.inf
        for t, (item, phase) in enumerate(zip(items, phases), 1):
            L += self.logmiss[:, phase, :]
            L[:, item] += (self.loghit[:, phase, item] -
                           self.logmiss[:, phase, item])
            M = _logsumexp_axis0(L) - self.logL
            cur_max = float(M.max()); max_logM = max(max_logM, cur_max)
            if not np.isfinite(t_fwer) and cur_max >= self.logb_fwer:
                t_fwer = t
            if not np.isfinite(t_fdr):
                if np.any(np.sort(M)[::-1] >= self.logb_fwer - self.log_ranks):
                    t_fdr = t
            if np.isfinite(t_fwer) and np.isfinite(t_fdr):
                break

        def users_at(event):
            if not np.isfinite(event) or owners is None:
                return np.inf if not np.isfinite(event) else float("nan")
            return int(owners[int(event) - 1]) + 1

        return DetectionResult(
            np.isfinite(t_fwer) or np.isfinite(t_fdr), t_fwer, t_fdr,
            users_at(t_fwer), users_at(t_fdr), len(items), max_logM / np.log(10))

    def monitor_profiles(self, profiles, seed: int = 0):
        rng = np.random.default_rng(seed)
        items, phases, owners = [], [], []
        for user, profile in enumerate(profiles):
            ordered = np.asarray(profile, dtype=int).copy(); rng.shuffle(ordered)
            for position, item in enumerate(ordered):
                items.append(int(item)); owners.append(user)
                phases.append(min(self.n_phases - 1,
                                  int(self.n_phases * position / max(1, len(ordered)))))
        return self._monitor_contextual(np.asarray(items), np.asarray(phases),
                                        np.asarray(owners))

    def benign_flag_rate(self, n_profiles=30, profile_size=20, n_trials=200, seed=0):
        rng = np.random.default_rng(seed)
        fwer = fdr = 0
        phases = np.tile(np.minimum(
            self.n_phases - 1,
            (self.n_phases * np.arange(profile_size) / profile_size).astype(int)), n_profiles)
        for _ in range(n_trials):
            items = np.asarray([rng.choice(self.n_items, p=self.p[phase])
                                for phase in phases])
            result = self._monitor_contextual(items, phases)
            fwer += int(np.isfinite(result.t_events_fwer))
            fdr += int(np.isfinite(result.t_events_fdr))
        return fwer / n_trials, fdr / n_trials


class MultiFamilyAdaptiveDefender:
    def __init__(self, dataset, eta: float = 0.5):
        self.entries = [
            ("global_a01", SequentialShillDetector(dataset, alpha=0.01, smoothing=0.5)),
            ("global_a05", SequentialShillDetector(dataset, alpha=0.05, smoothing=0.5)),
            ("global_s2", SequentialShillDetector(dataset, alpha=0.05, smoothing=2.0)),
            ("user_a01", UserConditionedShillDetector(dataset, alpha=0.01)),
            ("user_a05", UserConditionedShillDetector(dataset, alpha=0.05)),
            ("session_a01", SessionConditionedShillDetector(dataset, alpha=0.01)),
            ("session_a05", SessionConditionedShillDetector(dataset, alpha=0.05)),
        ]
        self.weights = np.full(len(self.entries), 1.0 / len(self.entries))
        self.eta = float(eta)
        self.round = 0

    def monitor_profiles(self, profiles, seed: int = 0, update: bool = True):
        results = [monitor.monitor_profiles(profiles, seed=seed)
                   for _, monitor in self.entries]
        survival = np.asarray([
            1.0 if not np.isfinite(result.t_events_fdr) else
            result.t_events_fdr / max(1, result.n_events) for result in results
        ])
        before = self.weights.copy()
        weighted = float(before @ survival)
        if update:
            self.weights *= np.exp(-self.eta * survival)
            self.weights /= self.weights.sum()
            self.round += 1
        names = [name for name, _ in self.entries]
        return {
            "weighted_survival": weighted,
            "worst_survival": float(survival.min()),
            "family_survival": dict(zip(names, survival.tolist())),
            "component_survival": survival.tolist(),
            "weights_before": before.tolist(), "weights_after": self.weights.tolist(),
            "round": self.round, "results": results,
        }


def _logsumexp_axis0(A: np.ndarray) -> np.ndarray:
    mx = A.max(axis=0)
    return mx + np.log(np.exp(A - mx).sum(axis=0))


def _genuine_item_distribution(dataset, floor: float = 1e-9) -> np.ndarray:
    genuine = dataset.R[:dataset.n_genuine_users]
    counts = np.asarray(genuine.sum(axis=0)).ravel().astype(float)
    total = counts.sum()
    q = counts / total if total > 0 else np.full(dataset.n_items, 1.0 / dataset.n_items)
    q = np.clip(q, floor, None)
    return q / q.sum()


def _genuine_inclusion_distribution(dataset, floor: float = 1e-9) -> np.ndarray:
    genuine = dataset.R[:dataset.n_genuine_users]
    included = np.asarray((genuine > 0).sum(axis=0)).ravel().astype(float)
    total = included.sum()
    q = included / total if total > 0 else np.full(dataset.n_items, 1.0 / dataset.n_items)
    q = np.clip(q, floor, None)
    return q / q.sum()


def _genuine_phase_distributions(dataset, n_phases: int = 3, seed: int = 1729,
                                 floor: float = 1e-9) -> np.ndarray:
    counts = np.zeros((n_phases, dataset.n_items), float)
    rng = np.random.default_rng(seed)
    for user in range(dataset.n_genuine_users):
        items = np.asarray(dataset.user_items(user), dtype=int).copy()
        rng.shuffle(items)
        for position, item in enumerate(items):
            phase = min(n_phases - 1, int(n_phases * position / max(1, len(items))))
            counts[phase, int(item)] += 1.0
    q = np.empty_like(counts)
    for phase in range(n_phases):
        total = counts[phase].sum()
        row = (counts[phase] / total if total > 0
               else np.full(dataset.n_items, 1.0 / dataset.n_items))
        q[phase] = np.clip(row, floor, None)
        q[phase] /= q[phase].sum()
    return q


class _DirichletMultinomialBayesFactor:
    # log E += log(a_i + n_i) - log(a0 + t) - log q0_i, a = concentration * q0.
    def __init__(self, q0, concentration):
        self.q0 = np.asarray(q0, float)
        self.logq0 = np.log(self.q0)
        self.prior = float(concentration) * self.q0
        self.prior_sum = float(concentration)
        self.counts = np.zeros_like(self.q0)
        self.total = 0.0
        self.log_e = 0.0

    def update(self, item):
        i = int(item)
        self.log_e += (np.log(self.prior[i] + self.counts[i])
                       - np.log(self.prior_sum + self.total) - self.logq0[i])
        self.counts[i] += 1.0
        self.total += 1.0
        return self.log_e


class GlobalFrequencyEMonitor:
    family = "global_frequency"

    def __init__(self, dataset, alpha: float = 0.05, concentration: float = 0.5):
        self.alpha = float(alpha)
        self.concentration = float(concentration)
        self.q0 = _genuine_item_distribution(dataset)
        self.log_threshold = np.log(1.0 / self.alpha)

    def monitor_profiles(self, profiles, seed: int = 0) -> DetectionResult:
        rng = np.random.default_rng(seed)
        items, _ = events_from_profiles(profiles, rng)
        n = len(items)
        if n == 0:
            return DetectionResult(False, np.inf, np.inf, np.inf, np.inf, 0, -np.inf)
        process = _DirichletMultinomialBayesFactor(self.q0, self.concentration)
        t_alarm, max_log = np.inf, -np.inf
        for t, item in enumerate(items, 1):
            log_e = process.update(item)
            max_log = max(max_log, log_e)
            if log_e >= self.log_threshold:
                t_alarm = t
                break
        return DetectionResult(np.isfinite(t_alarm), t_alarm, t_alarm, np.inf, np.inf,
                               n, max_log / np.log(10))


class UserInclusionEMonitor:
    family = "user_inclusion"

    def __init__(self, dataset, alpha: float = 0.05, concentration: float = 0.5):
        self.alpha = float(alpha)
        self.concentration = float(concentration)
        self.q0 = _genuine_inclusion_distribution(dataset)
        self.log_threshold = np.log(1.0 / self.alpha)

    def monitor_profiles(self, profiles, seed: int = 0) -> DetectionResult:
        del seed
        process = _DirichletMultinomialBayesFactor(self.q0, self.concentration)
        n_users = len(profiles)
        t_alarm, max_log = np.inf, -np.inf
        for user, profile in enumerate(profiles, 1):
            for item in np.unique(np.asarray(profile, dtype=int)):
                process.update(item)
            max_log = max(max_log, process.log_e)
            if process.log_e >= self.log_threshold:
                t_alarm = user
                break
        return DetectionResult(np.isfinite(t_alarm), t_alarm, t_alarm, t_alarm, t_alarm,
                               max(1, n_users), max_log / np.log(10))


class SessionPhaseEMonitor:
    family = "session_phase"

    def __init__(self, dataset, alpha: float = 0.05, concentration: float = 0.5,
                 n_phases: int = 3, seed: int = 1729):
        self.alpha = float(alpha)
        self.concentration = float(concentration)
        self.n_phases = int(n_phases)
        self.q0 = _genuine_phase_distributions(dataset, self.n_phases, seed)
        self.log_threshold = np.log(1.0 / self.alpha)

    def monitor_profiles(self, profiles, seed: int = 0) -> DetectionResult:
        rng = np.random.default_rng(seed)
        phase_processes = [_DirichletMultinomialBayesFactor(self.q0[phase], self.concentration)
                           for phase in range(self.n_phases)]
        t_alarm, max_log, n = np.inf, -np.inf, 0
        for profile in profiles:
            ordered = np.asarray(profile, dtype=int).copy()
            rng.shuffle(ordered)
            length = len(ordered)
            for position, item in enumerate(ordered):
                phase = min(self.n_phases - 1, int(self.n_phases * position / max(1, length)))
                phase_processes[phase].update(int(item))
                n += 1
                combined = sum(p.log_e for p in phase_processes)
                max_log = max(max_log, combined)
                if combined >= self.log_threshold:
                    t_alarm = n
                    break
            if np.isfinite(t_alarm):
                break
        total = sum(len(p) for p in profiles)
        return DetectionResult(np.isfinite(t_alarm), t_alarm, t_alarm, np.inf, np.inf,
                               max(1, total), max_log / np.log(10))


class PaperMonitorEnsemble:
    DEFAULT_ENTRIES = (
        {"name": "global_a01_c05", "kind": "global", "alpha": 0.01, "concentration": 0.5},
        {"name": "global_a05_c05", "kind": "global", "alpha": 0.05, "concentration": 0.5},
        {"name": "global_a05_c2", "kind": "global", "alpha": 0.05, "concentration": 2.0},
        {"name": "user_a01", "kind": "user", "alpha": 0.01},
        {"name": "user_a05", "kind": "user", "alpha": 0.05},
        {"name": "session_a01", "kind": "session", "alpha": 0.01},
        {"name": "session_a05", "kind": "session", "alpha": 0.05},
    )

    def __init__(self, dataset, settings=None):
        settings = settings or {}
        if settings.get("name", "paper_v1") != "paper_v1":
            raise ValueError("unknown detector ensemble")
        entries = settings.get("entries", self.DEFAULT_ENTRIES)
        if not entries or len({entry["name"] for entry in entries}) != len(entries):
            raise ValueError("detector entries require unique names and a nonempty ensemble")
        constructors = {"global": GlobalFrequencyEMonitor, "user": UserInclusionEMonitor,
                        "session": SessionPhaseEMonitor}
        self.entries = []
        for entry in entries:
            options = {key: value for key, value in entry.items() if key not in {"name", "kind"}}
            if entry["kind"] not in constructors or not 0 < options.get("alpha", 0.05) < 1:
                raise ValueError("invalid monitor kind or alpha")
            if options.get("concentration", 0.5) <= 0 or options.get("n_phases", 3) < 1:
                raise ValueError("concentration and phase count must be positive")
            self.entries.append((entry["name"], constructors[entry["kind"]](dataset, **options)))
        self.round = 0

    def monitor_profiles(self, profiles, seed: int = 0, update: bool = False):
        results = [monitor.monitor_profiles(profiles, seed=seed)
                   for _, monitor in self.entries]
        survival = np.asarray([
            1.0 if not np.isfinite(result.t_events_fdr) else
            result.t_events_fdr / max(1, result.n_events) for result in results
        ])
        if update:
            self.round += 1
        names = [name for name, _ in self.entries]
        return {
            "worst_survival": float(survival.min()),
            "component_survival": survival.tolist(),
            "family_survival": dict(zip(names, survival.tolist())),
            "round": self.round, "results": results,
        }
