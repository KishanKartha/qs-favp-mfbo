"""
Fidelity-Aware Verification Protocol (FAVP) and the catastrophic-noise
synthetic benchmark.

This module provides the verification layer that sits on top of the
QS-MFBO framework defined in qsmfbo.core. It contains:

  - OutlierInjector         wraps a test function with probabilistic
                            catastrophic outliers at the cheap fidelity
  - AnomalyEvent            record of a single FAVP verification episode
  - QueueSchedulerFAVP      QueueScheduler extended with Algorithm 2
                            (detect - repeat - resolve)
  - QueueSchedulerNaive     ablation baseline that rejects any z > tau
                            observation without verification
  - compute_true_regret     regret computed against the clean function,
                            so corrupted observations cannot produce
                            fake-good numbers
  - run_favp_benchmark      four-method benchmark used for Fig. 2 and
                            Extended Data Table 1 in the paper

All four methods share initialisation within a seed. The runner returns
both the per-seed logs and the diagnostics dict (FAVP event lists,
injection counts, naive-rejection counts) needed for the supplementary
tables.

Reference:
    Kartha, K. & James, A. P. Cost-aware multi-fidelity scheduling and
    cross-fidelity anomaly resolution for iterative learning under
    laboratory constraints. (2026). See Section 2.1, Algorithm 2.
"""

import copy
import time
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .core import (
    FIDELITY_CHEAP,
    FIDELITY_EXPENSIVE,
    Observation,
    QueueItem,
    QueueScheduler,
    StandardMFMES,
    FixedCostModel,
    QueueDependentCostModel,
    StyblinskiTang2D,
    Branin2D,
    Hartmann6D,
    generate_shared_init,
)


# =============================================================================
# Outlier Injection Wrapper
# =============================================================================

class OutlierInjector:
    """Wraps a test function and injects catastrophic outliers.

    With probability p_outlier, replaces the true observation at the cheap
    fidelity with a uniform random draw over the function's output range.
    The expensive fidelity is treated as ground truth and is not corrupted.

    Provides get_true_function() to access the underlying clean function
    for regret computation (see compute_true_regret below).
    """
    def __init__(self, test_function, p_outlier=0.08, seed=0):
        self.test_fn = test_function
        self.p_outlier = p_outlier
        self.dim = test_function.dim
        self.n_fidelities = test_function.n_fidelities
        self.optimal_value = test_function.optimal_value
        self.rng = np.random.RandomState(seed + 9999)

        self._estimate_output_range()

        self.injections = []
        self.total_evals = 0

    def _estimate_output_range(self):
        ys = []
        for _ in range(500):
            x = torch.rand(self.dim)
            y = self.test_fn.evaluate(x, FIDELITY_EXPENSIVE, noise_std=0.0)
            ys.append(y)
        self.y_min = min(ys)
        self.y_max = max(ys)
        margin = 0.2 * (self.y_max - self.y_min)
        self.y_min -= margin
        self.y_max += margin

    def evaluate(self, x_norm, fidelity, noise_std=0.0):
        """Evaluate with possible outlier injection at the cheap fidelity only.

        The highest fidelity is ground truth by definition; outliers are only
        injected into cheap measurements. This matches the FAVP design:
        anomalies at cheap fidelity get escalated to HF precisely because HF
        is the definitive measurement.
        """
        self.total_evals += 1
        y_true = self.test_fn.evaluate(x_norm, fidelity, noise_std)
        if fidelity < 0.5 and self.rng.rand() < self.p_outlier:
            y_outlier = self.rng.uniform(self.y_min, self.y_max)
            self.injections.append({
                'eval_num': self.total_evals,
                'x': x_norm.clone(),
                'fidelity': fidelity,
                'y_true': y_true,
                'y_outlier': y_outlier,
            })
            return y_outlier
        return y_true

    def evaluate_clean(self, x_norm, fidelity, noise_std=0.0):
        """Evaluate without any outlier injection (used for FAVP's repeat step
        so that a repeat measurement is not itself corrupted; corrupting the
        repeat would conflate FAVP's resolution logic with injection noise)."""
        self.total_evals += 1
        return self.test_fn.evaluate(x_norm, fidelity, noise_std)

    def get_true_function(self):
        """Return the underlying clean test function (for regret computation)."""
        return self.test_fn

    def reset(self, seed=0):
        self.rng = np.random.RandomState(seed + 9999)
        self.injections = []
        self.total_evals = 0


# =============================================================================
# True-Regret Helper
# =============================================================================

def compute_true_regret(observations, test_function_class, optimal_value):
    """Regret computed against the clean (noiseless, no-outlier) function.

    Finds the x that produced the best observed HF value in `observations`,
    then evaluates the true function at that x to get the actual regret. This
    prevents corrupted outliers from producing a fake-good regret number.
    """
    if hasattr(test_function_class, 'get_true_function'):
        true_fn = test_function_class.get_true_function()
    elif hasattr(test_function_class, 'test_fn'):
        true_fn = test_function_class.test_fn
    else:
        true_fn = test_function_class

    hf_obs = [o for o in observations if o.fidelity >= 0.5]
    if not hf_obs:
        hf_obs = list(observations)
    if not hf_obs:
        return float('inf')

    best_true_y = -float('inf')
    for o in hf_obs:
        true_y = true_fn.evaluate(o.x, FIDELITY_EXPENSIVE, noise_std=0.0)
        if true_y > best_true_y:
            best_true_y = true_y

    return optimal_value - best_true_y


# =============================================================================
# FAVP Anomaly Event
# =============================================================================

@dataclass
class AnomalyEvent:
    """Record of one FAVP verification episode.

    The `case` field takes values:
      'A1_genuine'  replicate confirms; both observations kept.
      'A2_escalate' replicate contradicts; point escalated to the next
                    fidelity queue; both observations still kept.
    """
    iteration: int
    x: torch.Tensor
    fidelity: float
    y_observed: float
    gp_mean: float
    gp_std: float
    z_score: float
    case: str
    y_repeat: Optional[float] = None
    was_real_outlier: bool = False


# =============================================================================
# QS-MFBO + FAVP
# =============================================================================

class QueueSchedulerFAVP(QueueScheduler):
    """QueueScheduler extended with the FAVP verification protocol.

    Implements Algorithm 2 of the paper:
      1. Compute standardised residual z of each new cheap observation.
      2. If z > tau, repeat at the same fidelity.
      3. If |y_repeat - y_original| <= gamma * sigma_t, Case A1 (genuine extreme).
      4. Otherwise Case A2: escalate the point to the expensive queue.

    Both the original and repeat observations are retained in all cases.
    Nothing is ever discarded. The consistency threshold
    epsilon = gamma * sigma_t(x, m) is self-calibrating: it widens when the
    surrogate is uncertain and tightens as the model sharpens.

    An N_min warmup period suppresses verification until enough observations
    have accumulated to compute meaningful residuals.
    """
    def __init__(self, test_function, cost_model, outlier_injector=None,
                 tau=3.0, gamma=1.5, n_min=15, **kwargs):
        super().__init__(test_function, cost_model, **kwargs)
        self.injector = outlier_injector
        self.tau = tau
        self.gamma = gamma
        self.n_min = n_min
        self.anomaly_events = []

    def _compute_z_score(self, x, fidelity, y_obs):
        if self.model is None:
            return 0.0, 0.0, 1.0
        x_fid = torch.cat([x, torch.tensor([fidelity])]).unsqueeze(0)
        with torch.no_grad():
            post = self.model.posterior(x_fid)
            mu = post.mean.squeeze().item()
            sigma = post.variance.squeeze().sqrt().item()
        if sigma < 1e-6:
            sigma = 1e-6
        return abs(y_obs - mu) / sigma, mu, sigma

    def _is_warmup(self):
        return len(self.observations) < self.n_min

    def _was_injected_outlier(self, n_inj_before):
        if self.injector is None:
            return False
        return len(self.injector.injections) > n_inj_before

    def run_iteration(self):
        self.iteration += 1
        q = len(self.queue)

        if not self._fit_model():
            x = torch.rand(self.dim)
            y = self.test_fn.evaluate(x, FIDELITY_CHEAP, self.noise_std)
            cost = self.cost_model.cost_per_sample(q, FIDELITY_CHEAP)
            self.cumulative_cost += cost
            self.observations.append(Observation(x, FIDELITY_CHEAP, y, cost, self.iteration))
            return self._make_log('execute_cheap', x=x, fidelity=FIDELITY_CHEAP, cost=cost)

        try:
            x_star, s_star, acq_imm = self._mfmes_suggest()
        except Exception:
            x = torch.rand(self.dim)
            y = self.test_fn.evaluate(x, FIDELITY_CHEAP, self.noise_std)
            cost = self.cost_model.cost_per_sample(q, FIDELITY_CHEAP)
            self.cumulative_cost += cost
            self.observations.append(Observation(x, FIDELITY_CHEAP, y, cost, self.iteration))
            return self._make_log('execute_cheap', x=x, fidelity=FIDELITY_CHEAP, cost=cost)

        if s_star >= 0.5:
            self.queue.append(QueueItem(x_star, FIDELITY_EXPENSIVE, self.iteration))
            if len(self.queue) >= self.q_max:
                return self._execute_batch_favp()
            return self._make_log('add_to_queue', x=x_star, fidelity=FIDELITY_EXPENSIVE,
                                  acq_value=acq_imm, cost=0.0)

        execute_batch = False
        if len(self.queue) >= self.q_min:
            if self.use_adaptive_trigger:
                G_hat = self._greedy_queue_valuation()
                C_batch = self.cost_model.batch_cost(len(self.queue))
                alpha_batch = G_hat / C_batch if C_batch > 0 else float('inf')
                if alpha_batch >= acq_imm:
                    execute_batch = True
            else:
                if len(self.queue) >= self.fixed_trigger_threshold:
                    execute_batch = True

        if execute_batch:
            return self._execute_batch_favp()

        # Execute a cheap observation, with FAVP verification after N_min.
        n_inj_before = len(self.injector.injections) if self.injector else 0
        y = self.test_fn.evaluate(x_star, FIDELITY_CHEAP, self.noise_std)
        is_injected_outlier = self._was_injected_outlier(n_inj_before)

        cost = self.cost_model.cost_per_sample(q, FIDELITY_CHEAP)
        self.cumulative_cost += cost

        if self._is_warmup():
            self.observations.append(Observation(x_star, FIDELITY_CHEAP, y, cost, self.iteration))
            return self._make_log('execute_cheap', x=x_star, fidelity=FIDELITY_CHEAP,
                                  acq_value=acq_imm, cost=cost)

        z, mu, sigma = self._compute_z_score(x_star, FIDELITY_CHEAP, y)

        if z > self.tau:
            # Flagged: repeat at the same fidelity.
            y_repeat = self.test_fn.evaluate_clean(x_star, FIDELITY_CHEAP, self.noise_std)
            repeat_cost = self.cost_model.cost_per_sample(q, FIDELITY_CHEAP)
            self.cumulative_cost += repeat_cost

            delta = abs(y_repeat - y)
            epsilon = self.gamma * sigma

            # Accept both observations in all cases.
            self.observations.append(Observation(x_star, FIDELITY_CHEAP, y, cost, self.iteration))
            self.observations.append(Observation(x_star, FIDELITY_CHEAP, y_repeat, repeat_cost, self.iteration))

            if delta <= epsilon:
                # Case A1: replicate confirms -- genuine extreme.
                self.anomaly_events.append(AnomalyEvent(
                    self.iteration, x_star, FIDELITY_CHEAP, y, mu, sigma, z,
                    'A1_genuine', y_repeat, is_injected_outlier))
                return self._make_log('favp_A1_genuine', x=x_star, fidelity=FIDELITY_CHEAP,
                                      cost=cost + repeat_cost)
            else:
                # Case A2: replicate contradicts -- escalate to HF queue.
                self.queue.append(QueueItem(x_star, FIDELITY_EXPENSIVE, self.iteration))
                self.anomaly_events.append(AnomalyEvent(
                    self.iteration, x_star, FIDELITY_CHEAP, y, mu, sigma, z,
                    'A2_escalate', y_repeat, is_injected_outlier))
                return self._make_log('favp_A2_escalate', x=x_star, fidelity=FIDELITY_CHEAP,
                                      cost=cost + repeat_cost)
        else:
            self.observations.append(Observation(x_star, FIDELITY_CHEAP, y, cost, self.iteration))
            return self._make_log('execute_cheap', x=x_star, fidelity=FIDELITY_CHEAP,
                                  acq_value=acq_imm, cost=cost)

    def _execute_batch_favp(self):
        """Execute the expensive batch. No anomaly check on the highest fidelity.

        The highest fidelity is the ground truth that we escalate TO. FAVP
        operates only on lower fidelities, where contradictions can be resolved
        by promotion. In a multi-fidelity setting every non-terminal fidelity
        gets detect - repeat - escalate; only the top fidelity is taken at face
        value.
        """
        q = len(self.queue)
        bc = self.cost_model.batch_cost(q)
        ps = bc / q if q > 0 else 0
        self.batch_counter += 1
        self.n_facility_visits += 1

        for item in self.queue:
            y = self.test_fn.evaluate(item.x, FIDELITY_EXPENSIVE, self.noise_std)
            self.observations.append(Observation(item.x, FIDELITY_EXPENSIVE, y, ps,
                                                 self.iteration, self.batch_counter))

        self.cumulative_cost += bc
        bs = q
        self.queue.clear()
        return self._make_log('execute_batch', batch_size=bs, cost=bc)


# =============================================================================
# QS-MFBO + Naive Rejection (ablation baseline)
# =============================================================================

class QueueSchedulerNaive(QueueScheduler):
    """Ablation: discard any observation with z > tau without verification.

    Uses the same N_min warmup as FAVP for a fair comparison. Tracks whether
    each rejected observation was a genuine extreme (injected_outlier == False)
    or a real catastrophic outlier (injected_outlier == True), so that the
    downstream summary can quantify how often naive rejection throws away
    real information.
    """
    def __init__(self, test_function, cost_model, outlier_injector=None,
                 tau=3.0, n_min=15, **kwargs):
        super().__init__(test_function, cost_model, **kwargs)
        self.injector = outlier_injector
        self.tau = tau
        self.n_min = n_min
        self.rejected_genuine = 0
        self.rejected_outliers = 0
        self.accepted_outliers = 0
        self.rejection_events = []

    def _compute_z_score(self, x, fidelity, y_obs):
        if self.model is None:
            return 0.0, 0.0, 1.0
        x_fid = torch.cat([x, torch.tensor([fidelity])]).unsqueeze(0)
        with torch.no_grad():
            post = self.model.posterior(x_fid)
            mu = post.mean.squeeze().item()
            sigma = post.variance.squeeze().sqrt().item()
        if sigma < 1e-6:
            sigma = 1e-6
        return abs(y_obs - mu) / sigma, mu, sigma

    def _is_warmup(self):
        return len(self.observations) < self.n_min

    def _was_injected_outlier(self, n_inj_before):
        if self.injector is None:
            return False
        return len(self.injector.injections) > n_inj_before

    def run_iteration(self):
        self.iteration += 1
        q = len(self.queue)

        if not self._fit_model():
            x = torch.rand(self.dim)
            y = self.test_fn.evaluate(x, FIDELITY_CHEAP, self.noise_std)
            cost = self.cost_model.cost_per_sample(q, FIDELITY_CHEAP)
            self.cumulative_cost += cost
            self.observations.append(Observation(x, FIDELITY_CHEAP, y, cost, self.iteration))
            return self._make_log('execute_cheap', x=x, fidelity=FIDELITY_CHEAP, cost=cost)

        try:
            x_star, s_star, acq_imm = self._mfmes_suggest()
        except Exception:
            x = torch.rand(self.dim)
            y = self.test_fn.evaluate(x, FIDELITY_CHEAP, self.noise_std)
            cost = self.cost_model.cost_per_sample(q, FIDELITY_CHEAP)
            self.cumulative_cost += cost
            self.observations.append(Observation(x, FIDELITY_CHEAP, y, cost, self.iteration))
            return self._make_log('execute_cheap', x=x, fidelity=FIDELITY_CHEAP, cost=cost)

        if s_star >= 0.5:
            self.queue.append(QueueItem(x_star, FIDELITY_EXPENSIVE, self.iteration))
            if len(self.queue) >= self.q_max:
                return self._execute_batch_naive()
            return self._make_log('add_to_queue', x=x_star, fidelity=FIDELITY_EXPENSIVE,
                                  acq_value=acq_imm, cost=0.0)

        execute_batch = False
        if len(self.queue) >= self.q_min:
            if self.use_adaptive_trigger:
                G_hat = self._greedy_queue_valuation()
                C_batch = self.cost_model.batch_cost(len(self.queue))
                alpha_batch = G_hat / C_batch if C_batch > 0 else float('inf')
                if alpha_batch >= acq_imm:
                    execute_batch = True
            else:
                if len(self.queue) >= self.fixed_trigger_threshold:
                    execute_batch = True

        if execute_batch:
            return self._execute_batch_naive()

        n_inj_before = len(self.injector.injections) if self.injector else 0
        y = self.test_fn.evaluate(x_star, FIDELITY_CHEAP, self.noise_std)
        is_injected_outlier = self._was_injected_outlier(n_inj_before)

        cost = self.cost_model.cost_per_sample(q, FIDELITY_CHEAP)
        self.cumulative_cost += cost

        # Warmup: accept everything.
        if self._is_warmup():
            if is_injected_outlier:
                self.accepted_outliers += 1
            self.observations.append(Observation(x_star, FIDELITY_CHEAP, y, cost, self.iteration))
            return self._make_log('execute_cheap', x=x_star, fidelity=FIDELITY_CHEAP,
                                  acq_value=acq_imm, cost=cost)

        z, mu, sigma = self._compute_z_score(x_star, FIDELITY_CHEAP, y)

        if z > self.tau:
            if is_injected_outlier:
                self.rejected_outliers += 1
            else:
                self.rejected_genuine += 1
            self.rejection_events.append({
                'iteration': self.iteration, 'fidelity': FIDELITY_CHEAP,
                'z': z, 'was_outlier': is_injected_outlier, 'y': y})
            return self._make_log('naive_reject', x=x_star, fidelity=FIDELITY_CHEAP, cost=cost)
        else:
            if is_injected_outlier:
                self.accepted_outliers += 1
            self.observations.append(Observation(x_star, FIDELITY_CHEAP, y, cost, self.iteration))
            return self._make_log('execute_cheap', x=x_star, fidelity=FIDELITY_CHEAP,
                                  acq_value=acq_imm, cost=cost)

    def _execute_batch_naive(self):
        q = len(self.queue)
        bc = self.cost_model.batch_cost(q)
        ps = bc / q if q > 0 else 0
        self.batch_counter += 1
        self.n_facility_visits += 1

        for item in self.queue:
            n_inj_before = len(self.injector.injections) if self.injector else 0
            y = self.test_fn.evaluate(item.x, FIDELITY_EXPENSIVE, self.noise_std)
            is_injected_outlier = self._was_injected_outlier(n_inj_before)

            if self._is_warmup():
                if is_injected_outlier:
                    self.accepted_outliers += 1
                self.observations.append(Observation(item.x, FIDELITY_EXPENSIVE, y, ps,
                                                     self.iteration, self.batch_counter))
                continue

            z, mu, sigma = self._compute_z_score(item.x, FIDELITY_EXPENSIVE, y)

            if z > self.tau:
                if is_injected_outlier:
                    self.rejected_outliers += 1
                else:
                    self.rejected_genuine += 1
                self.rejection_events.append({
                    'iteration': self.iteration, 'fidelity': FIDELITY_EXPENSIVE,
                    'z': z, 'was_outlier': is_injected_outlier, 'y': y})
            else:
                if is_injected_outlier:
                    self.accepted_outliers += 1
                self.observations.append(Observation(item.x, FIDELITY_EXPENSIVE, y, ps,
                                                     self.iteration, self.batch_counter))

        self.cumulative_cost += bc
        bs = q
        self.queue.clear()
        return self._make_log('execute_batch', batch_size=bs, cost=bc)


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_favp_benchmark(test_function_class, budget=200.0, lambda_cheap=1.0,
                       lambda_overhead=25.0, lambda_marginal=2.0, noise_std=0.1,
                       p_outlier=0.08, tau=3.0, gamma=1.5, n_min=15, n_seeds=10,
                       n_init_cheap=3, n_init_expensive=2, verbose=True):
    """Run the four-method catastrophic-noise benchmark.

    Compares:
      1. QS-MFBO + FAVP         full framework
      2. QS-MFBO (no FAVP)      scheduling only; outliers enter the GP
      3. QS-MFBO + naive        discard any z > tau observation
      4. MF-MES                 unaugmented baseline

    Returns (results, diagnostics). The four methods share initialisation
    within each seed. Regret is reported using compute_true_regret so that
    corrupted observations cannot produce fake-good numbers.
    """
    fc = FixedCostModel(lambda_cheap, lambda_overhead, lambda_marginal)
    qc_template = lambda: QueueDependentCostModel(lambda_cheap, lambda_overhead, lambda_marginal)
    init_cost = n_init_cheap * lambda_cheap + lambda_overhead + n_init_expensive * lambda_marginal

    results = {
        'QS-MFBO + FAVP': [],
        'QS-MFBO (no FAVP)': [],
        'QS-MFBO + naive': [],
        'MF-MES': [],
    }
    diagnostics = {
        'QS-MFBO + FAVP': [],
        'QS-MFBO (no FAVP)': [],
        'QS-MFBO + naive': [],
        'MF-MES': [],
    }

    for seed in range(n_seeds):
        if verbose:
            print("")
            print("=" * 60)
            print("SEED %d" % seed)
            print("=" * 60)

        init_data = generate_shared_init(test_function_class, n_cheap=n_init_cheap,
                                         n_expensive=n_init_expensive,
                                         noise_std=noise_std, seed=seed)

        # Method 1: QS-MFBO + FAVP
        if verbose:
            print("\n--- QS-MFBO + FAVP ---")
        injector1 = OutlierInjector(test_function_class, p_outlier=p_outlier, seed=seed)
        sched1 = QueueSchedulerFAVP(
            injector1, qc_template(), outlier_injector=injector1,
            tau=tau, gamma=gamma, n_min=n_min,
            noise_std=noise_std, seed=seed, init_data=copy.deepcopy(init_data))
        log1 = sched1.run(budget, verbose=verbose, init_cost=init_cost)
        true_regret1 = compute_true_regret(sched1.observations, injector1,
                                           test_function_class.optimal_value)
        results['QS-MFBO + FAVP'].append(log1)
        diagnostics['QS-MFBO + FAVP'].append({
            'anomaly_events': sched1.anomaly_events,
            'injections': list(injector1.injections),
            'n_injected': len(injector1.injections),
            'true_regret': true_regret1,
        })

        # Method 2: QS-MFBO (no FAVP)
        if verbose:
            print("\n--- QS-MFBO (no FAVP) ---")
        injector2 = OutlierInjector(test_function_class, p_outlier=p_outlier, seed=seed)
        sched2 = QueueScheduler(
            injector2, qc_template(),
            noise_std=noise_std, seed=seed, init_data=copy.deepcopy(init_data))
        log2 = sched2.run(budget, verbose=verbose, init_cost=init_cost)
        true_regret2 = compute_true_regret(sched2.observations, injector2,
                                           test_function_class.optimal_value)
        results['QS-MFBO (no FAVP)'].append(log2)
        diagnostics['QS-MFBO (no FAVP)'].append({
            'n_injected': len(injector2.injections),
            'n_corrupted_surrogate': len(injector2.injections),
            'true_regret': true_regret2,
        })

        # Method 3: QS-MFBO + naive rejection
        if verbose:
            print("\n--- QS-MFBO + naive rejection ---")
        injector3 = OutlierInjector(test_function_class, p_outlier=p_outlier, seed=seed)
        sched3 = QueueSchedulerNaive(
            injector3, qc_template(), outlier_injector=injector3,
            tau=tau, n_min=n_min,
            noise_std=noise_std, seed=seed, init_data=copy.deepcopy(init_data))
        log3 = sched3.run(budget, verbose=verbose, init_cost=init_cost)
        true_regret3 = compute_true_regret(sched3.observations, injector3,
                                           test_function_class.optimal_value)
        results['QS-MFBO + naive'].append(log3)
        diagnostics['QS-MFBO + naive'].append({
            'rejected_genuine': sched3.rejected_genuine,
            'rejected_outliers': sched3.rejected_outliers,
            'accepted_outliers': sched3.accepted_outliers,
            'rejection_events': sched3.rejection_events,
            'n_injected': len(injector3.injections),
            'true_regret': true_regret3,
        })

        # Method 4: MF-MES baseline
        if verbose:
            print("\n--- MF-MES baseline ---")
        injector4 = OutlierInjector(test_function_class, p_outlier=p_outlier, seed=seed)
        sched4 = StandardMFMES(
            injector4, fc,
            noise_std=noise_std, seed=seed, init_data=copy.deepcopy(init_data))
        log4 = sched4.run(budget, verbose=verbose, init_cost=init_cost)
        true_regret4 = compute_true_regret(sched4.observations, injector4,
                                           test_function_class.optimal_value)
        results['MF-MES'].append(log4)
        diagnostics['MF-MES'].append({
            'n_injected': len(injector4.injections),
            'n_corrupted_surrogate': len(injector4.injections),
            'true_regret': true_regret4,
        })

        if verbose:
            print("\n  TRUE REGRET (seed %d):" % seed)
            print("    QS-MFBO + FAVP:     %.4f" % true_regret1)
            print("    QS-MFBO (no FAVP):  %.4f" % true_regret2)
            print("    QS-MFBO + naive:    %.4f" % true_regret3)
            print("    MF-MES:             %.4f" % true_regret4)

    return results, diagnostics


# =============================================================================
# Summary
# =============================================================================

def print_summary(results, diagnostics, test_name=""):
    """Print a per-method summary using true regret (not observed regret)."""
    print("\n" + "=" * 70)
    print("FAVP BENCHMARK SUMMARY: %s" % test_name)
    print("=" * 70)

    for method in results:
        logs_list = results[method]
        diags = diagnostics[method]
        n = len(logs_list)

        obs_regrets = [max(l[-1].simple_regret, 0.0) for l in logs_list]
        true_regrets = [d['true_regret'] for d in diags]
        visits = [l[-1].n_facility_visits for l in logs_list]

        print("\n%s (%d seeds):" % (method, n))
        print("  True regret:     %.4f +/- %.4f" % (np.mean(true_regrets), np.std(true_regrets)))
        print("  Observed regret: %.4f +/- %.4f  (may be corrupted by outliers)" %
              (np.mean(obs_regrets), np.std(obs_regrets)))
        print("  Facility visits: %.1f +/- %.1f" % (np.mean(visits), np.std(visits)))

        if method == 'QS-MFBO + FAVP':
            n_inj = [d['n_injected'] for d in diags]
            cases = {}
            for d in diags:
                for evt in d['anomaly_events']:
                    cases[evt.case] = cases.get(evt.case, 0) + 1
            print("  Outliers injected: %.1f avg" % np.mean(n_inj))
            print("  FAVP cases: %s" % dict(cases))

        elif method == 'QS-MFBO + naive':
            rej_gen = [d['rejected_genuine'] for d in diags]
            rej_out = [d['rejected_outliers'] for d in diags]
            acc_out = [d['accepted_outliers'] for d in diags]
            n_inj = [d['n_injected'] for d in diags]
            print("  Outliers injected: %.1f avg" % np.mean(n_inj))
            print("  Genuine rejected:  %.1f avg" % np.mean(rej_gen))
            print("  Outliers rejected: %.1f avg" % np.mean(rej_out))
            print("  Outliers slipped:  %.1f avg" % np.mean(acc_out))

        elif method in ('QS-MFBO (no FAVP)', 'MF-MES'):
            n_inj = [d['n_injected'] for d in diags]
            print("  Outliers corrupted surrogate: %.1f avg" % np.mean(n_inj))


# =============================================================================
# Plotting
# =============================================================================

def plot_favp_results(results, diagnostics, title="FAVP Benchmark", save_path=None):
    """Three-panel diagnostic plot: regret vs cost, true regret, facility visits."""
    import matplotlib.pyplot as plt

    colors = {
        'QS-MFBO + FAVP':     '#2ecc71',
        'QS-MFBO (no FAVP)':  '#e74c3c',
        'QS-MFBO + naive':    '#f39c12',
        'MF-MES':             '#3498db',
    }
    linewidths = {
        'QS-MFBO + FAVP':     3,
        'QS-MFBO (no FAVP)':  1.5,
        'QS-MFBO + naive':    1.5,
        'MF-MES':             1.5,
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Regret vs cost (observed regret from logs)
    ax = axes[0]
    for method, logs_list in results.items():
        all_c, all_r = [], []
        for logs in logs_list:
            c, r, pc = [], [], -1
            for l in logs:
                if l.cumulative_cost > pc:
                    c.append(l.cumulative_cost)
                    r.append(max(l.simple_regret, 0.0))
                    pc = l.cumulative_cost
                elif c:
                    r[-1] = min(r[-1], max(l.simple_regret, 0.0))
            if c:
                all_c.append(c)
                all_r.append(r)
        if not all_c:
            continue
        mc = min(max(c_[-1] for c_ in all_c), 200)
        cg = np.linspace(0, mc, 100)
        ir = [np.interp(cg, c_, r_) for c_, r_ in zip(all_c, all_r)]
        mr, sr = np.mean(ir, axis=0), np.std(ir, axis=0)
        col = colors.get(method, '#333')
        lw = linewidths.get(method, 1.5)
        ax.plot(cg, mr, label=method, color=col, linewidth=lw)
        ax.fill_between(cg, mr - sr, mr + sr, alpha=0.15, color=col)
    ax.set_xlabel('Cumulative Cost')
    ax.set_ylabel('Simple Regret (observed)')
    ax.set_title(title + ': Regret vs Cost')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Panel 2: True final regret (box plot)
    ax = axes[1]
    mns = list(results.keys())
    true_r = [[d['true_regret'] for d in diagnostics[m]] for m in mns]
    short_labels = [m.replace('QS-MFBO ', 'QS\n') for m in mns]
    bp = ax.boxplot(true_r, labels=short_labels, patch_artist=True)
    for p, m in zip(bp['boxes'], mns):
        p.set_facecolor(colors.get(m, '#ccc'))
        p.set_alpha(0.7)
    ax.set_ylabel('True Final Regret')
    ax.set_title(title + ': True Regret (clean eval)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Facility visits
    ax = axes[2]
    v = [[logs[-1].n_facility_visits for logs in results[m]] for m in mns]
    bp = ax.boxplot(v, labels=short_labels, patch_artist=True)
    for p, m in zip(bp['boxes'], mns):
        p.set_facecolor(colors.get(m, '#ccc'))
        p.set_alpha(0.7)
    ax.set_ylabel('Facility Visits')
    ax.set_title(title + ': Facility Visits')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved to %s" % save_path)
    plt.show()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='FAVP synthetic benchmark')
    parser.add_argument('--function', type=str, default='all',
                        choices=['st2d', 'br2d', 'h6d', 'all'],
                        help='Test function to run (default: all)')
    parser.add_argument('--budget', type=float, default=200.0)
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--p_outlier', type=float, default=0.20)
    parser.add_argument('--tau', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument('--n_min', type=int, default=15)
    parser.add_argument('--lambda_overhead', type=float, default=25.0)
    parser.add_argument('--lambda_marginal', type=float, default=2.0)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--outdir', type=str, default='favp_results')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    function_map = {
        'st2d': ('ST-2D', StyblinskiTang2D),
        'br2d': ('BR-2D', Branin2D),
        'h6d':  ('H-6D',  Hartmann6D),
    }

    if args.function == 'all':
        functions_to_run = ['st2d', 'br2d', 'h6d']
    else:
        functions_to_run = [args.function]

    all_results = {}

    for fn_key in functions_to_run:
        fn_name, fn_class = function_map[fn_key]
        print('')
        print('#' * 60)
        print('# BENCHMARK: %s' % fn_name)
        print('# Budget=%.0f, Seeds=%d, p_outlier=%.2f' % (args.budget, args.n_seeds, args.p_outlier))
        print('#' * 60)

        t0 = time.time()
        results, diagnostics = run_favp_benchmark(
            fn_class,
            budget=args.budget,
            lambda_overhead=args.lambda_overhead,
            lambda_marginal=args.lambda_marginal,
            noise_std=args.noise_std,
            p_outlier=args.p_outlier,
            tau=args.tau,
            gamma=args.gamma,
            n_min=args.n_min,
            n_seeds=args.n_seeds,
            verbose=True,
        )
        elapsed = time.time() - t0

        print_summary(results, diagnostics, fn_name)
        print('Elapsed: %.1f seconds' % elapsed)

        pkl_path = os.path.join(args.outdir, 'favp_%s.pkl' % fn_key)
        with open(pkl_path, 'wb') as f:
            pickle.dump({'results': results, 'diagnostics': diagnostics,
                         'args': vars(args), 'function': fn_name}, f)
        print('Saved to %s' % pkl_path)

        if not args.no_plot:
            try:
                png_path = os.path.join(args.outdir, 'favp_%s.png' % fn_key)
                plot_favp_results(results, diagnostics,
                                  title='FAVP: %s (p=%.0f%%)' % (fn_name, args.p_outlier * 100),
                                  save_path=png_path)
            except Exception as e:
                print('Plotting failed: %s' % e)

        all_results[fn_key] = (results, diagnostics)

    # Combined summary table
    if len(functions_to_run) > 1:
        print('')
        print('=' * 100)
        print('COMBINED RESULTS TABLE')
        print('Settings: tau=%.1f, gamma=%.1f, N_min=%d, p_outlier=%.2f, budget=%.0f, %d seeds' % (
            args.tau, args.gamma, args.n_min, args.p_outlier, args.budget, args.n_seeds))
        print('=' * 100)
        print('%-20s | %-6s | %-16s | %-8s | %-12s | %-10s' % (
            'Method', 'Func', 'True Regret', 'Visits', 'Rej.Genuine', 'Corrupted'))
        print('-' * 100)

        for fn_key in functions_to_run:
            fn_name = function_map[fn_key][0]
            results, diagnostics = all_results[fn_key]

            for method in results:
                true_r = [d['true_regret'] for d in diagnostics[method]]
                vis = [l[-1].n_facility_visits for l in results[method]]

                rej_gen = '-'
                corrupted = '-'
                if method == 'QS-MFBO + naive':
                    rej_gen = '%.1f' % np.mean([d['rejected_genuine'] for d in diagnostics[method]])
                elif method in ('QS-MFBO (no FAVP)', 'MF-MES'):
                    corrupted = '%.1f' % np.mean([d['n_injected'] for d in diagnostics[method]])

                print('%-20s | %-6s | %6.4f+/-%.4f | %4.1f+/-%.1f | %-12s | %-10s' % (
                    method, fn_name,
                    np.mean(true_r), np.std(true_r),
                    np.mean(vis), np.std(vis),
                    rej_gen, corrupted))
            print('-' * 100)

    print('')
    print('All done.')
