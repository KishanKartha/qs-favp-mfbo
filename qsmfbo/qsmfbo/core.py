"""
Core QS-MFBO framework (two-fidelity implementation).

This module defines the queue-scheduled multi-fidelity Bayesian optimisation
layer. It contains:

  - QueueScheduler          the scheduling meta-layer wrapping MF-MES
  - StandardMFMES           the unaugmented baseline (immediate execution)
  - Cost models             QueueDependentCostModel, FixedCostModel
  - Test functions          Styblinski-Tang 2D, Branin 2D, Hartmann 6D
  - Shared utilities        initialisation, GP construction, plotting

The FAVP verification layer is kept separate in qsmfbo.favp. The three-fidelity
P3HT-CNT benchmark is also a separate module (benchmarks/p3ht_cnt/) because it
predates the refactor and remains self-contained.

Timing instrumentation: QueueScheduler and StandardMFMES record wall-clock
timings for model fits and queue valuations in self.timing_log, for the
computational-overhead analysis (each entry: iteration, op, q, n_obs, seconds).
"""

import torch
import numpy as np
import time
import warnings
from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.cost import AffineFidelityCostModel
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.acquisition.utils import project_to_target_fidelity
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TKWARGS = {"dtype": torch.double, "device": DEVICE}
if DEVICE.type == "cuda":
    print("Using GPU: %s" % torch.cuda.get_device_name(0))
else:
    print("Using CPU")

FIDELITY_CHEAP = 0.0
FIDELITY_EXPENSIVE = 1.0


# ++++++++++++++++++
# Data Structures
# ++++++++++++++++++

@dataclass
class Observation:
    x: torch.Tensor
    fidelity: float
    y: float
    cost: float
    iteration: int
    batch_id: Optional[int] = None

@dataclass
class QueueItem:
    x: torch.Tensor
    fidelity: float
    iteration_added: int

@dataclass
class IterationLog:
    iteration: int
    action: str
    x: Optional[torch.Tensor] = None
    fidelity: Optional[float] = None
    acq_value: Optional[float] = None
    queue_size: int = 0
    batch_size: int = 0
    cost_incurred: float = 0.0
    cumulative_cost: float = 0.0
    best_y: float = 0.0
    simple_regret: float = float('inf')
    n_facility_visits: int = 0


# ++++++++++++++++++
# Cost Models
# ++++++++++++++++++

class QueueDependentCostModel:
    """Contribution 1: lambda_m(q) = lambda_o/(q+1) + lambda_marginal"""
    def __init__(self, lambda_cheap, lambda_overhead, lambda_marginal):
        self.lambda_cheap = lambda_cheap
        self.lambda_overhead = lambda_overhead
        self.lambda_marginal = max(lambda_marginal, lambda_cheap)

    def cost_per_sample(self, q, fidelity):
        if fidelity < 0.5:
            return self.lambda_cheap
        return self.lambda_overhead / (q + 1) + self.lambda_marginal

    def batch_cost(self, q):
        if q == 0:
            return 0.0
        return self.lambda_overhead + q * self.lambda_marginal

    def single_expensive_cost(self):
        return self.lambda_overhead + self.lambda_marginal


class FixedCostModel:
    """Flat cost model for baselines."""
    def __init__(self, lambda_cheap, lambda_overhead, lambda_marginal):
        self.lambda_cheap = lambda_cheap
        self.lambda_overhead = lambda_overhead
        self.lambda_marginal = max(lambda_marginal, lambda_cheap)
        self.lambda_expensive = lambda_overhead + self.lambda_marginal

    def cost_per_sample(self, q, fidelity):
        if fidelity < 0.5:
            return self.lambda_cheap
        return self.lambda_expensive

    def batch_cost(self, q):
        if q == 0:
            return 0.0
        return self.lambda_overhead + q * self.lambda_marginal

    def single_expensive_cost(self):
        return self.lambda_expensive


# ++++++++++++++++++
# Test Functions
# ++++++++++++++++++

class StyblinskiTang2D:
    dim = 2
    n_fidelities = 2
    optimal_value = 78.33

    @staticmethod
    def _unnormalize(x):
        return x * 10.0 - 5.0

    @classmethod
    def evaluate(cls, x_norm, fidelity, noise_std=0.0):
        x = cls._unnormalize(x_norm)
        f_h = -0.5 * torch.sum(x**4 - 16*x**2 + 5*x).item()
        if fidelity >= 0.5:
            y = f_h
        else:
            y = 0.8 * f_h + 0.5 * torch.sin(3 * x[0]).item() + 2.0
        if noise_std > 0:
            y += np.random.randn() * noise_std
        return y


class Hartmann6D:
    dim = 6
    n_fidelities = 2
    optimal_value = 3.3224
    _alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
    _A = torch.tensor([
        [10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]
    ], dtype=torch.double)
    _P = 1e-4 * torch.tensor([
        [1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]
    ], dtype=torch.double)

    @classmethod
    def evaluate(cls, x_norm, fidelity, noise_std=0.0):
        x = x_norm.unsqueeze(0) if x_norm.dim() == 1 else x_norm
        inner = torch.sum(cls._A * (x - cls._P)**2, dim=-1)
        f_h = torch.sum(cls._alpha * torch.exp(-inner), dim=-1).item()
        if fidelity >= 0.5:
            y = f_h
        else:
            y = 0.75 * f_h + 0.3 * torch.sin(2 * np.pi * x_norm[0]).item() + 0.1
        if noise_std > 0:
            y += np.random.randn() * noise_std
        return y


class Branin2D:
    dim = 2
    n_fidelities = 2
    optimal_value = -0.397887

    @staticmethod
    def _unnormalize(x_norm):
        x = torch.zeros_like(x_norm)
        x[0] = x_norm[0] * 15.0 - 5.0
        x[1] = x_norm[1] * 15.0
        return x

    @classmethod
    def evaluate(cls, x_norm, fidelity, noise_std=0.0):
        x = cls._unnormalize(x_norm)
        x1, x2 = x[0].item(), x[1].item()
        a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
        r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
        f_branin = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
        f_h = -f_branin
        if fidelity >= 0.5:
            y = f_h
        else:
            y = 0.85 * f_h + 0.4 * np.sin(2.5 * x1) + 1.5
        if noise_std > 0:
            y += np.random.randn() * noise_std
        return y


# ++++++++++++++++++
# Shared Init & GP Utilities
# ++++++++++++++++++

def generate_shared_init(test_function, n_cheap=3, n_expensive=2, noise_std=0.1, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cheap_xs, cheap_ys = [], []
    for _ in range(n_cheap):
        x = torch.rand(test_function.dim)
        y = test_function.evaluate(x, FIDELITY_CHEAP, noise_std)
        cheap_xs.append(x)
        cheap_ys.append(y)
    expensive_xs, expensive_ys = [], []
    for _ in range(n_expensive):
        x = torch.rand(test_function.dim)
        y = test_function.evaluate(x, FIDELITY_EXPENSIVE, noise_std)
        expensive_xs.append(x)
        expensive_ys.append(y)
    return {'cheap_xs': cheap_xs, 'cheap_ys': cheap_ys,
            'expensive_xs': expensive_xs, 'expensive_ys': expensive_ys}


def build_mf_model(train_X, train_Y, fidelity_dim):
    model = SingleTaskMultiFidelityGP(
        train_X=train_X, train_Y=train_Y,
        outcome_transform=Standardize(m=1),
        data_fidelities=[fidelity_dim],
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def make_botorch_cost_utility(fidelity_dim, cost_cheap, cost_expensive):
    fid_weight = max(cost_expensive - cost_cheap, 0.01)
    cost_model = AffineFidelityCostModel(
        fidelity_weights={fidelity_dim: fid_weight}, fixed_cost=cost_cheap)
    return InverseCostWeightedUtility(cost_model=cost_model)


# ++++++++++++++++++
# MF-MES Acquisition (Takeno et al., ICML 2020)
# ++++++++++++++++++

def build_mfmes(model, fidelity_dim, dim, cost_utility, target_fidelities,
                num_mv_samples=10, num_fantasies=16, n_candidates=500):
    candidates = torch.rand(n_candidates, dim + 1)
    candidates[:, fidelity_dim] = 1.0
    d_total = dim + 1
    def project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities, d=d_total)
    return qMultiFidelityMaxValueEntropy(
        model=model, candidate_set=candidates,
        num_fantasies=num_fantasies, num_mv_samples=num_mv_samples,
        num_y_samples=128, cost_aware_utility=cost_utility, project=project)


def optimize_mfmes(acqf, dim, fidelity_dim, pending_X=None):
    if pending_X is not None:
        acqf.set_X_pending(pending_X)
    bounds = torch.zeros(2, dim + 1)
    bounds[1, :] = 1.0
    candidate, acq_value = optimize_acqf_mixed(
        acq_function=acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=128,
        fixed_features_list=[{fidelity_dim: f} for f in [FIDELITY_CHEAP, FIDELITY_EXPENSIVE]])
    x = candidate.squeeze()[:dim]
    fidelity = candidate.squeeze()[fidelity_dim].item()
    return x, fidelity, acq_value.item()


def evaluate_mfmes_info_gain(model, x, fidelity, fidelity_dim, dim, target_fidelities,
                              num_mv_samples=10, num_fantasies=16):
    """Raw MF-MES info gain (unit cost) for greedy sequential fantasisation."""
    unit_cost = AffineFidelityCostModel(fidelity_weights={fidelity_dim: 0.0}, fixed_cost=1.0)
    unit_utility = InverseCostWeightedUtility(cost_model=unit_cost)
    acqf = build_mfmes(model, fidelity_dim, dim, unit_utility, target_fidelities,
                       num_mv_samples=num_mv_samples, num_fantasies=num_fantasies)
    x_fid = torch.cat([x, torch.tensor([fidelity])]).unsqueeze(0)
    try:
        ig = acqf(x_fid).item()
    except Exception:
        ig = 0.0
    return max(ig, 0.0)


# ++++++++++++++++++
# QueueScheduler (Meta-Layer)
# ++++++++++++++++++

class QueueScheduler:
    """Queue scheduling meta-layer wrapping MF-MES.

    Implements three contributions:
      1. Queue-dependent cost model
      2. Greedy sequential fantasisation for queue valuation
      3. Information-theoretic batch trigger

    Ablation flags: use_queue_cost, use_adaptive_trigger

    Timing: self.timing_log records wall-clock seconds for 'model_fit' and
    'queue_valuation' operations, with queue size q and observation count
    n_obs at the time of the call.
    """
    def __init__(self, test_function, cost_model, noise_std=0.1, q_min=2, q_max=20,
                 n_init_cheap=3, n_init_expensive=2, seed=0, init_data=None,
                 num_mv_samples=10, num_fantasies=16,
                 use_queue_cost=True, use_adaptive_trigger=True,
                 fixed_trigger_threshold=3):
        self.test_fn = test_function
        self.cost_model = cost_model
        self.noise_std = noise_std
        self.q_min = q_min
        self.q_max = q_max
        self.n_init_cheap = n_init_cheap
        self.n_init_expensive = n_init_expensive
        self.seed = seed
        self.init_data = init_data
        self.num_mv_samples = num_mv_samples
        self.num_fantasies = num_fantasies
        self.use_queue_cost = use_queue_cost
        self.use_adaptive_trigger = use_adaptive_trigger
        self.fixed_trigger_threshold = fixed_trigger_threshold
        self.dim = test_function.dim
        self.fidelity_dim = self.dim
        self.target_fidelities = {self.fidelity_dim: 1.0}
        self.observations = []
        self.queue = []
        self.log = []
        self.timing_log = []
        self.iteration = 0
        self.cumulative_cost = 0.0
        self.n_facility_visits = 0
        self.batch_counter = 0
        self.model = None
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _get_train_tensors(self):
        if not self.observations:
            return None, None
        X = torch.stack([torch.cat([o.x, torch.tensor([o.fidelity])]) for o in self.observations])
        Y = torch.tensor([o.y for o in self.observations]).unsqueeze(-1)
        return X, Y

    def _get_pending_X(self):
        if not self.queue:
            return None
        return torch.stack([torch.cat([i.x, torch.tensor([i.fidelity])]) for i in self.queue])

    def _best_observed_hf(self):
        hf = [o.y for o in self.observations if o.fidelity >= 0.5]
        if hf:
            return max(hf)
        all_y = [o.y for o in self.observations]
        return max(all_y) if all_y else -float('inf')

    def _simple_regret(self):
        return self.test_fn.optimal_value - self._best_observed_hf()

    def _fit_model(self):
        t0 = time.time()
        train_X, train_Y = self._get_train_tensors()
        if train_X is None or train_X.shape[0] < 3:
            return False
        try:
            self.model = build_mf_model(train_X, train_Y, self.fidelity_dim)
            self.timing_log.append({'iteration': self.iteration,
                                    'op': 'model_fit',
                                    'q': len(self.queue),
                                    'n_obs': train_X.shape[0],
                                    'seconds': time.time() - t0})
            return True
        except Exception as e:
            print("  [WARN] Model fit failed: %s" % e)
            return False

    def _get_cost_utility(self):
        q = len(self.queue)
        if self.use_queue_cost:
            cost_cheap = self.cost_model.cost_per_sample(q, FIDELITY_CHEAP)
            cost_expensive = self.cost_model.cost_per_sample(q, FIDELITY_EXPENSIVE)
        else:
            cost_cheap = self.cost_model.lambda_cheap
            if hasattr(self.cost_model, 'lambda_expensive'):
                cost_expensive = self.cost_model.lambda_expensive
            else:
                cost_expensive = self.cost_model.lambda_overhead + self.cost_model.lambda_marginal
        return make_botorch_cost_utility(self.fidelity_dim, cost_cheap, cost_expensive)

    def _mfmes_suggest(self):
        cost_utility = self._get_cost_utility()
        acqf = build_mfmes(self.model, self.fidelity_dim, self.dim, cost_utility,
                           self.target_fidelities, self.num_mv_samples, self.num_fantasies)
        return optimize_mfmes(acqf, self.dim, self.fidelity_dim, self._get_pending_X())

    def _greedy_queue_valuation(self):
        """Contribution 2: Greedy sequential fantasisation."""
        t0 = time.time()
        if not self.queue:
            return 0.0
        q = len(self.queue)
        remaining = list(range(q))
        G_hat = 0.0
        current_model = self.model
        base_X, base_Y = self._get_train_tensors()
        acc_X, acc_Y = base_X.clone(), base_Y.clone()
        for step in range(q):
            gains = {}
            for idx in remaining:
                gains[idx] = evaluate_mfmes_info_gain(
                    current_model, self.queue[idx].x, self.queue[idx].fidelity,
                    self.fidelity_dim, self.dim, self.target_fidelities,
                    self.num_mv_samples, self.num_fantasies)
            if not gains:
                break
            best_idx = max(gains, key=gains.get)
            G_hat += gains[best_idx]
            remaining.remove(best_idx)
            if remaining:
                item = self.queue[best_idx]
                x_fid = torch.cat([item.x, torch.tensor([item.fidelity])]).unsqueeze(0)
                with torch.no_grad():
                    y_fan = current_model.posterior(x_fid).mean.squeeze()
                acc_X = torch.cat([acc_X, x_fid], dim=0)
                acc_Y = torch.cat([acc_Y, y_fan.reshape(1, 1)], dim=0)
                try:
                    current_model = build_mf_model(acc_X, acc_Y, self.fidelity_dim)
                except Exception:
                    break
        self.timing_log.append({'iteration': self.iteration,
                                'op': 'queue_valuation',
                                'q': q,
                                'n_obs': len(self.observations),
                                'seconds': time.time() - t0})
        return G_hat

    def _initialize(self, init_cost=None):
        if self.init_data is not None:
            d = self.init_data
            for x, y in zip(d['cheap_xs'], d['cheap_ys']):
                self.observations.append(Observation(x, FIDELITY_CHEAP, y, 0.0, 0))
            self.batch_counter += 1
            self.n_facility_visits += 1
            for x, y in zip(d['expensive_xs'], d['expensive_ys']):
                self.observations.append(Observation(x, FIDELITY_EXPENSIVE, y, 0.0, 0, self.batch_counter))
            self.cumulative_cost = init_cost if init_cost is not None else 0.0
        else:
            for _ in range(self.n_init_cheap):
                x = torch.rand(self.dim)
                y = self.test_fn.evaluate(x, FIDELITY_CHEAP, self.noise_std)
                cost = self.cost_model.cost_per_sample(0, FIDELITY_CHEAP)
                self.observations.append(Observation(x, FIDELITY_CHEAP, y, cost, 0))
                self.cumulative_cost += cost
            batch_xs = [torch.rand(self.dim) for _ in range(self.n_init_expensive)]
            bc = self.cost_model.batch_cost(len(batch_xs))
            ps = bc / len(batch_xs) if batch_xs else 0
            self.batch_counter += 1
            self.n_facility_visits += 1
            for x in batch_xs:
                y = self.test_fn.evaluate(x, FIDELITY_EXPENSIVE, self.noise_std)
                self.observations.append(Observation(x, FIDELITY_EXPENSIVE, y, ps, 0, self.batch_counter))
            self.cumulative_cost += bc

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
        except Exception as e:
            print("  [WARN] MF-MES failed iter %d: %s" % (self.iteration, e))
            x = torch.rand(self.dim)
            y = self.test_fn.evaluate(x, FIDELITY_CHEAP, self.noise_std)
            cost = self.cost_model.cost_per_sample(q, FIDELITY_CHEAP)
            self.cumulative_cost += cost
            self.observations.append(Observation(x, FIDELITY_CHEAP, y, cost, self.iteration))
            return self._make_log('execute_cheap', x=x, fidelity=FIDELITY_CHEAP, cost=cost)

        if s_star >= 0.5:
            self.queue.append(QueueItem(x_star, FIDELITY_EXPENSIVE, self.iteration))
            # q_max cap: if queue hits maximum, force execute batch
            if len(self.queue) >= self.q_max:
                return self._execute_batch()
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
            return self._execute_batch()

        y = self.test_fn.evaluate(x_star, FIDELITY_CHEAP, self.noise_std)
        cost = self.cost_model.cost_per_sample(q, FIDELITY_CHEAP)
        self.cumulative_cost += cost
        self.observations.append(Observation(x_star, FIDELITY_CHEAP, y, cost, self.iteration))
        return self._make_log('execute_cheap', x=x_star, fidelity=FIDELITY_CHEAP,
                              acq_value=acq_imm, cost=cost)

    def _execute_batch(self):
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

    def _make_log(self, action, x=None, fidelity=None, acq_value=None, cost=0.0, batch_size=0):
        return IterationLog(self.iteration, action, x, fidelity, acq_value,
            len(self.queue), batch_size, cost, self.cumulative_cost,
            self._best_observed_hf(), self._simple_regret(), self.n_facility_visits)

    def run(self, budget, verbose=True, init_cost=None):
        name = "QS-MFBO"
        if not self.use_queue_cost and not self.use_adaptive_trigger:
            name += " (-both)"
        elif not self.use_queue_cost:
            name += " (-cost)"
        elif not self.use_adaptive_trigger:
            name += " (-trigger)"
        if verbose:
            print("%s | Budget: %s | Seed: %d" % (name, str(budget), self.seed))
            if hasattr(self.cost_model, 'lambda_overhead'):
                print("  Costs: cheap=%s, overhead=%s, marginal=%s" % (
                    self.cost_model.lambda_cheap, self.cost_model.lambda_overhead,
                    self.cost_model.lambda_marginal))
        self._initialize(init_cost=init_cost)
        if verbose:
            print("  Init cost: %.1f, Best HF: %.4f" % (self.cumulative_cost, self._best_observed_hf()))
        while self.cumulative_cost < budget:
            entry = self.run_iteration()
            self.log.append(entry)
            if verbose and (self.iteration % 5 == 0 or entry.action == 'execute_batch'):
                print("  Iter %3d | %-16s | Q:%2d | Cost:%7.1f/%s | Regret:%.4f | V:%d" % (
                    self.iteration, entry.action, entry.queue_size,
                    self.cumulative_cost, str(int(budget)), entry.simple_regret,
                    self.n_facility_visits))
        if self.queue:
            if verbose:
                print("  Force-executing queue (%d items)" % len(self.queue))
            self.log.append(self._execute_batch())
        if verbose:
            print("  DONE | Cost:%.1f | Regret:%.4f | Visits:%d" % (
                self.cumulative_cost, self._simple_regret(), self.n_facility_visits))
        return self.log


# ++++++++++++++++++
# Standard MF-MES Baseline (No Queue)
# ++++++++++++++++++

class StandardMFMES:
    def __init__(self, test_function, cost_model, noise_std=0.1,
                 n_init_cheap=3, n_init_expensive=2, num_mv_samples=10,
                 seed=0, init_data=None):
        self.test_fn = test_function
        self.cost_model = cost_model
        self.noise_std = noise_std
        self.num_mv_samples = num_mv_samples
        self.seed = seed
        self.init_data = init_data
        self.n_init_cheap = n_init_cheap
        self.n_init_expensive = n_init_expensive
        self.dim = test_function.dim
        self.fidelity_dim = self.dim
        self.target_fidelities = {self.fidelity_dim: 1.0}
        self.observations = []
        self.log = []
        self.timing_log = []
        self.iteration = 0
        self.cumulative_cost = 0.0
        self.n_facility_visits = 0
        self.model = None
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _get_train_tensors(self):
        if not self.observations:
            return None, None
        X = torch.stack([torch.cat([o.x, torch.tensor([o.fidelity])]) for o in self.observations])
        Y = torch.tensor([o.y for o in self.observations]).unsqueeze(-1)
        return X, Y

    def _best_observed_hf(self):
        hf = [o.y for o in self.observations if o.fidelity >= 0.5]
        return max(hf) if hf else max((o.y for o in self.observations), default=-float('inf'))

    def _simple_regret(self):
        return self.test_fn.optimal_value - self._best_observed_hf()

    def _fit_model(self):
        t0 = time.time()
        X, Y = self._get_train_tensors()
        if X is None or X.shape[0] < 3:
            return False
        try:
            self.model = build_mf_model(X, Y, self.fidelity_dim)
            self.timing_log.append({'iteration': self.iteration,
                                    'op': 'model_fit',
                                    'q': 0,
                                    'n_obs': X.shape[0],
                                    'seconds': time.time() - t0})
            return True
        except Exception:
            return False

    def _initialize(self, init_cost=None):
        if self.init_data is not None:
            d = self.init_data
            for x, y in zip(d['cheap_xs'], d['cheap_ys']):
                self.observations.append(Observation(x, FIDELITY_CHEAP, y, 0.0, 0))
            for x, y in zip(d['expensive_xs'], d['expensive_ys']):
                self.observations.append(Observation(x, FIDELITY_EXPENSIVE, y, 0.0, 0))
                self.n_facility_visits += 1
            self.cumulative_cost = init_cost if init_cost is not None else 0.0
        else:
            for _ in range(self.n_init_cheap):
                x = torch.rand(self.dim)
                y = self.test_fn.evaluate(x, FIDELITY_CHEAP, self.noise_std)
                c = self.cost_model.cost_per_sample(0, FIDELITY_CHEAP)
                self.observations.append(Observation(x, FIDELITY_CHEAP, y, c, 0))
                self.cumulative_cost += c
            for _ in range(self.n_init_expensive):
                x = torch.rand(self.dim)
                y = self.test_fn.evaluate(x, FIDELITY_EXPENSIVE, self.noise_std)
                c = self.cost_model.single_expensive_cost()
                self.observations.append(Observation(x, FIDELITY_EXPENSIVE, y, c, 0))
                self.cumulative_cost += c
                self.n_facility_visits += 1

    def run(self, budget, verbose=True, init_cost=None):
        if verbose:
            print("MF-MES | Budget: %s | Seed: %d" % (str(budget), self.seed))
        self._initialize(init_cost=init_cost)
        while self.cumulative_cost < budget:
            self.iteration += 1
            if not self._fit_model():
                x = torch.rand(self.dim)
                y = self.test_fn.evaluate(x, FIDELITY_CHEAP, self.noise_std)
                c = self.cost_model.cost_per_sample(0, FIDELITY_CHEAP)
                self.cumulative_cost += c
                self.observations.append(Observation(x, FIDELITY_CHEAP, y, c, self.iteration))
                continue
            cu = make_botorch_cost_utility(self.fidelity_dim,
                self.cost_model.lambda_cheap, self.cost_model.lambda_expensive)
            try:
                acqf = build_mfmes(self.model, self.fidelity_dim, self.dim, cu,
                    self.target_fidelities, self.num_mv_samples)
                x, fid, _ = optimize_mfmes(acqf, self.dim, self.fidelity_dim)
            except Exception:
                x = torch.rand(self.dim)
                fid = FIDELITY_CHEAP
            y = self.test_fn.evaluate(x, fid, self.noise_std)
            if fid >= 0.5:
                c = self.cost_model.single_expensive_cost()
                self.n_facility_visits += 1
            else:
                c = self.cost_model.cost_per_sample(0, FIDELITY_CHEAP)
            self.cumulative_cost += c
            self.observations.append(Observation(x, fid, y, c, self.iteration))
            act = 'execute_cheap' if fid < 0.5 else 'execute_expensive'
            self.log.append(IterationLog(self.iteration, act, x, fid, cost_incurred=c,
                cumulative_cost=self.cumulative_cost, best_y=self._best_observed_hf(),
                simple_regret=self._simple_regret(), n_facility_visits=self.n_facility_visits))
            if verbose and self.iteration % 5 == 0:
                print("  Iter %3d | %-18s | Cost:%7.1f/%s | Regret:%.4f" % (
                    self.iteration, act, self.cumulative_cost, str(int(budget)),
                    self._simple_regret()))
        if verbose:
            print("  DONE | Cost:%.1f | Regret:%.4f | Visits:%d" % (
                self.cumulative_cost, self._simple_regret(), self.n_facility_visits))
        return self.log


# ++++++++++++++++++
# Experiment Runners
# ++++++++++++++++++

def run_experiment(test_function_class, budget=200.0, lambda_cheap=1.0,
                   lambda_overhead=25.0, lambda_marginal=2.0, noise_std=0.1,
                   n_seeds=10, n_init_cheap=3, n_init_expensive=2, verbose=True):
    """Benchmark: MF-MES vs QS-MFBO."""
    fc = FixedCostModel(lambda_cheap, lambda_overhead, lambda_marginal)
    qc = QueueDependentCostModel(lambda_cheap, lambda_overhead, lambda_marginal)
    init_cost = n_init_cheap * lambda_cheap + lambda_overhead + n_init_expensive * lambda_marginal
    results = {'MF-MES': [], 'QS-MFBO': []}
    for seed in range(n_seeds):
        if verbose:
            print("")
            print("=" * 60)
            print("SEED %d" % seed)
            print("=" * 60)
            print("  Shared init: %d cheap + %d expensive | cost=%.1f" % (
                n_init_cheap, n_init_expensive, init_cost))
        init_data = generate_shared_init(test_function_class, n_cheap=n_init_cheap,
            n_expensive=n_init_expensive, noise_std=noise_std, seed=seed)
        results['MF-MES'].append(
            StandardMFMES(test_function_class, fc, noise_std=noise_std,
                seed=seed, init_data=init_data)
            .run(budget, verbose, init_cost=init_cost))
        results['QS-MFBO'].append(
            QueueScheduler(test_function_class, qc, noise_std=noise_std,
                seed=seed, init_data=init_data)
            .run(budget, verbose, init_cost=init_cost))
    return results


def run_ablation(test_function_class, budget=200.0, lambda_cheap=1.0,
                 lambda_overhead=25.0, lambda_marginal=2.0, noise_std=0.1,
                 n_seeds=10, n_init_cheap=3, n_init_expensive=2, verbose=True):
    """Ablation: decompose QS-MFBO."""
    fc = FixedCostModel(lambda_cheap, lambda_overhead, lambda_marginal)
    qc = QueueDependentCostModel(lambda_cheap, lambda_overhead, lambda_marginal)
    init_cost = n_init_cheap * lambda_cheap + lambda_overhead + n_init_expensive * lambda_marginal
    results = {'QS-MFBO (full)': [], 'QS-MFBO (-cost)': [],
               'QS-MFBO (-trigger)': [], 'QS-MFBO (-both)': [], 'MF-MES': []}
    for seed in range(n_seeds):
        if verbose:
            print("")
            print("=" * 60)
            print("ABLATION SEED %d" % seed)
            print("=" * 60)
        init_data = generate_shared_init(test_function_class, n_cheap=n_init_cheap,
            n_expensive=n_init_expensive, noise_std=noise_std, seed=seed)
        results['QS-MFBO (full)'].append(
            QueueScheduler(test_function_class, qc, noise_std=noise_std,
                seed=seed, init_data=init_data,
                use_queue_cost=True, use_adaptive_trigger=True)
            .run(budget, verbose, init_cost=init_cost))
        results['QS-MFBO (-cost)'].append(
            QueueScheduler(test_function_class, qc, noise_std=noise_std,
                seed=seed, init_data=init_data,
                use_queue_cost=False, use_adaptive_trigger=True)
            .run(budget, verbose, init_cost=init_cost))
        results['QS-MFBO (-trigger)'].append(
            QueueScheduler(test_function_class, qc, noise_std=noise_std,
                seed=seed, init_data=init_data,
                use_queue_cost=True, use_adaptive_trigger=False, fixed_trigger_threshold=3)
            .run(budget, verbose, init_cost=init_cost))
        results['QS-MFBO (-both)'].append(
            QueueScheduler(test_function_class, qc, noise_std=noise_std,
                seed=seed, init_data=init_data,
                use_queue_cost=False, use_adaptive_trigger=False, fixed_trigger_threshold=3)
            .run(budget, verbose, init_cost=init_cost))
        results['MF-MES'].append(
            StandardMFMES(test_function_class, fc, noise_std=noise_std,
                seed=seed, init_data=init_data)
            .run(budget, verbose, init_cost=init_cost))
    return results


# ++++++++++++++++++
# Plotting
# ++++++++++++++++++

def plot_results(results, title="Benchmark", save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'MF-MES': '#3498db', 'QS-MFBO': '#2ecc71',
              'QS-MFBO (full)': '#2ecc71', 'QS-MFBO (-cost)': '#f39c12',
              'QS-MFBO (-trigger)': '#9b59b6', 'QS-MFBO (-both)': '#95a5a6'}
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
        lw = 3 if 'QS' in method else 1.5
        ax.plot(cg, mr, label=method, color=col, linewidth=lw)
        ax.fill_between(cg, mr - sr, mr + sr, alpha=0.15, color=col)
    ax.set_xlabel('Cumulative Cost')
    ax.set_ylabel('Simple Regret')
    ax.set_title(title + ': Regret vs Cost')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    mns = list(results.keys())
    v = [[logs[-1].n_facility_visits for logs in results[m]] for m in mns]
    bp = ax.boxplot(v, labels=mns, patch_artist=True)
    for p, m in zip(bp['boxes'], mns):
        p.set_facecolor(colors.get(m, '#ccc'))
        p.set_alpha(0.7)
    ax.set_ylabel('Facility Visits')
    ax.set_title(title + ': Facility Visits')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    ax = axes[2]
    fr = [[max(logs[-1].simple_regret, 0.0) for logs in results[m]] for m in mns]
    bp = ax.boxplot(fr, labels=mns, patch_artist=True)
    for p, m in zip(bp['boxes'], mns):
        p.set_facecolor(colors.get(m, '#ccc'))
        p.set_alpha(0.7)
    ax.set_ylabel('Final Regret')
    ax.set_title(title + ': Final Regret')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved to " + save_path)
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("qsmfbo.core smoke test")
    print("=" * 60)
    results = run_experiment(StyblinskiTang2D, budget=30.0, lambda_overhead=25.0,
        lambda_marginal=2.0, n_seeds=1, verbose=True)
    for m, ll in results.items():
        fr = [max(l[-1].simple_regret, 0.0) for l in ll]
        fv = [l[-1].n_facility_visits for l in ll]
        print("  %s: regret=%.4f, visits=%.1f" % (m, np.mean(fr), np.mean(fv)))
    print("Smoke test passed!")