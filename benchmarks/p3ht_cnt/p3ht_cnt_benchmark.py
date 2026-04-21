"""
Three-fidelity retrospective benchmark on the P3HT-CNT composite dataset.

Reproduces Section 2.2 of the paper and the associated Extended Data figures.
The dataset is Bash et al. (Adv. Funct. Mater. 2021); obtain the .xlsx from
the repository linked in this folder's README.md.

This module is a standalone parallel implementation to qsmfbo.core:
  - QueueScheduler3F         three-fidelity QS-MFBO with FAVP
  - StandardMFMES3F          three-fidelity MF-MES baseline
  - P3HTCNTLookupTable       replicate-pool oracle over the five-dimensional
                             composition simplex
  - ThreeFidelityCostModel   queue-dependent cost with floor constraint

The three-fidelity specialisation is kept separate from the two-fidelity
framework in qsmfbo.core so that the published numbers for Section 2.2
reproduce bit-for-bit from the code that generated them.

Fidelity mapping used by the paper:
  m = 0  LF  log absorption ratio 602 nm / 525 nm      cost = 1
  m = 1  MF  log sheet resistance (sign-inverted)      overhead 1.5, marginal 1.0
  m = 2  HF  log conductivity                          overhead 3.0, marginal 1.5

FAVP runs with tau = 4.0 on this benchmark (versus tau = 3.0 on the synthetic
benchmarks) to suppress spurious flags on genuinely noisy replicate draws;
see Supplementary Note 2 for the rationale.

Reference:
    Kartha, K. & James, A. P. Cost-aware multi-fidelity scheduling and
    cross-fidelity anomaly resolution for iterative learning under
    laboratory constraints. (2026). See Section 2.2.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.spatial.distance import cdist

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
print("Device: " + str(DEVICE) +
      (" (" + torch.cuda.get_device_name(0) + ")" if DEVICE.type == "cuda" else ""))

# Fidelity constants
FIDELITY_LF  = 0.0
FIDELITY_MF  = 0.5
FIDELITY_HF  = 1.0
FIDELITY_NAMES = {FIDELITY_LF: "LF", FIDELITY_MF: "MF", FIDELITY_HF: "HF"}
QUEUED_FIDELITIES  = [FIDELITY_MF, FIDELITY_HF]
ALL_FIDELITIES     = [FIDELITY_LF, FIDELITY_MF, FIDELITY_HF]
DIM = 5
FIDELITY_DIM = DIM
TARGET_FIDELITIES = {FIDELITY_DIM: FIDELITY_HF}

# Tunable hyperparameters
NUM_MV_SAMPLES = 10
NUM_FANTASIES  = 16
NUM_RESTARTS   = 10
RAW_SAMPLES    = 128
N_CANDIDATES   = 500


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
    comp_idx: int = -1
    batch_id: Optional[int] = None
    is_favp_repeat: bool = False
    is_favp_escalation: bool = False

@dataclass
class QueueItem:
    x: torch.Tensor
    fidelity: float
    iteration_added: int
    comp_idx: int
    source: str = "acquisition"

@dataclass
class FAVPEvent:
    iteration: int
    comp_idx: int
    fidelity: float
    y_original: float
    residual: float
    y_repeat: Optional[float] = None
    repeat_is_nan: bool = False
    case: Optional[str] = None   # "A1", "A2", "A2_dedup"
    escalated_to: Optional[float] = None

@dataclass
class IterationLog:
    iteration: int
    action: str
    fidelity: Optional[float] = None
    comp_idx: Optional[int] = None
    cost_incurred: float = 0.0
    cumulative_cost: float = 0.0
    best_y: float = float('-inf')
    true_regret: float = float('inf')
    queue_sizes: Dict[float, int] = field(default_factory=dict)
    batch_size: int = 0
    n_sessions: Dict[float, int] = field(default_factory=dict)


# ++++++++++++++++++
# Per-Fidelity Normalizer
# ++++++++++++++++++

class PerFidelityNormalizer:
    """
    Normalize observations per-fidelity to zero mean, unit std.

    Without this, the GP sees LF~[-0.5, 0.3], MF~[-9, -3], HF~[2, 7]
    and spends all its capacity modelling the between-fidelity offset
    rather than the within-fidelity variation that matters for optimisation.
    """
    def __init__(self):
        self.stats = {}

    def fit(self, observations):
        by_fid = defaultdict(list)
        for o in observations:
            by_fid[o.fidelity].append(o.y)
        self.stats = {}
        for fid, vals in by_fid.items():
            m = float(np.mean(vals))
            s = float(np.std(vals))
            if s < 1e-8:
                s = 1.0
            self.stats[fid] = (m, s)

    def transform(self, y, fidelity):
        if fidelity not in self.stats:
            return y
        m, s = self.stats[fidelity]
        return (y - m) / s

    def inverse_transform(self, y_norm, fidelity):
        if fidelity not in self.stats:
            return y_norm
        m, s = self.stats[fidelity]
        return y_norm * s + m


# ++++++++++++++++++
# Lookup Table
# ++++++++++++++++++

class P3HTCNTLookupTable:
    COMP_COLS = ['P3HT content (%)', 'D1 content (%)', 'D2 content (%)',
                 'D6 content (%)', 'D8 content (%)']
    FIDELITY_COL_MAP = {
        FIDELITY_LF: 'Absorption Ratio (log)',
        FIDELITY_MF: 'Sheet Resistance (ohm/sq) (log)',
        FIDELITY_HF: 'Conductivity (log)',
    }

    def __init__(self, filepath):
        df = pd.read_excel(filepath)
        df['comp_key'] = (df[self.COMP_COLS].round(2)
                          .apply(lambda r: tuple(r), axis=1))
        has_hf = df.groupby('comp_key')['Conductivity (log)'].transform(
            lambda s: s.notna().any())
        df_hf = df[has_hf].copy()

        comp_keys = list(df_hf['comp_key'].unique())
        self.n_compositions = len(comp_keys)
        self.comp_key_to_idx = {k: i for i, k in enumerate(comp_keys)}
        self.compositions = np.array([list(k) for k in comp_keys]) / 100.0
        self.compositions_tensor = torch.tensor(self.compositions,
                                                dtype=torch.double)
        self.dim = DIM

        self.replicate_pools = {}
        for comp_key, idx in self.comp_key_to_idx.items():
            group = df_hf[df_hf['comp_key'] == comp_key]
            self.replicate_pools[idx] = {}
            for fid, col in self.FIDELITY_COL_MAP.items():
                self.replicate_pools[idx][fid] = group[col].tolist()

        self.ground_truth = np.full(self.n_compositions, np.nan)
        for idx in range(self.n_compositions):
            hf_vals = [v for v in self.replicate_pools[idx][FIDELITY_HF]
                       if not np.isnan(v)]
            if hf_vals:
                self.ground_truth[idx] = np.mean(hf_vals)

        self.optimal_value = float(np.nanmax(self.ground_truth))
        self.optimal_idx   = int(np.nanargmax(self.ground_truth))

    def snap_to_nearest(self, x):
        x_np = x.detach().cpu().numpy().reshape(1, -1)
        return int(np.argmin(cdist(x_np, self.compositions)))

    def query(self, comp_idx, fidelity, rng):
        pool = self.replicate_pools[comp_idx][fidelity]
        draw = rng.choice(pool)
        if np.isnan(draw):
            return None, True
        if fidelity == FIDELITY_MF:
            draw = -draw
        return float(draw), False

    def get_composition_tensor(self, comp_idx):
        return self.compositions_tensor[comp_idx].clone()

    def print_info(self):
        opt = self.compositions[self.optimal_idx] * 100
        print("P3HT-CNT Lookup Table")
        print("  Compositions       : %d" % self.n_compositions)
        print("  Input dim          : %d (5D simplex)" % self.dim)
        print("  Optimal Y (log s)  : %.4f" % self.optimal_value)
        print("  Optimal composition: P3HT=%.1f%% D1=%.1f%% D2=%.1f%% D6=%.1f%% D8=%.1f%%" %
              (opt[0], opt[1], opt[2], opt[3], opt[4]))
        for fid, name in FIDELITY_NAMES.items():
            n_valid = sum(
                len([v for v in self.replicate_pools[i][fid] if not np.isnan(v)])
                for i in range(self.n_compositions))
            n_total = sum(len(self.replicate_pools[i][fid])
                          for i in range(self.n_compositions))
            print("  %s valid/total : %d/%d (%.1f%%)" %
                  (name, n_valid, n_total, 100*n_valid/n_total))


# ++++++++++++++++++
# Cost Models
# ++++++++++++++++++

class ThreeFidelityCostModel:
    def __init__(self, lambda_lf=1.0,
                 lambda_o_mf=1.5, lambda_mar_mf=1.0,
                 lambda_o_hf=3.0, lambda_mar_hf=1.5):
        self.lambda_lf = lambda_lf
        self.overheads  = {FIDELITY_MF: lambda_o_mf,  FIDELITY_HF: lambda_o_hf}
        self.marginals  = {FIDELITY_MF: max(lambda_mar_mf, lambda_lf),
                           FIDELITY_HF: max(lambda_mar_hf, lambda_mar_mf)}

    def cost_per_sample(self, queue_size, fidelity):
        if fidelity == FIDELITY_LF:
            return self.lambda_lf
        return self.overheads[fidelity] / (queue_size + 1) + self.marginals[fidelity]

    def batch_cost(self, queue_size, fidelity):
        if queue_size == 0:
            return 0.0
        return self.overheads[fidelity] + queue_size * self.marginals[fidelity]

    def single_cost(self, fidelity):
        if fidelity == FIDELITY_LF:
            return self.lambda_lf
        return self.overheads[fidelity] + self.marginals[fidelity]


class FixedThreeFidelityCostModel:
    def __init__(self, lambda_lf=1.0,
                 lambda_o_mf=1.5, lambda_mar_mf=1.0,
                 lambda_o_hf=3.0, lambda_mar_hf=1.5):
        self.lambda_lf = lambda_lf
        self.overheads  = {FIDELITY_MF: lambda_o_mf,  FIDELITY_HF: lambda_o_hf}
        self.marginals  = {FIDELITY_MF: max(lambda_mar_mf, lambda_lf),
                           FIDELITY_HF: max(lambda_mar_hf, lambda_mar_mf)}
        self.fixed = {
            FIDELITY_LF: lambda_lf,
            FIDELITY_MF: lambda_o_mf + max(lambda_mar_mf, lambda_lf),
            FIDELITY_HF: lambda_o_hf + max(lambda_mar_hf, lambda_mar_mf),
        }

    def cost_per_sample(self, queue_size, fidelity):
        return self.fixed[fidelity]

    def batch_cost(self, queue_size, fidelity):
        if queue_size == 0:
            return 0.0
        return self.overheads[fidelity] + queue_size * self.marginals[fidelity]

    def single_cost(self, fidelity):
        return self.fixed[fidelity]


# ++++++++++++++++++
# GP + MF-MES Utilities
# ++++++++++++++++++

def build_mf_model(train_X, train_Y):
    model = SingleTaskMultiFidelityGP(
        train_X=train_X, train_Y=train_Y,
        outcome_transform=Standardize(m=1),
        data_fidelities=[FIDELITY_DIM])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def make_cost_utility(cost_lf, cost_hf):
    w = max(cost_hf - cost_lf, 0.01)
    cm = AffineFidelityCostModel(fidelity_weights={FIDELITY_DIM: w},
                                 fixed_cost=cost_lf)
    return InverseCostWeightedUtility(cost_model=cm)


def build_mfmes(model, cost_utility):
    cands = torch.rand(N_CANDIDATES, DIM + 1, dtype=torch.double)
    cands[:, FIDELITY_DIM] = FIDELITY_HF
    def proj(X):
        return project_to_target_fidelity(
            X=X, target_fidelities=TARGET_FIDELITIES, d=DIM + 1)
    return qMultiFidelityMaxValueEntropy(
        model=model, candidate_set=cands,
        num_fantasies=NUM_FANTASIES, num_mv_samples=NUM_MV_SAMPLES,
        num_y_samples=128, cost_aware_utility=cost_utility, project=proj)


def optimize_mfmes(acqf, pending_X=None):
    if pending_X is not None:
        acqf.set_X_pending(pending_X)
    bounds = torch.zeros(2, DIM + 1, dtype=torch.double)
    bounds[1, :] = 1.0
    cand, val = optimize_acqf_mixed(
        acq_function=acqf, bounds=bounds, q=1,
        num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
        fixed_features_list=[{FIDELITY_DIM: f} for f in ALL_FIDELITIES])
    x = cand.squeeze()[:DIM]
    fid_raw = cand.squeeze()[FIDELITY_DIM].item()
    fid = min(ALL_FIDELITIES, key=lambda f: abs(f - fid_raw))
    return x, fid, val.item()


def evaluate_info_gain(model, x, fidelity):
    """Raw MF-MES info gain (unit cost) for greedy sequential fantasisation."""
    unit_cost = AffineFidelityCostModel(
        fidelity_weights={FIDELITY_DIM: 0.0}, fixed_cost=1.0)
    unit_utility = InverseCostWeightedUtility(cost_model=unit_cost)
    acqf = build_mfmes(model, unit_utility)
    x_fid = torch.cat([x, torch.tensor([fidelity], dtype=torch.double)]).unsqueeze(0)
    try:
        ig = acqf(x_fid).item()
    except Exception:
        ig = 0.0
    return max(ig, 0.0)


# ++++++++++++++++++
# Shared Initialisation
# ++++++++++++++++++

def generate_shared_init(lookup, n_lf=5, n_mf=3, n_hf=2, seed=0):
    rng = np.random.RandomState(seed)
    selected = rng.choice(lookup.n_compositions,
                          size=n_lf + n_mf + n_hf, replace=False)
    def draw_valid(idx, fid):
        for _ in range(20):
            v, nan = lookup.query(idx, fid, rng)
            if not nan:
                return v
        return None

    init = {'lf': [], 'mf': [], 'hf': []}
    for idx in selected[:n_lf]:
        v = draw_valid(idx, FIDELITY_LF)
        if v is not None:
            init['lf'].append((int(idx), v))
    for idx in selected[n_lf:n_lf + n_mf]:
        v = draw_valid(idx, FIDELITY_MF)
        if v is not None:
            init['mf'].append((int(idx), v))
    for idx in selected[n_lf + n_mf:]:
        v = draw_valid(idx, FIDELITY_HF)
        if v is not None:
            init['hf'].append((int(idx), v))
    return init


def compute_init_cost(cost_model, init_data):
    c = len(init_data['lf']) * cost_model.lambda_lf
    n_mf = len(init_data['mf'])
    if n_mf > 0:
        c += cost_model.batch_cost(n_mf, FIDELITY_MF)
    n_hf = len(init_data['hf'])
    if n_hf > 0:
        c += cost_model.batch_cost(n_hf, FIDELITY_HF)
    return c


# ++++++++++++++++++
# QueueScheduler - 3 Fidelity with FAVP (v4: escalation deduplication)
# ++++++++++++++++++

class QueueScheduler3F:
    def __init__(self, lookup, cost_model, seed=0, init_data=None,
                 q_min=2, q_max=20, use_favp=True,
                 tau=4.0, gamma=1.5, n_min=10):
        self.lookup     = lookup
        self.cost_model = cost_model
        self.seed       = seed
        self.rng        = np.random.RandomState(seed)
        self.init_data  = init_data
        self.q_min      = q_min
        self.q_max      = q_max
        self.use_favp  = use_favp
        self.tau        = tau
        self.gamma      = gamma
        self.n_min      = n_min

        self.observations = []
        self.queues = {FIDELITY_MF: [], FIDELITY_HF: []}
        self.log = []
        self.favp_events = []
        self.iteration     = 0
        self.cumulative_cost = 0.0
        self.n_sessions    = {FIDELITY_MF: 0, FIDELITY_HF: 0}
        self.batch_counter = 0
        self.model         = None
        self.normalizer    = PerFidelityNormalizer()
        torch.manual_seed(seed)

    def _train_tensors(self):
        if not self.observations:
            return None, None
        self.normalizer = PerFidelityNormalizer()
        self.normalizer.fit(self.observations)
        X = torch.stack([torch.cat([o.x, torch.tensor([o.fidelity])])
                         for o in self.observations])
        Y = torch.tensor([self.normalizer.transform(o.y, o.fidelity)
                          for o in self.observations]).unsqueeze(-1)
        return X, Y

    def _pending_X(self):
        items = []
        for fid in QUEUED_FIDELITIES:
            for qi in self.queues[fid]:
                items.append(torch.cat([qi.x, torch.tensor([qi.fidelity])]))
        return torch.stack(items) if items else None

    def _already_queued(self, comp_idx, fid):
        """Check if a composition is already present in a queue."""
        return any(qi.comp_idx == comp_idx for qi in self.queues[fid])

    def _best_hf(self):
        hf = [o.y for o in self.observations if o.fidelity == FIDELITY_HF]
        return max(hf) if hf else float('-inf')

    def _regret(self):
        b = self._best_hf()
        return max(self.lookup.optimal_value - b, 0.0) if b > float('-inf') else float('inf')

    def _fit(self):
        X, Y = self._train_tensors()
        if X is None or X.shape[0] < 4:
            return False
        try:
            self.model = build_mf_model(X, Y)
            return True
        except Exception:
            return False

    def _cost_utility(self):
        q_hf = len(self.queues[FIDELITY_HF])
        c_lf = self.cost_model.cost_per_sample(0, FIDELITY_LF)
        c_hf = self.cost_model.cost_per_sample(q_hf, FIDELITY_HF)
        return make_cost_utility(c_lf, c_hf)

    def _greedy_queue_valuation(self, fidelity):
        """
        Full greedy sequential fantasisation for one queue.

        At each round:
          1. Score all remaining items with MF-MES info gain on current model
          2. Pick the item with highest score; add its gain to G_hat
          3. Fantasise that item at GP posterior mean, refit GP
          4. Repeat until all items scored

        Returns total estimated queue value (sum of greedy selections).
        """
        queue = self.queues[fidelity]
        if not queue:
            return 0.0
        q = len(queue)
        remaining = list(range(q))
        G_hat = 0.0

        base_X, base_Y = self._train_tensors()
        if base_X is None:
            return 0.0
        acc_X, acc_Y = base_X.clone(), base_Y.clone()
        current_model = self.model

        for step in range(q):
            gains = {}
            for idx in remaining:
                gains[idx] = evaluate_info_gain(
                    current_model, queue[idx].x, queue[idx].fidelity)
            if not gains:
                break
            best_idx = max(gains, key=gains.get)
            G_hat += gains[best_idx]
            remaining.remove(best_idx)

            if remaining:
                item = queue[best_idx]
                x_fid = torch.cat([item.x, torch.tensor([item.fidelity])]).unsqueeze(0)
                with torch.no_grad():
                    y_fan_norm = current_model.posterior(x_fid).mean.squeeze()
                acc_X = torch.cat([acc_X, x_fid])
                acc_Y = torch.cat([acc_Y, y_fan_norm.reshape(1, 1)])
                try:
                    current_model = build_mf_model(acc_X, acc_Y)
                except Exception:
                    break
        return G_hat

    def _measure(self, comp_idx, fidelity, is_repeat=False, is_esc=False):
        x = self.lookup.get_composition_tensor(comp_idx)
        val, is_nan = self.lookup.query(comp_idx, fidelity, self.rng)
        if is_nan:
            return None
        obs = Observation(x=x, fidelity=fidelity, y=val, cost=0.0,
                          iteration=self.iteration, comp_idx=comp_idx,
                          is_favp_repeat=is_repeat,
                          is_favp_escalation=is_esc)
        self.observations.append(obs)
        return obs

    def _favp(self, obs):
        """
        FAVP with escalation deduplication.

        If an anomaly triggers an escalation to a higher fidelity, but the
        composition is already queued at that fidelity, the escalation is
        suppressed (case="A2_dedup"). The original and repeat observations
        are still kept in the GP, but no redundant queue entry is created.
        """
        if not self.use_favp or obs.fidelity == FIDELITY_HF:
            return
        if len(self.observations) < self.n_min:
            return

        xf = torch.cat([obs.x, torch.tensor([obs.fidelity])]).unsqueeze(0)
        with torch.no_grad():
            post = self.model.posterior(xf)
            mu_norm = post.mean.item()
            sigma_norm = post.variance.sqrt().item()
        if sigma_norm < 1e-8:
            return

        y_norm = self.normalizer.transform(obs.y, obs.fidelity)
        r = abs(y_norm - mu_norm) / sigma_norm
        if r <= self.tau:
            return

        ev = FAVPEvent(iteration=self.iteration, comp_idx=obs.comp_idx,
                        fidelity=obs.fidelity, y_original=obs.y, residual=r)

        rep_cost = self.cost_model.cost_per_sample(0, obs.fidelity)
        self.cumulative_cost += rep_cost

        rep = self._measure(obs.comp_idx, obs.fidelity, is_repeat=True)

        next_fid = FIDELITY_MF if obs.fidelity == FIDELITY_LF else FIDELITY_HF

        if rep is None:
            # Repeat was NaN - treat as contradicting, try to escalate
            ev.repeat_is_nan = True
            if self._already_queued(obs.comp_idx, next_fid):
                ev.case = "A2_dedup"
            else:
                ev.case = "A2"
                ev.escalated_to = next_fid
                x = self.lookup.get_composition_tensor(obs.comp_idx)
                self.queues[next_fid].append(QueueItem(
                    x=x, fidelity=next_fid, iteration_added=self.iteration,
                    comp_idx=obs.comp_idx, source="favp_escalation"))
        else:
            ev.y_repeat = rep.y
            delta = abs(rep.y - obs.y)
            # Consistency threshold in raw space: scale sigma_norm back
            _, s = self.normalizer.stats.get(obs.fidelity, (0, 1))
            epsilon = self.gamma * sigma_norm * s
            if delta <= epsilon:
                ev.case = "A1"  # genuine extreme, both kept
            else:
                if self._already_queued(obs.comp_idx, next_fid):
                    ev.case = "A2_dedup"  # contradicting but already queued
                else:
                    ev.case = "A2"
                    ev.escalated_to = next_fid
                    x = self.lookup.get_composition_tensor(obs.comp_idx)
                    self.queues[next_fid].append(QueueItem(
                        x=x, fidelity=next_fid, iteration_added=self.iteration,
                        comp_idx=obs.comp_idx, source="favp_escalation"))

        self.favp_events.append(ev)

    def _exec_batch(self, fidelity):
        queue = self.queues[fidelity]
        q  = len(queue)
        bc = self.cost_model.batch_cost(q, fidelity)
        self.cumulative_cost += bc
        self.batch_counter   += 1
        self.n_sessions[fidelity] += 1

        for item in queue:
            obs = self._measure(item.comp_idx, fidelity)
            if obs is not None:
                obs.batch_id = self.batch_counter
                if fidelity == FIDELITY_MF:
                    self._favp(obs)

        queue.clear()
        return self._make_log('batch_' + FIDELITY_NAMES[fidelity],
                              fidelity=fidelity, cost=bc, bs=q)

    def _do_init(self):
        d = self.init_data
        for ci, v in d['lf']:
            x = self.lookup.get_composition_tensor(ci)
            self.observations.append(Observation(
                x=x, fidelity=FIDELITY_LF, y=v, cost=0, iteration=0, comp_idx=ci))
        if d['mf']:
            self.batch_counter += 1
            self.n_sessions[FIDELITY_MF] += 1
            for ci, v in d['mf']:
                x = self.lookup.get_composition_tensor(ci)
                self.observations.append(Observation(
                    x=x, fidelity=FIDELITY_MF, y=v, cost=0, iteration=0,
                    comp_idx=ci, batch_id=self.batch_counter))
        if d['hf']:
            self.batch_counter += 1
            self.n_sessions[FIDELITY_HF] += 1
            for ci, v in d['hf']:
                x = self.lookup.get_composition_tensor(ci)
                self.observations.append(Observation(
                    x=x, fidelity=FIDELITY_HF, y=v, cost=0, iteration=0,
                    comp_idx=ci, batch_id=self.batch_counter))
        self.cumulative_cost = compute_init_cost(self.cost_model, d)

    def _make_log(self, action, fidelity=None, comp_idx=None, cost=0.0, bs=0):
        return IterationLog(
            iteration=self.iteration, action=action, fidelity=fidelity,
            comp_idx=comp_idx, cost_incurred=cost,
            cumulative_cost=self.cumulative_cost,
            best_y=self._best_hf(), true_regret=self._regret(),
            queue_sizes={f: len(self.queues[f]) for f in QUEUED_FIDELITIES},
            batch_size=bs, n_sessions=dict(self.n_sessions))

    def _iterate(self):
        self.iteration += 1

        if not self._fit():
            ci = self.rng.randint(self.lookup.n_compositions)
            self.cumulative_cost += self.cost_model.lambda_lf
            self._measure(ci, FIDELITY_LF)
            return self._make_log('random_lf', FIDELITY_LF, ci, self.cost_model.lambda_lf)

        try:
            x_star, fid_star, acq_imm = self._suggest()
        except Exception:
            ci = self.rng.randint(self.lookup.n_compositions)
            self.cumulative_cost += self.cost_model.lambda_lf
            self._measure(ci, FIDELITY_LF)
            return self._make_log('random_lf', FIDELITY_LF, ci, self.cost_model.lambda_lf)

        comp_idx = self.lookup.snap_to_nearest(x_star)

        # Queued fidelity selected -> add to queue
        if fid_star in QUEUED_FIDELITIES:
            x = self.lookup.get_composition_tensor(comp_idx)
            self.queues[fid_star].append(QueueItem(
                x=x, fidelity=fid_star, iteration_added=self.iteration,
                comp_idx=comp_idx))
            if len(self.queues[fid_star]) >= self.q_max:
                return self._exec_batch(fid_star)
            return self._make_log('queue_' + FIDELITY_NAMES[fid_star],
                                  fid_star, comp_idx, 0.0)

        # LF selected -> check batch triggers using full greedy valuation
        best_fid, best_alpha = None, -1.0
        for fid in QUEUED_FIDELITIES:
            if len(self.queues[fid]) >= self.q_min:
                G = self._greedy_queue_valuation(fid)
                C = self.cost_model.batch_cost(len(self.queues[fid]), fid)
                alpha = G / C if C > 0 else 0.0
                if alpha >= acq_imm and alpha > best_alpha:
                    best_alpha = alpha
                    best_fid   = fid

        if best_fid is not None:
            return self._exec_batch(best_fid)

        # ── Loop-prevention safeguard ──
        # If LF-at-comp is already queued at a higher fidelity, executing
        # yet another LF measurement would waste budget on a composition
        # whose uncertainty can only be resolved by the pending higher-
        # fidelity measurement. Force-execute the batch containing that
        # composition instead.
        for fid in QUEUED_FIDELITIES:
            if self._already_queued(comp_idx, fid) and len(self.queues[fid]) >= 1:
                return self._exec_batch(fid)

        # Execute LF
        self.cumulative_cost += self.cost_model.lambda_lf
        obs = self._measure(comp_idx, FIDELITY_LF)
        if obs is not None:
            self._favp(obs)
        return self._make_log('exec_LF', FIDELITY_LF, comp_idx,
                              self.cost_model.lambda_lf)

    def _suggest(self):
        cu   = self._cost_utility()
        acqf = build_mfmes(self.model, cu)
        return optimize_mfmes(acqf, self._pending_X())

    def run(self, budget, verbose=True):
        label = "QS-MFBO+FAVP" if self.use_favp else "QS-MFBO"
        if verbose:
            print("\n%s | Budget=%s | Seed=%d" % (label, str(budget), self.seed))
        self._do_init()
        if verbose:
            print("  Init: cost=%.1f  obs=%d  best_hf=%.3f" %
                  (self.cumulative_cost, len(self.observations), self._best_hf()))

        while self.cumulative_cost < budget:
            entry = self._iterate()
            self.log.append(entry)
            if verbose:
                qs = entry.queue_sizes
                print("  It%3d | %-14s | Qm:%2d Qh:%2d | C:%6.1f/%s | R:%.4f" %
                      (self.iteration, entry.action, qs.get(0.5, 0), qs.get(1.0, 0),
                       self.cumulative_cost, str(int(budget)), entry.true_regret))

        for fid in QUEUED_FIDELITIES:
            if self.queues[fid]:
                if verbose:
                    print("  Flush %s queue (%d items)" %
                          (FIDELITY_NAMES[fid], len(self.queues[fid])))
                self.log.append(self._exec_batch(fid))

        if verbose:
            print("  DONE | C:%.1f R:%.4f Sess MF:%d HF:%d FAVP:%d" %
                  (self.cumulative_cost, self._regret(),
                   self.n_sessions[FIDELITY_MF], self.n_sessions[FIDELITY_HF],
                   len(self.favp_events)))
        return self.log


# ++++++++++++++++++
# Standard MF-MES Baseline
# ++++++++++++++++++

class StandardMFMES3F:
    def __init__(self, lookup, cost_model, seed=0, init_data=None):
        self.lookup     = lookup
        self.cost_model = cost_model
        self.seed       = seed
        self.rng        = np.random.RandomState(seed)
        self.init_data  = init_data
        self.observations = []
        self.log = []
        self.iteration     = 0
        self.cumulative_cost = 0.0
        self.n_sessions    = {FIDELITY_MF: 0, FIDELITY_HF: 0}
        self.model = None
        self.normalizer = PerFidelityNormalizer()
        torch.manual_seed(seed)

    def _train_tensors(self):
        if not self.observations:
            return None, None
        self.normalizer = PerFidelityNormalizer()
        self.normalizer.fit(self.observations)
        X = torch.stack([torch.cat([o.x, torch.tensor([o.fidelity])])
                         for o in self.observations])
        Y = torch.tensor([self.normalizer.transform(o.y, o.fidelity)
                          for o in self.observations]).unsqueeze(-1)
        return X, Y

    def _best_hf(self):
        hf = [o.y for o in self.observations if o.fidelity == FIDELITY_HF]
        return max(hf) if hf else float('-inf')

    def _regret(self):
        b = self._best_hf()
        return max(self.lookup.optimal_value - b, 0.0) if b > float('-inf') else float('inf')

    def _fit(self):
        X, Y = self._train_tensors()
        if X is None or X.shape[0] < 4:
            return False
        try:
            self.model = build_mf_model(X, Y)
            return True
        except Exception:
            return False

    def _do_init(self):
        d = self.init_data
        for ci, v in d['lf']:
            x = self.lookup.get_composition_tensor(ci)
            self.observations.append(Observation(
                x=x, fidelity=FIDELITY_LF, y=v, cost=0, iteration=0, comp_idx=ci))
        for ci, v in d['mf']:
            x = self.lookup.get_composition_tensor(ci)
            self.observations.append(Observation(
                x=x, fidelity=FIDELITY_MF, y=v, cost=0, iteration=0, comp_idx=ci))
            self.n_sessions[FIDELITY_MF] += 1
        for ci, v in d['hf']:
            x = self.lookup.get_composition_tensor(ci)
            self.observations.append(Observation(
                x=x, fidelity=FIDELITY_HF, y=v, cost=0, iteration=0, comp_idx=ci))
            self.n_sessions[FIDELITY_HF] += 1
        self.cumulative_cost = compute_init_cost(self.cost_model, d)

    def _make_log(self, action, fidelity=None, comp_idx=None, cost=0.0):
        return IterationLog(
            iteration=self.iteration, action=action, fidelity=fidelity,
            comp_idx=comp_idx, cost_incurred=cost,
            cumulative_cost=self.cumulative_cost,
            best_y=self._best_hf(), true_regret=self._regret(),
            queue_sizes={FIDELITY_MF: 0, FIDELITY_HF: 0},
            n_sessions=dict(self.n_sessions))

    def run(self, budget, verbose=True):
        if verbose:
            print("\nMF-MES baseline | Budget=%s | Seed=%d" % (str(budget), self.seed))
        self._do_init()
        if verbose:
            print("  Init: cost=%.1f  obs=%d" %
                  (self.cumulative_cost, len(self.observations)))

        while self.cumulative_cost < budget:
            self.iteration += 1

            if not self._fit():
                ci = self.rng.randint(self.lookup.n_compositions)
                cost = self.cost_model.lambda_lf
                self.cumulative_cost += cost
                v, nan = self.lookup.query(ci, FIDELITY_LF, self.rng)
                if v is not None:
                    x = self.lookup.get_composition_tensor(ci)
                    self.observations.append(Observation(
                        x=x, fidelity=FIDELITY_LF, y=v, cost=cost,
                        iteration=self.iteration, comp_idx=ci))
                self.log.append(self._make_log('random_lf', FIDELITY_LF, ci, cost))
                continue

            c_lf = self.cost_model.single_cost(FIDELITY_LF)
            c_hf = self.cost_model.single_cost(FIDELITY_HF)
            cu   = make_cost_utility(c_lf, c_hf)

            try:
                acqf = build_mfmes(self.model, cu)
                x_star, fid_star, _ = optimize_mfmes(acqf)
            except Exception:
                ci = self.rng.randint(self.lookup.n_compositions)
                fid_star = FIDELITY_LF
                x_star = self.lookup.get_composition_tensor(ci)

            ci   = self.lookup.snap_to_nearest(x_star)
            cost = self.cost_model.single_cost(fid_star)
            self.cumulative_cost += cost

            if fid_star in QUEUED_FIDELITIES:
                self.n_sessions[fid_star] += 1

            v, nan = self.lookup.query(ci, fid_star, self.rng)
            if v is not None:
                x = self.lookup.get_composition_tensor(ci)
                self.observations.append(Observation(
                    x=x, fidelity=fid_star, y=v, cost=cost,
                    iteration=self.iteration, comp_idx=ci))

            act = 'exec_' + FIDELITY_NAMES[fid_star]
            self.log.append(self._make_log(act, fid_star, ci, cost))

            if verbose:
                print("  It%3d | %-8s | C:%6.1f/%s | R:%.4f" %
                      (self.iteration, act, self.cumulative_cost,
                       str(int(budget)), self._regret()))

        if verbose:
            print("  DONE | C:%.1f R:%.4f Sess MF:%d HF:%d" %
                  (self.cumulative_cost, self._regret(),
                   self.n_sessions[FIDELITY_MF], self.n_sessions[FIDELITY_HF]))
        return self.log


# ++++++++++++++++++
# Experiment Runner
# ++++++++++++++++++

def run_p3ht_benchmark(filepath, budget=200.0, n_seeds=10, verbose=True,
                       n_init_lf=5, n_init_mf=3, n_init_hf=2,
                       save_checkpoint=None):
    """
    Run the P3HT-CNT benchmark comparing:
      1. QS-MFBO + FAVP (full framework, full greedy valuation, tau=4, dedup)
      2. QS-MFBO         (scheduling only, no FAVP)
      3. MF-MES          (no scheduling, no FAVP)
    """
    lookup = P3HTCNTLookupTable(filepath)
    lookup.print_info()

    results = {'QS-MFBO+FAVP': [], 'QS-MFBO': [], 'MF-MES': []}
    all_favp = []

    for seed in range(n_seeds):
        t0 = time.time()
        if verbose:
            print("\n" + "="*60)
            print("SEED %d" % seed)
            print("="*60)

        init_data = generate_shared_init(lookup, n_init_lf, n_init_mf,
                                         n_init_hf, seed=seed)

        qc = ThreeFidelityCostModel()
        fc = FixedThreeFidelityCostModel()

        r1 = QueueScheduler3F(lookup, qc, seed=seed,
                              init_data=init_data, use_favp=True, tau=4.0)
        results['QS-MFBO+FAVP'].append(r1.run(budget, verbose))
        all_favp.append(r1.favp_events)

        r2 = QueueScheduler3F(lookup, qc, seed=seed,
                              init_data=init_data, use_favp=False, tau=4.0)
        results['QS-MFBO'].append(r2.run(budget, verbose))

        r3 = StandardMFMES3F(lookup, fc, seed=seed, init_data=init_data)
        results['MF-MES'].append(r3.run(budget, verbose))

        if verbose:
            print("  Seed %d elapsed: %.0fs" % (seed, time.time()-t0))

        if save_checkpoint is not None:
            try:
                with open(save_checkpoint, 'wb') as f:
                    pickle.dump({
                        'results': results,
                        'favp': all_favp,
                        'seeds_completed': seed + 1,
                        'budget': budget,
                    }, f)
                if verbose:
                    print("  Checkpoint saved: %s" % save_checkpoint)
            except Exception as e:
                print("  [WARN] Checkpoint save failed: %s" % e)

    return results, all_favp, lookup


# ++++++++++++++++++
# Plotting & Summary
# ++++++++++++++++++

def plot_p3ht_results(results, lookup, budget=200, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'QS-MFBO+FAVP': '#e74c3c', 'QS-MFBO': '#2ecc71',
              'MF-MES': '#3498db'}
    ls     = {'QS-MFBO+FAVP': '-', 'QS-MFBO': '--', 'MF-MES': ':'}
    lw     = {'QS-MFBO+FAVP': 2.5, 'QS-MFBO': 2.0, 'MF-MES': 1.5}

    ax = axes[0]
    for method, logs_list in results.items():
        all_c, all_r = [], []
        for logs in logs_list:
            c, r = [], []
            for l in logs:
                c.append(l.cumulative_cost)
                r.append(max(l.true_regret, 1e-4))
            all_c.append(c)
            all_r.append(r)
        cg = np.linspace(0, budget * 1.1, 100)
        ir = [np.interp(cg, c_, r_) for c_, r_ in zip(all_c, all_r)]
        mr, sr = np.mean(ir, 0), np.std(ir, 0)
        ax.plot(cg, mr, label=method, color=colors[method],
                linestyle=ls[method], linewidth=lw[method])
        ax.fill_between(cg, np.maximum(mr - sr, 1e-4), mr + sr,
                        alpha=.15, color=colors[method])
    ax.set_xlabel('Cumulative cost')
    ax.set_ylabel('True regret')
    ax.set_title('P3HT-CNT: Regret vs Cost')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=.3)

    ax = axes[1]
    methods = list(results.keys())
    x_pos   = np.arange(len(methods))
    w = 0.35
    mf_s = [[l[-1].n_sessions.get(0.5, 0) for l in results[m]] for m in methods]
    hf_s = [[l[-1].n_sessions.get(1.0, 0) for l in results[m]] for m in methods]
    ax.bar(x_pos - w/2, [np.mean(s) for s in mf_s], w,
           yerr=[np.std(s) for s in mf_s],
           label='MF sessions', color='#f39c12', alpha=.7, capsize=3)
    ax.bar(x_pos + w/2, [np.mean(s) for s in hf_s], w,
           yerr=[np.std(s) for s in hf_s],
           label='HF sessions', color='#9b59b6', alpha=.7, capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=8, rotation=15)
    ax.set_ylabel('Sessions')
    ax.set_title('Facility Sessions')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=.3, axis='y')

    ax = axes[2]
    fr = [[max(l[-1].true_regret, 1e-4) for l in results[m]] for m in methods]
    bp = ax.boxplot(fr, labels=methods, patch_artist=True)
    for p, m in zip(bp['boxes'], methods):
        p.set_facecolor(colors[m])
        p.set_alpha(0.7)
    ax.set_ylabel('Final regret')
    ax.set_title('Final Regret')
    ax.set_yscale('log')
    ax.grid(True, alpha=.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved: " + save_path)
    plt.show()


def print_summary(results, lookup):
    print("\n" + "="*75)
    print("P3HT-CNT Benchmark -- Optimal Y = %.4f" % lookup.optimal_value)
    print("="*75)
    print("%-18s %22s %12s %12s" % ("Method", "Regret (mean+/-std)", "MF sess", "HF sess"))
    print("-" * 65)
    for m, ll in results.items():
        reg = [max(l[-1].true_regret, 0) for l in ll]
        ms  = [l[-1].n_sessions.get(0.5, 0) for l in ll]
        hs  = [l[-1].n_sessions.get(1.0, 0) for l in ll]
        print("%-18s %8.4f +/- %6.4f     %5.1f+/-%.1f   %5.1f+/-%.1f" %
              (m, np.mean(reg), np.std(reg),
               np.mean(ms), np.std(ms),
               np.mean(hs), np.std(hs)))
    print("="*75)


def print_favp_events(all_favp):
    total = sum(len(e) for e in all_favp)
    print("\nFAVP events across all seeds: %d" % total)
    for si, events in enumerate(all_favp):
        if events:
            a1  = sum(1 for e in events if e.case == "A1")
            a2  = sum(1 for e in events if e.case == "A2")
            a2d = sum(1 for e in events if e.case == "A2_dedup")
            lf  = sum(1 for e in events if e.fidelity == FIDELITY_LF)
            mf  = sum(1 for e in events if e.fidelity == FIDELITY_MF)
            esc_mf = sum(1 for e in events if e.escalated_to == FIDELITY_MF)
            esc_hf = sum(1 for e in events if e.escalated_to == FIDELITY_HF)
            print("  Seed %d: %d events (A1:%d A2:%d A2dedup:%d | LF:%d MF:%d | esc->MF:%d esc->HF:%d)" %
                  (si, len(events), a1, a2, a2d, lf, mf, esc_mf, esc_hf))
