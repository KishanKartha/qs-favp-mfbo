"""
Run 3: heavy-tailed observation noise on Hartmann-6D.

Checkpointed per seed. Resumable.

Usage:
  python run_heavytail.py
  python run_heavytail.py --nu 3 --scale 0.1 --seed_end 10
"""

import argparse
import copy
import os
import pickle
import time

import numpy as np

from qsmfbo.core import (Hartmann6D, FixedCostModel, QueueDependentCostModel,
                         StandardMFMES, QueueScheduler, generate_shared_init,
                         FIDELITY_CHEAP, FIDELITY_EXPENSIVE)
from qsmfbo.favp import (QueueSchedulerFAVP, compute_true_regret)
from qsmfbo.robust import RobustMFMES, QueueSchedulerFAVPRobust

METHODS = ['QS-MFBO + FAVP', 'QS-MFBO (no FAVP)', 'Robust MFBO', 'MF-MES',
           'Stacked (FAVP + robust)']


class HeavyTailNoise:
    """Wraps a test function so cheap-fidelity evaluations carry
    Student-t(nu) noise (scaled) in place of Gaussian noise. The expensive
    fidelity remains uncorrupted ground truth. No catastrophic injection.

    Exposes the same interface the schedulers and compute_true_regret use:
      evaluate, evaluate_clean, get_true_function, dim, n_fidelities,
      optimal_value, and an .injections list (kept empty; present so the
      FAVP/robust diagnostic hooks that check it do not fail).
    """
    def __init__(self, test_function, nu=3.0, scale=0.1, seed=0):
        self.test_fn = test_function
        self.nu = nu
        self.scale = scale
        self.dim = test_function.dim
        self.n_fidelities = test_function.n_fidelities
        self.optimal_value = test_function.optimal_value
        self.rng = np.random.RandomState(seed + 4242)
        self.injections = []   # kept empty; heavy-tail noise is not "injected"
        self.total_evals = 0

    def evaluate(self, x_norm, fidelity, noise_std=0.0):
        # noise_std from the caller is ignored at the cheap fidelity; we
        # substitute Student-t noise. Expensive fidelity is clean ground truth.
        self.total_evals += 1
        y_true = self.test_fn.evaluate(x_norm, fidelity, noise_std=0.0)
        if fidelity < 0.5:
            y_true += self.scale * self.rng.standard_t(df=self.nu)
        return y_true

    def evaluate_clean(self, x_norm, fidelity, noise_std=0.0):
        # Used by FAVP's repeat step: return a fresh heavy-tailed draw at
        # cheap fidelity (a real repeat is itself noisy), clean at expensive.
        self.total_evals += 1
        y_true = self.test_fn.evaluate(x_norm, fidelity, noise_std=0.0)
        if fidelity < 0.5:
            y_true += self.scale * self.rng.standard_t(df=self.nu)
        return y_true

    def get_true_function(self):
        return self.test_fn


def run_one_seed(seed, args):
    fn_class = Hartmann6D
    fc = FixedCostModel(1.0, args.lambda_overhead, args.lambda_marginal)
    qc = lambda: QueueDependentCostModel(1.0, args.lambda_overhead,
                                         args.lambda_marginal)
    init_cost = 3 * 1.0 + args.lambda_overhead + 2 * args.lambda_marginal
    init_data = generate_shared_init(fn_class, n_cheap=3, n_expensive=2,
                                     noise_std=args.noise_std, seed=seed)
    opt = fn_class.optimal_value
    out = {}

    def wrapped():
        return HeavyTailNoise(fn_class, nu=args.nu, scale=args.scale, seed=seed)

    def cell(sched, injector, extra_fn=None):
        log = sched.run(args.budget, verbose=False, init_cost=init_cost)
        d = {'log': log,
             'true_regret': compute_true_regret(sched.observations, injector, opt),
             'final_visits': sched.n_facility_visits,
             'timing_log': getattr(sched, 'timing_log', [])}
        if extra_fn:
            d.update(extra_fn())
        return d

    # 1. QS-MFBO + FAVP
    w = wrapped()
    s = QueueSchedulerFAVP(w, qc(), outlier_injector=w,
                           tau=args.tau, gamma=args.gamma, n_min=args.n_min,
                           noise_std=args.noise_std, seed=seed,
                           init_data=copy.deepcopy(init_data))
    out['QS-MFBO + FAVP'] = cell(s, w,
        lambda: {'anomaly_events': len(s.anomaly_events)})

    # 2. QS-MFBO (no FAVP)
    w = wrapped()
    s = QueueScheduler(w, qc(), noise_std=args.noise_std, seed=seed,
                       init_data=copy.deepcopy(init_data))
    out['QS-MFBO (no FAVP)'] = cell(s, w)

    # 3. Robust MFBO
    w = wrapped()
    s = RobustMFMES(w, fc, noise_std=args.noise_std, seed=seed,
                    init_data=copy.deepcopy(init_data),
                    tau_r=args.tau_r, n_min_robust=args.n_min,
                    robust_verbose=False)
    out['Robust MFBO'] = cell(s, w, lambda: {'exclusion_log': s.exclusion_log})

    # 4. MF-MES baseline
    w = wrapped()
    s = StandardMFMES(w, fc, noise_std=args.noise_std, seed=seed,
                      init_data=copy.deepcopy(init_data))
    out['MF-MES'] = cell(s, w)

    # 5. Stacked
    w = wrapped()
    s = QueueSchedulerFAVPRobust(w, qc(), outlier_injector=w,
                                 tau=args.tau, gamma=args.gamma,
                                 n_min=args.n_min,
                                 noise_std=args.noise_std, seed=seed,
                                 init_data=copy.deepcopy(init_data),
                                 tau_r=args.tau_r, n_min_robust=args.n_min,
                                 robust_verbose=False)
    out['Stacked (FAVP + robust)'] = cell(s, w,
        lambda: {'anomaly_events': len(s.anomaly_events),
                 'exclusion_log': s.exclusion_log})

    return out


def summarise(store):
    seeds = sorted(store)
    print("\n%-26s | mean +/- sd | median" % "Method")
    print("-" * 70)
    for m in METHODS:
        tr = np.array([store[s][m]['true_regret'] for s in seeds])
        print("%-26s | %.3f +/- %.3f | %.3f"
              % (m, tr.mean(), tr.std(), np.median(tr)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nu', type=float, default=3.0)
    p.add_argument('--scale', type=float, default=0.1)
    p.add_argument('--seed_start', type=int, default=0)
    p.add_argument('--seed_end', type=int, default=10)
    p.add_argument('--budget', type=float, default=200.0)
    p.add_argument('--tau', type=float, default=3.0)
    p.add_argument('--gamma', type=float, default=1.5)
    p.add_argument('--tau_r', type=float, default=3.0)
    p.add_argument('--n_min', type=int, default=15)
    p.add_argument('--lambda_overhead', type=float, default=25.0)
    p.add_argument('--lambda_marginal', type=float, default=2.0)
    p.add_argument('--noise_std', type=float, default=0.1)
    p.add_argument('--outdir', default='sweep_results')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, 'heavytail_h6d.pkl')

    store = {}
    if os.path.exists(path):
        with open(path, 'rb') as f:
            store = pickle.load(f)['store']
        print("Resuming: %d seeds already done" % len(store))

    for seed in range(args.seed_start, args.seed_end):
        if seed in store:
            continue
        t0 = time.time()
        print("[seed %d] ..." % seed, flush=True)
        store[seed] = run_one_seed(seed, args)
        with open(path, 'wb') as f:
            pickle.dump({'store': store, 'args': vars(args)}, f)
        line = " | ".join("%s: %.2f" % (m.split(' ')[0],
                          store[seed][m]['true_regret']) for m in METHODS)
        print("[seed %d] done in %.0fs | %s"
              % (seed, time.time() - t0, line), flush=True)

    summarise(store)
    print("\nHeavy-tail run complete. Pickle at %s" % path)


if __name__ == "__main__":
    main()