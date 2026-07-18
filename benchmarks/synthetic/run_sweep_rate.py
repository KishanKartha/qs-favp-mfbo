"""
Run 2: outlier-rate sweep on Hartmann-6D.

p_outlier in {0.05, 0.10, 0.20, 0.30}, four methods, 10 seeds each:
  QS-MFBO + FAVP, QS-MFBO (no FAVP), Robust MFBO, MF-MES.

Checkpointed per (rate, seed). Resumable: rerun the same command and
completed cells are skipped.

Usage:
  python run_sweep_rate.py
  python run_sweep_rate.py --seed_end 10 --rates 0.05 0.10 0.20 0.30
"""

import argparse
import copy
import os
import pickle
import time

import numpy as np

from qsmfbo.core import (Hartmann6D, FixedCostModel, QueueDependentCostModel,
                         StandardMFMES, QueueScheduler, generate_shared_init)
from qsmfbo.favp import (OutlierInjector, QueueSchedulerFAVP,
                         compute_true_regret)
from qsmfbo.robust import RobustMFMES

METHODS = ['QS-MFBO + FAVP', 'QS-MFBO (no FAVP)', 'Robust MFBO', 'MF-MES']


def run_one_cell(seed, p_outlier, args):
    """All four methods for one (seed, p_outlier). Returns {method: dict}."""
    fn_class = Hartmann6D
    fc = FixedCostModel(1.0, args.lambda_overhead, args.lambda_marginal)
    qc = lambda: QueueDependentCostModel(1.0, args.lambda_overhead,
                                         args.lambda_marginal)
    init_cost = 3 * 1.0 + args.lambda_overhead + 2 * args.lambda_marginal
    init_data = generate_shared_init(fn_class, n_cheap=3, n_expensive=2,
                                     noise_std=args.noise_std, seed=seed)
    opt = fn_class.optimal_value
    out = {}

    def cell(sched, injector, extra_fn=None):
        log = sched.run(args.budget, verbose=False, init_cost=init_cost)
        d = {'log': log,
             'true_regret': compute_true_regret(sched.observations, injector, opt),
             'final_visits': sched.n_facility_visits,
             'n_injected': len(injector.injections),
             'timing_log': getattr(sched, 'timing_log', [])}
        if extra_fn:
            d.update(extra_fn())
        return d

    # 1. QS-MFBO + FAVP
    inj = OutlierInjector(fn_class, p_outlier=p_outlier, seed=seed)
    s = QueueSchedulerFAVP(inj, qc(), outlier_injector=inj,
                           tau=args.tau, gamma=args.gamma, n_min=args.n_min,
                           noise_std=args.noise_std, seed=seed,
                           init_data=copy.deepcopy(init_data))
    out['QS-MFBO + FAVP'] = cell(s, inj)

    # 2. QS-MFBO (no FAVP)
    inj = OutlierInjector(fn_class, p_outlier=p_outlier, seed=seed)
    s = QueueScheduler(inj, qc(), noise_std=args.noise_std, seed=seed,
                       init_data=copy.deepcopy(init_data))
    out['QS-MFBO (no FAVP)'] = cell(s, inj)

    # 3. Robust MFBO
    inj = OutlierInjector(fn_class, p_outlier=p_outlier, seed=seed)
    s = RobustMFMES(inj, fc, noise_std=args.noise_std, seed=seed,
                    init_data=copy.deepcopy(init_data),
                    tau_r=args.tau_r, n_min_robust=args.n_min,
                    robust_verbose=False)
    out['Robust MFBO'] = cell(s, inj, lambda: {'exclusion_log': s.exclusion_log})

    # 4. MF-MES baseline
    inj = OutlierInjector(fn_class, p_outlier=p_outlier, seed=seed)
    s = StandardMFMES(inj, fc, noise_std=args.noise_std, seed=seed,
                      init_data=copy.deepcopy(init_data))
    out['MF-MES'] = cell(s, inj)

    return out


def summarise(store):
    print("\n%-8s | %-18s | mean true regret by method" % ("p", "seeds"))
    print("-" * 90)
    for p in sorted(store):
        seeds = sorted(store[p])
        line = " | ".join(
            "%s: %.3f" % (m.split(' ')[0], np.mean(
                [store[p][s][m]['true_regret'] for s in seeds]))
            for m in METHODS)
        print("p=%-6.2f | n=%-16d | %s" % (p, len(seeds), line))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rates', type=float, nargs='+',
                   default=[0.05, 0.10, 0.20, 0.30])
    p.add_argument('--seed_start', type=int, default=0)
    p.add_argument('--seed_end', type=int, default=10)   # exclusive
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
    path = os.path.join(args.outdir, 'sweep_rate_h6d.pkl')

    store = {}   # store[p_outlier][seed] = {method: dict}
    if os.path.exists(path):
        with open(path, 'rb') as f:
            store = pickle.load(f)['store']
        done = sum(len(v) for v in store.values())
        print("Resuming: %d cells already done" % done)

    for p_outlier in args.rates:
        store.setdefault(p_outlier, {})
        for seed in range(args.seed_start, args.seed_end):
            if seed in store[p_outlier]:
                continue
            t0 = time.time()
            print("[p=%.2f seed %d] ..." % (p_outlier, seed), flush=True)
            store[p_outlier][seed] = run_one_cell(seed, p_outlier, args)
            with open(path, 'wb') as f:
                pickle.dump({'store': store, 'args': vars(args)}, f)
            line = " | ".join(
                "%s: %.2f" % (m.split(' ')[0],
                              store[p_outlier][seed][m]['true_regret'])
                for m in METHODS)
            print("[p=%.2f seed %d] done in %.0fs | %s"
                  % (p_outlier, seed, time.time() - t0, line), flush=True)

    summarise(store)
    print("\nSweep complete. Pickle at %s" % path)


if __name__ == "__main__":
    main()