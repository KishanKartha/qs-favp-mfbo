"""
Run 1: extended catastrophic-noise benchmark for the revision.

Six methods x three functions x 20 seeds, p_outlier = 0.20, budget 200.
Methods:
  1. QS-MFBO + FAVP            (full framework)
  2. QS-MFBO (no FAVP)         (scheduling only)
  3. QS-MFBO + naive rejection
  4. MF-MES                    (unaugmented baseline)
  5. Robust MFBO               (Martinez-Cantin-style, NEW)
  6. QS-MFBO + FAVP + robust   (stacked, NEW)

Checkpoints after every (function, seed): safe to interrupt and rerun,
completed cells are skipped. n_min_robust matches FAVP's N_min for
fairness.

Usage:
  python run_benchmark_extended.py                     # everything
  python run_benchmark_extended.py --function h6d      # one function
  python run_benchmark_extended.py --seed_start 10     # add seeds 10-19
"""

import argparse
import copy
import os
import pickle
import time

import numpy as np

from qsmfbo.core import (StyblinskiTang2D, Branin2D, Hartmann6D,
                         FixedCostModel, QueueDependentCostModel,
                         StandardMFMES, QueueScheduler, generate_shared_init)
from qsmfbo.favp import (OutlierInjector, QueueSchedulerFAVP,
                         QueueSchedulerNaive, compute_true_regret)
from qsmfbo.robust import RobustMFMES, QueueSchedulerFAVPRobust

FUNCTION_MAP = {
    'st2d': ('ST-2D', StyblinskiTang2D),
    'br2d': ('BR-2D', Branin2D),
    'h6d':  ('H-6D',  Hartmann6D),
}

METHODS = ['QS-MFBO + FAVP', 'QS-MFBO (no FAVP)', 'QS-MFBO + naive',
           'MF-MES', 'Robust MFBO', 'Stacked (FAVP + robust)']


def run_one_seed(fn_class, seed, args):
    """Run all six methods for one seed. Returns {method: cell_dict}."""
    fc = FixedCostModel(1.0, args.lambda_overhead, args.lambda_marginal)
    qc = lambda: QueueDependentCostModel(1.0, args.lambda_overhead,
                                         args.lambda_marginal)
    init_cost = 3 * 1.0 + args.lambda_overhead + 2 * args.lambda_marginal
    init_data = generate_shared_init(fn_class, n_cheap=3, n_expensive=2,
                                     noise_std=args.noise_std, seed=seed)
    opt = fn_class.optimal_value
    out = {}

    def cell(sched, injector, extra=None):
        log = sched.run(args.budget, verbose=False, init_cost=init_cost)
        d = {'log': log,
             'true_regret': compute_true_regret(sched.observations, injector, opt),
             'final_visits': sched.n_facility_visits,
             'n_injected': len(injector.injections),
             'timing_log': getattr(sched, 'timing_log', [])             }
        if extra:
            d.update(extra)
        return d

    # 1. QS-MFBO + FAVP
    inj = OutlierInjector(fn_class, p_outlier=args.p_outlier, seed=seed)
    s = QueueSchedulerFAVP(inj, qc(), outlier_injector=inj,
                           tau=args.tau, gamma=args.gamma, n_min=args.n_min,
                           noise_std=args.noise_std, seed=seed,
                           init_data=copy.deepcopy(init_data))
    out['QS-MFBO + FAVP'] = cell(s, inj,
        {'anomaly_events': len(s.anomaly_events)})

    # 2. QS-MFBO (no FAVP)
    inj = OutlierInjector(fn_class, p_outlier=args.p_outlier, seed=seed)
    s = QueueScheduler(inj, qc(), noise_std=args.noise_std, seed=seed,
                       init_data=copy.deepcopy(init_data))
    out['QS-MFBO (no FAVP)'] = cell(s, inj)

    # 3. QS-MFBO + naive rejection
    inj = OutlierInjector(fn_class, p_outlier=args.p_outlier, seed=seed)
    s = QueueSchedulerNaive(inj, qc(), outlier_injector=inj,
                            tau=args.tau, n_min=args.n_min,
                            noise_std=args.noise_std, seed=seed,
                            init_data=copy.deepcopy(init_data))
    out['QS-MFBO + naive'] = cell(s, inj,
        {'rejected_genuine': s.rejected_genuine,
         'rejected_outliers': s.rejected_outliers})

    # 4. MF-MES baseline
    inj = OutlierInjector(fn_class, p_outlier=args.p_outlier, seed=seed)
    s = StandardMFMES(inj, fc, noise_std=args.noise_std, seed=seed,
                      init_data=copy.deepcopy(init_data))
    out['MF-MES'] = cell(s, inj)

    # 5. Robust MFBO (NEW)
    inj = OutlierInjector(fn_class, p_outlier=args.p_outlier, seed=seed)
    s = RobustMFMES(inj, fc, noise_std=args.noise_std, seed=seed,
                    init_data=copy.deepcopy(init_data),
                    tau_r=args.tau_r, n_min_robust=args.n_min,
                    robust_verbose=False)
    out['Robust MFBO'] = cell(s, inj,
        {'exclusion_log': s.exclusion_log})

    # 6. Stacked (NEW)
    inj = OutlierInjector(fn_class, p_outlier=args.p_outlier, seed=seed)
    s = QueueSchedulerFAVPRobust(inj, qc(), outlier_injector=inj,
                                 tau=args.tau, gamma=args.gamma,
                                 n_min=args.n_min,
                                 noise_std=args.noise_std, seed=seed,
                                 init_data=copy.deepcopy(init_data),
                                 tau_r=args.tau_r, n_min_robust=args.n_min,
                                 robust_verbose=False)
    out['Stacked (FAVP + robust)'] = cell(s, inj,
        {'anomaly_events': len(s.anomaly_events),
         'exclusion_log': s.exclusion_log})

    return out


def summarise(store, fn_name):
    print("\n%s | seeds done: %d" % (fn_name, len(store)))
    print("%-26s | %-22s | %-18s | %s" % ("Method", "true regret mean+/-sd",
                                          "median (IQR)", "visits"))
    print("-" * 90)
    for m in METHODS:
        tr = [store[s][m]['true_regret'] for s in sorted(store)]
        v = [store[s][m]['final_visits'] for s in sorted(store)]
        q1, med, q3 = np.percentile(tr, [25, 50, 75])
        print("%-26s | %8.3f +/- %7.3f | %6.3f (%5.2f-%5.2f) | %.1f +/- %.1f"
              % (m, np.mean(tr), np.std(tr), med, q1, q3,
                 np.mean(v), np.std(v)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--function', default='all',
                   choices=['st2d', 'br2d', 'h6d', 'all'])
    p.add_argument('--seed_start', type=int, default=0)
    p.add_argument('--seed_end', type=int, default=20)   # exclusive
    p.add_argument('--budget', type=float, default=200.0)
    p.add_argument('--p_outlier', type=float, default=0.20)
    p.add_argument('--tau', type=float, default=3.0)
    p.add_argument('--gamma', type=float, default=1.5)
    p.add_argument('--tau_r', type=float, default=3.0)
    p.add_argument('--n_min', type=int, default=15)
    p.add_argument('--lambda_overhead', type=float, default=25.0)
    p.add_argument('--lambda_marginal', type=float, default=2.0)
    p.add_argument('--noise_std', type=float, default=0.1)
    p.add_argument('--outdir', default='extended_results')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fns = ['st2d', 'br2d', 'h6d'] if args.function == 'all' \
        else [args.function]

    for fk in fns:
        fn_name, fn_class = FUNCTION_MAP[fk]
        path = os.path.join(args.outdir, 'extended_%s.pkl' % fk)

        store = {}
        if os.path.exists(path):
            with open(path, 'rb') as f:
                store = pickle.load(f)['store']
            print("[%s] resuming: %d seeds already done" % (fn_name, len(store)))

        for seed in range(args.seed_start, args.seed_end):
            if seed in store:
                continue
            t0 = time.time()
            print("[%s] seed %d ..." % (fn_name, seed), flush=True)
            store[seed] = run_one_seed(fn_class, seed, args)
            with open(path, 'wb') as f:
                pickle.dump({'store': store, 'args': vars(args),
                             'function': fn_name}, f)
            line = " | ".join("%s: %.2f" % (m.split(' ')[0],
                              store[seed][m]['true_regret']) for m in METHODS)
            print("[%s] seed %d done in %.0fs | %s"
                  % (fn_name, seed, time.time() - t0, line), flush=True)

        summarise(store, fn_name)

    print("\nAll requested cells complete.")


if __name__ == "__main__":
    main()