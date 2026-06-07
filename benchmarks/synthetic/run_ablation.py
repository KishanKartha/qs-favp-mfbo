"""
Ablation of QS-MFBO components on Styblinski-Tang 2D.

"""

import argparse
import os
import pickle
import time

import numpy as np

from qsmfbo import StyblinskiTang2D, run_ablation


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--budget', type=float, default=200.0)
    parser.add_argument('--lambda_overhead', type=float, default=25.0)
    parser.add_argument('--lambda_marginal', type=float, default=2.0)
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("#" * 70)
    print("# Ablation on Styblinski-Tang 2D")
    print("# lambda_o=%.1f, lambda_m=%.1f, budget=%.0f, seeds=%d" % (
        args.lambda_overhead, args.lambda_marginal, args.budget, args.n_seeds))
    print("#" * 70)

    t0 = time.time()
    results = run_ablation(
        StyblinskiTang2D,
        budget=args.budget,
        lambda_overhead=args.lambda_overhead,
        lambda_marginal=args.lambda_marginal,
        n_seeds=args.n_seeds, verbose=True,
    )
    print("\nElapsed: %.1f s" % (time.time() - t0))

    pkl_path = os.path.join(args.outdir, "ablation_st2d.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({'results': results, 'settings': vars(args)}, f)
    print("Saved %s" % pkl_path)

    # Summary table: final regret and expensive-session count, mean +/- s.d.
    rows = []
    for method, logs_list in results.items():
        fr = [max(l[-1].simple_regret, 0.0) for l in logs_list]
        visits = [l[-1].n_facility_visits for l in logs_list]
        rows.append((method, np.mean(fr), np.std(fr), np.mean(visits), np.std(visits)))

    summary_path = os.path.join(args.outdir, "ablation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Ablation on Styblinski-Tang 2D\n")
        f.write("  lambda_o=%.1f, lambda_m=%.1f, budget=%.0f, seeds=%d\n\n" % (
            args.lambda_overhead, args.lambda_marginal, args.budget, args.n_seeds))
        f.write("%-22s  %-20s  %-20s\n" % ("Method", "Final regret", "Expensive sessions"))
        f.write("-" * 66 + "\n")
        for method, rm, rs, vm, vs in rows:
            f.write("%-22s  %6.3f +/- %5.3f       %4.1f +/- %3.1f\n" %
                    (method, rm, rs, vm, vs))

    print("\n" + "=" * 66)
    print("SUMMARY  (written to %s)" % summary_path)
    print("=" * 66)
    print("%-22s  %-20s  %-20s" % ("Method", "Final regret", "Expensive sessions"))
    print("-" * 66)
    for method, rm, rs, vm, vs in rows:
        print("%-22s  %6.3f +/- %5.3f       %4.1f +/- %3.1f" %
              (method, rm, rs, vm, vs))


if __name__ == "__main__":
    main()
