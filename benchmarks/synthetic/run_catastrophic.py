"""
Catastrophic-noise synthetic benchmark (20% outlier injection).

"""

import argparse
import os
import pickle
import time

import numpy as np

from qsmfbo import (
    StyblinskiTang2D, Branin2D, Hartmann6D,
    run_favp_benchmark,
)
from qsmfbo.favp import plot_favp_results, print_summary


FUNCTIONS = {
    'st2d': ('Styblinski-Tang 2D', StyblinskiTang2D),
    'br2d': ('Branin 2D',          Branin2D),
    'h6d':  ('Hartmann 6D',        Hartmann6D),
}


def format_row(method, results, diagnostics, fn_name):
    diags = diagnostics[method]
    logs_list = results[method]
    true_regrets = [d['true_regret'] for d in diags]
    visits = [l[-1].n_facility_visits for l in logs_list]
    return "%-22s  %-20s  regret = %.3f +/- %.3f   visits = %.1f +/- %.1f" % (
        method, fn_name,
        float(np.mean(true_regrets)), float(np.std(true_regrets)),
        float(np.mean(visits)), float(np.std(visits)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--function', choices=list(FUNCTIONS) + ['all'], default='all')
    parser.add_argument('--budget', type=float, default=200.0)
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--p_outlier', type=float, default=0.20)
    parser.add_argument('--tau', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument('--n_min', type=int, default=15)
    parser.add_argument('--lambda_overhead', type=float, default=25.0)
    parser.add_argument('--lambda_marginal', type=float, default=2.0)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fns = list(FUNCTIONS) if args.function == 'all' else [args.function]

    all_rows = []
    for fn_key in fns:
        fn_name, fn_class = FUNCTIONS[fn_key]
        print("\n" + "#" * 70)
        print("# %s" % fn_name)
        print("# budget=%.0f, seeds=%d, p_outlier=%.2f, tau=%.1f, gamma=%.1f" % (
            args.budget, args.n_seeds, args.p_outlier, args.tau, args.gamma))
        print("#" * 70)

        t0 = time.time()
        results, diagnostics = run_favp_benchmark(
            fn_class,
            budget=args.budget,
            lambda_overhead=args.lambda_overhead,
            lambda_marginal=args.lambda_marginal,
            noise_std=args.noise_std,
            p_outlier=args.p_outlier,
            tau=args.tau, gamma=args.gamma, n_min=args.n_min,
            n_seeds=args.n_seeds, verbose=True,
        )
        print("\nElapsed: %.1f s" % (time.time() - t0))

        print_summary(results, diagnostics, fn_name)

        pkl_path = os.path.join(args.outdir, "favp_%s.pkl" % fn_key)
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'results': results,
                'diagnostics': diagnostics,
                'function': fn_name,
                'settings': vars(args),
            }, f)
        print("Saved %s" % pkl_path)

        if not args.no_plot:
            png_path = os.path.join(args.outdir, "favp_%s.png" % fn_key)
            try:
                plot_favp_results(
                    results, diagnostics,
                    title="%s (p=%.0f%%)" % (fn_name, args.p_outlier * 100),
                    save_path=png_path,
                )
            except Exception as e:
                print("Plot failed: %s" % e)

        for method in results:
            all_rows.append(format_row(method, results, diagnostics, fn_name))

    summary_path = os.path.join(args.outdir, "favp_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Catastrophic-noise benchmark summary (TRUE regret on clean eval)\n")
        f.write("  budget=%.0f, seeds=%d, p_outlier=%.2f, tau=%.2f, gamma=%.2f, N_min=%d\n\n" % (
            args.budget, args.n_seeds, args.p_outlier, args.tau, args.gamma, args.n_min))
        for line in all_rows:
            f.write(line + "\n")
    print("\n" + "=" * 70)
    print("SUMMARY  (written to %s)" % summary_path)
    print("=" * 70)
    for line in all_rows:
        print(line)


if __name__ == "__main__":
    main()
