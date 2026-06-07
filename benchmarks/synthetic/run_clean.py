"""
Clean-condition synthetic benchmark.

"""

import argparse
import os
import pickle
import time

import numpy as np

from qsmfbo import (
    StyblinskiTang2D, Branin2D, Hartmann6D,
    run_experiment, plot_results,
)


FUNCTIONS = {
    'st2d': ('Styblinski-Tang 2D', StyblinskiTang2D),
    'br2d': ('Branin 2D',          Branin2D),
    'h6d':  ('Hartmann 6D',        Hartmann6D),
}

# (lambda_overhead, lambda_marginal) pairs matching the paper's Table 1
OVERHEAD_REGIMES = [
    (25.0, 2.0),
    (15.0, 1.5),
]


def format_summary_row(method, results, fn_name, lambda_o):
    logs_list = results[method]
    final_regret = [max(l[-1].simple_regret, 0.0) for l in logs_list]
    visits = [l[-1].n_facility_visits for l in logs_list]
    return "%-8s  %-22s  ov=%-5.1f  regret = %.3f +/- %.3f   visits = %.1f +/- %.1f" % (
        method, fn_name, lambda_o,
        float(np.mean(final_regret)), float(np.std(final_regret)),
        float(np.mean(visits)), float(np.std(visits)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--function', choices=list(FUNCTIONS) + ['all'], default='all')
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--budget', type=float, default=200.0)
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fns = list(FUNCTIONS) if args.function == 'all' else [args.function]

    summary_lines = []
    for fn_key in fns:
        fn_name, fn_class = FUNCTIONS[fn_key]
        for lambda_o, lambda_m in OVERHEAD_REGIMES:
            print("\n" + "#" * 70)
            print("# %s   lambda_o = %.1f, lambda_m = %.1f" % (fn_name, lambda_o, lambda_m))
            print("#" * 70)

            t0 = time.time()
            results = run_experiment(
                fn_class,
                budget=args.budget,
                lambda_overhead=lambda_o,
                lambda_marginal=lambda_m,
                n_seeds=args.n_seeds,
                verbose=True,
            )
            print("\nElapsed: %.1f s" % (time.time() - t0))

            tag = "%s_ov%d" % (fn_key, int(lambda_o))
            pkl_path = os.path.join(args.outdir, "clean_%s.pkl" % tag)
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'function': fn_name,
                    'lambda_overhead': lambda_o,
                    'lambda_marginal': lambda_m,
                    'n_seeds': args.n_seeds,
                    'budget': args.budget,
                }, f)
            print("Saved %s" % pkl_path)

            if not args.no_plot:
                png_path = os.path.join(args.outdir, "clean_%s.png" % tag)
                try:
                    plot_results(results,
                                 title="%s (lo=%.0f)" % (fn_name, lambda_o),
                                 save_path=png_path)
                except Exception as e:
                    print("Plot failed: %s" % e)

            for method in results:
                summary_lines.append(format_summary_row(method, results, fn_name, lambda_o))

    summary_path = os.path.join(args.outdir, "clean_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Clean-condition benchmark summary\n")
        f.write("  budget = %.0f, seeds = %d\n\n" % (args.budget, args.n_seeds))
        for line in summary_lines:
            f.write(line + "\n")
    print("\n" + "=" * 70)
    print("SUMMARY  (written to %s)" % summary_path)
    print("=" * 70)
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
