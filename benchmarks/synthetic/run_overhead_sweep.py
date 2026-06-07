"""
Overhead-sensitivity sweep on Styblinski-Tang 2D.

"""

import argparse
import os
import pickle
import time

import numpy as np

from qsmfbo import StyblinskiTang2D, run_experiment


OVERHEADS = [5.0, 10.0, 15.0, 25.0, 45.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--budget', type=float, default=200.0)
    parser.add_argument('--lambda_marginal', type=float, default=2.0)
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sweep = {}
    for lo in OVERHEADS:
        print("\n" + "#" * 70)
        print("# Overhead sweep: lambda_o = %.1f" % lo)
        print("#" * 70)

        t0 = time.time()
        results = run_experiment(
            StyblinskiTang2D,
            budget=args.budget,
            lambda_overhead=lo,
            lambda_marginal=args.lambda_marginal,
            n_seeds=args.n_seeds, verbose=True,
        )
        print("\nElapsed: %.1f s" % (time.time() - t0))
        sweep[lo] = results

    pkl_path = os.path.join(args.outdir, "overhead_sweep.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({'sweep': sweep, 'settings': vars(args)}, f)
    print("\nSaved %s" % pkl_path)

    # Build summary table (MFBO vs QS-MFBO final regret and sessions)
    rows = []
    for lo in OVERHEADS:
        row = {'lambda_o': lo}
        for method in ('MF-MES', 'QS-MFBO'):
            logs_list = sweep[lo][method]
            fr = [max(l[-1].simple_regret, 0.0) for l in logs_list]
            vis = [l[-1].n_facility_visits for l in logs_list]
            row[method] = (np.mean(fr), np.std(fr), np.mean(vis), np.std(vis))
        rows.append(row)

    # Write text table
    summary_path = os.path.join(args.outdir, "overhead_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Overhead sensitivity on Styblinski-Tang 2D (lambda_m=%.1f, budget=%.0f, %d seeds)\n\n"
                % (args.lambda_marginal, args.budget, args.n_seeds))
        f.write("%-8s   %-24s   %-24s\n" % ("lambda_o", "MFBO (regret / sessions)", "QS-MFBO (regret / sessions)"))
        f.write("-" * 68 + "\n")
        for r in rows:
            mr, ms, mvm, mvs = r['MF-MES']
            qr, qs_, qvm, qvs = r['QS-MFBO']
            f.write("%-8.1f   %6.3f+-%5.3f  %4.1f+-%3.1f     %6.3f+-%5.3f  %4.1f+-%3.1f\n" % (
                r['lambda_o'], mr, ms, mvm, mvs, qr, qs_, qvm, qvs))
    print("\n" + "=" * 68)
    print("SUMMARY  (written to %s)" % summary_path)
    print("=" * 68)
    with open(summary_path) as f:
        print(f.read())

    # Plot: final regret vs lambda_o, MFBO vs QS-MFBO
    if args.no_plot:
        return
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        los = np.array([r['lambda_o'] for r in rows])
        mfbo_m = np.array([r['MF-MES'][0] for r in rows])
        mfbo_s = np.array([r['MF-MES'][1] for r in rows])
        qs_m = np.array([r['QS-MFBO'][0] for r in rows])
        qs_s = np.array([r['QS-MFBO'][1] for r in rows])
        ax.errorbar(los, mfbo_m, yerr=mfbo_s, marker='s', mfc='white', mec='#e74c3c',
                    ecolor='#e74c3c', label='MFBO (baseline)', capsize=3)
        ax.errorbar(los, qs_m, yerr=qs_s, marker='o', color='#1f3a63',
                    label='QS-MFBO (full)', capsize=3)
        ax.axvline(25, color='k', linestyle='--', linewidth=0.6, alpha=0.5)
        ax.text(25, ax.get_ylim()[1] * 0.92, ' operating\n point',
                fontsize=8, alpha=0.6)
        ax.set_xlabel(r'Session overhead $\lambda_o$')
        ax.set_ylabel('Final simple regret')
        ax.legend(frameon=False)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        plt.tight_layout()
        png_path = os.path.join(args.outdir, "overhead_sweep.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print("Saved %s" % png_path)
    except Exception as e:
        print("Plot failed: %s" % e)


if __name__ == "__main__":
    main()
