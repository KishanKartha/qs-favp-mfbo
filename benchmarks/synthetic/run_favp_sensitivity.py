"""
FAVP threshold sensitivity sweep.

"""

import os
import copy
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from qsmfbo import (
    StyblinskiTang2D, Hartmann6D,
    QueueDependentCostModel,
    generate_shared_init,
)
from qsmfbo.favp import (
    OutlierInjector, QueueSchedulerFAVP, compute_true_regret,
)


# =============================================================================
# Configuration
# =============================================================================

TAU_GRID = [2.0, 2.5, 3.0, 3.5, 4.0]
GAMMA_GRID = [1.0, 1.5, 2.0, 2.5]
N_MIN_FIXED = 15

OPERATING_POINT = (3.0, 1.5)

FUNCTIONS = {
    "ST-2D": StyblinskiTang2D,
    "H-6D":  Hartmann6D,
}

# Paper settings
BUDGET = 200.0
LAMBDA_CHEAP = 1.0
LAMBDA_OVERHEAD = 25.0
LAMBDA_MARGINAL = 2.0
NOISE_STD = 0.1
P_OUTLIER = 0.20
N_INIT_CHEAP = 3
N_INIT_EXPENSIVE = 2


# =============================================================================
# Single-cell runner
# =============================================================================

def run_single_cell(test_function_class, tau, gamma, n_min, seed, verbose=False):
    """Run one FAVP seed at given (tau, gamma). Returns regret and event counts."""
    qc = QueueDependentCostModel(LAMBDA_CHEAP, LAMBDA_OVERHEAD, LAMBDA_MARGINAL)
    init_cost = (N_INIT_CHEAP * LAMBDA_CHEAP
                 + LAMBDA_OVERHEAD
                 + N_INIT_EXPENSIVE * LAMBDA_MARGINAL)

    init_data = generate_shared_init(
        test_function_class,
        n_cheap=N_INIT_CHEAP, n_expensive=N_INIT_EXPENSIVE,
        noise_std=NOISE_STD, seed=seed,
    )

    injector = OutlierInjector(test_function_class, p_outlier=P_OUTLIER, seed=seed)
    sched = QueueSchedulerFAVP(
        injector, qc, outlier_injector=injector,
        tau=tau, gamma=gamma, n_min=n_min,
        noise_std=NOISE_STD, seed=seed,
        init_data=copy.deepcopy(init_data),
    )
    sched.run(BUDGET, verbose=verbose, init_cost=init_cost)
    regret = compute_true_regret(
        sched.observations, injector, test_function_class.optimal_value,
    )

    # Partition AnomalyEvent records into Case A1 vs A2
    n_a1 = 0
    n_a2 = 0
    for evt in sched.anomaly_events:
        case = getattr(evt, "case", "") or ""
        case_str = str(case).upper()
        if "A1" in case_str or "CONFIRM" in case_str or "GENUINE" in case_str:
            n_a1 += 1
        elif "A2" in case_str or "ESCALAT" in case_str:
            n_a2 += 1

    return {
        "regret": float(regret),
        "n_a1": n_a1,
        "n_a2": n_a2,
        "n_total_events": len(sched.anomaly_events),
        "n_injected": len(injector.injections),
    }


# =============================================================================
# Full sweep
# =============================================================================

def run_sweep(n_seeds, out_dir, verbose=False):
    """Run the full tau x gamma grid across the two functions."""
    os.makedirs(out_dir, exist_ok=True)

    results = {
        fname: {
            "tau": np.array(TAU_GRID),
            "gamma": np.array(GAMMA_GRID),
            "regret": np.zeros((len(TAU_GRID), len(GAMMA_GRID), n_seeds)),
            "n_a1":   np.zeros((len(TAU_GRID), len(GAMMA_GRID), n_seeds), dtype=int),
            "n_a2":   np.zeros((len(TAU_GRID), len(GAMMA_GRID), n_seeds), dtype=int),
        }
        for fname in FUNCTIONS
    }

    total_cells = len(FUNCTIONS) * len(TAU_GRID) * len(GAMMA_GRID) * n_seeds
    counter = 0

    for fname, fclass in FUNCTIONS.items():
        print("\n" + "=" * 60)
        print("Function: %s" % fname)
        print("=" * 60)
        for i, tau in enumerate(TAU_GRID):
            for j, gamma in enumerate(GAMMA_GRID):
                for s in range(n_seeds):
                    counter += 1
                    print("  [%d/%d] %s tau=%.1f gamma=%.1f seed=%d ... "
                          % (counter, total_cells, fname, tau, gamma, s),
                          end="", flush=True)
                    try:
                        out = run_single_cell(fclass, tau, gamma, N_MIN_FIXED, s,
                                              verbose=verbose)
                        results[fname]["regret"][i, j, s] = out["regret"]
                        results[fname]["n_a1"][i, j, s] = out["n_a1"]
                        results[fname]["n_a2"][i, j, s] = out["n_a2"]
                        print("regret=%.4f  A1=%d A2=%d"
                              % (out["regret"], out["n_a1"], out["n_a2"]))
                    except Exception as e:
                        print("FAILED: %s" % e)
                        results[fname]["regret"][i, j, s] = np.nan
                        results[fname]["n_a1"][i, j, s] = -1
                        results[fname]["n_a2"][i, j, s] = -1

    np.savez(
        os.path.join(out_dir, "results_sensitivity.npz"),
        **{f"{fname}_{k}": v for fname, d in results.items() for k, v in d.items()},
    )

    # Per-seed CSV for Source Data
    csv_path = os.path.join(out_dir, "per_seed_regret.csv")
    with open(csv_path, "w") as f:
        f.write("function,tau,gamma,seed,regret,n_a1,n_a2\n")
        for fname, d in results.items():
            for i, tau in enumerate(TAU_GRID):
                for j, gamma in enumerate(GAMMA_GRID):
                    for s in range(n_seeds):
                        f.write("%s,%.2f,%.2f,%d,%.6f,%d,%d\n" % (
                            fname, tau, gamma, s,
                            d["regret"][i, j, s],
                            d["n_a1"][i, j, s],
                            d["n_a2"][i, j, s],
                        ))
    print("\nSaved raw arrays to %s" % out_dir)
    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_regret_heatmaps(results, out_dir):
    """Mean final true regret across the tau x gamma grid."""
    fig, axes = plt.subplots(1, len(FUNCTIONS), figsize=(5.5 * len(FUNCTIONS), 4.2))
    if len(FUNCTIONS) == 1:
        axes = [axes]

    for ax, (fname, d) in zip(axes, results.items()):
        mean_regret = np.nanmean(d["regret"], axis=2)       # (tau, gamma)
        plot_arr = mean_regret.T                             # gamma on y, tau on x

        pos_vals = plot_arr[plot_arr > 0]
        vmin = max(1e-3, np.nanmin(pos_vals)) if pos_vals.size else 1e-3
        vmax = np.nanmax(plot_arr)

        im = ax.imshow(
            plot_arr, origin="lower", aspect="auto", cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extent=[-0.5, len(TAU_GRID) - 0.5, -0.5, len(GAMMA_GRID) - 0.5],
        )
        ax.set_xticks(range(len(TAU_GRID)))
        ax.set_xticklabels(["%.1f" % t for t in TAU_GRID])
        ax.set_yticks(range(len(GAMMA_GRID)))
        ax.set_yticklabels(["%.1f" % g for g in GAMMA_GRID])
        ax.set_xlabel(r"$\tau$ (anomaly threshold)")
        ax.set_ylabel(r"$\gamma$ (consistency multiplier)")
        ax.set_title("%s: mean final true regret" % fname)

        # Annotate cells
        median = np.nanmedian(mean_regret)
        for i in range(len(TAU_GRID)):
            for j in range(len(GAMMA_GRID)):
                val = mean_regret[i, j]
                ax.text(i, j, "%.2f" % val,
                        ha="center", va="center",
                        color="white" if val > median else "black",
                        fontsize=8)

        # Mark operating point
        op_i = TAU_GRID.index(OPERATING_POINT[0])
        op_j = GAMMA_GRID.index(OPERATING_POINT[1])
        ax.plot(op_i, op_j, marker="*", color="red",
                markersize=18, markeredgecolor="white", markeredgewidth=1.2)

        plt.colorbar(im, ax=ax, label="final true regret (log scale)")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "heatmap_regret.pdf")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved regret heatmap to %s" % out_path)


def plot_event_heatmaps(results, out_dir):
    """A1 and A2 event counts across the tau x gamma grid."""
    fig, axes = plt.subplots(2, len(FUNCTIONS), figsize=(5.5 * len(FUNCTIONS), 8))
    if len(FUNCTIONS) == 1:
        axes = axes.reshape(2, 1)

    for col, (fname, d) in enumerate(results.items()):
        for row, (field, title) in enumerate([
            ("n_a1", "%s: mean Case A1 events" % fname),
            ("n_a2", "%s: mean Case A2 events" % fname),
        ]):
            ax = axes[row, col]
            mean_events = np.nanmean(d[field], axis=2).T   # (gamma, tau)
            im = ax.imshow(mean_events, origin="lower", aspect="auto", cmap="magma")
            ax.set_xticks(range(len(TAU_GRID)))
            ax.set_xticklabels(["%.1f" % t for t in TAU_GRID])
            ax.set_yticks(range(len(GAMMA_GRID)))
            ax.set_yticklabels(["%.1f" % g for g in GAMMA_GRID])
            ax.set_xlabel(r"$\tau$")
            ax.set_ylabel(r"$\gamma$")
            ax.set_title(title)

            median = np.nanmedian(mean_events)
            for i in range(len(TAU_GRID)):
                for j in range(len(GAMMA_GRID)):
                    val = mean_events.T[i, j]
                    ax.text(i, j, "%.1f" % val,
                            ha="center", va="center",
                            color="white" if val > median else "black",
                            fontsize=8)

            op_i = TAU_GRID.index(OPERATING_POINT[0])
            op_j = GAMMA_GRID.index(OPERATING_POINT[1])
            ax.plot(op_i, op_j, marker="*", color="cyan",
                    markersize=16, markeredgecolor="white", markeredgewidth=1.2)
            plt.colorbar(im, ax=ax, label="mean count")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "heatmap_events.pdf")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved event heatmap to %s" % out_path)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="results/sensitivity")
    parser.add_argument("--verbose", action="store_true",
                        help="Per-iteration scheduler log inside each run.")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip sweep, reload npz and regenerate plots.")
    args = parser.parse_args()

    if args.plot_only:
        data = np.load(os.path.join(args.out_dir, "results_sensitivity.npz"))
        results = {}
        for fname in FUNCTIONS:
            results[fname] = {
                "tau":    data[f"{fname}_tau"],
                "gamma":  data[f"{fname}_gamma"],
                "regret": data[f"{fname}_regret"],
                "n_a1":   data[f"{fname}_n_a1"],
                "n_a2":   data[f"{fname}_n_a2"],
            }
    else:
        results = run_sweep(args.n_seeds, args.out_dir, verbose=args.verbose)

    plot_regret_heatmaps(results, args.out_dir)
    plot_event_heatmaps(results, args.out_dir)

    # Print operating-point and corner summaries
    print("\n" + "=" * 70)
    print("SUMMARY: final true regret (mean +/- s.d.)")
    print("=" * 70)
    for fname, d in results.items():
        op_i = TAU_GRID.index(OPERATING_POINT[0])
        op_j = GAMMA_GRID.index(OPERATING_POINT[1])
        op_mean = np.nanmean(d["regret"][op_i, op_j])
        op_std = np.nanstd(d["regret"][op_i, op_j])
        print("\n%s:" % fname)
        print("  Operating point (tau=%.1f, gamma=%.1f): %.3f +/- %.3f"
              % (OPERATING_POINT[0], OPERATING_POINT[1], op_mean, op_std))
        for i_name, i in [("min-tau", 0), ("max-tau", len(TAU_GRID) - 1)]:
            for j_name, j in [("min-gamma", 0), ("max-gamma", len(GAMMA_GRID) - 1)]:
                m = np.nanmean(d["regret"][i, j])
                s = np.nanstd(d["regret"][i, j])
                print("  Corner (tau=%.1f, gamma=%.1f) [%s, %s]: %.3f +/- %.3f"
                      % (TAU_GRID[i], GAMMA_GRID[j], i_name, j_name, m, s))


if __name__ == "__main__":
    main()
