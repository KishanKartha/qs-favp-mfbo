"""
Runner for the P3HT-CNT three-fidelity benchmark.

Reproduces:
  - Fig. 3 (main text, Section 2.2)
  - Extended Data Fig. 2 (per-seed convergence trajectories)
  - Extended Data Fig. 3 (FAVP event timeline across the campaign)

Before running, place the Bash et al. dataset file (`data_all_back_A.xlsx`)
next to this script. See README.md in this folder for where to obtain it.

Usage (from this folder):
    python run_benchmark.py

To capture the log in real time (Windows PowerShell):
    python -u run_benchmark.py 2>&1 | Tee-Object -FilePath run_log.txt

To capture the log in real time (Linux / macOS / plain Windows):
    python -u run_benchmark.py > run_log.txt 2>&1

Ten seeds at budget = 200 takes roughly 4 to 7 hours on a single NVIDIA
Quadro RTX 5000. CPU-only runs are substantially longer.
"""

import pickle
import time

import p3ht_cnt_benchmark as bench
from p3ht_cnt_benchmark import (
    run_p3ht_benchmark,
    print_summary,
    print_favp_events,
    plot_p3ht_results,
)

# ++++++++++++++++++++++++++
# Configuration (paper defaults)
# ++++++++++++++++++++++++++

FILEPATH        = "data_all_back_A.xlsx"
BUDGET          = 200
N_SEEDS         = 10
NUM_FANTASIES   = 16

CHECKPOINT_FILE = "p3ht_checkpoint.pkl"
RESULTS_FILE    = "p3ht_results.pkl"
PLOT_FILE       = "p3ht_results.png"

bench.NUM_FANTASIES = NUM_FANTASIES

# ++++++++++++++++++++++++++
# Run
# ++++++++++++++++++++++++++

print("=" * 70)
print("P3HT-CNT three-fidelity benchmark (QS-MFBO + FAVP)")
print("=" * 70)
print("Config:")
print("  Data file      : %s" % FILEPATH)
print("  Budget         : %s" % BUDGET)
print("  Seeds          : %s" % N_SEEDS)
print("  NUM_FANTASIES  : %s" % NUM_FANTASIES)
print("  Checkpoint     : %s" % CHECKPOINT_FILE)
print("=" * 70)

t_start = time.time()

results, favp, lookup = run_p3ht_benchmark(
    FILEPATH,
    budget=BUDGET,
    n_seeds=N_SEEDS,
    verbose=True,
    save_checkpoint=CHECKPOINT_FILE,
)

elapsed = time.time() - t_start
print("\n" + "=" * 70)
print("TOTAL RUN TIME: %.1f hours (%.0f seconds)" % (elapsed / 3600, elapsed))
print("=" * 70)

# ++++++++++++++++++++++++++
# Summary and FAVP events
# ++++++++++++++++++++++++++

print_summary(results, lookup)
print_favp_events(favp)

# ++++++++++++++++++++++++++
# Save results and plot
# ++++++++++++++++++++++++++

try:
    with open(RESULTS_FILE, "wb") as f:
        pickle.dump({"results": results, "favp": favp}, f)
    print("\nResults saved to: %s" % RESULTS_FILE)
except Exception as e:
    print("\n[WARN] Failed to save results: %s" % e)

try:
    plot_p3ht_results(results, lookup, budget=BUDGET, save_path=PLOT_FILE)
    print("Plot saved to: %s" % PLOT_FILE)
except Exception as e:
    print("[WARN] Failed to save plot: %s" % e)

print("\nDone.")
