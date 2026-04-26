"""
Runner script for the P3HT-CNT three-fidelity benchmark.This reproduces the P3HT-CNT retrospective benchmark reported in Section 1.2 of the paper: ten random seeds, three methods
(QS-MFBO + FAVP, QS-MFBO without FAVP, MF-MES baseline) under identical surrogates, acquisition function, and shared initialisation.

Place this file in the same directory as:
  - p3ht_cnt_3f_benchmark.py   (the benchmark module)
  - data_all_back_A.xlsx       (Bash et al. 2021 dataset; see data/README.md)

Run (Windows PowerShell, unbuffered for real-time output):
  python -u run_benchmark.py 2>&1 | Tee-Object -FilePath run_log.txt

Or plain redirect (Linux/macOS or PowerShell):
  python -u run_benchmark.py > run_log.txt 2>&1

Outputs:
  - p3ht_results.pkl     pickled (results, favop, lookup) for figure scripts
  - p3ht_checkpoint.pkl  intermediate state, in case the run is interrupted
  - p3ht_results.png     diagnostic plot from the benchmark module
"""

import pickle
import time

import p3ht_cnt_3f_benchmark as bench
from p3ht_cnt_3f_benchmark import (
    run_p3ht_benchmark,
    print_summary,
    print_favop_events,
    plot_p3ht_results,
)

# =====================================================================
# Configuration
# =====================================================================

FILEPATH         = "data_all_back_A.xlsx"
BUDGET           = 250
N_SEEDS          = 10
NUM_FANTASIES    = 16

CHECKPOINT_FILE  = "p3ht_checkpoint.pkl"
RESULTS_FILE     = "p3ht_results.pkl"
PLOT_FILE        = "p3ht_results.png"

# Cost values: typical shared-facility overheads at this lab and
# equipment scale; see Supplementary Note 2 for justification.
# These are the defaults inside ThreeFidelityCostModel in the
# benchmark module, listed here for reference.
#   lambda_lf      = 1.0
#   lambda_o_mf    = 6.0,   lambda_mar_mf = 1.0
#   lambda_o_hf    = 8.0,   lambda_mar_hf = 1.0

# =========================================================
# Set NUM_FANTASIES globally in the benchmark module
# ================================================
bench.NUM_FANTASIES = NUM_FANTASIES

# ======================================
# Run
# ==============================================
print("=" * 70)
print("P3HT-CNT 3-Fidelity QS-MFBO + FAVP Benchmark")
print("=" * 70)
print("Config:")
print("  Data file      : %s" % FILEPATH)
print("  Budget         : %s" % BUDGET)
print("  Seeds          : %s" % N_SEEDS)
print("  NUM_FANTASIES  : %s" % NUM_FANTASIES)
print("  Checkpoint     : %s" % CHECKPOINT_FILE)
print("=" * 70)




t_start = time.time()

results, favop, lookup = run_p3ht_benchmark(
    FILEPATH,
    budget=BUDGET,
    n_seeds=N_SEEDS,
    verbose=True,
    save_checkpoint=CHECKPOINT_FILE,   )

elapsed = time.time() - t_start
print("\n" + "=" * 70)
print("TOTAL RUN TIME: %.1f hours (%.0f seconds)" % (elapsed / 3600, elapsed))
print("=" * 70)

# ===================================================
# Summary & FAVP events
# ====================================================
print_summary(results, lookup)
print_favop_events(favop)

# ==================================================
# Save results and plot
# =============================================
try:
    with open(RESULTS_FILE, "wb") as f:
        pickle.dump({"results": results, "favop": favop}, f)
    print("\nResults saved to: %s" % RESULTS_FILE)
except Exception as e:
    print("\n[WARN] Failed to save results: %s" % e)

try:
    plot_p3ht_results(results, lookup, budget=BUDGET, save_path=PLOT_FILE)
    print("Plot saved to: %s" % PLOT_FILE)
except Exception as e:
    print("[WARN] Failed to save plot: %s" % e)

print("\nDone.")
