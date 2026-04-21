# Synthetic benchmarks

Scripts in this folder reproduce the synthetic-benchmark results in the paper.
Each script is self-contained and writes its output (pickles, PNGs, text
summaries) to `results/` by default.

Run order is not important — the four experiments are independent.

| Script | Reproduces |
|---|---|
| `run_clean.py` | Table 1 and Extended Data Fig. 1 (clean-condition benchmark, MF-MES vs QS-MFBO, two overhead regimes). |
| `run_catastrophic.py` | Fig. 2 and Extended Data Table 1 (four-method comparison under 20% catastrophic outlier injection, true regret on clean evaluation). |
| `run_ablation.py` | Supplementary Note 4, Supplementary Table 1 (component ablation of QS-MFBO on Styblinski-Tang 2D). |
| `run_overhead_sweep.py` | Supplementary Note 4, Supplementary Table 2 and Supplementary Fig. 3 (MFBO vs QS-MFBO advantage across five overhead regimes). |
| `run_favp_sensitivity.py` | Supplementary Note 3, Supplementary Figs. 1–2 (FAVP threshold sensitivity; 5 x 4 grid over tau and gamma on ST-2D and H-6D). |

## Invocation

From the repository root, install the package once in editable mode:

```bash
pip install -e .
```

Then run any of the benchmarks:

```bash
cd benchmarks/synthetic

# Full runs (default: 10 seeds for the first four, 5 seeds for sensitivity)
python run_clean.py
python run_catastrophic.py
python run_ablation.py
python run_overhead_sweep.py
python run_favp_sensitivity.py
```

Each script also accepts `--n_seeds N` for a smaller check and `--no_plot` if
only the numerical output is needed.

## Approximate runtime

On a single NVIDIA Quadro RTX 5000 (the machine used to produce the paper's
numbers), typical wall-clock times are:

| Script | Seeds | Runtime |
|---|---|---|
| `run_clean.py` | 10 | 3–4 h (6 configurations) |
| `run_catastrophic.py` | 10 | 6–8 h (3 functions, 4 methods) |
| `run_ablation.py` | 10 | 2 h |
| `run_overhead_sweep.py` | 10 | 3–4 h (5 overhead values) |
| `run_favp_sensitivity.py` | 5 | 6–8 h (200 cells) |

Runs are substantially slower on CPU.

## Output layout

Each script writes to `results/` (created if absent):

```
results/
├── clean_st2d_ov25.pkl                      run_clean.py
├── clean_st2d_ov25.png
├── clean_summary.txt
├── favp_st2d.pkl                            run_catastrophic.py
├── favp_st2d.png
├── favp_summary.txt
├── ablation_st2d.pkl                        run_ablation.py
├── ablation_summary.txt
├── overhead_sweep.pkl                       run_overhead_sweep.py
├── overhead_sweep.png
├── overhead_summary.txt
└── sensitivity/                             run_favp_sensitivity.py
    ├── results_sensitivity.npz
    ├── per_seed_regret.csv
    ├── heatmap_regret.pdf
    └── heatmap_events.pdf
```

Pickle files contain the per-seed logs and diagnostics dictionaries used to
regenerate figures and tables.
