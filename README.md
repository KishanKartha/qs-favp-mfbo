# qs-favp-mfbo

Code accompanying the paper:

> Kartha, K. and James, A. P. *Cost-aware multi-fidelity scheduling and
> cross-fidelity anomaly resolution for iterative learning under laboratory
> constraints.* (2026)

Two ideas are implemented here. The first is queue-scheduling, which
batches expensive measurements so the fixed overhead of a shared facility
is paid once per batch instead of once per sample. The second is FAVP, a
verification protocol that flags anomalous cheap observations, repeats
them, and if the replicate disagrees, promotes the point to the next
fidelity for arbitration rather than discarding it. Neither mechanism is
tied to a specific acquisition function or kernel, any multi-fidelity
optimiser that accepts a per-evaluation cost can plug in. In this
repository they sit on top of MF-MES as the acquisition function with a
semiparametric latent factor (SLFM) multi-fidelity kernel, which is the
combination used for the numbers in the paper. Both mechanisms are
evaluated on three synthetic functions and on the P3HT-CNT dataset of Bash
et al. (2021). The live laboratory deployment described in the paper used
the same decision logic inside a Gradio interface; that part of the code
is specific to our lab and is not included here.

## Install

Python 3.9 or newer. From a fresh virtual environment:

```bash
git clone https://github.com/KishanKartha/qs-favp-mfbo.git
cd qs-favp-mfbo
pip install -e .
```

This installs the `qsmfbo` package and pulls in botorch, gpytorch, and the
usual numerical/plotting stack. A CUDA-capable GPU is not required but will
make things considerably faster; the benchmarks auto-detect it and print
`Using GPU` or `Using CPU` on import.

## Quickstart

The fastest way to check everything is working end-to-end is the
catastrophic-noise benchmark on the two-dimensional Styblinski-Tang
function with three seeds (a few minutes on GPU, longer on CPU):

```bash
cd benchmarks/synthetic
python run_catastrophic.py --function st2d --n_seeds 3
```

Output lands in `benchmarks/synthetic/results/`: a pickle with per-seed
logs, a three-panel PNG, and a text summary. Numbers will not match the
paper exactly at three seeds, but the method ordering (FAVP < no-FAVP <
naive) should already be visible.

## What reproduces what

All paper numbers come from the scripts in `benchmarks/`. Defaults are set
to the values used in the paper, so no arguments are needed for a faithful
reproduction.

| Paper artefact | Script |
|---|---|
| Table 1 | `benchmarks/synthetic/run_clean.py` |
| Figure 2 | `benchmarks/synthetic/run_catastrophic.py` |
| Figure 3 | `benchmarks/p3ht_cnt/run_benchmark.py` |
| Extended Data Fig. 1 | `benchmarks/synthetic/run_clean.py` |
| Extended Data Fig. 2 | `benchmarks/p3ht_cnt/run_benchmark.py` |
| Extended Data Fig. 3 | `benchmarks/p3ht_cnt/run_benchmark.py` |
| Extended Data Table 1 | `benchmarks/synthetic/run_catastrophic.py` |
| Supp. Figs. 1, 2 | `benchmarks/synthetic/run_favp_sensitivity.py` |
| Supp. Fig. 3 | `benchmarks/synthetic/run_overhead_sweep.py` |
| Supp. Table 1 | `benchmarks/synthetic/run_ablation.py` |
| Supp. Table 2 | `benchmarks/synthetic/run_overhead_sweep.py` |

Figures 4 and 5 and Extended Data Figs. 4 and 5 are from the live
experimental campaign (optical micrographs, Raman spectra, and the
trial-by-trial trajectory) and are not generated from code.

See `benchmarks/synthetic/README.md` and `benchmarks/p3ht_cnt/README.md` for
runtime estimates, CLI options, and output layout for each script. The
P3HT-CNT benchmark needs the Bash et al. dataset, which is not redistributed
here; that README has the link.

## Layout

```
qsmfbo/
    core.py         queue-scheduling framework and the MF-MES baseline
    favp.py         FAVP protocol and the catastrophic-noise benchmark
benchmarks/
    synthetic/      five runners for the synthetic experiments
    p3ht_cnt/       three-fidelity benchmark and its runner
```

The package `qsmfbo` contains the general two-fidelity framework. The
three-fidelity P3HT benchmark is intentionally kept as a separate standalone
module under `benchmarks/p3ht_cnt/`  it is the same code that produced the
numbers in Section 2.2 of the paper, preserved so the results reproduce
exactly.

## Reproducibility notes

Ten-seed runs are not short. Rough wall-clock times on a single RTX 5000:

- clean-condition synthetic benchmark: 3–4 h
- catastrophic-noise synthetic benchmark: 6–8 h
- P3HT-CNT three-fidelity benchmark: 4–7 h
- FAVP sensitivity sweep (200 cells, 5 seeds each): 6–8 h

On CPU these run roughly an order of magnitude slower. If you want to
sanity-check a runner without waiting, every script accepts `--n_seeds 2` or
`--n_seeds 3`.

Every runner writes its output under `results/` in its own folder. Re-runs
overwrite. Pickles include all per-seed logs and diagnostics, so plots and
tables can be regenerated from a single finished run.

## Citation

If you use this code, please cite the repo and paper:


## Licence

MIT. See [LICENSE](LICENSE).
