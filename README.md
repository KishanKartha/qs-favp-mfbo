# qs-favp-mfbo


## Install

Python 3.9 or newer. From a fresh virtual environment:

```bash
#git clone <this repo url> # the code will be released in the github post acceptance.
cd qs-favp-mfbo
pip install -e .
```

## Quickstart

The fastest way to check everything is working end-to-end is by running the catastrophic-noise benchmark on the two-dimensional Styblinski-Tang
function with three seeds (a few minutes on GPU, longer on CPU):

```bash
cd benchmarks/synthetic
python run_catastrophic.py --function st2d --n_seeds 3
```

Output lands in the `benchmarks/synthetic/results/`: a pickle with per-seed logs, a three-panel PNG, and a text summary. Numbers will not match the
paper exactly at three seeds, but the method ordering (FAVP < no-FAVP < naive) should already be observable.

## What reproduces what

All paper numbers come from the scripts in `benchmarks/`. Defaults are set to the values used in the paper, so no arguments are needed for a faithful
reproduction.


See `benchmarks/synthetic/README.md` and `benchmarks/p3ht_cnt/README.md`. TheP3HT-CNT benchmark needs the Bash et al. dataset, which is not redistributed
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
numbers in the paper, preserved so the results reproduceexactly.

