# P3HT-CNT retrospective benchmark

Reproduces Section 2.2 of the paper and the associated Extended Data figures
using the P3HT-CNT composite dataset of Bash et al. (*Adv. Funct. Mater.*,
2021, **31**, 2102606).

## Dataset

This folder does **not** include the dataset. The file used by the paper is
`data_all_back_A.xlsx` from the upstream repository:

> https://github.com/Lightmann/GraphModel_for_CNTDesign

Download it and place it alongside `run_benchmark.py` before launching a run.

The benchmark uses the dataset in the following way:

| Fidelity | m | Dataset column | Cost |
|---|---|---|---|
| LF | 0 | log absorption ratio 602 nm / 525 nm | 1 (immediate) |
| MF | 1 | log four-point-probe sheet resistance (sign-inverted) | overhead 1.5, marginal 1.0 |
| HF | 2 | log conductivity (from sheet resistance and profilometry) | overhead 3.0, marginal 1.5 |

At each oracle query the module draws one droplet uniformly at random from
the composition's replicate pool and returns the recorded value at the
requested fidelity. Replicate draws where the requested measurement is
missing return a missing value (the cost is still charged), reproducing the
occasional measurement failures of the original campaign without synthetic
noise injection. See Supplementary Note 2 for the rationale.

## Running

From this folder, once the dataset file is in place:

```bash
python run_benchmark.py
```

Defaults reproduce the paper: 10 seeds, budget 200, τ = 4.0, with the
queue-scheduling + FAVP framework, a scheduling-alone ablation, and an
MF-MES baseline. Output:

```
p3ht_checkpoint.pkl    per-seed checkpoint (updated after every seed)
p3ht_results.pkl       final results and FAVP event logs
p3ht_results.png       convergence plot
```

Checkpointing lets the run resume after an interruption. If the run completes
cleanly, `p3ht_results.pkl` carries the same content plus the plot.

## Runtime

On a single NVIDIA Quadro RTX 5000, ten seeds take roughly **4 to 7 hours**.
CPU-only runs are substantially longer. GPU availability is auto-detected;
the module prints `Using GPU: ...` or `Using CPU` at import.

## Files

```
p3ht_cnt_benchmark.py     benchmark module (standalone; does not import qsmfbo)
run_benchmark.py          runner script
data_all_back_A.xlsx      Bash et al. dataset (not committed; see above)
```

## Note on the parallel implementation

The three-fidelity benchmark is intentionally a separate standalone module
rather than a generalisation of the two-fidelity framework in `qsmfbo/`. This
is a deliberate choice: it is the exact code that produced the published
numbers for Section 2.2, so the results reproduce bit-for-bit. The paper's
Methods section describes the generalised M-fidelity formulation that both
implementations instantiate as special cases.
