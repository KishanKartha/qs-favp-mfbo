# P3HT-CNT three-fidelity benchmark

Retrospective benchmark on the P3HT-CNT composite dataset of Bash et al. (*Adv. Funct. Mater.* **31**, 2102606 (2021)), corresponding to Section 1.2
of the paper. Reproduces three methods on real noisy three-fidelity data: the full framework (QS-MFBO + FAVP), scheduling alone (QS-MFBO without
FAVP), and the unaugmented MF-MES baseline.

## Files

```
p3ht_cnt_3f_benchmark.py       Benchmark module: oracle, cost model,
                               scheduler, FAVP, MF-MES baseline.
run_benchmark.py               Entry point. Runs 10 seeds × 3 methods.
p3ht_figure.py                 Main-text Figure 3 (regret vs cost +
                               facility sessions).

```

## Reproducing the run (~7-10 hours on a single GPU)

To regenerate `p3ht_results.pkl` from scratch, place the Bash et al.
dataset (see `../../data/README.md`) in this directory as
`data_all_back_A.xlsx`, then:

```
python -u run_benchmark.py > run_log.txt 2>&1
```

The runner writes `p3ht_results.pkl`, `p3ht_checkpoint.pkl`, and a
diagnostic plot. Configuration (budget, number of seeds, MC samples)
sits at the top of `run_benchmark.py`.

## Configuration

Default cost values (set in `ThreeFidelityCostModel`):

```
lambda_lf      = 1.0
lambda_o_mf    = 6.0,   lambda_mar_mf = 1.0
lambda_o_hf    = 8.0,   lambda_mar_hf = 1.0
budget         = 250
```


