# Synthetic benchmarks

Scripts in this folder reproduce the synthetic-benchmark results. Each script writes its output to results/ by default.

Run order is not important, the experiments are independent.

## Scripts

- `run_clean.py`
- `run_catastrophic.py`
- `run_ablation.py`
- `run_overhead_sweep.py`
- `run_favp_sensitivity.py`
- `run_cost_ablation.py`
- `run_heavytail.py`
- `run_sweep_rate.py`
- `run_benchmark_extended.py`
- `smoke_robust.py`

## Invocation

From the repository root, install the package once in editable mode:

```bash
pip install -e .
```

Then run any of the benchmarks:

```bash
cd benchmarks/synthetic
python <script_name>.py
```

Each script also accepts `n_seeds N` for a smaller check and `no_plot` if only the numerical output is needed.

## Output layout

Each script writes to `results/` (created if absent).