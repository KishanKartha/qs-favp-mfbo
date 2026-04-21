"""
qsmfbo: Queue-Scheduled Multi-Fidelity Bayesian Optimisation with
Fidelity-Aware Verification Protocol (FAVP).

Accompanies the paper:
    Kartha, K. & James, A. P. (2026). Cost-aware multi-fidelity scheduling
    and cross-fidelity anomaly resolution for iterative learning under
    laboratory constraints.

The top-level package re-exports the core classes and utilities so that
typical use looks like:

    from qsmfbo import QueueScheduler, StandardMFMES, run_favp_benchmark

Submodules:
    qsmfbo.core   -- base QS-MFBO framework (2-fidelity, MF-MES base)
    qsmfbo.favp   -- FAVP extensions: outlier injection, verification protocol,
                     and the synthetic catastrophic-noise benchmark runner
"""

from .core import (
    # data structures
    Observation,
    QueueItem,
    IterationLog,
    # cost models
    QueueDependentCostModel,
    FixedCostModel,
    # test functions
    StyblinskiTang2D,
    Branin2D,
    Hartmann6D,
    # schedulers
    QueueScheduler,
    StandardMFMES,
    # utilities
    generate_shared_init,
    build_mf_model,
    run_experiment,
    run_ablation,
    plot_results,
)

from .favp import (
    OutlierInjector,
    AnomalyEvent,
    QueueSchedulerFAVP,
    QueueSchedulerNaive,
    compute_true_regret,
    run_favp_benchmark,
)

__version__ = "1.0.0"
