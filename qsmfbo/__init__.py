"""
qsmfbo: Queue-Scheduled Multi-Fidelity Bayesian Optimisation with
Fidelity-Aware Verification Protocol (FAVP).
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
