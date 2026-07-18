"""
run_cost_ablation.py

Cost-model ablation for Reviewer 1, Comment 2.
Three per-sample cost models, identical batch_cost (actual expenditure).
Only the decision quantity fed to the acquisition differs.

Run from the outer qsmfbo folder:
    python run_cost_ablation.py --function st2d --seeds 10
"""
import argparse, os, pickle, time
from collections import Counter
import numpy as np

from qsmfbo.core import (QueueScheduler, QueueDependentCostModel, FixedCostModel,
                         StyblinskiTang2D, Branin2D, Hartmann6D,
                         generate_shared_init, FIDELITY_CHEAP)

FUNCS = {'st2d': StyblinskiTang2D, 'br2d': Branin2D, 'h6d': Hartmann6D}


# ---------------------------------------------------------------
# New cost model: true marginal cost of adding one more item
# ---------------------------------------------------------------
class MarginalCostModel:
    """lambda_m(q) = lambda_marginal if q >= 1 else lambda_o + lambda_marginal.

    The true incremental cost of adding one more sample once a session is
    already committed. batch_cost is unchanged: the actual expenditure is
    identical to the other models.
    """
    def __init__(self, lambda_cheap, lambda_overhead, lambda_marginal):
        self.lambda_cheap = lambda_cheap
        self.lambda_overhead = lambda_overhead
        self.lambda_marginal = max(lambda_marginal, lambda_cheap)
        self.lambda_expensive = lambda_overhead + self.lambda_marginal

    def cost_per_sample(self, q, fidelity):
        if fidelity < 0.5:
            return self.lambda_cheap
        if q >= 1:
            return self.lambda_marginal
        return self.lambda_overhead + self.lambda_marginal

    def batch_cost(self, q):
        if q == 0:
            return 0.0
        return self.lambda_overhead + q * self.lambda_marginal

    def single_expensive_cost(self):
        return self.lambda_overhead + self.lambda_marginal


COST_MODELS = {
    'shapley':  QueueDependentCostModel,   # lambda_o/(q+1) + lambda_mar  (ours)
    'marginal': MarginalCostModel,         # lambda_mar once q >= 1
    'fixed':    FixedCostModel,            # lambda_o + lambda_mar always
}


# ---------------------------------------------------------------
# Instrumentation: why did each batch fire?
# ---------------------------------------------------------------
class InstrumentedQueueScheduler(QueueScheduler):
    """Records dispatch reason: 'trigger', 'cap' (q_max hit), or 'flush'
    (post-budget force-execute in QueueScheduler.run)."""

    def __init__(self, *args, budget=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dispatch_events = []
        self._budget = budget

    def _execute_batch(self):
        q = len(self.queue)
        if self._budget is not None and self.cumulative_cost >= self._budget:
            reason = 'flush'
        elif q >= self.q_max:
            reason = 'cap'
        else:
            reason = 'trigger'
        cost_before = self.cumulative_cost
        entry = super()._execute_batch()
        self.dispatch_events.append({'reason': reason,
                                     'iteration': self.iteration,
                                     'batch_size': q,
                                     'cost_at_dispatch': cost_before})
        return entry


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def summarise(sched, budget):
    log = sched.log
    ev = sched.dispatch_events
    in_budget = [e for e in ev if e['reason'] != 'flush']
    first = in_budget[0] if in_budget else None
    return dict(
        regret_at_budget = next((l.simple_regret for l in reversed(log)
                                 if l.cumulative_cost <= budget), np.nan),
        final_regret     = log[-1].simple_regret,
        final_cost       = log[-1].cumulative_cost,
        final_visits     = log[-1].n_facility_visits,
        n_dispatch       = len(in_budget),
        first_disp_cost  = first['cost_at_dispatch'] if first else np.nan,
        first_disp_iter  = first['iteration']        if first else np.nan,
        batch_sizes      = [e['batch_size'] for e in in_budget],
        peak_q           = max([l.queue_size for l in log] + [0]),
        reasons          = [e['reason'] for e in ev],
        dispatch_events  = ev,
        log              = log,
    )


def sd(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return a.std(ddof=1) if len(a) > 1 else 0.0


# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--function', default='st2d', choices=list(FUNCS))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--budget', type=float, default=200.0)
    ap.add_argument('--lambda_cheap', type=float, default=1.0)
    ap.add_argument('--lambda_overhead', type=float, default=25.0)
    ap.add_argument('--lambda_marginal', type=float, default=2.0)
    ap.add_argument('--noise_std', type=float, default=0.1)
    ap.add_argument('--outdir', default='ablation_results')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fn = FUNCS[args.function]
    path = os.path.join(args.outdir, 'cost_ablation_%s.pkl' % args.function)
    store = {}
    if os.path.exists(path):
        with open(path, 'rb') as f:
            store = pickle.load(f)['store']

    init_cost = (3 * args.lambda_cheap + args.lambda_overhead
                 + 2 * args.lambda_marginal)

    for seed in range(args.seeds):
        store.setdefault(seed, {})
        init_data = generate_shared_init(fn, n_cheap=3, n_expensive=2,
                                         noise_std=args.noise_std, seed=seed)
        for name, CM in COST_MODELS.items():
            if name in store[seed]:
                print("[%s] seed %d %-9s cached" % (args.function, seed, name))
                continue
            t0 = time.time()
            cm = CM(args.lambda_cheap, args.lambda_overhead, args.lambda_marginal)
            sched = InstrumentedQueueScheduler(
                fn, cm, noise_std=args.noise_std, seed=seed,
                init_data=init_data, budget=args.budget,
                use_queue_cost=True, use_adaptive_trigger=True)
            sched.run(args.budget, verbose=False, init_cost=init_cost)
            m = summarise(sched, args.budget)
            store[seed][name] = m
            print("  [%s] seed %d %-9s regret@budget %8.3f | first disp cost %6.1f "
                  "| dispatches %d | reasons %s | %.0fs"
                  % (args.function, seed, name, m['regret_at_budget'],
                     m['first_disp_cost'], m['n_dispatch'],
                     dict(Counter(m['reasons'])), time.time() - t0))
            with open(path, 'wb') as f:
                pickle.dump({'store': store, 'args': vars(args)}, f)

    # ---------------- report ----------------
    seeds = sorted(store.keys())
    print("\n" + "=" * 104)
    print("COST MODEL ABLATION | %s | %d seeds | budget %.0f | lambda_o=%.0f lambda_mar=%.0f"
          % (args.function.upper(), len(seeds), args.budget,
             args.lambda_overhead, args.lambda_marginal))
    print("=" * 104)
    print("%-9s | %-17s | %-16s | %-14s | %-12s | %s"
          % ("model", "regret@budget", "first disp cost", "batch size",
             "sessions", "no dispatch"))
    print("-" * 104)
    for name in COST_MODELS:
        rg  = [store[s][name]['regret_at_budget'] for s in seeds]
        fdc = [store[s][name]['first_disp_cost']  for s in seeds]
        vis = [store[s][name]['final_visits']     for s in seeds]
        bs  = [b for s in seeds for b in store[s][name]['batch_sizes']]
        never = sum(1 for s in seeds if store[s][name]['n_dispatch'] == 0)
        print("%-9s | %7.3f +/- %6.3f | %6.1f +/- %6.1f | %5.2f +/- %5.2f | "
              "%4.1f +/- %4.1f | %d/%d"
              % (name, np.nanmean(rg), sd(rg), np.nanmean(fdc), sd(fdc),
                 np.mean(bs) if bs else 0, sd(bs) if bs else 0,
                 np.mean(vis), sd(vis), never, len(seeds)))

    print("\nDispatch reasons (all seeds):")
    for name in COST_MODELS:
        rs = [r for s in seeds for r in store[s][name]['reasons']]
        print("  %-9s %s" % (name, dict(Counter(rs))))

    print("\nPeak queue size per seed:")
    for name in COST_MODELS:
        pq = [store[s][name]['peak_q'] for s in seeds]
        print("  %-9s mean %5.2f +/- %4.2f | max %d"
              % (name, np.mean(pq), sd(pq), max(pq)))

    print("\nBatch-size distribution (trigger/cap dispatches, flush excluded):")
    for name in COST_MODELS:
        bs = [b for s in seeds for b in store[s][name]['batch_sizes']]
        if bs:
            v, c = np.unique(bs, return_counts=True)
            print("  %-9s %s" % (name, ", ".join("%d:%d" % (a, b) for a, b in zip(v, c))))

    print("\nFinal cost (flush inflation check):")
    for name in COST_MODELS:
        fc = [store[s][name]['final_cost'] for s in seeds]
        print("  %-9s %6.1f +/- %5.1f" % (name, np.mean(fc), sd(fc)))


if __name__ == '__main__':
    main()