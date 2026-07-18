"""
Robust-GP baseline (Martinez-Cantin et al., AISTATS 2018 style).

Iterative classify-and-refit: fit the GP, flag observations whose
standardised residual exceeds tau_r, refit on the inliers, repeat until
the inlier mask stops changing.

Detection uses leave-one-out (LOO) residuals for points inside the
current fit (Rasmussen & Williams Sec. 5.4.2, closed form), and ordinary
posterior residuals for points currently excluded. LOO removes the
self-influence problem: a GP that interpolates an outlier gives it a
near-zero in-sample residual, but its LOO residual remains large.

Two method classes:
  RobustMFMES              MFBO baseline + robust refit   (Comment 7 baseline)
  QueueSchedulerFAVPRobust QS-MFBO + FAVP + robust refit  (stacked variant)

Ground-truth protection: expensive-fidelity observations are never
excluded, consistent with the benchmark design in which the expensive
fidelity is uncorrupted ground truth.
"""

import torch

from .core import StandardMFMES, build_mf_model
from .favp import QueueSchedulerFAVP


def loo_z_scores(model, jitter=1e-8):
    """Closed-form leave-one-out standardised residuals for an exact GP.

    Using A = K_f(X,X) + sigma^2 I on the training inputs:
      y_i - mu_loo_i = [A^{-1}(y - m)]_i / [A^{-1}]_ii
      var_loo_i      = 1 / [A^{-1}]_ii
      z_i            = |alpha_i| / sqrt([A^{-1}]_ii),  alpha = A^{-1}(y - m)

    Computed in the model's (standardised) output space, which is the
    space the likelihood noise refers to.
    """
    Xtr = model.train_inputs[0]
    ytr = model.train_targets.squeeze(-1) if model.train_targets.dim() > 1 \
        else model.train_targets
    n = Xtr.shape[0]
    with torch.no_grad():
        mean = model.mean_module(Xtr).squeeze(-1) if \
            model.mean_module(Xtr).dim() > 1 else model.mean_module(Xtr)
        Kf = model.covar_module(Xtr).to_dense()
        noise = model.likelihood.noise.view(-1)[0]
        A = Kf + (noise + jitter) * torch.eye(n, dtype=Kf.dtype,
                                              device=Kf.device)
        L = torch.linalg.cholesky(A)
        Ainv = torch.cholesky_inverse(L)
        alpha = Ainv @ (ytr - mean)
        diag = Ainv.diagonal().clamp_min(1e-12)
        z = alpha.abs() / diag.sqrt()
    return z


class RobustFitMixin:
    """Overrides _fit_model with an iterative robust classify-and-refit.

    Extra kwargs (popped before passing to the parent __init__):
      tau_r             residual threshold for exclusion (default 3.0)
      max_robust_rounds max classify/refit iterations     (default 3)
      n_min_robust      observations required before any filtering
                        (plain fit during warmup, default 10)
      robust_verbose    print per-fit diagnostics          (default True)
    """

    def __init__(self, *args, tau_r=3.0, max_robust_rounds=3,
                 n_min_robust=10, robust_verbose=False, **kwargs):
        self.tau_r = tau_r
        self.max_robust_rounds = max_robust_rounds
        self.n_min_robust = n_min_robust
        self.robust_verbose = robust_verbose
        self.exclusion_log = []
        super().__init__(*args, **kwargs)

    def _robust_z_all(self, model, X, yv, mask):
        """z for every observation: LOO for in-fit points, posterior z for
        excluded points."""
        n = X.shape[0]
        z_all = torch.zeros(n, dtype=yv.dtype)

        # In-fit points: LOO residuals.
        try:
            z_in = loo_z_scores(model)
            in_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            z_all[in_idx] = z_in.to(z_all.dtype)
        except Exception:
            # Fall back to posterior z for everything.
            mask = torch.zeros_like(mask)

        # Excluded points: ordinary posterior z against the fitted model.
        out_idx = torch.nonzero(~mask, as_tuple=False).squeeze(-1)
        if out_idx.numel() > 0:
            with torch.no_grad():
                post = model.posterior(X[out_idx])
                mu = post.mean.squeeze(-1)
                sd = post.variance.clamp_min(1e-12).sqrt().squeeze(-1)
            z_all[out_idx] = ((yv[out_idx] - mu).abs() / sd).to(z_all.dtype)

        return z_all

    def _fit_model(self):
        import time as _time
        _t0 = _time.time()
        X, Y = self._get_train_tensors()
        if X is None or X.shape[0] < 3:
            return False

        n = X.shape[0]

        # Warmup: plain fit, no filtering.
        if n < self.n_min_robust:
            try:
                self.model = build_mf_model(X, Y, self.fidelity_dim)
                return True
            except Exception:
                return False

        yv = Y.squeeze(-1)
        protected = X[:, self.fidelity_dim] >= 0.5   # never drop ground truth
        mask = torch.ones(n, dtype=torch.bool)
        fitted_mask = None
        model = None
        z = torch.zeros(n)

        for rnd in range(self.max_robust_rounds):
            try:
                model = build_mf_model(X[mask], Y[mask], self.fidelity_dim)
                fitted_mask = mask.clone()
            except Exception:
                if model is None:
                    return False
                break

            z = self._robust_z_all(model, X, yv, mask)

            if self.robust_verbose:
                zc = z[~protected]
                if zc.numel() > 0:
                    print("    [robust] iter %d rnd %d: max z=%.2f | "
                          "z>2:%d z>2.5:%d z>3:%d | cheap pts:%d | noise=%.4f"
                          % (self.iteration, rnd, zc.max().item(),
                             int((zc > 2.0).sum()), int((zc > 2.5).sum()),
                             int((zc > 3.0).sum()), zc.numel(),
                             model.likelihood.noise.mean().item()))

            new_mask = (z <= self.tau_r) | protected
            if new_mask.sum() < 3:          # safety: never starve the GP
                new_mask = mask
            if torch.equal(new_mask, mask):
                break
            mask = new_mask

        # If the loop advanced the mask after the last fit, refit once.
        if fitted_mask is None or not torch.equal(fitted_mask, mask):
            try:
                model = build_mf_model(X[mask], Y[mask], self.fidelity_dim)
            except Exception:
                if model is None:
                    return False

        self.model = model
        n_excl = int((~mask).sum())
        self.exclusion_log.append(
            {'iteration': self.iteration, 'n_total': n, 'n_excluded': n_excl,
             'max_z': float(z[~protected].max()) if (~protected).sum() > 0 else 0.0})
        if self.robust_verbose and n_excl > 0:
            print("    [robust] iter %d: excluded %d/%d observations"
                  % (self.iteration, n_excl, n))

        self.timing_log.append({'iteration': self.iteration,
                                'op': 'robust_fit',
                                'q': len(getattr(self, 'queue', [])),
                                'n_obs': n,
                                'seconds': _time.time() - _t0})
        return True

    def exclusion_summary(self):
        if not self.exclusion_log:
            return "no fits logged"
        tot = self.exclusion_log[-1]
        peak = max(e['n_excluded'] for e in self.exclusion_log)
        peak_z = max(e['max_z'] for e in self.exclusion_log)
        n_active = sum(1 for e in self.exclusion_log if e['n_excluded'] > 0)
        return ("final: %d/%d excluded | peak: %d | peak z: %.2f | "
                "fits with exclusions: %d/%d"
                % (tot['n_excluded'], tot['n_total'], peak, peak_z,
                   n_active, len(self.exclusion_log)))


class RobustMFMES(RobustFitMixin, StandardMFMES):
    """MFBO baseline with Martinez-Cantin-style robust refit."""
    pass


class QueueSchedulerFAVPRobust(RobustFitMixin, QueueSchedulerFAVP):
    """Stacked variant: QS-MFBO + FAVP + robust refit."""
    pass