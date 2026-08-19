"""Stan-comparison testing framework: checks that a `StatisticalModel`'s log
density, gradient, and Hessian agree with a live BridgeStan-compiled Stan
model, in both constrained and unconstrained space.

Since every constraint type implemented in `constraints.py`/`stan_bijectors.py`
replicates Stan's own transform exactly (see those modules' docstrings),
this compares the *entire* flat unconstrained vector at once -- there is no
per-block skip logic here (a `ParameterSpec` with
`matches_stan_unconstrained == False` would need its own bespoke handling,
but no constraint type used by any converted model is in that state).

Draw sourcing is synthetic by default (sample a random unconstrained vector,
let BridgeStan's own `param_constrain` produce the corresponding constrained
values) -- this works even for posteriors with no stored reference draws
(e.g. dogs_nonhierarchical), and, since the same unconstrained vector is fed
to both the JAX model and BridgeStan directly, sidesteps any need to know
posteriordb's stored draw naming at all for this check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


class StanComparisonHarness:
    """Thin wrapper around one compiled BridgeStan model."""

    def __init__(self, stan_file, data: dict):
        import bridgestan

        self.bs_model = bridgestan.StanModel(str(stan_file), _jsonable(data))

    def param_unc_num(self) -> int:
        return self.bs_model.param_unc_num()

    def param_num(self, include_tp=False, include_gq=False) -> int:
        return self.bs_model.param_num(include_tp=include_tp, include_gq=include_gq)

    def param_names(self, include_tp=False, include_gq=False) -> List[str]:
        return self.bs_model.param_names(include_tp=include_tp, include_gq=include_gq)

    def param_unc_names(self) -> List[str]:
        return self.bs_model.param_unc_names()

    def param_constrain(self, x, include_tp=False, include_gq=False):
        return np.asarray(self.bs_model.param_constrain(np.asarray(x, dtype=np.float64), include_tp=include_tp, include_gq=include_gq))

    def param_unconstrain(self, constrained_flat):
        return np.asarray(self.bs_model.param_unconstrain(np.asarray(constrained_flat, dtype=np.float64)))

    def log_density(self, x, propto=False, jacobian=True):
        return self.bs_model.log_density(np.asarray(x, dtype=np.float64), propto=propto, jacobian=jacobian)

    def log_density_gradient(self, x, propto=False, jacobian=True):
        lp, grad = self.bs_model.log_density_gradient(np.asarray(x, dtype=np.float64), propto=propto, jacobian=jacobian)
        return lp, np.asarray(grad)

    def log_density_hessian(self, x, propto=False, jacobian=True):
        lp, grad, hess = self.bs_model.log_density_hessian(np.asarray(x, dtype=np.float64), propto=propto, jacobian=jacobian)
        return lp, np.asarray(grad), np.asarray(hess)


def _jsonable(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        out[k] = np.asarray(v).tolist() if hasattr(v, "tolist") else v
    return out


@dataclass
class ComparisonReport:
    n_draws: int
    check_hessian: bool
    max_abs_err_log_density: float = 0.0
    max_abs_err_grad: float = 0.0
    max_abs_err_hessian: float = 0.0
    max_abs_err_constrained_flat: float = 0.0

    def assert_within(
        self,
        atol: float,
        atol_hessian: Optional[float] = None,
        atol_log_density: Optional[float] = None,
    ):
        """`atol_hessian` (defaults to `atol`) can be set looser than `atol`:
        for models with a SIMPLEX parameter, the Hessian is a *second*
        derivative through the simplex bijector's autodiff-based Jacobian
        (itself a jacfwd+slogdet), which is numerically noisier than the
        log-density/gradient paths -- typically still ~1e-7 *relative*
        error, just larger in absolute terms when Hessian entries are large
        (e.g. from a long `lax.scan` recursion). Not a formula bug; verify
        by comparing `max_abs_err_hessian` against the Hessian's own
        magnitude before loosening this further than that."""
        lp_bound = atol if atol_log_density is None else atol_log_density
        assert self.max_abs_err_log_density < lp_bound, f"log_density max err {self.max_abs_err_log_density}"
        assert self.max_abs_err_grad < atol, f"grad max err {self.max_abs_err_grad}"
        assert self.max_abs_err_constrained_flat < atol, f"constrained flat max err {self.max_abs_err_constrained_flat}"
        if self.check_hessian:
            bound = atol if atol_hessian is None else atol_hessian
            assert self.max_abs_err_hessian < bound, f"hessian max err {self.max_abs_err_hessian}"


def assert_matches_stan(
    model,
    harness: StanComparisonHarness,
    *,
    n_synthetic: int = 20,
    seed: int = 0,
    atol: float = 1e-6,
    check_hessian: bool = True,
    scale: float = 1.0,
) -> ComparisonReport:
    """Draws `n_synthetic` random unconstrained vectors and, for each,
    compares `model`'s log density / gradient / (optionally) Hessian and
    constrained-space flat parameter vector against BridgeStan's, using the
    *same* unconstrained vector for both -- this requires the JAX model's
    `ParameterSpec`s to be declared in the same order as Stan's
    `parameters`-block variables (so the flat unconstrained/constrained
    vectors line up index-for-index)."""
    import jax.numpy as jnp

    assert harness.param_unc_num() == model.model_spec.free_dim, (
        f"free dim mismatch: Stan={harness.param_unc_num()} vs model={model.model_spec.free_dim}"
    )
    rng = np.random.default_rng(seed)
    report = ComparisonReport(n_draws=n_synthetic, check_hessian=check_hessian)
    for _ in range(n_synthetic):
        x_np = rng.normal(scale=scale, size=harness.param_unc_num())
        x = jnp.asarray(x_np)

        c_stan = harness.param_constrain(x_np, include_tp=False, include_gq=False)
        params = model.constrain(x)
        c_jax = np.asarray(model.model_spec.flatten_constrained(params))
        report.max_abs_err_constrained_flat = max(report.max_abs_err_constrained_flat, float(np.max(np.abs(c_jax - c_stan))))

        lp_stan, grad_stan = harness.log_density_gradient(x_np, propto=False, jacobian=True)
        lp_jax = float(model.log_prob_unconstrained(x))
        grad_jax = np.asarray(model.grad_unconstrained(x))
        report.max_abs_err_log_density = max(report.max_abs_err_log_density, abs(lp_jax - lp_stan))
        report.max_abs_err_grad = max(report.max_abs_err_grad, float(np.max(np.abs(grad_jax - grad_stan))))

        if check_hessian:
            _, _, hess_stan = harness.log_density_hessian(x_np, propto=False, jacobian=True)
            hess_jax = np.asarray(model.hessian_unconstrained(x))
            report.max_abs_err_hessian = max(report.max_abs_err_hessian, float(np.max(np.abs(hess_jax - hess_stan))))

    return report
