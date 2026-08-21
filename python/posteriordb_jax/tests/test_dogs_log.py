import jax.numpy as jnp
import numpy as np

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.dogs_log import DogsLog
from posteriordb_jax.testing import StanComparisonHarness


def _load_model():
    raw = load_data("dogs")
    data = {
        "n_dogs": raw["n_dogs"],
        "n_trials": raw["n_trials"],
        "y": raw["y"],
    }
    return DogsLog(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("dogs_log"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["beta.1", "beta.2"]


def test_matches_bridgestan_log_density_grad_hessian():
    """`beta` is a plain unconstrained `vector[2]`, but its *priors* are
    `uniform(-100, 0)` (beta[1]) / `uniform(0, 100)` (beta[2]) -- outside
    those one-sided intervals, both Stan's and this model's log density are
    `-inf` by construction (see dogs_log.py's docstring), which is correct,
    not a bug, but means `assert_matches_stan`'s default zero-mean synthetic
    draws would land beta[1]>0 or beta[2]<0 on ~75% of draws and spuriously
    "mismatch" on -inf/-inf comparisons (nan differences). Draw synthetic
    points with the right sign per component instead, replicating
    `assert_matches_stan`'s checks directly."""
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("dogs_log"), data)
    rng = np.random.default_rng(0)

    max_err_lp = max_err_grad = max_err_hess = max_err_flat = 0.0
    for _ in range(20):
        x_np = np.array([rng.uniform(-30.0, -0.5), rng.uniform(0.5, 30.0)])
        x = jnp.asarray(x_np)

        c_stan = harness.param_constrain(x_np)
        c_jax = np.asarray(model.constrain(x)["beta"])
        max_err_flat = max(max_err_flat, float(np.max(np.abs(c_jax - c_stan))))

        lp_stan, grad_stan = harness.log_density_gradient(x_np)
        lp_jax = float(model.log_prob_unconstrained(x))
        grad_jax = np.asarray(model.grad_unconstrained(x))
        max_err_lp = max(max_err_lp, abs(lp_jax - lp_stan))
        max_err_grad = max(max_err_grad, float(np.max(np.abs(grad_jax - grad_stan))))

        _, _, hess_stan = harness.log_density_hessian(x_np)
        hess_jax = np.asarray(model.hessian_unconstrained(x))
        max_err_hess = max(max_err_hess, float(np.max(np.abs(hess_jax - hess_stan))))

    assert max_err_flat < 1e-8, max_err_flat
    assert max_err_lp < 1e-6, max_err_lp
    assert max_err_grad < 1e-6, max_err_grad
    assert max_err_hess < 1e-4, max_err_hess
