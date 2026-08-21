import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.GLM_Poisson_model import GLMPoissonModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("GLM_Poisson_Data")
    data = {
        "n": raw["n"],
        "C": raw["C"],
        "year": jnp.asarray(raw["year"], dtype=jnp.float64),
    }
    return GLMPoissonModel(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("GLM_Poisson_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["alpha", "beta1", "beta2", "beta3"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("GLM_Poisson_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    # Cubic-in-year log_lambda amplifies large |alpha,beta| synthetic draws
    # into large Poisson log-density/gradient/Hessian magnitudes -- same
    # scale-driven float64 noise as test_earn_height.py, not a formula bug
    # (constrained-flat error stays at machine epsilon).
    report.assert_within(atol=5e-4, atol_hessian=3000.0)
