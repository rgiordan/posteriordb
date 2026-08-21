import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.low_dim_gauss_mix_collapse import LowDimGaussMixCollapse
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("low_dim_gauss_mix_collapse")
    data = {"N": raw["N"], "y": jnp.asarray(raw["y"], dtype=jnp.float64)}
    return LowDimGaussMixCollapse(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("low_dim_gauss_mix_collapse"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["mu.1", "mu.2", "sigma.1", "sigma.2", "theta"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("low_dim_gauss_mix_collapse"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.5)
    report.assert_within(atol=1e-5, atol_hessian=1e-4)
