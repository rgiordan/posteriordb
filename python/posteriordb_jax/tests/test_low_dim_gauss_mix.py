import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.low_dim_gauss_mix import LowDimGaussMix
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("low_dim_gauss_mix")
    data = {"N": raw["N"], "y": jnp.asarray(raw["y"], dtype=jnp.float64)}
    return LowDimGaussMix(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("low_dim_gauss_mix"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == [
        "mu.1",
        "mu.2",
        "sigma.1",
        "sigma.2",
        "theta",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("low_dim_gauss_mix"), data)
    # A mixture log-density is only piecewise-smooth in general, but is
    # smooth (and its Hessian well-defined) at any specific parameter draw --
    # a small scale keeps synthetic draws away from any pathological region.
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.5)
    # N=1000 data points each contribute a log_mix second-derivative term;
    # summing that many terms accumulates floating-point noise beyond 1e-5
    # even though each term itself is correct (grad error stays ~1e-11).
    report.assert_within(atol=1e-5, atol_hessian=1e-4)
