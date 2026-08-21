import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.logmesquite_logvas import LogmesquiteLogvas
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("mesquite")
    data = {
        k: (jnp.asarray(v, dtype=jnp.float64) if isinstance(v, list) else v)
        for k, v in raw.items()
    }
    return LogmesquiteLogvas(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("logmesquite_logvas"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == [
        "beta.1", "beta.2", "beta.3", "beta.4", "beta.5", "beta.6", "beta.7", "sigma",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("logmesquite_logvas"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    report.assert_within(atol=1e-5, atol_hessian=1e-1)
