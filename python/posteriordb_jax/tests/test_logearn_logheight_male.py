import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.logearn_logheight_male import LogearnLogheightMale
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("earnings")
    data = {
        "N": raw["N"],
        "earn": jnp.asarray(raw["earn"], dtype=jnp.float64),
        "height": jnp.asarray(raw["height"], dtype=jnp.float64),
        "male": jnp.asarray(raw["male"], dtype=jnp.float64),
    }
    return LogearnLogheightMale(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("logearn_logheight_male"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["beta.1", "beta.2", "beta.3", "sigma"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("logearn_logheight_male"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    report.assert_within(atol=1e-5, atol_hessian=1e-2)
