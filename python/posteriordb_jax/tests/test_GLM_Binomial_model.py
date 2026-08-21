import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.GLM_Binomial_model import GLMBinomialModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("GLM_Binomial_data")
    data = {
        "nyears": raw["nyears"],
        "C": raw["C"],
        "N": raw["N"],
        "year": jnp.asarray(raw["year"], dtype=jnp.float64),
    }
    return GLMBinomialModel(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("GLM_Binomial_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["alpha", "beta1", "beta2"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("GLM_Binomial_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-4, atol_hessian=1.0)
