import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.wells_dae_inter_model import WellsDaeInterModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("wells_data")
    data = {
        "N": raw["N"],
        "switched": raw["switched"],
        "dist": jnp.asarray(raw["dist"], dtype=jnp.float64),
        "arsenic": jnp.asarray(raw["arsenic"], dtype=jnp.float64),
        "educ": jnp.asarray(raw["educ"], dtype=jnp.float64),
    }
    return WellsDaeInterModel(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("wells_dae_inter_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == [
        "alpha",
        "beta.1",
        "beta.2",
        "beta.3",
        "beta.4",
        "beta.5",
        "beta.6",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("wells_dae_inter_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    report.assert_within(atol=5e-5, atol_hessian=0.05)
