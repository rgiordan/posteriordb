import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.radon_partially_pooled_noncentered import RadonPartiallyPooledNoncentered
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("radon_all")
    data = {
        "N": raw["N"],
        "J": raw["J"],
        "county_idx": raw["county_idx"],
        "log_radon": jnp.asarray(raw["log_radon"], dtype=jnp.float64),
    }
    return RadonPartiallyPooledNoncentered(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("radon_partially_pooled_noncentered"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    j = data["J"]
    assert harness.param_unc_names() == [f"alpha_raw.{i}" for i in range(1, j + 1)] + [
        "mu_alpha",
        "sigma_alpha",
        "sigma_y",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("radon_partially_pooled_noncentered"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    report.assert_within(atol=1e-4, atol_hessian=1.0)
