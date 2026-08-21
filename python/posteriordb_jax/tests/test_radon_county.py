import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.radon_county import RadonCounty
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("radon_mod")
    data = {
        "N": raw["N"],
        "J": raw["J"],
        "county": raw["county"],
        "y": jnp.asarray(raw["y"], dtype=jnp.float64),
    }
    return RadonCounty(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("radon_county"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    j = data["J"]
    assert harness.param_unc_names() == [f"a.{i}" for i in range(1, j + 1)] + [
        "mu_a",
        "sigma_a",
        "sigma_y",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("radon_county"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    # N=12573 summed terms; floating-point noise accumulates past 1e-5.
    report.assert_within(atol=1e-4, atol_hessian=1.0)
