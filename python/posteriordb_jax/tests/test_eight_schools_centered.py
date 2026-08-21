import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.eight_schools_centered import EightSchoolsCentered
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("eight_schools")
    data = {
        "J": raw["J"],
        "y": jnp.asarray(raw["y"], dtype=jnp.float64),
        "sigma": jnp.asarray(raw["sigma"], dtype=jnp.float64),
    }
    return EightSchoolsCentered(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("eight_schools_centered"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == [
        "theta.1",
        "theta.2",
        "theta.3",
        "theta.4",
        "theta.5",
        "theta.6",
        "theta.7",
        "theta.8",
        "mu",
        "tau",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("eight_schools_centered"), data)
    report = assert_matches_stan(model, harness, n_synthetic=25, seed=0, check_hessian=True)
    report.assert_within(atol=1e-6)
