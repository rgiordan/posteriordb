import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.radon_pooled import RadonPooled
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("radon_all")
    data = {
        "N": raw["N"],
        "floor_measure": jnp.asarray(raw["floor_measure"], dtype=jnp.float64),
        "log_radon": jnp.asarray(raw["log_radon"], dtype=jnp.float64),
    }
    return RadonPooled(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("radon_pooled"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["alpha", "beta", "sigma_y"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("radon_pooled"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    # N=12573 summed terms; floating-point noise accumulates past 1e-5.
    report.assert_within(atol=1e-4, atol_hessian=1.0)
