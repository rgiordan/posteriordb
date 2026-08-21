import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.diamonds import Diamonds
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("diamonds")
    data = {
        "N": raw["N"],
        "Y": jnp.asarray(raw["Y"], dtype=jnp.float64),
        "K": raw["K"],
        "X": jnp.asarray(raw["X"], dtype=jnp.float64),
        "prior_only": raw["prior_only"],
    }
    return Diamonds(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("diamonds"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    kc = data["K"] - 1
    assert harness.param_unc_names() == [f"b.{i}" for i in range(1, kc + 1)] + [
        "Intercept",
        "sigma",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("diamonds"), data)
    report = assert_matches_stan(model, harness, n_synthetic=10, seed=0, check_hessian=True, scale=0.3)
    # N=5000 summed terms; floating-point noise accumulates past 1e-5.
    report.assert_within(atol=1e-3, atol_hessian=10.0)
