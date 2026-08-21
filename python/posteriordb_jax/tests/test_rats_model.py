import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.rats_model import RatsModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("rats_data")
    data = {
        "N": raw["N"],
        "Npts": raw["Npts"],
        "rat": raw["rat"],
        "x": jnp.asarray(raw["x"], dtype=jnp.float64),
        "y": jnp.asarray(raw["y"], dtype=jnp.float64),
        "xbar": raw["xbar"],
    }
    return RatsModel(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("rats_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    n = data["N"]
    assert harness.param_unc_names() == (
        [f"alpha.{i}" for i in range(1, n + 1)]
        + [f"beta.{i}" for i in range(1, n + 1)]
        + ["mu_alpha", "mu_beta", "sigma_y", "sigma_alpha", "sigma_beta"]
    )


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("rats_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    # x/y are in raw units (weights up to ~150, x up to ~36); loosen
    # atol_hessian for the same scale reason as test_blr.py/test_mesquite.py.
    report.assert_within(atol=1e-4, atol_hessian=5.0)
