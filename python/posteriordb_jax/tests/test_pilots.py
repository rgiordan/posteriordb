import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.pilots import Pilots
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("pilots")
    data = {
        "N": raw["N"],
        "n_groups": raw["n_groups"],
        "n_scenarios": raw["n_scenarios"],
        "group_id": raw["group_id"],
        "scenario_id": raw["scenario_id"],
        "y": jnp.asarray(raw["y"], dtype=jnp.float64),
    }
    return Pilots(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("pilots"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    ng = data["n_groups"]
    ns = data["n_scenarios"]
    assert harness.param_unc_names() == (
        [f"a.{i}" for i in range(1, ng + 1)]
        + [f"b.{i}" for i in range(1, ns + 1)]
        + ["mu_a", "mu_b", "sigma_a", "sigma_b", "sigma_y"]
    )


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("pilots"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-4, atol_hessian=1.0)
