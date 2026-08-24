import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.election88_full import Election88Full
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("election88")
    data = {
        "N": raw["N"],
        "n_age": raw["n_age"],
        "n_edu": raw["n_edu"],
        "n_age_edu": raw["n_age_edu"],
        "n_region_full": raw["n_region_full"],
        "n_state": raw["n_state"],
        "age": raw["age"],
        "age_edu": raw["age_edu"],
        "black": raw["black"],
        "edu": raw["edu"],
        "female": raw["female"],
        "region_full": raw["region_full"],
        "state": raw["state"],
        "v_prev_full": jnp.asarray(raw["v_prev_full"], dtype=jnp.float64),
        "y": raw["y"],
    }
    return Election88Full(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("election88_full"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    names = (
        [f"a.{i}" for i in range(1, data["n_age"] + 1)]
        + [f"b.{i}" for i in range(1, data["n_edu"] + 1)]
        + [f"c.{i}" for i in range(1, data["n_age_edu"] + 1)]
        + [f"d.{i}" for i in range(1, data["n_state"] + 1)]
        + [f"e.{i}" for i in range(1, data["n_region_full"] + 1)]
        + [f"beta.{i}" for i in range(1, 6)]
        + ["sigma_a", "sigma_b", "sigma_c", "sigma_d", "sigma_e"]
    )
    assert harness.param_unc_names() == names


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("election88_full"), data)
    report = assert_matches_stan(model, harness, n_synthetic=5, seed=0, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-3, atol_hessian=10.0)
