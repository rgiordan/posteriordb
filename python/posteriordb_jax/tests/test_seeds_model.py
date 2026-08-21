import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.seeds_model import SeedsModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("seeds_data")
    data = {
        "I": raw["I"],
        "n": raw["n"],
        "N": raw["N"],
        "x1": jnp.asarray(raw["x1"], dtype=jnp.float64),
        "x2": jnp.asarray(raw["x2"], dtype=jnp.float64),
    }
    return SeedsModel(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("seeds_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    i = data["I"]
    assert harness.param_unc_names() == ["alpha0", "alpha1", "alpha12", "alpha2", "tau"] + [
        f"b.{k}" for k in range(1, i + 1)
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("seeds_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-5, atol_hessian=1e-2)
