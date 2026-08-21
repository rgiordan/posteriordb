import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.kidscore_momhs import KidscoreMomhs
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("kidiq")
    data = {
        "N": raw["N"],
        "kid_score": jnp.asarray(raw["kid_score"], dtype=jnp.float64),
        "mom_hs": jnp.asarray(raw["mom_hs"], dtype=jnp.float64),
    }
    return KidscoreMomhs(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("kidscore_momhs"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["beta.1", "beta.2", "sigma"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("kidscore_momhs"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    report.assert_within(atol=1e-5, atol_hessian=5e-3)
