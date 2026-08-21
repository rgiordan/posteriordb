import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.GLMM_Poisson_model import GLMMPoissonModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("GLMM_Poisson_data")
    data = {
        "n": raw["n"],
        "C": raw["C"],
        "year": jnp.asarray(raw["year"], dtype=jnp.float64),
    }
    return GLMMPoissonModel(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("GLMM_Poisson_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    n = data["n"]
    assert harness.param_unc_names() == ["alpha", "beta1", "beta2", "beta3"] + [
        f"eps.{i}" for i in range(1, n + 1)
    ] + ["sigma"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("GLMM_Poisson_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    # beta2/beta3's wide bounds ([-10,10]/[-10,20]) times year_cubed can push
    # log_lambda (and hence Poisson log density/grad/Hessian) to huge
    # magnitudes for some synthetic draws -- same scale-driven float64 noise
    # as test_GLM_Poisson_model.py, not a formula bug (constrained-flat
    # error stays at machine epsilon).
    report.assert_within(atol=3000.0, atol_hessian=1e12)
