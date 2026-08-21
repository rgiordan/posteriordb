import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.logistic_regression_rhs import LogisticRegressionRhs
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("prostate")
    data = {
        "n": raw["n"],
        "d": raw["d"],
        "y": raw["y"],
        "x": jnp.asarray(raw["x"], dtype=jnp.float64),
        "scale_icept": raw["scale_icept"],
        "scale_global": raw["scale_global"],
        "nu_global": raw["nu_global"],
        "nu_local": raw["nu_local"],
        "slab_scale": raw["slab_scale"],
        "slab_df": raw["slab_df"],
    }
    return LogisticRegressionRhs(data), data


def test_free_dim_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("logistic_regression_rhs"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim


def test_matches_bridgestan_log_density_and_grad():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("logistic_regression_rhs"), data)
    # d=5966 -> free_dim ~= 11935; a Hessian comparison at this size is
    # infeasible (O(free_dim^2) jax.hessian passes). check_hessian=False,
    # small n_synthetic -- same reasoning as test_ldaK2.py/test_ldaK5.py.
    report = assert_matches_stan(model, harness, n_synthetic=3, seed=0, check_hessian=False, scale=0.1)
    report.assert_within(atol=1e-3)
