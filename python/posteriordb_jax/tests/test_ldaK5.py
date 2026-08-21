import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.ldaK5 import LdaK5
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("prideprejudice_chapter")
    data = {
        "V": raw["V"],
        "M": raw["M"],
        "N": raw["N"],
        "w": raw["w"],
        "doc": raw["doc"],
        "alpha": jnp.asarray(raw["alpha"], dtype=jnp.float64),
        "beta": jnp.asarray(raw["beta"], dtype=jnp.float64),
    }
    return LdaK5(data), data


def test_free_dim_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("ldaK5"), data)
    # See test_ldaK2.py -- param_unc_names() content isn't meaningful here.
    assert harness.param_unc_num() == model.model_spec.free_dim


def test_matches_bridgestan_log_density_and_grad():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("ldaK5"), data)
    # free_dim = M*(K-1) + K*(V-1) = 61*4 + 5*1494 = 7714 -- far too large
    # for a Hessian comparison; see test_ldaK2.py's comment.
    report = assert_matches_stan(model, harness, n_synthetic=3, seed=0, check_hessian=False)
    report.assert_within(atol=1e-3)
