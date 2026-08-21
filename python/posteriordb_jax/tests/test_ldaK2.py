import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.ldaK2 import LdaK2
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("three_men2")
    data = {
        "V": raw["V"],
        "M": raw["M"],
        "N": raw["N"],
        "w": raw["w"],
        "doc": raw["doc"],
    }
    return LdaK2(data), data


def test_free_dim_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("ldaK2"), data)
    # NOTE: don't assert on harness.param_unc_names() content here -- for
    # "array of simplex" parameters, BridgeStan's own param_unc_names()
    # reports a misleading order that doesn't match the actual free-vector
    # layout (see ParameterSpec.free_flatten_order's docstring and
    # tests/test_stan_bijectors.py::test_simplex_array_matches_bridgestan).
    # Only the count is a meaningful cross-check here.
    assert harness.param_unc_num() == model.model_spec.free_dim


def test_matches_bridgestan_log_density_and_grad():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("ldaK2"), data)
    # free_dim = M*(K-1) + K*(V-1) = 14*1 + 2*248 = 510 -- a full
    # jax.hessian/BridgeStan-Hessian comparison at this size is expensive
    # relative to the value it adds on top of the log-density/gradient/
    # constrained-flat checks (which already exercise every element of the
    # array-of-simplex machinery); check_hessian=False here, exercised
    # instead by the dedicated, much smaller StanSimplex/array unit tests
    # in test_stan_bijectors.py.
    report = assert_matches_stan(model, harness, n_synthetic=5, seed=0, check_hessian=False)
    report.assert_within(atol=1e-4)
