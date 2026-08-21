import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.lsat_model import LsatModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("lsat_data")
    data = {
        "response": raw["response"],
        "culm": raw["culm"],
        "N": raw["N"],
        "R": raw["R"],
        "T": raw["T"],
    }
    return LsatModel(data), data


def test_free_dim_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("lsat_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim


def test_matches_bridgestan_log_density_and_grad():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("lsat_model"), data)
    # N=1000 -> free_dim ~= 1006; Hessian comparison skipped for the same
    # size reason as test_logistic_regression_rhs.py.
    report = assert_matches_stan(model, harness, n_synthetic=10, seed=0, check_hessian=False, scale=0.3)
    report.assert_within(atol=1e-3)
