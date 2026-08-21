import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.Rate_4_model import Rate4Model
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("Rate_4_data")
    data = {"n": raw["n"], "k": raw["k"]}
    return Rate4Model(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("Rate_4_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["theta", "thetaprior"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("Rate_4_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    report.assert_within(atol=1e-5)
