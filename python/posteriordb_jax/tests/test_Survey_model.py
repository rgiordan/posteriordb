import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.Survey_model import SurveyModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("Survey_data")
    data = {"nmax": raw["nmax"], "m": raw["m"], "k": raw["k"]}
    return SurveyModel(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("Survey_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["theta"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("Survey_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.5)
    report.assert_within(atol=1e-6)
