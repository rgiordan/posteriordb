import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.GLMM1_model import GLMM1Model
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("GLMM_data")
    data = {
        "nobs": raw["nobs"],
        "nmis": raw["nmis"],
        "nyear": raw["nyear"],
        "nsite": raw["nsite"],
        "obs": raw["obs"],
        "obsyear": raw["obsyear"],
        "obssite": raw["obssite"],
        "misyear": raw["misyear"],
        "missite": raw["missite"],
    }
    return GLMM1Model(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("GLMM1_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    nsite = data["nsite"]
    assert harness.param_unc_names() == [f"alpha.{i}" for i in range(1, nsite + 1)] + [
        "mu_alpha",
        "sd_alpha",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("GLMM1_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-4, atol_hessian=1.0)
