import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.surgical_model import SurgicalModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("surgical_data")
    data = {"N": raw["N"], "r": raw["r"], "n": raw["n"]}
    return SurgicalModel(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("surgical_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    n = data["N"]
    assert harness.param_unc_names() == ["mu", "sigmasq"] + [f"b.{k}" for k in range(1, n + 1)]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("surgical_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-5, atol_hessian=1e-2)
