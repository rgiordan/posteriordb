import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.irt_2pl import Irt2pl
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("irt_2pl")
    data = {"I": raw["I"], "J": raw["J"], "y": raw["y"]}
    return Irt2pl(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("irt_2pl"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    i = data["I"]
    j = data["J"]
    expected = (
        ["sigma_theta"]
        + [f"theta.{k}" for k in range(1, j + 1)]
        + ["sigma_a"]
        + [f"a.{k}" for k in range(1, i + 1)]
        + ["mu_b", "sigma_b"]
        + [f"b.{k}" for k in range(1, i + 1)]
    )
    assert harness.param_unc_names() == expected


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("irt_2pl"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-4, atol_hessian=1.0)
