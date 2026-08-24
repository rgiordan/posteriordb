import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.Mh_model import MhModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("Mh_data")
    data = {"y": raw["y"], "M": raw["M"], "T": raw["T"]}
    return MhModel(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("Mh_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    m = data["M"]
    assert harness.param_unc_names() == ["omega", "mean_p", "sigma"] + [
        f"eps_raw.{i}" for i in range(1, m + 1)
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("Mh_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=10, seed=0, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-4, atol_hessian=1.0)
