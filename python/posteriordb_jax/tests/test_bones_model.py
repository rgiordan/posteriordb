import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.bones_model import BonesModel
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("bones_data")
    data = {
        "nChild": raw["nChild"],
        "nInd": raw["nInd"],
        "gamma": raw["gamma"],
        "delta": raw["delta"],
        "ncat": raw["ncat"],
        "grade": raw["grade"],
    }
    return BonesModel(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("bones_model"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    n_child = data["nChild"]
    assert harness.param_unc_names() == [f"theta.{i}" for i in range(1, n_child + 1)]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("bones_model"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=1.0)
    # Gradient/Hessian errors are at machine epsilon (~1e-14 / ~1e-9); the
    # log-density error (~1.9e-6) is summed float64 accumulation noise over
    # ~420 `log(p[i,j,grade[i,j]])` terms (13 children x 34 indicators, minus
    # 20 missing), not a formula bug -- see README's note on Hessian/large-N
    # tolerances for the same pattern in other models.
    report.assert_within(atol=1e-8, atol_log_density=5e-6, atol_hessian=1e-7)
