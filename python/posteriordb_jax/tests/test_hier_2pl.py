import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.hier_2pl import Hier2pl
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("sat")
    data = {
        "I": raw["I"],
        "J": raw["J"],
        "N": raw["N"],
        "ii": raw["ii"],
        "jj": raw["jj"],
        "y": raw["y"],
    }
    return Hier2pl(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("hier_2pl"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    i = data["I"]
    j = data["J"]
    assert harness.param_unc_names() == (
        [f"theta.{k}" for k in range(1, j + 1)]
        + [f"xi1.{k}" for k in range(1, i + 1)]
        + [f"xi2.{k}" for k in range(1, i + 1)]
        + ["mu.1", "mu.2", "tau.1", "tau.2", "L_Omega.1"]
    )


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("hier_2pl"), data)
    report = assert_matches_stan(model, harness, n_synthetic=10, seed=0, check_hessian=True, scale=0.2)
    report.assert_within(atol=1e-4, atol_hessian=1.0)
