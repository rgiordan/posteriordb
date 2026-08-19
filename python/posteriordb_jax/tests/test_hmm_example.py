import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.hmm_example import HmmExample
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("hmm_example")
    data = {"N": raw["N"], "K": raw["K"], "y": jnp.asarray(raw["y"], dtype=jnp.float64)}
    return HmmExample(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("hmm_example"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["theta1.1", "theta2.1", "mu.1", "mu.2"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("hmm_example"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.5)
    # Hessian gets a looser bound: it's a second derivative through the
    # SIMPLEX bijector's autodiff-based Jacobian, which is numerically
    # noisier in absolute terms (still ~1e-7 relative -- see
    # ComparisonReport.assert_within's docstring) than log-density/gradient.
    report.assert_within(atol=1e-6, atol_hessian=2e-4)
