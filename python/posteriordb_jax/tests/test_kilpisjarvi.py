import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.kilpisjarvi import Kilpisjarvi
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("kilpisjarvi_mod")
    data = {
        "N": raw["N"],
        "x": jnp.asarray(raw["x"], dtype=jnp.float64),
        "y": jnp.asarray(raw["y"], dtype=jnp.float64),
        "xpred": raw["xpred"],
        "pmualpha": raw["pmualpha"],
        "psalpha": raw["psalpha"],
        "pmubeta": raw["pmubeta"],
        "psbeta": raw["psbeta"],
    }
    return Kilpisjarvi(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("kilpisjarvi"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["alpha", "beta", "sigma"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("kilpisjarvi"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    # x is in raw units (O(3900-4000)); a standard-normal unconstrained beta
    # draw times x implies large residuals, pushing log density/grad/Hessian
    # magnitude up and float64 absolute noise with it -- same scale-driven
    # effect as test_earn_height.py, not a formula bug (constrained_flat
    # error stays at machine epsilon).
    report.assert_within(atol=1e-3, atol_hessian=1.0)
