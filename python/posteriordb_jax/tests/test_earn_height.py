import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.earn_height import EarnHeight
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("earnings")
    data = {
        "N": raw["N"],
        "earn": jnp.asarray(raw["earn"], dtype=jnp.float64),
        "height": jnp.asarray(raw["height"], dtype=jnp.float64),
    }
    return EarnHeight(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("earn_height"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == ["beta.1", "beta.2", "sigma"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("earn_height"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    # `earn` is in raw dollars (O(1e4-1e5)) with no informative prior on
    # `beta`/`sigma`, so a standard-normal *unconstrained* synthetic draw
    # implies residuals of O(1e4) at sigma~1 -- log density/grad/Hessian
    # values themselves run up to O(1e11-1e13), so float64's ~1e-16
    # relative precision alone produces absolute errors in the 1e-3 to
    # 1e3 range. `max_abs_err_constrained_flat` (checked separately, not
    # loosened) stays at float64 machine epsilon, confirming the
    # constrain/unconstrain transform itself is exact -- this is pure
    # magnitude-driven float noise, not a formula bug. See
    # `ComparisonReport.assert_within`'s docstring.
    report.assert_within(atol=0.05, atol_hessian=3000.0)
