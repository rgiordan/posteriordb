import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.blr import Blr
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("sblrc")
    data = {
        "N": raw["N"],
        "D": raw["D"],
        "X": jnp.asarray(raw["X"], dtype=jnp.float64),
        "y": jnp.asarray(raw["y"], dtype=jnp.float64),
    }
    return Blr(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("blr"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == [
        "beta.1",
        "beta.2",
        "beta.3",
        "beta.4",
        "beta.5",
        "sigma",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("blr"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True)
    # Data covariate magnitudes here are O(1e2), so raw Hessian entries run
    # up to O(1e6-1e7) (second derivatives through 1/sigma^2 * X'X terms);
    # atol_hessian is loosened accordingly -- relative error stays ~1e-9,
    # log density/gradient/constrained-flat (still tight) confirm the
    # formula itself is correct. See `ComparisonReport.assert_within`'s
    # docstring.
    report.assert_within(atol=1e-5, atol_hessian=0.05)
