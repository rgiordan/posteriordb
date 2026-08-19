import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.dogs_nonhierarchical import DogsNonhierarchical
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("dogs")
    data = {
        "n_dogs": raw["n_dogs"],
        "n_trials": raw["n_trials"],
        "y": jnp.asarray(raw["y"], dtype=jnp.float64),
    }
    harness_data = {"n_dogs": raw["n_dogs"], "n_trials": raw["n_trials"], "y": raw["y"]}
    return DogsNonhierarchical(data), data, harness_data


def test_param_specs_order_and_free_dim_matches_stan():
    model, data, harness_data = _load_model()
    harness = StanComparisonHarness(stan_file_path("dogs_nonhierarchical"), harness_data)
    assert harness.param_unc_num() == model.model_spec.free_dim == 65
    names = harness.param_unc_names()
    assert names[:2] == ["mu_logit_ab.1", "mu_logit_ab.2"]
    assert names[2:4] == ["sigma_logit_ab.1", "sigma_logit_ab.2"]
    assert names[4] == "L_logit_ab.1"
    assert len(names) == 65


def test_matches_bridgestan_log_density_and_grad():
    model, data, harness_data = _load_model()
    harness = StanComparisonHarness(stan_file_path("dogs_nonhierarchical"), harness_data)
    # No Hessian here: 65x65 finite/analytic Hessian via BridgeStan + JAX for
    # a model with a length-750 bernoulli likelihood is unnecessarily slow
    # for a routine test; log density + full 65-dim gradient already pins
    # down correctness of every block (including the previously-"skipped"
    # cholesky_factor_corr block, now fully validated like everything else).
    report = assert_matches_stan(model, harness, n_synthetic=8, seed=0, check_hessian=False, scale=0.3)
    # log_density gets a slightly looser bound: TFP's CholeskyLKJ and Stan's
    # lkj_corr_cholesky_lpdf differ by a tiny, L-independent normalizing-
    # constant precision difference (~1.4e-7, confirmed during development,
    # gradient-invariant since it's additive-constant). Gradient (matched to
    # 0.0 here) and constrained values (matched to ~2e-16) are the real
    # correctness signal.
    report.assert_within(atol=1e-6, atol_log_density=2e-6)


def test_matches_bridgestan_hessian_single_draw():
    """One draw, with the Hessian check enabled -- the expensive check, kept
    to a single draw rather than the full n_synthetic sweep above."""
    model, data, harness_data = _load_model()
    harness = StanComparisonHarness(stan_file_path("dogs_nonhierarchical"), harness_data)
    report = assert_matches_stan(model, harness, n_synthetic=1, seed=1, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-6, atol_hessian=1e-3, atol_log_density=2e-6)
