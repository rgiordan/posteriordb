"""New-style (modelgami) models: the contract every model must meet, plus
agreement with the old `StatisticalModel` translations and with Stan."""

import jax
import jax.numpy as jnp
import numpy as np
import paragami as pg
import pytest

import posteriordb_jax  # noqa: F401  (enables float64)
from posteriordb_jax import catalog
from posteriordb_jax.io import load_data, load_posterior_info, stan_file_path
from posteriordb_jax.models import MODEL_REGISTRY
from posteriordb_jax.testing import StanComparisonHarness, assert_model_matches_stan
from modelgami.testing import check_model_contract

NAMES = catalog.list_models()
# Models translated after the old StatisticalModel versions were retired have
# no old translation to compare against; BridgeStan still checks them.
OLD_NAMES = [name for name in NAMES if catalog.find_module(name).STAN_MODEL in MODEL_REGISTRY]

# Per-posterior exceptions to the BridgeStan comparison:
# - Stan's bernoulli_logit_lpmf and bernoulli_logit_glm_lpmf switch to cutoff
#   approximations for |logit| > 20 and, above the cutoff, return the partial
#   -exp(-logit) instead of the exact exp(-logit) / (1 + exp(-logit)), an error
#   of up to ~4e-9 per term, so their gradients get a looser bound. The log
#   densities still agree to 1e-11. nn_rbm1bJ100's gradient entries are sums
#   over the whole MNIST training set through 100 hidden units, and differ
#   from Stan's by ~1e-9 from summation order alone.
# - For the largest models the Hessian is skipped: jax.hessian is slow at
#   thousands of free parameters, and BridgeStan's Hessian is finite
#   differences of gradients, which is too noisy there to compare. The same
#   finite differences are too noisy for GLMM_Poisson_model, whose cubic
#   trend gives Hessian entries of order 1e11.
STAN_GRAD_ATOL = {
    "irt_2pl-irt_2pl": 1e-7,
    "sat-hier_2pl": 1e-7,
    "ovarian-logistic_regression_rhs": 1e-7,
    "prostate-logistic_regression_rhs": 1e-7,
    "fims_Aus_Jpn_irt-2pl_latent_reg_irt": 1e-7,
    "mnist-nn_rbm1bJ100": 1e-8,
}
# Looser Hessian bounds where Hessian entries reach 1e4 to 4e5 (the hidden
# Markov models' forward recursions; up to 2e8 for accel_gp and hierarchical_gp
# at some synthetic points): BridgeStan's finite-difference Hessian
# is then off by up to ~1e-5 of the largest entry, and the bounds are set
# from that relative error (checked against jax.hessian at random points).
STAN_HESSIAN_ATOL = {
    "bball_drive_event_0-hmm_drive_0": 0.1,
    "bball_drive_event_1-hmm_drive_1": 2.0,
    "hmm_gaussian_simulated-hmm_gaussian": 2.0,
    "iohmm_reg_simulated-iohmm_reg": 0.5,
    "mcycle_gp-accel_gp": 1e-3,
    "state_wide_presidential_votes-hierarchical_gp": 1e-2,
}
# Smaller synthetic free vectors where a standard normal one often lands where
# Stan's density is -inf: GLMM_Poisson_model's beta2 is bounded to (-10, 20)
# but has a uniform(-10, 10) prior.
SYNTHETIC_SCALE = {"GLMM_Poisson_data-GLMM_Poisson_model": 0.3}
# The old translations of these give nan gradients: p is exactly 1 on each
# dog's first trial, and their Bernoulli log density differentiates log(1 - p)
# there. The new versions' gradients match Stan (test_matches_stan).
OLD_GRADIENT_NAN = {"dogs-dogs_hierarchical", "dogs-dogs_nonhierarchical"}
# Stan's own values are inaccurate here: kronecker_gp takes eigendecompositions
# of two matrices with nearly repeated eigenvalues (the kernel matrix's sit at
# the 1e-5 jitter, and at random points the correlation matrix is nearly
# singular), so Stan's log density is off by ~1e-8 relative (up to 3e-6) and
# its eigenvector derivatives by O(1). The catalog version uses a Cholesky
# factor of the full covariance instead, which agrees to ~1e-11 with a second,
# independent evaluation (one Cholesky per eigenvalue of the correlation
# matrix). Its log density is compared with Stan's at a looser bound, and its
# gradient against finite differences.
STAN_INACCURATE = {"synthetic_grid_RBF_kernels-kronecker_gp"}
NO_HESSIAN = {"GLMM_Poisson_data-GLMM_Poisson_model", "ovarian-logistic_regression_rhs", "prostate-logistic_regression_rhs",
              "traffic_accident_nyc-bym2_offset_only", "prideprejudice_chapter-ldaK5",
              "prideprejudice_paragraph-ldaK5", "three_docs1200-ldaK2", "three_men1-ldaK2",
              "three_men2-ldaK2", "three_men3-ldaK2", *STAN_INACCURATE, "mnist_100-nn_rbm1bJ10", "mnist-nn_rbm1bJ100"}


@pytest.mark.parametrize("name", NAMES)
def test_contract(name):
    check_model_contract(catalog.load(name))


@pytest.mark.parametrize("name", OLD_NAMES)
def test_matches_old_translation(name):
    """Same free vector, same log density and gradient as the existing
    (BridgeStan-verified) StatisticalModel version. The tolerance allows
    TFP's small constant offsets in some log densities (e.g. Cauchy, Beta, or
    a normal summed over the ~12,000 radon houses) and in gradients of
    the vague gamma and inverse-gamma priors (e.g. gamma(1e-4, 1e-4)), which
    the new versions don't have: they match Stan to ~1e-14."""
    model = catalog.load(name)
    mod = catalog.find_module(name)
    raw = load_data(load_posterior_info(name)["data_name"])
    # Lists become float arrays, except integer fields the new model either
    # casts to int32 (indices and 0/1 outcomes) or doesn't take at all
    # (raw encodings it expands itself, e.g. lsat's run lengths).
    data_fields = model.data if isinstance(model.data, dict) else {}
    int_fields = {k for k, v in raw.items()
                  if isinstance(v, list) and (jnp.asarray(data_fields[k]).dtype == jnp.int32 if k in data_fields
                                              else jnp.issubdtype(jnp.asarray(v).dtype, jnp.integer))}
    data = {k: (jnp.asarray(v, dtype=jnp.int32 if k in int_fields else jnp.float64) if isinstance(v, list) else v)
            for k, v in raw.items()}
    old = MODEL_REGISTRY[mod.STAN_MODEL](data)
    f = model.eval_free_log_joint
    for k in jax.random.split(jax.random.PRNGKey(0), 5):
        x = jax.random.normal(k, (model.params_pattern.get_free_length(),))
        np.testing.assert_allclose(f(x), old.log_prob_unconstrained(x), rtol=1e-8, atol=1e-5)
        if name not in OLD_GRADIENT_NAN:
            np.testing.assert_allclose(jax.grad(f)(x), old.grad_unconstrained(x), rtol=1e-7, atol=1e-6)


@pytest.mark.stan
@pytest.mark.parametrize("name", NAMES)
def test_matches_stan(name):
    model = catalog.load(name)
    raw = load_data(load_posterior_info(name)["data_name"])
    harness = StanComparisonHarness(stan_file_path(catalog.find_module(name).STAN_MODEL), raw)
    report = assert_model_matches_stan(model, harness, n_synthetic=10, check_hessian=name not in NO_HESSIAN,
                                       scale=SYNTHETIC_SCALE.get(name, 1.0))
    if name in STAN_INACCURATE:
        assert report.max_abs_err_log_density < 1e-5 and report.max_abs_err_constrained_flat < 1e-9
        return
    # Scaled errors (see assert_model_matches_stan). BridgeStan's Hessian is
    # finite differences of gradients, hence the looser Hessian bound.
    report.assert_within(atol=STAN_GRAD_ATOL.get(name, 1e-9), atol_hessian=STAN_HESSIAN_ATOL.get(name, 1e-4), atol_log_density=1e-11)


@pytest.mark.parametrize("name", sorted(STAN_INACCURATE))
def test_gradient_matches_finite_differences(name):
    model = catalog.load(name)
    f = jax.jit(model.eval_free_log_joint)
    rng = np.random.default_rng(0)
    x = rng.normal(size=model.params_pattern.get_free_length())
    grad = np.asarray(jax.grad(f)(jnp.asarray(x)))
    step = 1e-5
    for i in rng.choice(x.size, size=10, replace=False):
        shift = step * np.eye(x.size)[i]
        fd = (float(f(jnp.asarray(x + shift))) - float(f(jnp.asarray(x - shift)))) / (2.0 * step)
        assert abs(fd - grad[i]) / (1.0 + abs(grad[i])) < 1e-7


def test_list_and_filter():
    assert "kidiq-kidscore_momiq" in catalog.list_models(exchangeable=True)
    assert "eight_schools-eight_schools_noncentered" not in catalog.list_models(exchangeable=True)
    with pytest.raises(KeyError):
        catalog.load("not-a-posterior")


def test_reference_draws():
    draws = catalog.load_reference_draws("kidiq-kidscore_momiq")
    assert draws["beta"].shape[-1] == 2 and draws["beta"].shape[:2] == draws["sigma"].shape
    idata = pg.folded_draws_to_arviz(catalog.load("kidiq-kidscore_momiq").params_pattern, draws)
    assert idata["posterior"]["beta"].shape[:2] == draws["sigma"].shape
