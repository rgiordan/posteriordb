"""Unit tests for posteriordb_jax.stan_bijectors.

Two layers of checking, per the plan:
1. Stan-free: forward/inverse round trips, and the analytic
   forward_log_det_jacobian cross-checked against an independent
   finite-difference Jacobian.
2. BridgeStan-backed: forward map and forward_log_det_jacobian cross-checked
   against a live compiled Stan model's param_constrain and
   log_density(jacobian=True/False), at several K and several random points.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import posteriordb_jax  # noqa: F401  (enables x64)
from posteriordb_jax.stan_bijectors import (
    StanCholeskyFactorCorr,
    StanCholeskyFactorCov,
    StanSimplex,
)

from conftest import STAN_FIXTURES, bridgestan_model

jax.config.update("jax_enable_x64", True)


def _finite_diff_logdet(chart_fn, x, eps=1e-6):
    n = x.shape[-1]
    j = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        j[:, i] = (np.asarray(chart_fn(x + dx)) - np.asarray(chart_fn(x - dx))) / (2 * eps)
    _, logdet = np.linalg.slogdet(j)
    return logdet


# ---------------------------------------------------------------------------
# StanSimplex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_simplex_round_trip(rng, k):
    b = StanSimplex(k)
    n = k - 1
    x = jnp.asarray(rng.normal(size=(n,)))
    y = b.forward(x)
    assert y.shape == (k,)
    np.testing.assert_allclose(np.sum(y), 1.0, atol=1e-10)
    assert np.all(np.asarray(y) >= 0)
    x_back = b.inverse(y)
    np.testing.assert_allclose(x_back, x, atol=1e-8)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_simplex_jacobian_vs_finite_diff(rng, k):
    b = StanSimplex(k)
    n = k - 1
    x = np.asarray(rng.normal(size=(n,)))

    def chart(x1d):
        return np.asarray(b.forward(jnp.asarray(x1d)))[:-1]

    fd_logdet = _finite_diff_logdet(chart, x)
    analytic = float(b.forward_log_det_jacobian(jnp.asarray(x), event_ndims=1))
    np.testing.assert_allclose(analytic, fd_logdet, atol=1e-5)


def test_simplex_batching(rng):
    k = 4
    b = StanSimplex(k)
    x = jnp.asarray(rng.normal(size=(5, k - 1)))
    y = b.forward(x)
    assert y.shape == (5, k)
    logdet = b.forward_log_det_jacobian(x, event_ndims=1)
    assert logdet.shape == (5,)
    for i in range(5):
        expected = b.forward_log_det_jacobian(x[i], event_ndims=1)
        np.testing.assert_allclose(logdet[i], expected, atol=1e-10)


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_simplex_matches_bridgestan(rng, k):
    stan_file = STAN_FIXTURES / "probe_simplex.stan"
    model = bridgestan_model(stan_file, {"K": k})
    b = StanSimplex(k)
    for _ in range(5):
        x = rng.normal(size=k - 1)
        x_stan_order = x  # BridgeStan's param_unc order matches declaration order
        constrained_stan = np.asarray(model.param_constrain(x_stan_order))
        constrained_jax = np.asarray(b.forward(jnp.asarray(x)))
        np.testing.assert_allclose(constrained_jax, constrained_stan, atol=1e-8)

        lp_true = model.log_density(x_stan_order, propto=False, jacobian=True)
        lp_false = model.log_density(x_stan_order, propto=False, jacobian=False)
        stan_logdet = lp_true - lp_false
        jax_logdet = float(b.forward_log_det_jacobian(jnp.asarray(x), event_ndims=1))
        np.testing.assert_allclose(jax_logdet, stan_logdet, atol=1e-6)

        x_back = np.asarray(b.inverse(jnp.asarray(constrained_stan)))
        np.testing.assert_allclose(x_back, x, atol=1e-6)


# ---------------------------------------------------------------------------
# StanCholeskyFactorCorr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_chol_corr_round_trip(rng, k):
    b = StanCholeskyFactorCorr(k)
    m = k * (k - 1) // 2
    x = jnp.asarray(rng.normal(size=(m,)))
    y = b.forward(x)
    assert y.shape == (k, k)
    x_back = b.inverse(y)
    np.testing.assert_allclose(x_back, x, atol=1e-7)


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_chol_corr_jacobian_vs_finite_diff(rng, k):
    b = StanCholeskyFactorCorr(k)
    m = k * (k - 1) // 2
    if m == 0:
        pytest.skip("no free parameters")
    x = np.asarray(rng.normal(size=(m,)))
    rows, cols = np.tril_indices(k, k=-1)

    def chart(x1d):
        l = np.asarray(b.forward(jnp.asarray(x1d)))
        return l[rows, cols]

    fd_logdet = _finite_diff_logdet(chart, x)
    analytic = float(b.forward_log_det_jacobian(jnp.asarray(x), event_ndims=1))
    np.testing.assert_allclose(analytic, fd_logdet, atol=1e-5)


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_chol_corr_matches_bridgestan(rng, k):
    stan_file = STAN_FIXTURES / "probe_corr.stan"
    model = bridgestan_model(stan_file, {"K": k})
    b = StanCholeskyFactorCorr(k)
    m = k * (k - 1) // 2
    if m == 0:
        pytest.skip("no free parameters")
    for _ in range(5):
        x = rng.normal(size=m)
        l_stan_full = np.asarray(model.param_constrain(x)).reshape(k, k, order="F")
        l_jax_full = np.asarray(b.forward(jnp.asarray(x)))
        np.testing.assert_allclose(l_jax_full, l_stan_full, atol=1e-8)

        lp_true = model.log_density(x, propto=False, jacobian=True)
        lp_false = model.log_density(x, propto=False, jacobian=False)
        stan_logdet = lp_true - lp_false
        jax_logdet = float(b.forward_log_det_jacobian(jnp.asarray(x), event_ndims=1))
        np.testing.assert_allclose(jax_logdet, stan_logdet, atol=1e-6)

        x_back = np.asarray(b.inverse(jnp.asarray(l_stan_full)))
        np.testing.assert_allclose(x_back, x, atol=1e-6)


# ---------------------------------------------------------------------------
# StanCholeskyFactorCov
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_chol_cov_round_trip(rng, k):
    b = StanCholeskyFactorCov(k)
    m = k * (k + 1) // 2
    x = jnp.asarray(rng.normal(size=(m,)) * 0.5)
    y = b.forward(x)
    assert y.shape == (k, k)
    x_back = b.inverse(y)
    np.testing.assert_allclose(x_back, x, atol=1e-8)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_chol_cov_jacobian_vs_finite_diff(rng, k):
    b = StanCholeskyFactorCov(k)
    m = k * (k + 1) // 2
    x = np.asarray(rng.normal(size=(m,)) * 0.3)
    rows, cols = np.tril_indices(k)

    def chart(x1d):
        l = np.asarray(b.forward(jnp.asarray(x1d)))
        return l[rows, cols]

    fd_logdet = _finite_diff_logdet(chart, x)
    analytic = float(b.forward_log_det_jacobian(jnp.asarray(x), event_ndims=1))
    np.testing.assert_allclose(analytic, fd_logdet, atol=1e-5)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_chol_cov_matches_bridgestan(rng, k):
    stan_file = STAN_FIXTURES / "probe_cholesky_cov.stan"
    model = bridgestan_model(stan_file, {"K": k})
    b = StanCholeskyFactorCov(k)
    m = k * (k + 1) // 2
    for _ in range(5):
        x = rng.normal(size=m) * 0.5
        l_stan_full = np.asarray(model.param_constrain(x)).reshape(k, k, order="F")
        l_jax_full = np.asarray(b.forward(jnp.asarray(x)))
        np.testing.assert_allclose(l_jax_full, l_stan_full, atol=1e-7)

        lp_true = model.log_density(x, propto=False, jacobian=True)
        lp_false = model.log_density(x, propto=False, jacobian=False)
        stan_logdet = lp_true - lp_false
        jax_logdet = float(b.forward_log_det_jacobian(jnp.asarray(x), event_ndims=1))
        np.testing.assert_allclose(jax_logdet, stan_logdet, atol=1e-6)

        x_back = np.asarray(b.inverse(jnp.asarray(l_stan_full)))
        np.testing.assert_allclose(x_back, x, atol=1e-6)


# ---------------------------------------------------------------------------
# Composed chains: cov_matrix (works); corr_matrix (does NOT -- see below)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k,expect_match", [(2, True), (3, True), (4, False)])
def test_corr_matrix_cholesky_outer_product_boundary(rng, k, expect_match):
    """Documents a real (initially surprising) finding: Stan's `corr_matrix[K]`
    transform is NOT, in general, `cholesky_factor_corr` followed by
    L @ L.T, despite both using K*(K-1)/2 unconstrained inputs and
    cholesky_factor_corr's own transform composing correctly with
    CholeskyOuterProduct for exactly that purpose (see
    test_chol_corr_matches_bridgestan). Confirmed by direct comparison
    against BridgeStan during development: the two constructions agree for
    K<=3 (coincidentally -- there's little room for an internal-ordering
    difference to matter at these sizes) but diverge starting at K=4,
    consistent with `corr_matrix` using a genuinely different internal
    construction (e.g. an "onion"/growth-by-one-row method) for larger K.

    No model in this repo has a *free* `corr_matrix` parameter (only as a
    derived `transformed parameters` quantity, e.g. in
    dogs_nonhierarchical, always at K=2), so a fully general Stan-exact
    `corr_matrix` bijector is deferred rather than implemented here -- this
    test just pins down and documents the boundary so it isn't silently
    rediscovered later.
    """
    import tensorflow_probability.substrates.jax as tfp

    tfb = tfp.bijectors
    stan_file = STAN_FIXTURES / "probe_corr_matrix.stan"
    model = bridgestan_model(stan_file, {"K": k})
    b = tfb.Chain([tfb.CholeskyOuterProduct(), StanCholeskyFactorCorr(k)])
    m = k * (k - 1) // 2
    x = rng.normal(size=m)
    c_stan = np.asarray(model.param_constrain(x)).reshape(k, k, order="F")
    c_jax = np.asarray(b.forward(jnp.asarray(x)))
    matches = np.allclose(c_jax, c_stan, atol=1e-4)
    assert matches == expect_match


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_cov_matrix_composed_matches_bridgestan(rng, k):
    import tensorflow_probability.substrates.jax as tfp

    tfb = tfp.bijectors
    stan_file = STAN_FIXTURES / "probe_cov_matrix.stan"
    model = bridgestan_model(stan_file, {"K": k})
    b = tfb.Chain([tfb.CholeskyOuterProduct(), StanCholeskyFactorCov(k)])
    m = k * (k + 1) // 2
    for _ in range(5):
        x = rng.normal(size=m) * 0.3
        c_stan = np.asarray(model.param_constrain(x)).reshape(k, k, order="F")
        c_jax = np.asarray(b.forward(jnp.asarray(x)))
        np.testing.assert_allclose(c_jax, c_stan, atol=1e-5)

        lp_true = model.log_density(x, propto=False, jacobian=True)
        lp_false = model.log_density(x, propto=False, jacobian=False)
        stan_logdet = lp_true - lp_false
        jax_logdet = float(b.forward_log_det_jacobian(jnp.asarray(x), event_ndims=1))
        np.testing.assert_allclose(jax_logdet, stan_logdet, atol=1e-4)

        x_back = np.asarray(b.inverse(jnp.asarray(c_stan)))
        np.testing.assert_allclose(x_back, x, atol=1e-5)
