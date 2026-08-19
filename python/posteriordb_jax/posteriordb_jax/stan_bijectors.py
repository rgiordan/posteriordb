"""Custom TFP-compatible bijectors that replicate Stan's own constraining
transforms exactly, for constraint types where no off-the-shelf
`tensorflow_probability.substrates.jax` bijector is formula-identical to
Stan's.

Every forward map and every `forward_log_det_jacobian` formula below was
confirmed against a live BridgeStan-compiled Stan model during development
(not just derived from documentation): the forward map was checked to
reproduce `StanModel.param_constrain` bit-for-bit, and the
`forward_log_det_jacobian` was checked against the difference between
`StanModel.log_density(..., jacobian=True)` and `log_density(...,
jacobian=False)`, both directly and (where the formula was hand-derived)
via an independent finite-difference Jacobian. See
`tests/test_stan_bijectors.py` for the reusable version of those checks.

Two things turned out to differ from what a naive reading of general Stan
documentation/off-the-shelf TFP bijectors would suggest -- worth recording
here since they're easy to get wrong silently:

- **Simplex**: as of Stan 2.37, the simplex transform is no longer the
  "stick-breaking" logistic transform (still documented in some older Stan
  references) -- it is a "sum-to-zero vector" linear transform followed by a
  softmax. `StanSimplex` targets the current (>=2.37) transform.
- **Cholesky factors** (`cholesky_factor_cov`/`cholesky_factor_corr`): TFP's
  own `tfb.FillScaleTriL` (built on `tfb.FillTriangular`) fills a triangular
  matrix from a flat vector using an internal "zigzag" element order chosen
  for compute efficiency, NOT Stan's plain row-major order. Using it
  directly silently produces a *different, still-valid* Cholesky
  parameterization -- support-equivalent, but numerically different from
  Stan's own unconstrained space. `StanCholeskyFactorCov` below fills in
  Stan's actual row-major order instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors


def _sum_to_zero_matrix(n: int) -> np.ndarray:
    """The (n+1, n) matrix A such that z = A @ y is Stan's sum-to-zero
    transform of an unconstrained y in R^n to a sum-to-zero z in R^(n+1).

    Built directly from the Stan Reference Manual's formula:
        z_1     = sum_{k=1}^{n} y_k / sqrt(k*(k+1))
        z_{k+1} = sum_{i=k+1}^{n} y_i / sqrt(i*(i+1)) - k*y_k / sqrt(k*(k+1)),  k=1..n

    Computed once (n is static, tied to a parameter's declared shape) with
    plain numpy -- never traced by JAX.
    """
    if n == 0:
        return np.zeros((1, 0))
    c = np.array([1.0 / np.sqrt(k * (k + 1)) for k in range(1, n + 1)])  # c[k-1] = c_k
    a = np.zeros((n + 1, n))
    a[0, :] = c
    for k in range(1, n + 1):
        row = k  # z_{k+1} -> row index k (0-indexed)
        if k < n:
            a[row, k:] = c[k:]  # sum_{i=k+1}^n y_i c_i  (0-indexed cols k..n-1)
        a[row, k - 1] = -k * c[k - 1]  # -k*y_k*c_k
    return a


class StanSimplex(tfb.Bijector):
    """Stan's simplex transform (Stan >=2.37): unconstrained y in R^(K-1) ->
    sum-to-zero z in R^K (linear) -> simplex x = softmax(z) in R^K.

    The forward log-det-Jacobian is computed via autodiff on the map
    y -> x[..., :-1] (the first K-1 simplex coordinates, treating x_K as the
    redundant/determined one) rather than a hand-derived closed form -- this
    was empirically confirmed to reproduce BridgeStan's own
    log_density(jacobian=True) convention exactly (to ~1e-9), and sidesteps
    the risk of a transcription error in a closed-form formula for a
    genuinely nontrivial change-of-variables.
    """

    def __init__(self, k: int, validate_args: bool = False, name: str = "stan_simplex"):
        self.k = k
        self._n = k - 1
        self._a = jnp.asarray(_sum_to_zero_matrix(self._n))  # shape (K, K-1)
        self._a_pinv = jnp.asarray(np.linalg.pinv(_sum_to_zero_matrix(self._n)))  # (K-1, K)
        super().__init__(
            forward_min_event_ndims=1,
            validate_args=validate_args,
            name=name,
        )

    def _forward(self, x):
        z = jnp.einsum("...n,kn->...k", x, self._a)
        return jax.nn.softmax(z, axis=-1)

    def _inverse(self, y):
        logy = jnp.log(y)
        z = logy - jnp.mean(logy, axis=-1, keepdims=True)
        return jnp.einsum("...k,nk->...n", z, self._a_pinv)

    def _forward_log_det_jacobian(self, x):
        def chart(x1d):
            return self._forward(x1d)[..., :-1]

        batch_shape = x.shape[:-1]
        x_flat = x.reshape((-1, x.shape[-1])) if batch_shape else x[None, :]

        def per_sample(x1d):
            j = jax.jacfwd(chart)(x1d)
            _, logdet = jnp.linalg.slogdet(j)
            return logdet

        logdets = jax.vmap(per_sample)(x_flat)
        return logdets.reshape(batch_shape) if batch_shape else logdets[0]


def _cholesky_corr_forward(y: jnp.ndarray, k: int) -> jnp.ndarray:
    """y: (..., K*(K-1)/2) unconstrained (row-major lower-triangle order,
    i.e. entries for row i=0..K-1, column j=0..i-1, in that nested order) ->
    L: (..., K, K), a valid Cholesky factor of a correlation matrix.
    """
    z = jnp.tanh(y)
    batch_shape = y.shape[:-1]
    l = jnp.zeros(batch_shape + (k, k), dtype=y.dtype)
    idx = 0
    for i in range(k):
        remaining = jnp.ones(batch_shape, dtype=y.dtype)
        for j in range(i):
            zij = z[..., idx]
            idx += 1
            lij = zij * jnp.sqrt(remaining)
            l = l.at[..., i, j].set(lij)
            remaining = remaining * (1 - zij**2)
        l = l.at[..., i, i].set(jnp.sqrt(remaining))
    return l


class StanCholeskyFactorCorr(tfb.Bijector):
    """Stan's `cholesky_factor_corr[K]` transform: unconstrained y in
    R^(K*(K-1)/2) -> canonical partial correlations z = tanh(y) -> a valid
    Cholesky factor L of a correlation matrix, built row by row:
        L[i,j] = z[i,j] * sqrt(1 - sum_{j'<j} L[i,j']^2),   j < i
        L[i,i] = sqrt(1 - sum_{j'<i} L[i,j']^2)

    Forward output is the full K*K matrix (zeros above the diagonal,
    diagonal filled in) -- matching Stan's own printed/constrained
    representation exactly (confirmed via BridgeStan's `param_names()`,
    which reports all K*K entries, not just the K*(K-1)/2 free ones) and
    making this bijector directly composable via `tfb.Chain` with
    `tfb.CholeskyOuterProduct()` for `corr_matrix`.

    The closed-form log-det-Jacobian below is the Stan Reference Manual's
    formula:
        log|det J| = -2 * sum_n log(cosh(y_n))
                     + 0.5 * sum_{i>j} log(1 - sum_{j'<j} L[i,j']^2)
    Confirmed against BridgeStan's log_density(jacobian=True), both
    directly and via an independent finite-difference check.

    NOTE on `corr_matrix`: despite using the same number of unconstrained
    inputs (K*(K-1)/2), Stan's `corr_matrix[K]` transform is NOT this
    bijector followed by `tfb.CholeskyOuterProduct()` -- confirmed by direct
    comparison against BridgeStan during development (they agree at K=2,
    where there's only one free parameter and no room for an ordering
    difference, but diverge for K>=3). No model in this repo has a *free*
    `corr_matrix` parameter, so a Stan-exact `corr_matrix` bijector is
    deferred; see `tests/test_stan_bijectors.py::
    test_corr_matrix_is_not_simply_cholesky_outer_product` for the pinned
    negative result.
    """

    def __init__(self, k: int, validate_args: bool = False, name: str = "stan_cholesky_factor_corr"):
        self.k = k
        self._rows, self._cols = np.tril_indices(k, k=-1)
        super().__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=2,
            validate_args=validate_args,
            name=name,
        )

    def _forward(self, x):
        return _cholesky_corr_forward(x, self.k)

    def _inverse(self, y):
        k = self.k
        batch_shape = y.shape[:-2]
        out = jnp.zeros(batch_shape + (k * (k - 1) // 2,), dtype=y.dtype)
        idx = 0
        for i in range(k):
            remaining = jnp.ones(batch_shape, dtype=y.dtype)
            for j in range(i):
                lij = y[..., i, j]
                zij = lij / jnp.sqrt(remaining)
                out = out.at[..., idx].set(jnp.arctanh(zij))
                idx += 1
                remaining = remaining * (1 - zij**2)
        return out

    def _forward_log_det_jacobian(self, x):
        k = self.k
        z = jnp.tanh(x)
        batch_shape = x.shape[:-1]
        logdet = -2.0 * jnp.sum(jnp.log(jnp.cosh(x)), axis=-1)
        idx = 0
        for i in range(k):
            remaining = jnp.ones(batch_shape, dtype=x.dtype)
            for j in range(i):
                zij = z[..., idx]
                idx += 1
                logdet = logdet + 0.5 * jnp.log(remaining)
                remaining = remaining * (1 - zij**2)
        return logdet


class StanCholeskyFactorCov(tfb.Bijector):
    """Stan's `cholesky_factor_cov[K]` transform (square case): unconstrained
    y in R^(K*(K+1)/2) -> lower-triangular K*K matrix L, filled in Stan's
    plain row-major order (row i=0..K-1, column j=0..i), with `exp` applied
    to the diagonal entries only:
        L[i,j] = y[i,j]        j < i
        L[i,i] = exp(y[i,i])

    NOTE: this is deliberately NOT `tfb.FillScaleTriL` -- that bijector
    fills the triangular matrix using TFP's internal `FillTriangular`
    "zigzag" element order (chosen for compute efficiency), which is a
    different, but equally valid, Cholesky parameterization from Stan's
    plain row-major fill. Using `FillScaleTriL` directly reproduces the
    right *support* but a numerically different unconstrained space --
    confirmed to disagree with BridgeStan during development, which is why
    this simple direct-fill bijector exists instead.

    `cov_matrix[K]` is `tfb.Chain([tfb.CholeskyOuterProduct(),
    StanCholeskyFactorCov(k)])`.
    """

    def __init__(self, k: int, validate_args: bool = False, name: str = "stan_cholesky_factor_cov"):
        self.k = k
        self._rows, self._cols = np.tril_indices(k)
        self._diag_positions = np.where(self._rows == self._cols)[0]
        super().__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=2,
            validate_args=validate_args,
            name=name,
        )

    def _forward(self, x):
        k = self.k
        batch_shape = x.shape[:-1]
        l = jnp.zeros(batch_shape + (k, k), dtype=x.dtype)
        l = l.at[..., self._rows, self._cols].set(x)
        diag = np.arange(k)
        l = l.at[..., diag, diag].set(jnp.exp(l[..., diag, diag]))
        return l

    def _inverse(self, y):
        flat = y[..., self._rows, self._cols]
        diag = self._diag_positions
        return flat.at[..., diag].set(jnp.log(flat[..., diag]))

    def _forward_log_det_jacobian(self, x):
        return jnp.sum(x[..., self._diag_positions], axis=-1)
