"""Matrix patterns: Cholesky factors, covariance and correlation matrices.

With `bijector="stan"` (the default) the transforms are Stan's (Stan
Reference Manual, "Constraint transforms"):

- `CholeskyCovPattern(K)`: lower-triangular, positive diagonal. The free
  vector fills the lower triangle row by row, with `exp` on the diagonal.
- `PSDSymmetricMatrixPattern(K)`: a covariance matrix, `L @ L.T` with `L`
  from `CholeskyCovPattern` (Stan's `cov_matrix`).
- `CholeskyCorrPattern(K)`: Cholesky factor of a correlation matrix, from
  canonical partial correlations `tanh(u)` filled row by row (Stan's
  `cholesky_factor_corr`).
- `CorrMatrixPattern(K)`: a correlation matrix (Stan's `corr_matrix`). The
  same construction as `CholeskyCorrPattern` followed by `L @ L.T`, but
  Stan fills the partial correlations column by column and uses the
  Jacobian of the partial correlations -> correlation matrix map.

With `bijector="tfp"` they are TensorFlow Probability's:

- `CholeskyCovPattern`: `FillScaleTriL()` (TFP's fill order, softplus on the
  diagonal plus 1e-5).
- `PSDSymmetricMatrixPattern`: `CholeskyOuterProduct()` after `FillScaleTriL()`.
- `CholeskyCorrPattern`: `CorrelationCholesky()`.
- `CorrMatrixPattern`: `CholeskyOuterProduct()` after `CorrelationCholesky()`,
  with the log det TFP's `Chain` gives (the sum of the two).

The flat (constrained) vector is always the full K*K matrix, row-major.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .base import Pattern, check_bijector, check_shape, format_call

TFP_DIAG_SHIFT = 1e-5


def tfp_fill_triangular_indices(n: int) -> np.ndarray:
    """An (n, n) integer matrix P such that TFP's `fill_triangular(x)` has
    x[P[i, j]] at each lower-triangular (i, j); -1 above the diagonal."""
    if n == 0:
        return np.zeros((0, 0), dtype=int)
    x = np.arange(n * (n + 1) // 2)
    filled = np.concatenate([x[n:], x[::-1]]).reshape(n, n)
    return np.where(np.tril(np.ones((n, n), dtype=bool)), filled, -1)


def _log_det_outer_product(chol):
    """log |d(L L^T) / dL| over lower-triangular entries: K log 2 + sum_i (K - i) log L_ii."""
    k = chol.shape[-1]
    weights = jnp.arange(k, 0, -1, dtype=chol.dtype)
    return k * math.log(2.0) + jnp.sum(weights * jnp.log(jnp.diag(chol)))


class _SquareMatrixPattern(Pattern):
    def __init__(self, size: int, *, free_length: int, bijector: str):
        self.size = int(size)
        self.shape = (self.size, self.size)
        self.bijector = check_bijector(bijector)
        super().__init__(flat_length=self.size * self.size, free_length=free_length)

    def as_dict(self):
        return {"size": self.size, "bijector": self.bijector}

    def code_to_generate(self, *, module: str = "pg", indent: int = 4) -> str:
        kwargs = {"bijector": repr(self.bijector)} if self.bijector != "stan" else {}
        return format_call(module=module, name=type(self).__name__, args=[str(self.size)], kwargs=kwargs)

    def _flat_to_folded(self, flat_val):
        return jnp.reshape(flat_val, self.shape)

    def _folded_to_flat(self, folded_val):
        return jnp.reshape(jnp.asarray(folded_val), (-1,))

    def _names(self, free):
        if free:
            return super()._names(free=free)
        return [f"[{i},{j}]" for i, j in np.ndindex(*self.shape)]

    def _validate_lower_triangular(self, x):
        if np.any(np.triu(x, 1) != 0):
            return "must be lower triangular"
        if np.any(np.diag(x) <= 0):
            return "diagonal must be positive"
        return None


# -- covariance ----------------------------------------------------------------------

class CholeskyCovPattern(_SquareMatrixPattern):
    """A lower-triangular matrix with a positive diagonal."""

    def __init__(self, size: int, *, bijector: str = "stan"):
        super().__init__(size, free_length=size * (size + 1) // 2, bijector=bijector)
        rows, cols = np.tril_indices(self.size)
        if self.bijector == "tfp":
            index = tfp_fill_triangular_indices(self.size)
            order = np.argsort(index[rows, cols])  # free entry k sits at (rows[order[k]], cols[order[k]])
            rows, cols = rows[order], cols[order]
        self._rows, self._cols = rows, cols
        self._diag_pos = np.where(rows == cols)[0]

    def _positive(self, u):
        return jax.nn.softplus(u) + TFP_DIAG_SHIFT if self.bijector == "tfp" else jnp.exp(u)

    def _positive_inverse(self, d):
        if self.bijector == "tfp":
            d = d - TFP_DIAG_SHIFT
            return d + jnp.log(-jnp.expm1(-d))
        return jnp.log(d)

    def _free_to_folded(self, u):
        values = u.at[self._diag_pos].set(self._positive(u[self._diag_pos]))
        return jnp.zeros(self.shape, dtype=u.dtype).at[self._rows, self._cols].set(values)

    def _free_to_folded_log_det(self, u):
        diag = u[self._diag_pos]
        return jnp.sum(jax.nn.log_sigmoid(diag)) if self.bijector == "tfp" else jnp.sum(diag)

    def _folded_to_free(self, mat):
        values = jnp.asarray(mat)[self._rows, self._cols]
        return values.at[self._diag_pos].set(self._positive_inverse(values[self._diag_pos]))

    def _validate(self, x, validate_value):
        msg = check_shape(x, self.shape)
        if msg or not validate_value:
            return msg
        return self._validate_lower_triangular(np.asarray(x))


class PSDSymmetricMatrixPattern(_SquareMatrixPattern):
    """A symmetric positive definite matrix, `L @ L.T` for `L` from
    `CholeskyCovPattern` (Stan's `cov_matrix`)."""

    def __init__(self, size: int, *, bijector: str = "stan"):
        super().__init__(size, free_length=size * (size + 1) // 2, bijector=bijector)
        self._chol = CholeskyCovPattern(size, bijector=bijector)

    def _free_to_folded(self, u):
        chol = self._chol._free_to_folded(u)
        return chol @ chol.T

    def _free_to_folded_log_det(self, u):
        return self._chol._free_to_folded_log_det(u) + _log_det_outer_product(self._chol._free_to_folded(u))

    def _folded_to_free(self, mat):
        return self._chol._folded_to_free(jnp.linalg.cholesky(jnp.asarray(mat)))

    def _validate(self, x, validate_value):
        msg = check_shape(x, self.shape)
        if msg or not validate_value:
            return msg
        x = np.asarray(x)
        if not np.allclose(x, x.T, atol=1e-10, rtol=1e-10):
            return "must be symmetric"
        if np.any(np.linalg.eigvalsh(x) <= 0):
            return "must be positive definite"
        return None


# -- correlation ------------------------------------------------------------------

class _CorrelationTransform:
    """The Cholesky factor L of a correlation matrix from K(K-1)/2 free
    values, Stan's or TFP's way. `stan_order` is "row" (Stan's
    `cholesky_factor_corr`) or "column" (Stan's `corr_matrix`)."""

    def __init__(self, *, size: int, bijector: str, stan_order: str = "row"):
        self.size, self.bijector = size, bijector
        rows, cols = np.tril_indices(size, k=-1)
        if bijector == "tfp":
            index = tfp_fill_triangular_indices(size - 1)  # filled into L[1:, :-1]
            sub_rows, sub_cols = np.tril_indices(size - 1)
            order = np.argsort(index[sub_rows, sub_cols])
            rows, cols = sub_rows[order] + 1, sub_cols[order]
        elif stan_order == "column":
            order = np.lexsort((rows, cols))
            rows, cols = rows[order], cols[order]
        self.rows, self.cols = rows, cols

    def _strictly_lower(self, values):
        return jnp.zeros((self.size, self.size), dtype=values.dtype).at[self.rows, self.cols].set(values)

    def stan_partial_correlations(self, u):
        """(K, K) matrix of tanh(u) below the diagonal, and the remaining
        variance before each column: prod_{m<j} (1 - z[i, m]^2)."""
        z = self._strictly_lower(jnp.tanh(u))
        one_minus = 1.0 - z**2
        remaining = jnp.cumprod(jnp.concatenate([jnp.ones((self.size, 1), dtype=u.dtype), one_minus[:, :-1]],
                                                axis=1), axis=1)
        return z, remaining

    def forward(self, u):
        if self.bijector == "tfp":
            m = self._strictly_lower(u) + jnp.eye(self.size, dtype=u.dtype)
            return m / jnp.linalg.norm(m, axis=1, keepdims=True)
        z, remaining = self.stan_partial_correlations(u)
        return z * jnp.sqrt(remaining) + jnp.diag(jnp.sqrt(jnp.diag(remaining)))

    def inverse(self, chol):
        chol = jnp.asarray(chol)
        if self.bijector == "tfp":
            return (chol / jnp.diag(chol)[:, None])[self.rows, self.cols]
        squares = jnp.cumsum(chol**2, axis=1)
        before = jnp.concatenate([jnp.zeros((self.size, 1), dtype=chol.dtype), squares[:, :-1]], axis=1)
        z = chol / jnp.sqrt(1.0 - before)
        return jnp.arctanh(z[self.rows, self.cols])

    def tfp_log_det(self, chol):
        """TFP's `CorrelationCholesky` log det: sum_i (i + 2) log L_ii (0-indexed)."""
        weights = jnp.arange(2, self.size + 2, dtype=chol.dtype)
        return jnp.sum(weights * jnp.log(jnp.diag(chol)))


class CholeskyCorrPattern(_SquareMatrixPattern):
    """The Cholesky factor of a correlation matrix: lower triangular,
    positive diagonal, rows of unit norm."""

    def __init__(self, size: int, *, bijector: str = "stan"):
        super().__init__(size, free_length=size * (size - 1) // 2, bijector=bijector)
        self._transform = _CorrelationTransform(size=self.size, bijector=self.bijector, stan_order="row")

    def _free_to_folded(self, u):
        return self._transform.forward(u)

    def _free_to_folded_log_det(self, u):
        if self.bijector == "tfp":
            return self._transform.tfp_log_det(self._transform.forward(u))
        _, remaining = self._transform.stan_partial_correlations(u)
        below = remaining[self._transform.rows, self._transform.cols]
        return -2.0 * jnp.sum(jnp.log(jnp.cosh(u))) + 0.5 * jnp.sum(jnp.log(below))

    def _folded_to_free(self, mat):
        return self._transform.inverse(mat)

    def _validate(self, x, validate_value):
        msg = check_shape(x, self.shape)
        if msg or not validate_value:
            return msg
        x = np.asarray(x)
        msg = self._validate_lower_triangular(x)
        if msg:
            return msg
        if not np.allclose(np.sum(x**2, axis=1), 1.0, atol=1e-8):
            return "rows must have unit norm"
        return None


class CorrMatrixPattern(_SquareMatrixPattern):
    """A correlation matrix: symmetric, positive definite, unit diagonal."""

    def __init__(self, size: int, *, bijector: str = "stan"):
        super().__init__(size, free_length=size * (size - 1) // 2, bijector=bijector)
        self._transform = _CorrelationTransform(size=self.size, bijector=self.bijector, stan_order="column")
        # Stan's Jacobian weight (K - c - 2) for the partial correlation in column c.
        self._stan_weights = np.maximum(self.size - self._transform.cols - 2, 0).astype(float)

    def _free_to_folded(self, u):
        chol = self._transform.forward(u)
        return chol @ chol.T

    def _free_to_folded_log_det(self, u):
        if self.bijector == "tfp":
            chol = self._transform.forward(u)
            return self._transform.tfp_log_det(chol) + _log_det_outer_product(chol)
        z = jnp.tanh(u)
        return (-2.0 * jnp.sum(jnp.log(jnp.cosh(u)))
                + 0.5 * jnp.sum(jnp.asarray(self._stan_weights) * jnp.log1p(-(z**2))))

    def _folded_to_free(self, mat):
        return self._transform.inverse(jnp.linalg.cholesky(jnp.asarray(mat)))

    def _validate(self, x, validate_value):
        msg = check_shape(x, self.shape)
        if msg or not validate_value:
            return msg
        x = np.asarray(x)
        if not np.allclose(x, x.T, atol=1e-10, rtol=1e-10):
            return "must be symmetric"
        if not np.allclose(np.diag(x), 1.0, atol=1e-8):
            return "diagonal must be one"
        if np.any(np.linalg.eigvalsh(x) <= 0):
            return "must be positive definite"
        return None
