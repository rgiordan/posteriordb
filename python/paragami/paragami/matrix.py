"""Matrix patterns: Cholesky factors and positive-definite matrices, with
Stan's transforms (Stan Reference Manual, "Constraint transforms").

- `CholeskyCovPattern(K)`: lower-triangular, positive diagonal. The free
  vector fills the lower triangle row by row, with `exp` on the diagonal.
  This is Stan's fill order, which differs from TFP's `FillScaleTriL`.
- `PSDSymmetricMatrixPattern(K)`: a covariance matrix, `L @ L.T` with `L`
  from `CholeskyCovPattern` (Stan's `cov_matrix`).
- `CholeskyCorrPattern(K)`: Cholesky factor of a correlation matrix, built
  from `tanh` of the free values (Stan's `cholesky_factor_corr`).

The flat (constrained) vector is always the full K*K matrix, row-major.
"""

from __future__ import annotations

import math
from typing import Optional

import jax.numpy as jnp
import numpy as np

from .base import Pattern, check_shape


class _SquareMatrixPattern(Pattern):
    def __init__(self, size: int, free_length: int, free_default: Optional[bool]):
        self.size = int(size)
        self.shape = (self.size, self.size)
        super().__init__(self.size * self.size, free_length, free_default)

    def as_dict(self):
        return {"size": self.size}

    def _flat_to_folded(self, flat_val):
        return jnp.reshape(flat_val, self.shape)

    def _folded_to_flat(self, folded_val):
        return jnp.reshape(jnp.asarray(folded_val), (-1,))

    def _names(self, free):
        if free:
            return super()._names(free)
        return [f"[{i},{j}]" for i, j in np.ndindex(*self.shape)]


class CholeskyCovPattern(_SquareMatrixPattern):
    def __init__(self, size: int, free_default: Optional[bool] = None):
        super().__init__(size, size * (size + 1) // 2, free_default)
        self._rows, self._cols = np.tril_indices(self.size)
        self._diag_pos = np.where(self._rows == self._cols)[0]

    def _free_to_folded(self, u):
        k = self.size
        mat = jnp.zeros((k, k), dtype=u.dtype).at[self._rows, self._cols].set(u)
        d = np.arange(k)
        mat = mat.at[d, d].set(jnp.exp(mat[d, d]))
        return mat, jnp.sum(u[self._diag_pos])

    def _folded_to_free(self, mat):
        flat = jnp.asarray(mat)[self._rows, self._cols]
        return flat.at[self._diag_pos].set(jnp.log(flat[self._diag_pos]))

    def _validate(self, x, validate_value):
        msg = check_shape(x, self.shape)
        if msg or not validate_value:
            return msg
        x = np.asarray(x)
        if np.any(np.triu(x, 1) != 0):
            return "must be lower triangular"
        if np.any(np.diag(x) <= 0):
            return "diagonal must be positive"
        return None


class PSDSymmetricMatrixPattern(_SquareMatrixPattern):
    def __init__(self, size: int, free_default: Optional[bool] = None):
        super().__init__(size, size * (size + 1) // 2, free_default)
        self._chol = CholeskyCovPattern(size)

    def _free_to_folded(self, u):
        l, logdet = self._chol._free_to_folded(u)
        k = self.size
        # log|d(LL^T)/dL| = K log 2 + sum_i (K - i) log L_ii   (i = 0..K-1)
        weights = jnp.arange(k, 0, -1, dtype=u.dtype)
        logdet = logdet + k * math.log(2.0) + jnp.sum(weights * jnp.log(jnp.diag(l)))
        return l @ l.T, logdet

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


class CholeskyCorrPattern(_SquareMatrixPattern):
    def __init__(self, size: int, free_default: Optional[bool] = None):
        super().__init__(size, size * (size - 1) // 2, free_default)

    def _free_to_folded(self, u):
        k = self.size
        z = jnp.tanh(u)
        mat = jnp.zeros((k, k), dtype=u.dtype)
        logdet = -2.0 * jnp.sum(jnp.log(jnp.cosh(u)))
        idx = 0
        for i in range(k):
            remaining = jnp.ones((), dtype=u.dtype)
            for j in range(i):
                mat = mat.at[i, j].set(z[idx] * jnp.sqrt(remaining))
                logdet = logdet + 0.5 * jnp.log(remaining)
                remaining = remaining * (1 - z[idx] ** 2)
                idx += 1
            mat = mat.at[i, i].set(jnp.sqrt(remaining))
        return mat, logdet

    def _folded_to_free(self, mat):
        mat = jnp.asarray(mat)
        out = []
        for i in range(self.size):
            remaining = jnp.ones((), dtype=mat.dtype)
            for j in range(i):
                zij = mat[i, j] / jnp.sqrt(remaining)
                out.append(jnp.arctanh(zij))
                remaining = remaining * (1 - zij**2)
        return jnp.stack(out) if out else jnp.zeros((0,), dtype=mat.dtype)

    def _validate(self, x, validate_value):
        msg = check_shape(x, self.shape)
        if msg or not validate_value:
            return msg
        x = np.asarray(x)
        if np.any(np.triu(x, 1) != 0):
            return "must be lower triangular"
        if np.any(np.diag(x) <= 0):
            return "diagonal must be positive"
        if not np.allclose(np.sum(x**2, axis=1), 1.0, atol=1e-8):
            return "rows must have unit norm"
        return None
