"""Simplices, stochastic matrices and sum-to-zero vectors and matrices.

Stan (>= 2.37) builds a simplex of dimension K as the softmax of a
sum-to-zero vector, which is itself a fixed linear map of K-1 unconstrained
values (Stan Reference Manual, "Unit simplex" and "Zero sum vector"); this
replaced Stan's older stick-breaking transform. With `bijector="tfp"`, a
simplex uses TensorFlow Probability's `SoftmaxCentered` instead: the
softmax of the K-1 free values with a zero appended. Both log Jacobians are
for the first K-1 coordinates plus log(K)/2, the convention Stan and TFP
share.

Free vectors follow Stan's layout: for an array of simplices
(`array[M] simplex[K]`) all K-1 free values of the first simplex, then the
second, and so on; for matrices, the free matrix in column-major order.
"""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import Pattern, check_bijector, check_shape, format_call


def make_sum_to_zero_matrix(n: int) -> np.ndarray:
    """The (n+1, n) matrix A with orthonormal columns for which z = A @ y is
    Stan's sum-to-zero transform of y."""
    if n == 0:
        return np.zeros((1, 0))
    c = np.array([1.0 / np.sqrt(k * (k + 1)) for k in range(1, n + 1)])
    a = np.zeros((n + 1, n))
    a[0, :] = c
    for k in range(1, n + 1):
        if k < n:
            a[k, k:] = c[k:]
        a[k, k - 1] = -k * c[k - 1]
    return a


def make_sum_to_zero_matrix_basis(n: int) -> np.ndarray:
    """The (n, n-1) orthonormal basis Stan's `sum_to_zero_matrix` uses along
    each axis (different from the vector transform's basis). Column k has
    -1 in rows 0..k-1, k+1 in row k and -1 in the last row, scaled by
    1 / sqrt((k+1)(k+2))."""
    basis = np.zeros((n, max(n - 1, 0)))
    for k in range(n - 1):
        basis[:k, k] = -1.0
        basis[k, k] = k + 1.0
        basis[n - 1, k] = -1.0
        basis[:, k] /= np.sqrt((k + 1) * (k + 2))
    return basis


class _SimplexTransform:
    """One simplex of size K from K-1 free values, Stan's or TFP's way."""

    def __init__(self, *, size: int, bijector: str):
        self.size = size
        self.bijector = bijector
        self._a = jnp.asarray(make_sum_to_zero_matrix(size - 1))

    def constrain(self, y):
        if self.bijector == "tfp":
            return jax.nn.softmax(jnp.concatenate([y, jnp.zeros((1,), dtype=y.dtype)]))
        return jax.nn.softmax(self._a @ y)

    def compute_log_det(self, y):
        return jnp.sum(jnp.log(self.constrain(y))) + 0.5 * math.log(self.size)

    def unconstrain(self, x):
        logx = jnp.log(x)
        if self.bijector == "tfp":
            return logx[:-1] - logx[-1]
        z = logx - jnp.mean(logx)
        return self._a.T @ z


class SimplexArrayPattern(Pattern):
    """An array of shape `array_shape + (simplex_size,)` whose last axis
    lies on the simplex (nonnegative, sums to one)."""

    def __init__(self, simplex_size: int, *, array_shape: Tuple[int, ...] = (), bijector: str = "stan"):
        self.simplex_size = int(simplex_size)
        self.array_shape = tuple(int(s) for s in array_shape)
        if self.simplex_size < 1:
            raise ValueError("simplex_size must be >= 1")
        self.bijector = check_bijector(bijector)
        self.shape = self.array_shape + (self.simplex_size,)
        self._m = int(math.prod(self.array_shape)) if self.array_shape else 1
        self._simplex = _SimplexTransform(size=self.simplex_size, bijector=self.bijector)
        super().__init__(flat_length=self._m * self.simplex_size, free_length=self._m * (self.simplex_size - 1))

    def to_dict(self):
        return {"simplex_size": self.simplex_size, "array_shape": self.array_shape, "bijector": self.bijector}

    def generate_code(self, *, module: str = "pg", indent: int = 4) -> str:
        kwargs = {}
        if self.array_shape:
            kwargs["array_shape"] = repr(self.array_shape)
        if self.bijector != "stan":
            kwargs["bijector"] = repr(self.bijector)
        return format_call(module=module, name="SimplexArrayPattern", args=[str(self.simplex_size)], kwargs=kwargs)

    def _flat_to_folded(self, flat_val):
        return jnp.reshape(flat_val, self.shape)

    def _folded_to_flat(self, folded_val):
        return jnp.reshape(jnp.asarray(folded_val), (-1,))

    def _free_to_folded(self, free_val):
        y = jnp.reshape(free_val, (self._m, self.simplex_size - 1))
        return jnp.reshape(jax.vmap(self._simplex.constrain)(y), self.shape)

    def _free_to_folded_log_det(self, free_val):
        if self.simplex_size == 1:
            return jnp.zeros((), dtype=free_val.dtype)
        y = jnp.reshape(free_val, (self._m, self.simplex_size - 1))
        return jnp.sum(jax.vmap(self._simplex.compute_log_det)(y))

    def _folded_to_free(self, folded_val):
        x = jnp.reshape(jnp.asarray(folded_val), (self._m, self.simplex_size))
        return jnp.reshape(jax.vmap(self._simplex.unconstrain)(x), (-1,))

    def _validate(self, folded_val, validate_value):
        msg = check_shape(folded_val, self.shape)
        if msg or not validate_value:
            return msg
        return _check_simplices(x=np.asarray(folded_val), axis=-1)

    def _names(self, free):
        shape = self.array_shape + ((self.simplex_size - 1,) if free else (self.simplex_size,))
        return ["[" + ",".join(str(i) for i in idx) + "]" for idx in np.ndindex(*shape)]


class StochasticMatrixPattern(Pattern):
    """A (rows, cols) matrix whose columns (`kind="column"`, Stan's
    `column_stochastic_matrix`) or rows (`kind="row"`, Stan's
    `row_stochastic_matrix`) are simplices. The flat vector is row-major;
    the free matrix is laid out column-major, as in Stan."""

    def __init__(self, rows: int, cols: int, *, kind: str = "column", bijector: str = "stan"):
        if kind not in ("column", "row"):
            raise ValueError("kind must be 'column' or 'row'")
        self.rows, self.cols, self.kind = int(rows), int(cols), kind
        self.bijector = check_bijector(bijector)
        self.shape = (self.rows, self.cols)
        simplex_size = self.rows if kind == "column" else self.cols
        self._simplex = _SimplexTransform(size=simplex_size, bijector=self.bijector)
        self._free_shape = (self.rows - 1, self.cols) if kind == "column" else (self.rows, self.cols - 1)
        super().__init__(flat_length=self.rows * self.cols, free_length=int(np.prod(self._free_shape)))

    def to_dict(self):
        return {"rows": self.rows, "cols": self.cols, "kind": self.kind, "bijector": self.bijector}

    def generate_code(self, *, module: str = "pg", indent: int = 4) -> str:
        kwargs = {"kind": repr(self.kind)}
        if self.bijector != "stan":
            kwargs["bijector"] = repr(self.bijector)
        return format_call(module=module, name="StochasticMatrixPattern", args=[str(self.rows), str(self.cols)],
                           kwargs=kwargs)

    def _flat_to_folded(self, flat_val):
        return jnp.reshape(flat_val, self.shape)

    def _folded_to_flat(self, folded_val):
        return jnp.reshape(jnp.asarray(folded_val), (-1,))

    def _free_matrix(self, free_val):
        return jnp.reshape(free_val, self._free_shape, order="F")

    def _free_to_folded(self, free_val):
        y = self._free_matrix(free_val)
        if self.kind == "column":
            return jax.vmap(self._simplex.constrain, in_axes=1, out_axes=1)(y)
        return jax.vmap(self._simplex.constrain)(y)

    def _free_to_folded_log_det(self, free_val):
        y = self._free_matrix(free_val)
        return jnp.sum(jax.vmap(self._simplex.compute_log_det, in_axes=1 if self.kind == "column" else 0)(y))

    def _folded_to_free(self, folded_val):
        x = jnp.asarray(folded_val)
        if self.kind == "column":
            y = jax.vmap(self._simplex.unconstrain, in_axes=1, out_axes=1)(x)
        else:
            y = jax.vmap(self._simplex.unconstrain)(x)
        return jnp.reshape(y, (-1,), order="F")

    def _validate(self, folded_val, validate_value):
        msg = check_shape(folded_val, self.shape)
        if msg or not validate_value:
            return msg
        return _check_simplices(x=np.asarray(folded_val), axis=0 if self.kind == "column" else 1)

    def _names(self, free):
        if free:  # column-major, like the free vector
            rows, cols = self._free_shape
            return [f"[{i},{j}]" for j in range(cols) for i in range(rows)]
        return [f"[{i},{j}]" for i, j in np.ndindex(*self.shape)]


class SumToZeroVectorPattern(Pattern):
    """A vector of length K whose entries sum to zero (Stan's
    `sum_to_zero_vector`): an isometric linear map of K-1 free values, so
    the log Jacobian is zero."""

    def __init__(self, length: int):
        self.length = int(length)
        if self.length < 1:
            raise ValueError("length must be >= 1")
        self._a = jnp.asarray(make_sum_to_zero_matrix(self.length - 1))
        super().__init__(flat_length=self.length, free_length=self.length - 1)

    def to_dict(self):
        return {"length": self.length}

    def generate_code(self, *, module: str = "pg", indent: int = 4) -> str:
        return format_call(module=module, name="SumToZeroVectorPattern", args=[str(self.length)])

    def _flat_to_folded(self, flat_val):
        return flat_val

    def _folded_to_flat(self, folded_val):
        return jnp.asarray(folded_val)

    def _free_to_folded(self, free_val):
        return self._a @ free_val

    def _free_to_folded_log_det(self, free_val):
        return jnp.zeros((), dtype=free_val.dtype)

    def _folded_to_free(self, folded_val):
        return self._a.T @ jnp.asarray(folded_val)

    def _validate(self, folded_val, validate_value):
        msg = check_shape(folded_val, (self.length,))
        if msg or not validate_value:
            return msg
        if abs(float(np.sum(folded_val))) > 1e-8:
            return "entries must sum to zero"
        return None


class SumToZeroMatrixPattern(Pattern):
    """A (rows, cols) matrix whose rows and columns all sum to zero (Stan's
    `sum_to_zero_matrix`): Z = B_rows @ Y @ B_cols.T for a free
    (rows-1, cols-1) matrix Y, laid out column-major, with B from
    `sum_to_zero_matrix_basis`. The log Jacobian is zero."""

    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = int(rows), int(cols)
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be >= 1")
        self.shape = (self.rows, self.cols)
        self._b_rows = jnp.asarray(make_sum_to_zero_matrix_basis(self.rows))
        self._b_cols = jnp.asarray(make_sum_to_zero_matrix_basis(self.cols))
        super().__init__(flat_length=self.rows * self.cols, free_length=(self.rows - 1) * (self.cols - 1))

    def to_dict(self):
        return {"rows": self.rows, "cols": self.cols}

    def generate_code(self, *, module: str = "pg", indent: int = 4) -> str:
        return format_call(module=module, name="SumToZeroMatrixPattern", args=[str(self.rows), str(self.cols)])

    def _flat_to_folded(self, flat_val):
        return jnp.reshape(flat_val, self.shape)

    def _folded_to_flat(self, folded_val):
        return jnp.reshape(jnp.asarray(folded_val), (-1,))

    def _free_to_folded(self, free_val):
        y = jnp.reshape(free_val, (self.rows - 1, self.cols - 1), order="F")
        return self._b_rows @ y @ self._b_cols.T

    def _free_to_folded_log_det(self, free_val):
        return jnp.zeros((), dtype=free_val.dtype)

    def _folded_to_free(self, folded_val):
        y = self._b_rows.T @ jnp.asarray(folded_val) @ self._b_cols
        return jnp.reshape(y, (-1,), order="F")

    def _validate(self, folded_val, validate_value):
        msg = check_shape(folded_val, self.shape)
        if msg or not validate_value:
            return msg
        x = np.asarray(folded_val)
        if np.any(np.abs(x.sum(axis=0)) > 1e-8) or np.any(np.abs(x.sum(axis=1)) > 1e-8):
            return "rows and columns must sum to zero"
        return None


def _check_simplices(*, x, axis):
    if np.any(x < 0):
        return "simplex entries must be nonnegative"
    if np.any(np.abs(x.sum(axis=axis) - 1) > 1e-8):
        return f"simplex entries must sum to one along axis {axis}"
    return None
