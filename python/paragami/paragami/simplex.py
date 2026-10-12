"""Arrays of simplices, with Stan's (>= 2.37) simplex transform.

Each simplex of dimension K is the softmax of a sum-to-zero vector, which is
itself a fixed linear map of K-1 unconstrained values (Stan Reference
Manual, "Unit simplex"). This replaced Stan's older stick-breaking
transform in 2.37; it was checked against a compiled Stan model through
BridgeStan.

For an array of simplices (Stan's `array[M] simplex[K]`), the free vector is
laid out simplex by simplex: all K-1 free values of the first simplex, then
the second, and so on. That also matches Stan.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import Pattern, check_shape


def sum_to_zero_matrix(n: int) -> np.ndarray:
    """The (n+1, n) matrix A with z = A @ y Stan's sum-to-zero transform."""
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


class SimplexArrayPattern(Pattern):
    """An array of shape `array_shape + (simplex_size,)` whose last axis
    lies on the simplex (nonnegative, sums to one)."""

    def __init__(self, simplex_size: int, array_shape: Tuple[int, ...] = (), free_default: Optional[bool] = None):
        self.simplex_size = int(simplex_size)
        self.array_shape = tuple(int(s) for s in array_shape)
        if self.simplex_size < 1:
            raise ValueError("simplex_size must be >= 1")
        self.shape = self.array_shape + (self.simplex_size,)
        m = int(math.prod(self.array_shape)) if self.array_shape else 1
        self._m = m
        a = sum_to_zero_matrix(self.simplex_size - 1)
        self._a = a
        self._a_pinv = np.linalg.pinv(a)
        super().__init__(m * self.simplex_size, m * (self.simplex_size - 1), free_default)

    def as_dict(self):
        return {"simplex_size": self.simplex_size, "array_shape": self.array_shape}

    def _flat_to_folded(self, flat_val):
        return jnp.reshape(flat_val, self.shape)

    def _folded_to_flat(self, folded_val):
        return jnp.reshape(jnp.asarray(folded_val), (-1,))

    def _one_forward(self, y):
        return jax.nn.softmax(jnp.asarray(self._a) @ y)

    def _one_log_det(self, y):
        # Density of the first K-1 coordinates (the last is determined).
        # Autodiff Jacobian, which matches Stan's convention exactly.
        jac = jax.jacfwd(lambda y_: self._one_forward(y_)[:-1])(y)
        return jnp.linalg.slogdet(jac)[1]

    def _free_to_folded(self, free_val):
        y = jnp.reshape(free_val, (self._m, self.simplex_size - 1))
        x = jax.vmap(self._one_forward)(y)
        logdet = jnp.sum(jax.vmap(self._one_log_det)(y)) if self.simplex_size > 1 else jnp.zeros(())
        return jnp.reshape(x, self.shape), logdet

    def _folded_to_free(self, folded_val):
        x = jnp.reshape(jnp.asarray(folded_val), (self._m, self.simplex_size))
        logx = jnp.log(x)
        z = logx - jnp.mean(logx, axis=-1, keepdims=True)
        return jnp.reshape(z @ jnp.asarray(self._a_pinv).T, (-1,))

    def _validate(self, folded_val, validate_value):
        msg = check_shape(folded_val, self.shape)
        if msg or not validate_value:
            return msg
        x = np.asarray(folded_val)
        if np.any(x < 0):
            return "simplex entries must be nonnegative"
        if np.any(np.abs(x.sum(axis=-1) - 1) > 1e-8):
            return "simplex entries must sum to one along the last axis"
        return None

    def _names(self, free):
        shape = self.array_shape + ((self.simplex_size - 1,) if free else (self.simplex_size,))
        return ["[" + ",".join(str(i) for i in idx) + "]" for idx in np.ndindex(*shape)]
