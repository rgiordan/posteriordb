"""Numeric arrays with optional elementwise bounds, and ordered vectors.

The unconstraining transforms are Stan's (Stan Reference Manual,
"Constraint transforms"), so free vectors and log Jacobians match Stan's
unconstrained space exactly:

- lower bound only:  x = lb + exp(u)
- upper bound only:  x = ub - exp(u)
- both:              x = lb + (ub - lb) * inv_logit(u)
- ordered:           x[0] = u[0],      x[k] = x[k-1] + exp(u[k])
- positive ordered:  x[0] = exp(u[0]), x[k] = x[k-1] + exp(u[k])
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import Pattern, check_shape


def _size(shape) -> int:
    return int(math.prod(shape)) if shape else 1


class NumericArrayPattern(Pattern):
    """An array of a given shape with optional scalar bounds `lb`, `ub`.

    `order` ("C" row-major, or "F" column-major) is the order in which the
    array's entries are laid out in both the flat and free vectors. Use "F"
    to match Stan's unconstrained layout for matrix-shaped parameters."""

    def __init__(self, shape: Tuple[int, ...], lb: float = -math.inf, ub: float = math.inf,
                 order: str = "C", free_default: Optional[bool] = None):
        self.shape = tuple(int(s) for s in shape)
        self.lb, self.ub = float(lb), float(ub)
        if not self.lb < self.ub:
            raise ValueError(f"lb ({lb}) must be < ub ({ub})")
        if order not in ("C", "F"):
            raise ValueError("order must be 'C' or 'F'")
        self.order = order
        n = _size(self.shape)
        super().__init__(n, n, free_default)

    def as_dict(self):
        return {"shape": self.shape, "lb": self.lb, "ub": self.ub, "order": self.order}

    @property
    def has_lb(self):
        return self.lb > -math.inf

    @property
    def has_ub(self):
        return self.ub < math.inf

    # flat <-> folded is a reshape
    def _fold_flat(self, flat_val):
        return jnp.reshape(flat_val, self.shape, order=self.order)

    def _flatten_flat(self, folded_val):
        return jnp.reshape(jnp.asarray(folded_val), (-1,), order=self.order)

    def _constrain(self, u):
        if self.has_lb and self.has_ub:
            x = self.lb + (self.ub - self.lb) * jax.nn.sigmoid(u)
            logdet = jnp.sum(math.log(self.ub - self.lb) + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))
        elif self.has_lb:
            x, logdet = self.lb + jnp.exp(u), jnp.sum(u)
        elif self.has_ub:
            x, logdet = self.ub - jnp.exp(u), jnp.sum(u)
        else:
            x, logdet = u, jnp.zeros((), dtype=u.dtype)
        return x, logdet

    def _unconstrain(self, x):
        if self.has_lb and self.has_ub:
            p = (x - self.lb) / (self.ub - self.lb)
            return jnp.log(p) - jnp.log1p(-p)
        if self.has_lb:
            return jnp.log(x - self.lb)
        if self.has_ub:
            return jnp.log(self.ub - x)
        return x

    def _fold_free(self, free_val):
        x, logdet = self._constrain(free_val)
        return self._fold_flat(x), logdet

    def _flatten_free(self, folded_val):
        return self._unconstrain(self._flatten_flat(folded_val))

    def _validate(self, folded_val, validate_value):
        msg = check_shape(folded_val, self.shape)
        if msg or not validate_value:
            return msg
        x = np.asarray(folded_val)
        if np.any(np.isnan(x)):
            return "contains NaN"
        if self.has_lb and np.any(x <= self.lb):
            return f"values must be > lb = {self.lb} (min {x.min()})"
        if self.has_ub and np.any(x >= self.ub):
            return f"values must be < ub = {self.ub} (max {x.max()})"
        return None

    def flat_names(self, free=None):
        if not self.shape:
            return [""]
        idx = np.array(list(np.ndindex(*self.shape)))
        if self.order == "F":
            idx = np.array(list(np.ndindex(*self.shape[::-1])))[:, ::-1]
        return ["[" + ",".join(str(i) for i in row) + "]" for row in idx]


class NumericVectorPattern(NumericArrayPattern):
    def __init__(self, length: int, lb: float = -math.inf, ub: float = math.inf, free_default=None):
        super().__init__((length,), lb=lb, ub=ub, free_default=free_default)


class NumericScalarPattern(NumericArrayPattern):
    def __init__(self, lb: float = -math.inf, ub: float = math.inf, free_default=None):
        super().__init__((), lb=lb, ub=ub, free_default=free_default)


class OrderedVectorPattern(Pattern):
    """A strictly increasing vector of length K. With `positive=True`, also
    positive (Stan's `positive_ordered`)."""

    def __init__(self, length: int, positive: bool = False, free_default: Optional[bool] = None):
        self.length = int(length)
        self.positive = bool(positive)
        super().__init__(self.length, self.length, free_default)

    def as_dict(self):
        return {"length": self.length, "positive": self.positive}

    def _fold_flat(self, flat_val):
        return flat_val

    def _flatten_flat(self, folded_val):
        return jnp.asarray(folded_val)

    def _fold_free(self, u):
        if self.positive:
            return jnp.cumsum(jnp.exp(u)), jnp.sum(u)
        steps = jnp.concatenate([u[:1], jnp.exp(u[1:])])
        return jnp.cumsum(steps), jnp.sum(u[1:])

    def _flatten_free(self, x):
        x = jnp.asarray(x)
        first = jnp.log(x[:1]) if self.positive else x[:1]
        return jnp.concatenate([first, jnp.log(jnp.diff(x))])

    def _validate(self, folded_val, validate_value):
        msg = check_shape(folded_val, (self.length,))
        if msg or not validate_value:
            return msg
        x = np.asarray(folded_val)
        if np.any(np.diff(x) <= 0):
            return "must be strictly increasing"
        if self.positive and x.size and x[0] <= 0:
            return "must be positive"
        return None
