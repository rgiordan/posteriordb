"""Numeric arrays with optional elementwise bounds or an affine transform,
ordered vectors, and unit vectors.

With `bijector="stan"` (the default) the unconstraining transforms are
Stan's (Stan Reference Manual, "Constraint transforms"), so free vectors and
log Jacobians match Stan's unconstrained space exactly:

- lower bound only:     x = lb + exp(u)
- upper bound only:     x = ub - exp(u)
- both:                 x = lb + (ub - lb) * inv_logit(u)
- offset / multiplier:  x = offset + multiplier * u
- ordered:              x[0] = u[0],      x[k] = x[k-1] + exp(u[k])
- positive ordered:     x[0] = exp(u[0]), x[k] = x[k-1] + exp(u[k])
- unit vector:          x = u / |u|, with "log det" -|u|^2 / 2 (Stan's
                        convention: a standard normal on u, as Stan adds)

With `bijector="tfp"`, one-sided bounds use TensorFlow Probability's
softplus (x = lb + softplus(u), x = ub - softplus(u)), which is what TFP's
default event-space bijectors use for positive quantities. The other
transforms are the same in Stan and TFP (TFP's `Sigmoid(low, high)`,
`Shift`/`Scale` and `Ascending`), so they have no `bijector` option.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import Pattern, check_bijector, check_shape, format_call, format_value


def _size(shape) -> int:
    return int(math.prod(shape)) if shape else 1


class NumericArrayPattern(Pattern):
    """An array of a given shape, optionally with scalar bounds `lb`, `ub`
    or an affine transform `offset`, `multiplier` (scalars, or arrays of the
    pattern's shape). Bounds and the affine transform cannot be combined,
    as in Stan.

    `order` ("C" row-major, or "F" column-major) is the order in which the
    array's entries are laid out in both the flat and free vectors. Use "F"
    to match Stan's unconstrained layout for matrix-shaped parameters."""

    def __init__(self, shape: Tuple[int, ...], *, lb: float = -math.inf, ub: float = math.inf,
                 offset=0.0, multiplier=1.0, order: str = "C", bijector: str = "stan"):
        self.shape = tuple(int(s) for s in shape)
        self.lb, self.ub = float(lb), float(ub)
        if not self.lb < self.ub:
            raise ValueError(f"lb ({lb}) must be < ub ({ub})")
        self.offset = np.asarray(offset, dtype=float)
        self.multiplier = np.asarray(multiplier, dtype=float)
        for name, value in (("offset", self.offset), ("multiplier", self.multiplier)):
            if value.ndim and value.shape != self.shape:
                raise ValueError(f"{name} must be a scalar or have shape {self.shape}, got {value.shape}")
        if np.any(self.multiplier <= 0):
            raise ValueError("multiplier must be positive")
        if self.has_affine and (self.has_lb or self.has_ub):
            raise ValueError("bounds (lb, ub) and offset/multiplier cannot be combined")
        if order not in ("C", "F"):
            raise ValueError("order must be 'C' or 'F'")
        self.order = order
        self.bijector = check_bijector(bijector)
        n = _size(self.shape)
        super().__init__(flat_length=n, free_length=n)

    def as_dict(self):
        return {"shape": self.shape, "lb": self.lb, "ub": self.ub, "offset": self.offset.tolist(),
                "multiplier": self.multiplier.tolist(), "order": self.order, "bijector": self.bijector}

    @property
    def has_lb(self):
        return self.lb > -math.inf

    @property
    def has_ub(self):
        return self.ub < math.inf

    @property
    def has_affine(self):
        return bool(np.any(self.offset != 0.0) or np.any(self.multiplier != 1.0))

    # -- code generation -----------------------------------------------------------

    def _code_kwargs(self):
        kwargs = {}
        if self.has_lb:
            kwargs["lb"] = format_value(self.lb)
        if self.has_ub:
            kwargs["ub"] = format_value(self.ub)
        if np.any(self.offset != 0.0):
            kwargs["offset"] = format_value(self.offset)
        if np.any(self.multiplier != 1.0):
            kwargs["multiplier"] = format_value(self.multiplier)
        if self.order != "C":
            kwargs["order"] = repr(self.order)
        if self.bijector != "stan":
            kwargs["bijector"] = repr(self.bijector)
        return kwargs

    def code_to_generate(self, *, module: str = "pg", indent: int = 4) -> str:
        return format_call(module=module, name="NumericArrayPattern", args=[repr(self.shape)],
                           kwargs=self._code_kwargs())

    # -- transforms ------------------------------------------------------------------

    def _flat_to_folded(self, flat_val):
        return jnp.reshape(flat_val, self.shape, order=self.order)

    def _folded_to_flat(self, folded_val):
        return jnp.reshape(jnp.asarray(folded_val), (-1,), order=self.order)

    def _flat_affine(self):
        """Offset and multiplier laid out like the flat vector."""
        shift = jnp.reshape(jnp.broadcast_to(self.offset, self.shape), (-1,), order=self.order)
        scale = jnp.reshape(jnp.broadcast_to(self.multiplier, self.shape), (-1,), order=self.order)
        return shift, scale

    def _constrain(self, u):
        if self.has_lb and self.has_ub:
            return self.lb + (self.ub - self.lb) * jax.nn.sigmoid(u)
        positive = jax.nn.softplus(u) if self.bijector == "tfp" else jnp.exp(u)
        if self.has_lb:
            return self.lb + positive
        if self.has_ub:
            return self.ub - positive
        if self.has_affine:
            shift, scale = self._flat_affine()
            return shift + scale * u
        return u

    def _constrain_log_det(self, u):
        if self.has_lb and self.has_ub:
            return jnp.sum(math.log(self.ub - self.lb) + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))
        if self.has_lb or self.has_ub:
            return jnp.sum(jax.nn.log_sigmoid(u)) if self.bijector == "tfp" else jnp.sum(u)
        if self.has_affine:
            return jnp.sum(jnp.log(self._flat_affine()[1]))
        return jnp.zeros((), dtype=u.dtype)

    def _unconstrain(self, x):
        if self.has_lb and self.has_ub:
            p = (x - self.lb) / (self.ub - self.lb)
            return jnp.log(p) - jnp.log1p(-p)
        if self.has_lb or self.has_ub:
            positive = x - self.lb if self.has_lb else self.ub - x
            if self.bijector == "tfp":
                return positive + jnp.log(-jnp.expm1(-positive))  # inverse softplus
            return jnp.log(positive)
        if self.has_affine:
            shift, scale = self._flat_affine()
            return (x - shift) / scale
        return x

    def _free_to_folded(self, free_val):
        return self._flat_to_folded(self._constrain(free_val))

    def _free_to_folded_log_det(self, free_val):
        return self._constrain_log_det(free_val)

    def _folded_to_free(self, folded_val):
        return self._unconstrain(self._folded_to_flat(folded_val))

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

    def _names(self, free):
        if not self.shape:
            return [""]
        idx = np.array(list(np.ndindex(*self.shape)))
        if self.order == "F":
            idx = np.array(list(np.ndindex(*self.shape[::-1])))[:, ::-1]
        return ["[" + ",".join(str(i) for i in row) + "]" for row in idx]


class NumericVectorPattern(NumericArrayPattern):
    def __init__(self, length: int, *, lb: float = -math.inf, ub: float = math.inf,
                 offset=0.0, multiplier=1.0, bijector: str = "stan"):
        super().__init__((length,), lb=lb, ub=ub, offset=offset, multiplier=multiplier, bijector=bijector)

    def code_to_generate(self, *, module: str = "pg", indent: int = 4) -> str:
        return format_call(module=module, name="NumericVectorPattern", args=[str(self.shape[0])],
                           kwargs=self._code_kwargs())


class NumericScalarPattern(NumericArrayPattern):
    def __init__(self, *, lb: float = -math.inf, ub: float = math.inf, offset=0.0, multiplier=1.0,
                 bijector: str = "stan"):
        super().__init__((), lb=lb, ub=ub, offset=offset, multiplier=multiplier, bijector=bijector)

    def code_to_generate(self, *, module: str = "pg", indent: int = 4) -> str:
        return format_call(module=module, name="NumericScalarPattern", kwargs=self._code_kwargs())


class OrderedVectorPattern(Pattern):
    """A strictly increasing vector of length K. With `positive=True`, also
    positive (Stan's `positive_ordered`). TFP's `Ascending` bijector is the
    same transform as Stan's `ordered`."""

    def __init__(self, length: int, *, positive: bool = False):
        self.length = int(length)
        self.positive = bool(positive)
        super().__init__(flat_length=self.length, free_length=self.length)

    def as_dict(self):
        return {"length": self.length, "positive": self.positive}

    def code_to_generate(self, *, module: str = "pg", indent: int = 4) -> str:
        return format_call(module=module, name="OrderedVectorPattern", args=[str(self.length)],
                           kwargs={"positive": "True"} if self.positive else {})

    def _flat_to_folded(self, flat_val):
        return flat_val

    def _folded_to_flat(self, folded_val):
        return jnp.asarray(folded_val)

    def _free_to_folded(self, u):
        if self.positive:
            return jnp.cumsum(jnp.exp(u))
        return jnp.cumsum(jnp.concatenate([u[:1], jnp.exp(u[1:])]))

    def _free_to_folded_log_det(self, u):
        return jnp.sum(u) if self.positive else jnp.sum(u[1:])

    def _folded_to_free(self, x):
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


class UnitVectorPattern(Pattern):
    """A vector of length K with Euclidean norm one (Stan's `unit_vector`).

    The free vector has length K too: x = u / |u|. The map is not a
    bijection, so as in Stan the "log det" is -|u|^2 / 2, which puts a
    standard normal on u and makes the radius irrelevant. `folded_to_free`
    returns x itself, as Stan does. TFP has no unit-vector bijector."""

    def __init__(self, length: int):
        self.length = int(length)
        if self.length < 2:
            raise ValueError("a unit vector needs length >= 2")
        super().__init__(flat_length=self.length, free_length=self.length)

    def as_dict(self):
        return {"length": self.length}

    def code_to_generate(self, *, module: str = "pg", indent: int = 4) -> str:
        return format_call(module=module, name="UnitVectorPattern", args=[str(self.length)])

    def _flat_to_folded(self, flat_val):
        return flat_val

    def _folded_to_flat(self, folded_val):
        return jnp.asarray(folded_val)

    def _free_to_folded(self, u):
        return u / jnp.sqrt(jnp.sum(u**2))

    def _free_to_folded_log_det(self, u):
        return -0.5 * jnp.sum(u**2)

    def _folded_to_free(self, x):
        return jnp.asarray(x)

    def _validate(self, folded_val, validate_value):
        msg = check_shape(folded_val, (self.length,))
        if msg or not validate_value:
            return msg
        if abs(float(np.sum(np.asarray(folded_val) ** 2)) - 1.0) > 1e-8:
            return "must have unit norm"
        return None
