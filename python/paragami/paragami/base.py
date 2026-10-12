"""The `Pattern` base class.

A pattern describes one structured value (an array, a matrix, a dict of
arrays, ...) and converts it to and from flat vectors in two ways:

- `free=False`: the *flat* vector, simply the folded value's entries
  concatenated (row-major unless a pattern says otherwise). It has the same
  constraints as the folded value.
- `free=True`: the *free* vector, an unconstrained point in R^d obtained by
  a smooth bijection. `fold_with_log_det` also returns the log absolute
  Jacobian determinant of the free-to-constrained map, which is what a
  density on the free vector needs.

All transforms are pure JAX, so `fold`/`flatten` can be differentiated,
`jit`-ed and `vmap`-ed. Patterns act on a single value; batch with
`jax.vmap`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class ValidationError(ValueError):
    """A folded value does not match its pattern."""


def is_concrete(x) -> bool:
    """False if any leaf of `x` is a JAX tracer (inside jit/grad/vmap), in
    which case value checks are impossible and are skipped."""
    return not any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(x))


class Pattern(ABC):
    """Base class. Subclasses implement the six underscore methods below;
    everything public is derived here."""

    def __init__(self, flat_length: int, free_flat_length: int, free_default: Optional[bool] = None):
        self._flat_length = int(flat_length)
        self._free_flat_length = int(free_flat_length)
        self.free_default = free_default

    # -- subclass interface ----------------------------------------------------

    @abstractmethod
    def _fold_flat(self, flat_val):
        """Constrained flat vector -> folded value."""

    @abstractmethod
    def _flatten_flat(self, folded_val):
        """Folded value -> constrained flat vector."""

    @abstractmethod
    def _fold_free(self, free_val) -> Tuple[Any, jnp.ndarray]:
        """Free vector -> (folded value, log |det d folded_flat / d free|)."""

    @abstractmethod
    def _flatten_free(self, folded_val):
        """Folded value -> free vector."""

    @abstractmethod
    def _validate(self, folded_val, validate_value: bool) -> Optional[str]:
        """Return None if valid, else a message. Shape checks always run;
        value checks only when `validate_value`."""

    @abstractmethod
    def as_dict(self) -> dict:
        """A JSON-able description, used for equality and printing."""

    # -- public API ----------------------------------------------------------------

    def flat_length(self, free: Optional[bool] = None) -> int:
        return self._free_flat_length if self._free(free) else self._flat_length

    def fold(self, flat_val, free: Optional[bool] = None, validate_value: Optional[bool] = None):
        flat_val = jnp.asarray(flat_val)
        free = self._free(free)
        if flat_val.shape != (self.flat_length(free),):
            raise ValidationError(f"{self}: expected a flat vector of length {self.flat_length(free)}, got shape {flat_val.shape}")
        if free:
            folded, _ = self._fold_free(flat_val)
            return folded
        folded = self._fold_flat(flat_val)
        self._check(folded, validate_value)
        return folded

    def fold_with_log_det(self, free_val) -> Tuple[Any, jnp.ndarray]:
        """Fold a free vector, also returning the log absolute determinant of
        the Jacobian of the free -> constrained map."""
        free_val = jnp.asarray(free_val)
        if free_val.shape != (self.flat_length(True),):
            raise ValidationError(f"{self}: expected a free vector of length {self.flat_length(True)}, got shape {free_val.shape}")
        return self._fold_free(free_val)

    def log_abs_det_unfreeing_jacobian(self, free_val) -> jnp.ndarray:
        return self.fold_with_log_det(free_val)[1]

    def flatten(self, folded_val, free: Optional[bool] = None, validate_value: Optional[bool] = None):
        self._check(folded_val, validate_value)
        return self._flatten_free(folded_val) if self._free(free) else self._flatten_flat(folded_val)

    def validate_folded(self, folded_val, validate_value: Optional[bool] = None) -> Tuple[bool, str]:
        """(is_valid, message). Value checks are skipped under tracing."""
        try:
            self._check(folded_val, validate_value)
        except ValidationError as e:
            return False, str(e)
        return True, ""

    def empty(self, valid: bool = True):
        """A folded value of the right shape: valid if `valid`, else NaNs."""
        folded = self.fold(jnp.zeros(self.flat_length(True)), free=True)
        if valid:
            return folded
        return jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.nan), folded)

    def random(self, key, scale: float = 1.0):
        """A random valid folded value: a standard normal free vector, folded."""
        return self.fold(scale * jax.random.normal(key, (self.flat_length(True),)), free=True)

    def flat_names(self, free: Optional[bool] = None) -> List[str]:
        """Names for each entry of the flat (or free) vector, e.g. "[0]",
        "[1,2]". Containers prepend their keys."""
        n = self.flat_length(free)
        return [f"[{i}]" for i in range(n)] if n != 1 else [""]

    # -- helpers ------------------------------------------------------------------

    def _free(self, free: Optional[bool]) -> bool:
        if free is None:
            if self.free_default is None:
                raise ValueError(f"{self}: pass `free=True/False` (no free_default set)")
            return self.free_default
        return bool(free)

    def _check(self, folded_val, validate_value: Optional[bool]):
        value = (validate_value is None or validate_value) and is_concrete(folded_val)
        msg = self._validate(folded_val, value)
        if msg is not None:
            raise ValidationError(f"{self}: {msg}")

    def __eq__(self, other):
        return type(self) is type(other) and self.as_dict() == other.as_dict()

    def __hash__(self):
        return hash(repr(self.as_dict()))

    def __repr__(self):
        d = {k: v for k, v in self.as_dict().items() if k != "pattern"}
        return f"{type(self).__name__}({', '.join(f'{k}={v}' for k, v in d.items())})"


def check_shape(x, shape) -> Optional[str]:
    shp = np.shape(x)
    if tuple(shp) != tuple(shape):
        return f"expected shape {tuple(shape)}, got {tuple(shp)}"
    return None
