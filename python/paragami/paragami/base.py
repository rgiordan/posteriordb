"""The `Pattern` base class.

A pattern describes one structured value (an array, a matrix, a dict of
arrays, ...) in three representations:

- *folded*: the value itself.
- *flat*: the folded value's entries concatenated into a vector (row-major
  unless a pattern says otherwise). It has the same constraints as the
  folded value.
- *free*: an unconstrained point in R^d, related to the folded value by a
  smooth bijection.

Conversions are named `<from>_to_<to>`: `flat_to_folded`, `folded_to_flat`,
`free_to_folded`, `folded_to_free`. `free_to_folded_with_log_det` also
returns the log absolute Jacobian determinant of the free-to-folded map,
which is what a density on the free vector needs. The generic `fold` and
`flatten` take `from_free` and `to_free` to pick the vector representation.

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
    """Base class. Subclasses implement the underscore methods below (and
    `_names` when the default entry names don't fit); everything public is
    derived here.

    `free_default` is the vector representation `fold` and `flatten` use
    when `from_free` / `to_free` is not given (None means it must be given).
    """

    def __init__(self, flat_length: int, free_length: int, free_default: Optional[bool] = None):
        self._flat_length = int(flat_length)
        self._free_length = int(free_length)
        self.free_default = free_default

    # -- subclass interface ----------------------------------------------------

    @abstractmethod
    def _flat_to_folded(self, flat_val):
        """Constrained flat vector -> folded value."""

    @abstractmethod
    def _folded_to_flat(self, folded_val):
        """Folded value -> constrained flat vector."""

    @abstractmethod
    def _free_to_folded(self, free_val) -> Tuple[Any, jnp.ndarray]:
        """Free vector -> (folded value, log |det d folded_flat / d free|)."""

    @abstractmethod
    def _folded_to_free(self, folded_val):
        """Folded value -> free vector."""

    @abstractmethod
    def _validate(self, folded_val, validate_value: bool) -> Optional[str]:
        """Return None if valid, else a message. Shape checks always run;
        value checks only when `validate_value`."""

    @abstractmethod
    def as_dict(self) -> dict:
        """A JSON-able description, used for equality and printing."""

    def _names(self, free: bool) -> List[str]:
        """Names for each entry of the flat (or free) vector, e.g. "[0]",
        "[1,2]". Containers prepend their keys."""
        n = self._length(free)
        return [f"[{i}]" for i in range(n)] if n != 1 else [""]

    # -- lengths and names -------------------------------------------------------

    def flat_length(self) -> int:
        return self._flat_length

    def free_length(self) -> int:
        return self._free_length

    def flat_names(self) -> List[str]:
        return self._names(False)

    def free_names(self) -> List[str]:
        return self._names(True)

    # -- conversions ---------------------------------------------------------------

    def flat_to_folded(self, flat_val, validate_value: Optional[bool] = None):
        flat_val = self._check_vector(flat_val, free=False)
        folded = self._flat_to_folded(flat_val)
        self._check(folded, validate_value)
        return folded

    def folded_to_flat(self, folded_val, validate_value: Optional[bool] = None):
        self._check(folded_val, validate_value)
        return self._folded_to_flat(folded_val)

    def free_to_folded(self, free_val):
        """Every free vector folds to a valid value, so there is nothing to validate."""
        return self.free_to_folded_with_log_det(free_val)[0]

    def folded_to_free(self, folded_val, validate_value: Optional[bool] = None):
        self._check(folded_val, validate_value)
        return self._folded_to_free(folded_val)

    def free_to_folded_with_log_det(self, free_val) -> Tuple[Any, jnp.ndarray]:
        """Fold a free vector, also returning the log absolute determinant of
        the Jacobian of the free -> folded map."""
        return self._free_to_folded(self._check_vector(free_val, free=True))

    def free_to_folded_log_abs_det(self, free_val) -> jnp.ndarray:
        """The log absolute Jacobian determinant of the free -> folded map."""
        return self.free_to_folded_with_log_det(free_val)[1]

    def fold(self, vector, from_free: Optional[bool] = None, validate_value: Optional[bool] = None):
        """`free_to_folded` if `from_free`, else `flat_to_folded`."""
        if self._free(from_free, "from_free"):
            return self.free_to_folded(vector)
        return self.flat_to_folded(vector, validate_value)

    def flatten(self, folded_val, to_free: Optional[bool] = None, validate_value: Optional[bool] = None):
        """`folded_to_free` if `to_free`, else `folded_to_flat`."""
        if self._free(to_free, "to_free"):
            return self.folded_to_free(folded_val, validate_value)
        return self.folded_to_flat(folded_val, validate_value)

    # -- other ------------------------------------------------------------------------

    def validate_folded(self, folded_val, validate_value: Optional[bool] = None) -> Tuple[bool, str]:
        """(is_valid, message). Value checks are skipped under tracing."""
        try:
            self._check(folded_val, validate_value)
        except ValidationError as e:
            return False, str(e)
        return True, ""

    def empty(self, valid: bool = True):
        """A folded value of the right shape: valid if `valid`, else NaNs."""
        folded = self.free_to_folded(jnp.zeros(self.free_length()))
        if valid:
            return folded
        return jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.nan), folded)

    def random(self, key, scale: float = 1.0):
        """A random valid folded value: a standard normal free vector, folded."""
        return self.free_to_folded(scale * jax.random.normal(key, (self.free_length(),)))

    # -- helpers ------------------------------------------------------------------

    def _length(self, free: bool) -> int:
        return self._free_length if free else self._flat_length

    def _free(self, free: Optional[bool], name: str) -> bool:
        if free is None:
            if self.free_default is None:
                raise ValueError(f"{self}: pass `{name}=True/False` (no free_default set)")
            return self.free_default
        return bool(free)

    def _check_vector(self, vector, free: bool):
        vector = jnp.asarray(vector)
        n = self._length(free)
        if vector.shape != (n,):
            kind = "free" if free else "flat"
            raise ValidationError(f"{self}: expected a {kind} vector of length {n}, got shape {vector.shape}")
        return vector

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
