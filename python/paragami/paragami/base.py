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
`free_to_folded`, `folded_to_free`. `free_to_folded` does not compute the log
Jacobian determinant of the free-to-folded map; ask for it with
`free_to_folded_log_abs_det` or `free_to_folded_with_log_det`. The generic
`fold` and `flatten` take a required `from_free` / `to_free`.

Inputs are validated by default. `disable_validation()` turns this off (for
a container, for all of its children too) so conversions in an inner loop
run with no checks at all.

All transforms are pure JAX, so conversions can be differentiated,
`jit`-ed and `vmap`-ed. Patterns act on a single value; batch with
`jax.vmap`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

BIJECTORS = ("stan", "tfp")


class ValidationError(ValueError):
    """A value does not match its pattern."""


def is_concrete(x) -> bool:
    """False if any leaf of `x` is a JAX tracer (inside jit/grad/vmap), in
    which case value checks are impossible and are skipped."""
    return not any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(x))


class Pattern(ABC):
    """Base class. Subclasses implement the underscore methods below (and
    `_names` when the default entry names don't fit); everything public is
    derived here."""

    def __init__(self, *, flat_length: int, free_length: int):
        self._flat_length = int(flat_length)
        self._free_length = int(free_length)
        self.validate = True

    # -- subclass interface ----------------------------------------------------

    @abstractmethod
    def _flat_to_folded(self, flat_val):
        """Constrained flat vector -> folded value."""

    @abstractmethod
    def _folded_to_flat(self, folded_val):
        """Folded value -> constrained flat vector."""

    @abstractmethod
    def _free_to_folded(self, free_val):
        """Free vector -> folded value."""

    @abstractmethod
    def _free_to_folded_log_det(self, free_val) -> jnp.ndarray:
        """Free vector -> log |det d folded / d free|, in the coordinates
        that parameterize the folded value (for example the lower triangle
        of a Cholesky factor)."""

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

    @abstractmethod
    def code_to_generate(self, *, module: str = "pg", indent: int = 4) -> str:
        """Python code that constructs this pattern, assuming paragami is
        imported as `module` (`module=""` for names imported directly).
        Arguments at their defaults are left out."""

    def _names(self, free: bool) -> List[str]:
        """Names for each entry of the flat (or free) vector, e.g. "[0]",
        "[1,2]". Containers prepend their keys."""
        n = self._length(free=free)
        return [f"[{i}]" for i in range(n)] if n != 1 else [""]

    def _set_validate(self, validate: bool):
        self.validate = validate

    # -- validation switch -------------------------------------------------------

    def enable_validation(self):
        """Check inputs to conversions (the default). Returns the pattern."""
        self._set_validate(validate=True)
        return self

    def disable_validation(self):
        """Skip all input checks in conversions, for speed. A container
        pattern also disables its children. Returns the pattern."""
        self._set_validate(validate=False)
        return self

    # -- lengths and names -------------------------------------------------------

    def flat_length(self) -> int:
        return self._flat_length

    def free_length(self) -> int:
        return self._free_length

    def flat_names(self) -> List[str]:
        return self._names(free=False)

    def free_names(self) -> List[str]:
        return self._names(free=True)

    # -- conversions ---------------------------------------------------------------

    def flat_to_folded(self, flat_val):
        if self.validate:
            flat_val = self._check_vector(vector=flat_val, free=False)
        folded = self._flat_to_folded(flat_val)
        if self.validate:
            self._check(folded)
        return folded

    def folded_to_flat(self, folded_val):
        if self.validate:
            self._check(folded_val)
        return self._folded_to_flat(folded_val)

    def free_to_folded(self, free_val):
        """Every free vector folds to a valid value, so only its length is checked."""
        if self.validate:
            free_val = self._check_vector(vector=free_val, free=True)
        return self._free_to_folded(free_val)

    def folded_to_free(self, folded_val):
        if self.validate:
            self._check(folded_val)
        return self._folded_to_free(folded_val)

    def free_to_folded_log_abs_det(self, free_val) -> jnp.ndarray:
        """The log absolute Jacobian determinant of the free -> folded map."""
        if self.validate:
            free_val = self._check_vector(vector=free_val, free=True)
        return self._free_to_folded_log_det(free_val)

    def free_to_folded_with_log_det(self, free_val) -> Tuple[Any, jnp.ndarray]:
        """`(free_to_folded(free_val), free_to_folded_log_abs_det(free_val))`."""
        if self.validate:
            free_val = self._check_vector(vector=free_val, free=True)
        return self._free_to_folded(free_val), self._free_to_folded_log_det(free_val)

    def fold(self, vector, *, from_free: bool):
        """`free_to_folded` if `from_free`, else `flat_to_folded`."""
        return self.free_to_folded(vector) if from_free else self.flat_to_folded(vector)

    def flatten(self, folded_val, *, to_free: bool):
        """`folded_to_free` if `to_free`, else `folded_to_flat`."""
        return self.folded_to_free(folded_val) if to_free else self.folded_to_flat(folded_val)

    # -- other ------------------------------------------------------------------------

    def validate_folded(self, folded_val) -> Tuple[bool, str]:
        """(is_valid, message), whatever the validation switch says. Value
        checks are skipped under tracing."""
        try:
            self._check(folded_val)
        except ValidationError as e:
            return False, str(e)
        return True, ""

    def empty(self, *, valid: bool = True):
        """A folded value of the right shape: valid if `valid`, else NaNs."""
        folded = self._free_to_folded(jnp.zeros(self.free_length()))
        if valid:
            return folded
        return jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.nan), folded)

    def random(self, key, *, scale: float = 1.0):
        """A random valid folded value: a standard normal free vector, folded."""
        return self._free_to_folded(scale * jax.random.normal(key, (self.free_length(),)))

    # -- helpers ------------------------------------------------------------------

    def _length(self, *, free: bool) -> int:
        return self._free_length if free else self._flat_length

    def _check_vector(self, *, vector, free: bool):
        vector = jnp.asarray(vector)
        n = self._length(free=free)
        if vector.shape != (n,):
            kind = "free" if free else "flat"
            raise ValidationError(f"{self}: expected a {kind} vector of length {n}, got shape {vector.shape}")
        return vector

    def _check(self, folded_val):
        msg = self._validate(folded_val, validate_value=is_concrete(folded_val))
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


def check_bijector(bijector: str) -> str:
    if bijector not in BIJECTORS:
        raise ValueError(f"bijector must be one of {BIJECTORS}, got {bijector!r}")
    return bijector


# -- helpers for code_to_generate -------------------------------------------------

def format_value(x) -> str:
    """A Python literal for a number or array of numbers."""
    if np.ndim(x) == 0:
        x = float(x)
        if math.isinf(x):
            return 'float("inf")' if x > 0 else '-float("inf")'
        return repr(x)
    return "[" + ", ".join(format_value(v) for v in np.asarray(x)) + "]"


def format_call(*, module: str, name: str, args=(), kwargs=None) -> str:
    """`module.name(arg, ..., key=value, ...)`, given already-formatted values."""
    prefix = f"{module}." if module else ""
    parts = list(args) + [f"{k}={v}" for k, v in (kwargs or {}).items()]
    return f"{prefix}{name}({', '.join(parts)})"


def indent_lines(text: str, *, spaces: int) -> str:
    """Indent every line of `text` but the first (for nesting generated code)."""
    return text.replace("\n", "\n" + " " * spaces)
