"""Container patterns: `PatternDict` (named sub-patterns) and
`PatternArray` (an array of one sub-pattern)."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import Pattern


class PatternDict(Pattern):
    """An ordered dict of patterns. Folded values are dicts with the same
    keys; flat and free vectors concatenate the entries in insertion order.

        p = PatternDict(beta=NumericVectorPattern(3), sigma=NumericScalarPattern(lb=0))
        p["tau"] = NumericScalarPattern(lb=0)   # or add items later
    """

    def __init__(self, patterns=None, free_default: Optional[bool] = None, **kwargs):
        self._patterns = OrderedDict()
        self._locked = False
        super().__init__(0, 0, free_default)
        for k, v in list((patterns or {}).items()) + list(kwargs.items()):
            self[k] = v

    def __setitem__(self, key, pattern):
        if self._locked:
            raise ValueError("PatternDict is locked")
        if not isinstance(pattern, Pattern):
            raise TypeError(f"{key}: expected a Pattern, got {type(pattern)}")
        self._patterns[key] = pattern
        self._update()

    def __delitem__(self, key):
        if self._locked:
            raise ValueError("PatternDict is locked")
        del self._patterns[key]
        self._update()

    def __getitem__(self, key):
        return self._patterns[key]

    def __contains__(self, key):
        return key in self._patterns

    def __iter__(self):
        return iter(self._patterns)

    def __len__(self):
        return len(self._patterns)

    def keys(self):
        return self._patterns.keys()

    def items(self):
        return self._patterns.items()

    def lock(self):
        self._locked = True
        return self

    def _update(self):
        self._flat_length = sum(p.flat_length() for p in self._patterns.values())
        self._free_length = sum(p.free_length() for p in self._patterns.values())

    def as_dict(self):
        return {"patterns": {k: (type(p).__name__, p.as_dict()) for k, p in self._patterns.items()}}

    def _split(self, flat_val, free):
        out, offset = {}, 0
        for k, p in self._patterns.items():
            n = p._length(free)
            out[k] = flat_val[offset:offset + n]
            offset += n
        return out

    def _flat_to_folded(self, flat_val):
        return {k: self._patterns[k]._flat_to_folded(v) for k, v in self._split(flat_val, False).items()}

    def _folded_to_flat(self, folded_val):
        return _concat([p._folded_to_flat(folded_val[k]) for k, p in self._patterns.items()])

    def _free_to_folded(self, free_val):
        out, logdet = {}, jnp.zeros((), dtype=free_val.dtype)
        for k, v in self._split(free_val, True).items():
            out[k], ld = self._patterns[k]._free_to_folded(v)
            logdet = logdet + ld
        return out, logdet

    def _folded_to_free(self, folded_val):
        return _concat([p._folded_to_free(folded_val[k]) for k, p in self._patterns.items()])

    def _validate(self, folded_val, validate_value):
        if not isinstance(folded_val, dict):
            return f"expected a dict, got {type(folded_val).__name__}"
        missing = set(self._patterns) - set(folded_val)
        extra = set(folded_val) - set(self._patterns)
        if missing or extra:
            return f"keys differ from the pattern (missing {sorted(missing)}, unexpected {sorted(extra)})"
        for k, p in self._patterns.items():
            msg = p._validate(folded_val[k], validate_value)
            if msg:
                return f"[{k!r}] {msg}"
        return None

    def _names(self, free):
        return [f"{k}{n}" for k, p in self._patterns.items() for n in p._names(free)]


class PatternArray(Pattern):
    """An array of `array_shape` copies of `base_pattern`, whose folded value
    must itself be an array. The folded value has shape
    `array_shape + base_shape`; flat and free vectors are laid out element
    by element (row-major over `array_shape`)."""

    def __init__(self, array_shape: Tuple[int, ...], base_pattern: Pattern, free_default: Optional[bool] = None):
        self.array_shape = tuple(int(s) for s in array_shape)
        self.base_pattern = base_pattern
        self._m = int(math.prod(self.array_shape)) if self.array_shape else 1
        base_shape = np.shape(base_pattern.empty(valid=True))
        self.shape = self.array_shape + tuple(base_shape)
        self._base_shape = tuple(base_shape)
        super().__init__(self._m * base_pattern.flat_length(), self._m * base_pattern.free_length(), free_default)

    def as_dict(self):
        return {"array_shape": self.array_shape, "base": (type(self.base_pattern).__name__, self.base_pattern.as_dict())}

    def _flat_to_folded(self, flat_val):
        rows = jnp.reshape(flat_val, (self._m, -1))
        return jnp.reshape(jax.vmap(self.base_pattern._flat_to_folded)(rows), self.shape)

    def _folded_to_flat(self, folded_val):
        items = jnp.reshape(jnp.asarray(folded_val), (self._m,) + self._base_shape)
        return jnp.reshape(jax.vmap(self.base_pattern._folded_to_flat)(items), (-1,))

    def _free_to_folded(self, free_val):
        rows = jnp.reshape(free_val, (self._m, -1))
        x, logdet = jax.vmap(self.base_pattern._free_to_folded)(rows)
        return jnp.reshape(x, self.shape), jnp.sum(logdet)

    def _folded_to_free(self, folded_val):
        items = jnp.reshape(jnp.asarray(folded_val), (self._m,) + self._base_shape)
        return jnp.reshape(jax.vmap(self.base_pattern._folded_to_free)(items), (-1,))

    def _validate(self, folded_val, validate_value):
        if tuple(np.shape(folded_val)) != self.shape:
            return f"expected shape {self.shape}, got {tuple(np.shape(folded_val))}"
        items = np.reshape(np.asarray(folded_val), (self._m,) + self._base_shape) if validate_value else None
        if validate_value:
            for i, item in enumerate(items):
                msg = self.base_pattern._validate(item, True)
                if msg:
                    return f"element {np.unravel_index(i, self.array_shape)}: {msg}"
        return None

    def _names(self, free):
        base = self.base_pattern._names(free)
        return [f"[{','.join(map(str, idx))}]{n}" for idx in np.ndindex(*self.array_shape) for n in base]


def _concat(chunks):
    return jnp.concatenate([jnp.atleast_1d(c) for c in chunks]) if chunks else jnp.zeros((0,))
