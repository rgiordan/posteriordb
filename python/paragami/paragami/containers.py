"""Container patterns: `PatternDict` (named sub-patterns) and
`PatternArray` (an array of one sub-pattern)."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import Pattern, format_call, indent_lines


class PatternDict(Pattern):
    """An ordered dict of patterns. Folded values are dicts with the same
    keys; flat and free vectors concatenate the entries in insertion order.
    Entries can themselves be `PatternDict`s.

        p = PatternDict(beta=NumericVectorPattern(3), sigma=NumericScalarPattern(lb=0))
        p["tau"] = NumericScalarPattern(lb=0)   # or add items later
    """

    def __init__(self, patterns=None, **kwargs):
        self._patterns = OrderedDict()
        self._locked = False
        super().__init__(flat_length=0, free_length=0)
        for k, v in list((patterns or {}).items()) + list(kwargs.items()):
            self[k] = v

    def __setitem__(self, key, pattern):
        if self._locked:
            raise ValueError("PatternDict is locked")
        if not isinstance(pattern, Pattern):
            raise TypeError(f"{key}: expected a Pattern, got {type(pattern)}")
        if not self.validate:
            pattern.disable_validation()
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

    def _set_validate(self, validate):
        self.validate = validate
        for p in self._patterns.values():
            p._set_validate(validate=validate)

    def _update(self):
        self._flat_length = sum(p.flat_length() for p in self._patterns.values())
        self._free_length = sum(p.free_length() for p in self._patterns.values())

    def as_dict(self):
        return {"patterns": {k: (type(p).__name__, p.as_dict()) for k, p in self._patterns.items()}}

    def code_to_generate(self, *, module: str = "pg", indent: int = 4) -> str:
        prefix = f"{module}." if module else ""
        if not self._patterns:
            return f"{prefix}PatternDict()"
        pad = " " * indent
        codes = {k: p.code_to_generate(module=module, indent=indent) for k, p in self._patterns.items()}
        if all(isinstance(k, str) and k.isidentifier() and k != "patterns" for k in codes):
            body = "".join(f"{pad}{k}={indent_lines(code, spaces=indent)},\n" for k, code in codes.items())
            return f"{prefix}PatternDict(\n{body})"
        body = "".join(f"{pad}{pad}{k!r}: {indent_lines(code, spaces=2 * indent)},\n" for k, code in codes.items())
        return f"{prefix}PatternDict(\n{pad}{{\n{body}{pad}}},\n)"

    def _split(self, *, vector, free):
        out, offset = {}, 0
        for k, p in self._patterns.items():
            n = p._length(free=free)
            out[k] = vector[offset:offset + n]
            offset += n
        return out

    def _flat_to_folded(self, flat_val):
        return {k: self._patterns[k]._flat_to_folded(v) for k, v in self._split(vector=flat_val, free=False).items()}

    def _folded_to_flat(self, folded_val):
        return _concat([p._folded_to_flat(folded_val[k]) for k, p in self._patterns.items()])

    def _free_to_folded(self, free_val):
        return {k: self._patterns[k]._free_to_folded(v) for k, v in self._split(vector=free_val, free=True).items()}

    def _free_to_folded_log_det(self, free_val):
        logdet = jnp.zeros((), dtype=free_val.dtype)
        for k, v in self._split(vector=free_val, free=True).items():
            logdet = logdet + self._patterns[k]._free_to_folded_log_det(v)
        return logdet

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
            msg = p._validate(folded_val[k], validate_value=validate_value)
            if msg:
                return f"[{k!r}] {msg}"
        return None

    def _names(self, free):
        # Nested dicts are joined with ".": "outer.inner[0]".
        return [k + ("." if n and not n.startswith("[") else "") + n
                for k, p in self._patterns.items() for n in p._names(free=free)]


class PatternArray(Pattern):
    """An array of `array_shape` copies of `base_pattern`, whose folded value
    must itself be an array. The folded value has shape
    `array_shape + base_shape`; flat and free vectors are laid out element
    by element (row-major over `array_shape`)."""

    def __init__(self, array_shape: Tuple[int, ...], base_pattern: Pattern):
        self.array_shape = tuple(int(s) for s in array_shape)
        self.base_pattern = base_pattern
        self._m = int(math.prod(self.array_shape)) if self.array_shape else 1
        base_shape = np.shape(base_pattern.empty(valid=True))
        self.shape = self.array_shape + tuple(base_shape)
        self._base_shape = tuple(base_shape)
        super().__init__(flat_length=self._m * base_pattern.flat_length(),
                         free_length=self._m * base_pattern.free_length())

    def as_dict(self):
        return {"array_shape": self.array_shape, "base": (type(self.base_pattern).__name__, self.base_pattern.as_dict())}

    def code_to_generate(self, *, module: str = "pg", indent: int = 4) -> str:
        base = self.base_pattern.code_to_generate(module=module, indent=indent)
        return format_call(module=module, name="PatternArray", args=[repr(self.array_shape), base])

    def _set_validate(self, validate):
        self.validate = validate
        self.base_pattern._set_validate(validate=validate)

    def _flat_to_folded(self, flat_val):
        rows = jnp.reshape(flat_val, (self._m, -1))
        return jnp.reshape(jax.vmap(self.base_pattern._flat_to_folded)(rows), self.shape)

    def _folded_to_flat(self, folded_val):
        items = jnp.reshape(jnp.asarray(folded_val), (self._m,) + self._base_shape)
        return jnp.reshape(jax.vmap(self.base_pattern._folded_to_flat)(items), (-1,))

    def _free_to_folded(self, free_val):
        rows = jnp.reshape(free_val, (self._m, -1))
        return jnp.reshape(jax.vmap(self.base_pattern._free_to_folded)(rows), self.shape)

    def _free_to_folded_log_det(self, free_val):
        rows = jnp.reshape(free_val, (self._m, -1))
        return jnp.sum(jax.vmap(self.base_pattern._free_to_folded_log_det)(rows))

    def _folded_to_free(self, folded_val):
        items = jnp.reshape(jnp.asarray(folded_val), (self._m,) + self._base_shape)
        return jnp.reshape(jax.vmap(self.base_pattern._folded_to_free)(items), (-1,))

    def _validate(self, folded_val, validate_value):
        if tuple(np.shape(folded_val)) != self.shape:
            return f"expected shape {self.shape}, got {tuple(np.shape(folded_val))}"
        if validate_value:
            items = np.reshape(np.asarray(folded_val), (self._m,) + self._base_shape)
            for i, item in enumerate(items):
                msg = self.base_pattern._validate(item, validate_value=True)
                if msg:
                    return f"element {np.unravel_index(i, self.array_shape)}: {msg}"
        return None

    def _names(self, free):
        base = self.base_pattern._names(free=free)
        return [f"[{','.join(map(str, idx))}]{n}" for idx in np.ndindex(*self.array_shape) for n in base]


def _concat(chunks):
    return jnp.concatenate([jnp.atleast_1d(c) for c in chunks]) if chunks else jnp.zeros((0,))
