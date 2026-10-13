"""Write patterns for data structures you already have.

`make_unconstrained_pattern_like(value)` builds an unconstrained pattern that
matches a nested structure of dicts, lists and tuples with arrays or
numbers at the leaves. Every pattern's `generate_code()` prints Python
code that rebuilds it, so `make_unconstrained_pattern_like(value).generate_code()`
gives code to paste and then edit (for example to add constraints).
"""

from __future__ import annotations

import numbers

import numpy as np

from .base import Pattern
from .containers import PatternDict
from .numeric import NumericArrayPattern, NumericScalarPattern, NumericVectorPattern


def make_unconstrained_pattern_like(value, *, default_name: str = "x") -> Pattern:
    """An unconstrained pattern whose folded values look like `value`.

    `value` is a pytree of dicts, lists and tuples with arrays or numbers at
    the leaves:

    - a leaf becomes a pattern of its shape with no bounds (`NumericScalarPattern`,
      `NumericVectorPattern` or `NumericArrayPattern`);
    - a dict becomes a `PatternDict` with the same keys (converted to str);
    - a list or tuple becomes a `PatternDict` keyed `f"{default_name}_{i}"`,
      so its folded values are dicts, not lists; inside a dict, the default
      is the dict key, so `{"w": [a, b]}` gives keys "w_0" and "w_1".

    Integer leaves give float patterns.
    """
    if isinstance(value, dict):
        out = PatternDict()
        for k, v in value.items():
            out[str(k)] = make_unconstrained_pattern_like(v, default_name=str(k))
        return out
    if isinstance(value, (list, tuple)):
        out = PatternDict()
        for i, v in enumerate(value):
            name = f"{default_name}_{i}"
            out[name] = make_unconstrained_pattern_like(v, default_name=name)
        return out
    if isinstance(value, numbers.Number) or hasattr(value, "shape"):
        if np.issubdtype(np.asarray(value).dtype, np.complexfloating):
            raise TypeError(f"{default_name}: complex values are not supported")
        shape = np.shape(value)
        if shape == ():
            return NumericScalarPattern()
        if len(shape) == 1:
            return NumericVectorPattern(shape[0])
        return NumericArrayPattern(shape)
    raise TypeError(f"{default_name}: cannot build a pattern for a leaf of type {type(value).__name__}")

