"""Python source code that rebuilds a pattern."""

from __future__ import annotations

import math

from .base import Pattern
from .containers import PatternArray, PatternDict
from .matrix import CholeskyCorrPattern, CholeskyCovPattern, PSDSymmetricMatrixPattern
from .numeric import NumericArrayPattern, NumericScalarPattern, NumericVectorPattern, OrderedVectorPattern
from .simplex import SimplexArrayPattern


def pattern_to_code(pattern: Pattern, module: str = "pg", indent: int = 4) -> str:
    """Python code that constructs `pattern`, assuming paragami is imported
    as `module` (use `module=""` for names imported directly). Evaluating the
    code gives a pattern equal to `pattern`; arguments at their defaults are
    left out.

        print(pg.pattern_to_code(pattern))
    """
    return _code(pattern, (module + ".") if module else "", " " * indent, 0)


def _num(x) -> str:
    if math.isinf(x):
        return 'float("inf")' if x > 0 else '-float("inf")'
    return repr(float(x))


def _call(prefix, name, args, kwargs):
    parts = [*args] + [f"{k}={v}" for k, v in kwargs.items()]
    return f"{prefix}{name}({', '.join(parts)})"


def _code(p: Pattern, prefix: str, ind: str, depth: int) -> str:
    kw = {}
    if isinstance(p, PatternDict):
        if not len(p):
            return _call(prefix, "PatternDict", [], {"free_default": p.free_default} if p.free_default is not None else {})
        inner, outer = ind * (depth + 1), ind * depth
        entries = [(k, _code(v, prefix, ind, depth + 1)) for k, v in p.items()]
        if all(k.isidentifier() for k, _ in entries):
            body = ",\n".join(f"{inner}{k}={c}" for k, c in entries)
        else:
            body = f"{inner}{{\n" + ",\n".join(f"{inner}{ind}{k!r}: {c}" for k, c in entries) + f",\n{inner}}}"
        if p.free_default is not None:
            body += f",\n{inner}free_default={p.free_default}"
        return f"{prefix}PatternDict(\n{body},\n{outer})"

    if p.free_default is not None:
        kw["free_default"] = p.free_default
    t = type(p)
    if t in (NumericScalarPattern, NumericVectorPattern, NumericArrayPattern):
        bounds = {}
        if p.lb != -math.inf:
            bounds["lb"] = _num(p.lb)
        if p.ub != math.inf:
            bounds["ub"] = _num(p.ub)
        if t is NumericScalarPattern:
            return _call(prefix, t.__name__, [], {**bounds, **kw})
        if t is NumericVectorPattern:
            return _call(prefix, t.__name__, [str(p.shape[0])], {**bounds, **kw})
        order = {"order": repr(p.order)} if p.order != "C" else {}
        return _call(prefix, t.__name__, [repr(p.shape)], {**bounds, **order, **kw})
    if t is OrderedVectorPattern:
        return _call(prefix, t.__name__, [str(p.length)], {**({"positive": "True"} if p.positive else {}), **kw})
    if t is SimplexArrayPattern:
        shape = {"array_shape": repr(p.array_shape)} if p.array_shape else {}
        return _call(prefix, t.__name__, [str(p.simplex_size)], {**shape, **kw})
    if t in (CholeskyCovPattern, CholeskyCorrPattern, PSDSymmetricMatrixPattern):
        return _call(prefix, t.__name__, [str(p.size)], kw)
    if t is PatternArray:
        base = _code(p.base_pattern, prefix, ind, depth)
        return _call(prefix, t.__name__, [repr(p.array_shape), base], kw)
    raise TypeError(f"pattern_to_code does not know how to write a {t.__name__}")
