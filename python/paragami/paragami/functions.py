"""Change the representation a function takes or returns.

Each wrapper is named for what changes: `function_input_<from>_to_<to>`
changes the representation of some positional arguments, and
`function_output_<from>_to_<to>` changes the representation of the result.

    def objective(params, data): ...
    f = function_input_folded_to_free(objective, params_pattern, argnums=0)
    f(free_vector, data)  ==  objective(params_pattern.free_to_folded(free_vector), data)

The wrappers are ordinary Python functions built from JAX operations, so the
result can be passed to `jax.grad`, `jax.jit` and `jax.vmap`.
"""

from __future__ import annotations

import functools
from typing import Sequence, Union

from .base import Pattern

Patterns = Union[Pattern, Sequence[Pattern]]


def _as_list(x, n=None):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x] * (n if n is not None else 1)


def _convert_inputs(fun, patterns: Patterns, argnums, convert):
    """Apply `convert(pattern, value)` to positional arguments `argnums` before calling `fun`."""
    argnums = _as_list(argnums)
    patterns = _as_list(patterns, len(argnums))
    if len(patterns) != len(argnums):
        raise ValueError("patterns and argnums must have the same length")

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        args = list(args)
        for i, pattern in zip(argnums, patterns):
            args[i] = convert(pattern, args[i])
        return fun(*args, **kwargs)

    return wrapped


def _convert_output(fun, pattern: Pattern, convert):
    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        return convert(pattern, fun(*args, **kwargs))

    return wrapped


# -- inputs: `fun` takes the first representation; the wrapper takes the second ------------

def function_input_folded_to_free(fun, patterns: Patterns, argnums=0):
    """`fun` takes folded values; the returned function takes free vectors."""
    return _convert_inputs(fun, patterns, argnums, lambda p, v: p.free_to_folded(v))


def function_input_folded_to_flat(fun, patterns: Patterns, argnums=0):
    """`fun` takes folded values; the returned function takes flat vectors."""
    return _convert_inputs(fun, patterns, argnums, lambda p, v: p.flat_to_folded(v, validate_value=False))


def function_input_free_to_folded(fun, patterns: Patterns, argnums=0):
    """`fun` takes free vectors; the returned function takes folded values."""
    return _convert_inputs(fun, patterns, argnums, lambda p, v: p.folded_to_free(v, validate_value=False))


def function_input_flat_to_folded(fun, patterns: Patterns, argnums=0):
    """`fun` takes flat vectors; the returned function takes folded values."""
    return _convert_inputs(fun, patterns, argnums, lambda p, v: p.folded_to_flat(v, validate_value=False))


# -- outputs: `fun` returns the first representation; the wrapper returns the second --------

def function_output_free_to_folded(fun, pattern: Pattern):
    """`fun` returns a free vector; the returned function returns it folded."""
    return _convert_output(fun, pattern, lambda p, v: p.free_to_folded(v))


def function_output_flat_to_folded(fun, pattern: Pattern):
    """`fun` returns a flat vector; the returned function returns it folded."""
    return _convert_output(fun, pattern, lambda p, v: p.flat_to_folded(v, validate_value=False))


def function_output_folded_to_free(fun, pattern: Pattern):
    """`fun` returns a folded value; the returned function returns its free vector."""
    return _convert_output(fun, pattern, lambda p, v: p.folded_to_free(v, validate_value=False))


def function_output_folded_to_flat(fun, pattern: Pattern):
    """`fun` returns a folded value; the returned function returns its flat vector."""
    return _convert_output(fun, pattern, lambda p, v: p.folded_to_flat(v, validate_value=False))
