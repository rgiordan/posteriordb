"""Wrap a function of folded values so it takes flat or free vectors
instead (and the reverse), argument by argument.

    def objective(params, data): ...
    f = FoldFunctionInput(objective, patterns=params_pattern, free=True, argnums=0)
    f(free_vector, data)  ==  objective(params_pattern.fold(free_vector, free=True), data)

The wrappers are ordinary Python callables built from JAX operations, so the
result can be passed to `jax.grad`, `jax.jit` and `jax.vmap`.
"""

from __future__ import annotations

from typing import Sequence, Union

from .base import Pattern


def _as_list(x, n=None):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x] * (n if n is not None else 1)


class TransformFunctionInput:
    """Convert positional arguments `argnums` before calling `original_fun`.
    If `original_is_flat`, the wrapped function takes folded values and
    flattens them; otherwise it takes flat values and folds them."""

    def __init__(self, original_fun, patterns: Union[Pattern, Sequence[Pattern]], free, original_is_flat: bool,
                 argnums=0):
        self.original_fun = original_fun
        self.argnums = _as_list(argnums)
        self.patterns = _as_list(patterns, len(self.argnums))
        self.free = _as_list(free, len(self.argnums))
        self.original_is_flat = original_is_flat
        if not (len(self.argnums) == len(self.patterns) == len(self.free)):
            raise ValueError("argnums, patterns and free must have the same length")

    def __call__(self, *args, **kwargs):
        args = list(args)
        for i, pattern, free in zip(self.argnums, self.patterns, self.free):
            if self.original_is_flat:
                args[i] = pattern.flatten(args[i], free=free, validate_value=False)
            else:
                args[i] = pattern.fold(args[i], free=free, validate_value=False)
        return self.original_fun(*args, **kwargs)


class FoldFunctionInput(TransformFunctionInput):
    """`original_fun` takes folded values; the wrapper takes flat/free vectors."""

    def __init__(self, original_fun, patterns, free, argnums=0):
        super().__init__(original_fun, patterns, free, original_is_flat=False, argnums=argnums)


class FlattenFunctionInput(TransformFunctionInput):
    """`original_fun` takes flat/free vectors; the wrapper takes folded values."""

    def __init__(self, original_fun, patterns, free, argnums=0):
        super().__init__(original_fun, patterns, free, original_is_flat=True, argnums=argnums)


class FoldFunctionOutput:
    """`original_fun` returns a flat/free vector; the wrapper returns it folded."""

    def __init__(self, original_fun, pattern: Pattern, free: bool):
        self.original_fun, self.pattern, self.free = original_fun, pattern, free

    def __call__(self, *args, **kwargs):
        return self.pattern.fold(self.original_fun(*args, **kwargs), free=self.free, validate_value=False)


class FlattenFunctionOutput:
    """`original_fun` returns a folded value; the wrapper returns it flattened."""

    def __init__(self, original_fun, pattern: Pattern, free: bool):
        self.original_fun, self.pattern, self.free = original_fun, pattern, free

    def __call__(self, *args, **kwargs):
        return self.pattern.flatten(self.original_fun(*args, **kwargs), free=self.free, validate_value=False)
