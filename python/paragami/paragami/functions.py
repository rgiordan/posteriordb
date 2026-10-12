"""Wrap a function of folded values so it takes flat or free vectors
instead (and the reverse), argument by argument.

    def objective(params, data): ...
    f = FoldFunctionInput(objective, patterns=params_pattern, from_free=True, argnums=0)
    f(free_vector, data)  ==  objective(params_pattern.free_to_folded(free_vector), data)

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
    If `original_takes_vectors`, the wrapped function takes folded values and
    flattens them; otherwise it takes vectors and folds them. `vector_is_free`
    says whether those vectors are free (True) or flat (False)."""

    def __init__(self, original_fun, patterns: Union[Pattern, Sequence[Pattern]], vector_is_free,
                 original_takes_vectors: bool, argnums=0):
        self.original_fun = original_fun
        self.argnums = _as_list(argnums)
        self.patterns = _as_list(patterns, len(self.argnums))
        self.vector_is_free = _as_list(vector_is_free, len(self.argnums))
        self.original_takes_vectors = original_takes_vectors
        if not (len(self.argnums) == len(self.patterns) == len(self.vector_is_free)):
            raise ValueError("argnums, patterns and vector_is_free must have the same length")

    def __call__(self, *args, **kwargs):
        args = list(args)
        for i, pattern, free in zip(self.argnums, self.patterns, self.vector_is_free):
            if self.original_takes_vectors:
                args[i] = pattern.flatten(args[i], to_free=free, validate_value=False)
            else:
                args[i] = pattern.fold(args[i], from_free=free, validate_value=False)
        return self.original_fun(*args, **kwargs)


class FoldFunctionInput(TransformFunctionInput):
    """`original_fun` takes folded values; the wrapper takes flat vectors, or
    free vectors if `from_free`."""

    def __init__(self, original_fun, patterns, from_free, argnums=0):
        super().__init__(original_fun, patterns, from_free, original_takes_vectors=False, argnums=argnums)


class FlattenFunctionInput(TransformFunctionInput):
    """`original_fun` takes flat vectors, or free vectors if `to_free`; the
    wrapper takes folded values."""

    def __init__(self, original_fun, patterns, to_free, argnums=0):
        super().__init__(original_fun, patterns, to_free, original_takes_vectors=True, argnums=argnums)


class FoldFunctionOutput:
    """`original_fun` returns a flat vector, or a free vector if `from_free`;
    the wrapper returns it folded."""

    def __init__(self, original_fun, pattern: Pattern, from_free: bool):
        self.original_fun, self.pattern, self.from_free = original_fun, pattern, from_free

    def __call__(self, *args, **kwargs):
        return self.pattern.fold(self.original_fun(*args, **kwargs), from_free=self.from_free, validate_value=False)


class FlattenFunctionOutput:
    """`original_fun` returns a folded value; the wrapper returns it as a flat
    vector, or a free vector if `to_free`."""

    def __init__(self, original_fun, pattern: Pattern, to_free: bool):
        self.original_fun, self.pattern, self.to_free = original_fun, pattern, to_free

    def __call__(self, *args, **kwargs):
        return self.pattern.flatten(self.original_fun(*args, **kwargs), to_free=self.to_free, validate_value=False)
