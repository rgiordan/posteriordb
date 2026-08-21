"""Declarative parameter-constraint specifications and their bijectors.

`ParameterSpec` mirrors Stan's own constraint vocabulary
(`data`/`parameters`-block type declarations) so a per-model `StatisticalModel`
subclass only ever needs to *declare* each parameter's shape and constraint
type -- never write constrain/unconstrain/Jacobian code itself.

Every bijector dispatched here either (a) is an off-the-shelf
`tensorflow_probability.substrates.jax` bijector (or `Chain` of them)
confirmed to be formula-identical to Stan's own transform, or (b) is one of
the custom bijectors in `stan_bijectors.py`, built for exactly this purpose.
See that module's docstring for what was checked and how.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from . import stan_bijectors as sb

tfb = tfp.bijectors


def _f64(v: float) -> jnp.ndarray:
    """Cast a bound literal to an explicit float64 array. TFP bijectors
    built from bare Python floats default to float32 regardless of
    `jax_enable_x64` -- this avoids a dtype mismatch against float64 model
    parameters."""
    return jnp.asarray(v, dtype=jnp.float64)


class ConstraintType(Enum):
    UNCONSTRAINED = auto()
    LOWER = auto()
    UPPER = auto()
    LOWER_UPPER = auto()
    ORDERED = auto()
    POSITIVE_ORDERED = auto()
    SIMPLEX = auto()
    CHOLESKY_FACTOR_CORR = auto()
    CHOLESKY_FACTOR_COV = auto()
    COV_MATRIX = auto()
    UNIT_VECTOR = auto()  # deferred: no Stan-exact bijector implemented yet
    CORR_MATRIX = auto()  # deferred: Stan's transform is not simply
    # cholesky_factor_corr + CholeskyOuterProduct for K>=4 (confirmed against
    # BridgeStan; see stan_bijectors.StanCholeskyFactorCorr's docstring and
    # tests/test_stan_bijectors.py::test_corr_matrix_cholesky_outer_product_boundary).
    # Not needed by any seed model (only appears as a *derived* quantity, at
    # K=2, in dogs_nonhierarchical).


_DEFERRED = {ConstraintType.UNIT_VECTOR, ConstraintType.CORR_MATRIX}

# Constraint types whose bijector.forward() already returns an array of the
# parameter's declared `shape` (SIMPLEX -> (K,); the Cholesky-ish types ->
# (K,K)). Every other type's bijector is shape-preserving/elementwise and
# operates on the flat (free_size,) == (constrained_size,) vector -- its
# output must be explicitly reshaped to `shape` by the caller (see
# `flatten.py`).
_SHAPED_OUTPUT = {
    ConstraintType.SIMPLEX,
    ConstraintType.CHOLESKY_FACTOR_CORR,
    ConstraintType.CHOLESKY_FACTOR_COV,
    ConstraintType.COV_MATRIX,
}


@dataclass(frozen=True)
class ParameterSpec:
    """One named model parameter's declared (constrained) shape and Stan
    constraint type.

    `shape` is always the *constrained*-space array shape, matching how Stan
    itself declares/prints the parameter: `()` for a scalar, `(J,)` for a
    `vector[J]` or `simplex[J]`/`ordered[J]`, `(K, K)` for
    `cholesky_factor_corr[K]`/`cholesky_factor_cov[K]`/`cov_matrix[K]`.

    `bounds` is only used for LOWER/UPPER/LOWER_UPPER.
    """

    name: str
    shape: Tuple[int, ...]
    constraint: ConstraintType
    bounds: Tuple[Optional[float], Optional[float]] = (None, None)

    def __post_init__(self):
        if self.constraint == ConstraintType.SIMPLEX:
            # Leading dims beyond the last are Stan's "array of simplex"
            # (e.g. `array[M] simplex[K]`) -- the last dim is always the
            # simplex dimension K. A bare `simplex[K]` is just the M=()
            # case.
            if len(self.shape) < 1:
                raise ValueError(f"{self.constraint} requires at least a 1-D shape, got {self.shape}")
        if self.constraint in (
            ConstraintType.ORDERED,
            ConstraintType.POSITIVE_ORDERED,
        ):
            if len(self.shape) != 1:
                raise ValueError(f"{self.constraint} requires a 1-D shape, got {self.shape}")
        if self.constraint in (
            ConstraintType.CHOLESKY_FACTOR_CORR,
            ConstraintType.CHOLESKY_FACTOR_COV,
            ConstraintType.COV_MATRIX,
            ConstraintType.CORR_MATRIX,
        ):
            if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
                raise ValueError(f"{self.constraint} requires a square shape, got {self.shape}")

    @property
    def constrained_size(self) -> int:
        return int(math.prod(self.shape)) if self.shape else 1

    @property
    def free_size(self) -> int:
        c = self.constraint
        if c == ConstraintType.SIMPLEX:
            # shape[:-1] is Stan's "array of" leading dims (empty for a
            # bare simplex[K]); each independent K-simplex contributes K-1
            # free coordinates.
            return int(math.prod(self.shape[:-1])) * (self.shape[-1] - 1)
        if c in (ConstraintType.ORDERED, ConstraintType.POSITIVE_ORDERED):
            return self.shape[0]
        if c == ConstraintType.CHOLESKY_FACTOR_CORR:
            k = self.shape[0]
            return k * (k - 1) // 2
        if c in (ConstraintType.CHOLESKY_FACTOR_COV, ConstraintType.COV_MATRIX):
            k = self.shape[0]
            return k * (k + 1) // 2
        if c == ConstraintType.CORR_MATRIX:
            k = self.shape[0]
            return k * (k - 1) // 2
        if c == ConstraintType.UNIT_VECTOR:
            return self.shape[0]
        # UNCONSTRAINED, LOWER, UPPER, LOWER_UPPER
        return self.constrained_size

    @property
    def matches_stan_unconstrained(self) -> bool:
        """Whether `.bijector` replicates Stan's own unconstrained-space
        transform exactly (vs. merely being support-equivalent). Every
        constraint type actually implemented here is Stan-exact by
        construction; this stays `True` unless a future constraint type is
        added without yet verifying its bijector against BridgeStan."""
        return self.constraint not in _DEFERRED

    @property
    def bijector_output_is_shaped(self) -> bool:
        """Whether `.bijector.forward()` already returns an array of `shape`
        (True for SIMPLEX and the Cholesky-ish types) vs. a flat
        (free_size,) == (constrained_size,) vector that must be reshaped to
        `shape` by the caller (True for every shape-preserving/elementwise
        type -- UNCONSTRAINED, LOWER, UPPER, LOWER_UPPER, ORDERED,
        POSITIVE_ORDERED)."""
        return self.constraint in _SHAPED_OUTPUT

    @property
    def free_shape(self) -> Tuple[int, ...]:
        """Shape a flat `(free_size,)` chunk is reshaped into before being
        handed to `.bijector.forward` (and the shape `.bijector.inverse`
        returns), as opposed to `.shape` (the full constrained array
        shape). Equal to `(free_size,)` -- a flat vector, i.e. a no-op
        reshape -- for every constraint type except SIMPLEX with leading
        "array of simplex" dims (e.g. `array[M] simplex[K]`), where it's
        `shape[:-1] + (K-1,)`: the same leading dims as `.shape`, with the
        trailing simplex dimension K replaced by its free dimension K-1.

        `ModelSpec.constrain`/`.unconstrain` pass `len(free_shape)` as
        `event_ndims` to the bijector, relying on TFP's standard behavior
        of summing a bijector's own per-instance log-det-Jacobian over any
        leading dims beyond its declared `forward_min_event_ndims` -- so
        `StanSimplex` (which already supports arbitrary leading batch dims
        in `_forward`/`_inverse`/`_forward_log_det_jacobian`) handles the
        "array of" case with no per-type special-casing in `flatten.py`.
        See `free_flatten_order` for the (surprising) element order used
        to reshape a flat chunk into this shape."""
        if self.constraint == ConstraintType.SIMPLEX:
            return self.shape[:-1] + (self.shape[-1] - 1,)
        return (self.free_size,)

    @property
    def free_flatten_order(self) -> str:
        """Element order used to reshape a flat `(free_size,)` chunk into
        `.free_shape` (and back) -- always "C" (row-major / blockwise: all
        of array-element 1's free coordinates, then element 2's, ...).

        This is a **real, empirically-confirmed discovery, not a design
        choice**: for `array[M] simplex[K]`, BridgeStan's own
        `param_unc_names()` reports an interleaved, "M-fastest" name order
        for the *unconstrained* parameters (matching `.flatten_order` ==
        "F", the order the *constrained* array actually uses) -- but the
        free parameter VECTOR'S ACTUAL VALUES do not follow that order.
        Feeding a BridgeStan-compiled `array[M] simplex[K]` model a known
        unconstrained vector and comparing `param_constrain` against an
        isolated single `simplex[K]` model fed each blockwise chunk
        confirms the free vector is laid out row-major (element 1's K-1
        free coords, then element 2's, ...), NOT in the order its own
        `param_unc_names()` would suggest. Trust values, not names, for
        this constraint type -- see
        `tests/test_stan_bijectors.py::test_simplex_array_matches_bridgestan`
        for the verification methodology. Irrelevant (a no-op either way)
        for every constraint type where `free_shape` is already 1-D."""
        return "C"

    @property
    def flatten_order(self) -> str:
        """Element order used when flattening this parameter's constrained
        array to/from posteriordb's flat per-element naming convention.
        Stan prints matrix-shaped parameters in column-major order."""
        return "F" if len(self.shape) >= 2 else "C"

    @functools.cached_property
    def bijector(self):
        c = self.constraint
        if c == ConstraintType.UNCONSTRAINED:
            return tfb.Identity()
        if c == ConstraintType.LOWER:
            lo, _ = self.bounds
            return tfb.Chain([tfb.Shift(_f64(lo)), tfb.Exp()])
        if c == ConstraintType.UPPER:
            _, hi = self.bounds
            return tfb.Chain([tfb.Shift(_f64(hi)), tfb.Scale(_f64(-1.0)), tfb.Exp()])
        if c == ConstraintType.LOWER_UPPER:
            lo, hi = self.bounds
            return tfb.Sigmoid(low=_f64(lo), high=_f64(hi))
        if c == ConstraintType.ORDERED:
            k = self.shape[0]
            return tfb.Chain(
                [tfb.Cumsum(), tfb.Blockwise([tfb.Identity(), tfb.Exp()], block_sizes=[1, k - 1])]
            )
        if c == ConstraintType.POSITIVE_ORDERED:
            return tfb.Chain([tfb.Cumsum(), tfb.Exp()])
        if c == ConstraintType.SIMPLEX:
            return sb.StanSimplex(self.shape[-1])
        if c == ConstraintType.CHOLESKY_FACTOR_CORR:
            return sb.StanCholeskyFactorCorr(self.shape[0])
        if c == ConstraintType.CHOLESKY_FACTOR_COV:
            return sb.StanCholeskyFactorCov(self.shape[0])
        if c == ConstraintType.COV_MATRIX:
            return tfb.Chain([tfb.CholeskyOuterProduct(), sb.StanCholeskyFactorCov(self.shape[0])])
        raise NotImplementedError(
            f"{c} has no Stan-exact bijector implemented yet (deferred -- "
            "not required by any seed model)."
        )
