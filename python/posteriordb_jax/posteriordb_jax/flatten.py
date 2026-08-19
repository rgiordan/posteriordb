"""Flatten/unflatten utilities shared by every model: convert between a
dict of named, shaped, constrained (or unconstrained) parameter arrays and
(a) a single flat vector per draw, and (b) posteriordb's own flat
per-element naming convention (e.g. "theta[1]", "z[3,2]", 1-indexed,
column-major for matrix-shaped parameters -- matching how Stan itself
prints them).

Entirely constraint-type-agnostic: built only on `ParameterSpec.shape`,
`.constrained_size`, `.free_size`, `.flatten_order`, and `.bijector`. A
per-model `StatisticalModel` subclass never needs its own shape or
flattening code.

Every leaf array may carry an optional leading batch dimension (i.e. shape
`batch_shape + spec.shape`, with `batch_shape` either `()` or `(N,)`) --
every function here works uniformly for a single draw or a stack of draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from .constraints import ParameterSpec


def _flatten_last_dims(x: jnp.ndarray, param_ndim: int, order: str) -> jnp.ndarray:
    """Flatten the last `param_ndim` axes of `x` into one axis, in `order`
    ("C" or "F"), leaving any leading batch axes untouched."""
    if param_ndim == 0:
        return x[..., None]
    batch_ndim = x.ndim - param_ndim
    if order == "F":
        axes = list(range(batch_ndim)) + list(reversed(range(batch_ndim, x.ndim)))
        x = jnp.transpose(x, axes)
    batch_shape = x.shape[:batch_ndim]
    return x.reshape(batch_shape + (-1,))


def _unflatten_last_dim(flat: jnp.ndarray, shape: Tuple[int, ...], order: str) -> jnp.ndarray:
    """Inverse of `_flatten_last_dims`: expand the last axis of `flat` into
    `shape`, in `order`, leaving any leading batch axes untouched."""
    batch_shape = flat.shape[:-1]
    if len(shape) == 0:
        return flat[..., 0]
    if order == "F":
        x = flat.reshape(batch_shape + tuple(reversed(shape)))
        batch_ndim = len(batch_shape)
        axes = list(range(batch_ndim)) + list(reversed(range(batch_ndim, batch_ndim + len(shape))))
        return jnp.transpose(x, axes)
    return flat.reshape(batch_shape + tuple(shape))


def _element_keys(name: str, shape: Tuple[int, ...], order: str) -> List[str]:
    """1-indexed posteriordb-style element keys for a parameter, in the
    given flatten order (matching Stan's own print convention: "F" i.e.
    column-major/first-index-fastest for matrix-shaped parameters)."""
    if len(shape) == 0:
        return [name]
    ranges = [range(1, d + 1) for d in shape]
    if order == "F":
        # first index varies fastest
        idx_tuples = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(len(shape), -1, order="F").T
    else:
        idx_tuples = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(len(shape), -1, order="C").T
    return [f"{name}[{','.join(str(i) for i in idx)}]" for idx in idx_tuples]


@dataclass(frozen=True)
class ModelSpec:
    specs: Tuple[ParameterSpec, ...]

    def __post_init__(self):
        object.__setattr__(self, "specs", tuple(self.specs))

    @property
    def free_dim(self) -> int:
        return sum(s.free_size for s in self.specs)

    @property
    def constrained_dim(self) -> int:
        return sum(s.constrained_size for s in self.specs)

    # -- constrained <-> flat -------------------------------------------------

    def flatten_constrained(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        chunks = []
        for s in self.specs:
            chunks.append(_flatten_last_dims(jnp.asarray(params[s.name]), len(s.shape), s.flatten_order))
        return jnp.concatenate(chunks, axis=-1)

    def unflatten_constrained(self, flat: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        out = {}
        offset = 0
        for s in self.specs:
            chunk = flat[..., offset : offset + s.constrained_size]
            out[s.name] = _unflatten_last_dim(chunk, s.shape, s.flatten_order)
            offset += s.constrained_size
        return out

    # -- unconstrained <-> flat -------------------------------------------------

    def flatten_unconstrained(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([jnp.asarray(params[s.name]) for s in self.specs], axis=-1)

    def unflatten_unconstrained(self, flat: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        out = {}
        offset = 0
        for s in self.specs:
            out[s.name] = flat[..., offset : offset + s.free_size]
            offset += s.free_size
        return out

    # -- constrain / unconstrain -------------------------------------------------

    def constrain(self, unconstrained_flat: jnp.ndarray) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        params = {}
        logdet = jnp.zeros(unconstrained_flat.shape[:-1])
        offset = 0
        for s in self.specs:
            chunk = unconstrained_flat[..., offset : offset + s.free_size]
            raw = s.bijector.forward(chunk)
            params[s.name] = raw if s.bijector_output_is_shaped else _unflatten_last_dim(
                raw, s.shape, s.flatten_order
            )
            logdet = logdet + s.bijector.forward_log_det_jacobian(chunk, event_ndims=1)
            offset += s.free_size
        return params, logdet

    def unconstrain(self, params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        chunks = []
        logdet = None
        for s in self.specs:
            x = jnp.asarray(params[s.name])
            if s.bijector_output_is_shaped:
                inp, event_ndims = x, len(s.shape)
            else:
                inp, event_ndims = _flatten_last_dims(x, len(s.shape), s.flatten_order), 1
            chunk = s.bijector.inverse(inp)
            chunks.append(chunk)
            ildj = s.bijector.inverse_log_det_jacobian(inp, event_ndims=event_ndims)
            logdet = ildj if logdet is None else logdet + ildj
        return jnp.concatenate(chunks, axis=-1), logdet

    # -- posteriordb dict interop -------------------------------------------------

    def to_posteriordb_dict(self, params: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        out = {}
        for s in self.specs:
            flat = _flatten_last_dims(jnp.asarray(params[s.name]), len(s.shape), s.flatten_order)
            keys = _element_keys(s.name, s.shape, s.flatten_order)
            values = np.asarray(flat)
            for k, v in zip(keys, np.moveaxis(values, -1, 0)):
                out[k] = v if v.ndim > 0 else float(v)
        return out

    def from_posteriordb_dict(self, flat_dict: Dict[str, float]) -> Dict[str, jnp.ndarray]:
        out = {}
        for s in self.specs:
            keys = _element_keys(s.name, s.shape, s.flatten_order)
            values = jnp.stack([jnp.asarray(flat_dict[k]) for k in keys], axis=-1)
            out[s.name] = _unflatten_last_dim(values, s.shape, s.flatten_order)
        return out
