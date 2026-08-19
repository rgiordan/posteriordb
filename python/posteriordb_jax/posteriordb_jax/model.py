"""`StatisticalModel`: the base class every hand-converted Stan model
subclasses.

A subclass writes exactly one thing: `log_density(params, data)`, a scalar
log-density function of a *structured* dict of constrained parameter arrays
plus an **explicit** data pytree argument (never read from `self.data`
internally). That explicitness is what lets `jax.grad` differentiate with
respect to data, not just parameters -- something Stan itself cannot do.

Everything else (constrain/unconstrain, flattening, log-prob in either
parameterization, gradients, Hessians, Hessian-vector products, batching,
and gradients with respect to data) is derived generically here via JAX
transforms, using the shared `constraints.py`/`flatten.py` machinery.
"""

from __future__ import annotations

import abc
from typing import Dict, List

import jax
import jax.numpy as jnp

from .constraints import ParameterSpec
from .flatten import ModelSpec


class StatisticalModel(abc.ABC):
    def __init__(self, data):
        self.data = data
        self.specs: List[ParameterSpec] = self.param_specs(data)
        self.model_spec = ModelSpec(tuple(self.specs))

    @abc.abstractmethod
    def param_specs(self, data) -> List[ParameterSpec]:
        """Declare each parameter's shape and Stan constraint type. May
        depend on `data`'s shapes (e.g. `J = len(data["y"])`) -- must NOT
        depend on external metadata like posteriordb's `posteriors/*.json`
        `"dimensions"` field, which is not a reliable shape source."""

    @abc.abstractmethod
    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        """User-written: scalar log density (prior + likelihood, matching
        the Stan model's `model` + `transformed parameters` blocks) as a
        function of a structured dict of *constrained*-space parameter
        arrays (keyed by `ParameterSpec.name`, each with its declared
        `shape`) and an explicit `data` pytree. No Jacobian term -- that's
        added generically by `log_prob_unconstrained` below."""

    # -- core flat-scalar functions, single draw -----------------------------

    def _lp_unconstrained_flat(self, x: jnp.ndarray, data) -> jnp.ndarray:
        params, logdet = self.model_spec.constrain(x)
        return self.log_density(params, data) + logdet

    def _lp_constrained_flat(self, y: jnp.ndarray, data) -> jnp.ndarray:
        params = self.model_spec.unflatten_constrained(y)
        return self.log_density(params, data)

    # -- public single-draw API ---------------------------------------------

    def log_prob_unconstrained(self, x: jnp.ndarray, data=None) -> jnp.ndarray:
        return self._lp_unconstrained_flat(x, self.data if data is None else data)

    def log_prob_constrained(self, y: jnp.ndarray, data=None) -> jnp.ndarray:
        return self._lp_constrained_flat(y, self.data if data is None else data)

    def grad_unconstrained(self, x: jnp.ndarray, data=None) -> jnp.ndarray:
        return jax.grad(self._lp_unconstrained_flat)(x, self.data if data is None else data)

    def hessian_unconstrained(self, x: jnp.ndarray, data=None) -> jnp.ndarray:
        return jax.hessian(self._lp_unconstrained_flat)(x, self.data if data is None else data)

    def hvp_unconstrained(self, x: jnp.ndarray, v: jnp.ndarray, data=None) -> jnp.ndarray:
        d = self.data if data is None else data
        _, jvp_out = jax.jvp(lambda x_: self.grad_unconstrained(x_, d), (x,), (v,))
        return jvp_out

    def grad_constrained(self, y: jnp.ndarray, data=None) -> jnp.ndarray:
        return jax.grad(self._lp_constrained_flat)(y, self.data if data is None else data)

    def hessian_constrained(self, y: jnp.ndarray, data=None) -> jnp.ndarray:
        return jax.hessian(self._lp_constrained_flat)(y, self.data if data is None else data)

    def hvp_constrained(self, y: jnp.ndarray, v: jnp.ndarray, data=None) -> jnp.ndarray:
        d = self.data if data is None else data
        _, jvp_out = jax.jvp(lambda y_: self.grad_constrained(y_, d), (y,), (v,))
        return jvp_out

    def grad_wrt_data(self, x_or_y: jnp.ndarray, data=None, space: str = "unconstrained"):
        """Gradient of the log density with respect to `data` (a pytree
        matching `self.data`'s structure). `allow_int=True` lets static
        integer fields (e.g. a dimension like `J`) sit in the same pytree as
        the differentiable float arrays (e.g. `y`) without erroring --
        int-dtype leaves come back as JAX's placeholder `float0` gradient
        rather than a real (and meaningless) value."""
        d = self.data if data is None else data
        fn = self._lp_unconstrained_flat if space == "unconstrained" else self._lp_constrained_flat
        return jax.grad(fn, argnums=1, allow_int=True)(x_or_y, d)

    # -- batched (vmap) variants ---------------------------------------------

    def log_prob_unconstrained_batch(self, x: jnp.ndarray, data=None) -> jnp.ndarray:
        d = self.data if data is None else data
        return jax.vmap(self._lp_unconstrained_flat, in_axes=(0, None))(x, d)

    def log_prob_constrained_batch(self, y: jnp.ndarray, data=None) -> jnp.ndarray:
        d = self.data if data is None else data
        return jax.vmap(self._lp_constrained_flat, in_axes=(0, None))(y, d)

    def grad_unconstrained_batch(self, x: jnp.ndarray, data=None) -> jnp.ndarray:
        d = self.data if data is None else data
        return jax.vmap(jax.grad(self._lp_unconstrained_flat), in_axes=(0, None))(x, d)

    def grad_constrained_batch(self, y: jnp.ndarray, data=None) -> jnp.ndarray:
        d = self.data if data is None else data
        return jax.vmap(jax.grad(self._lp_constrained_flat), in_axes=(0, None))(y, d)

    def hessian_unconstrained_batch(self, x: jnp.ndarray, data=None) -> jnp.ndarray:
        """O(n_draws * free_dim^2) -- fine for the small models in this repo,
        but prefer `hvp_unconstrained_batch` when only Hessian-vector
        products are needed, which stays O(n_draws * free_dim)."""
        d = self.data if data is None else data
        return jax.vmap(jax.hessian(self._lp_unconstrained_flat), in_axes=(0, None))(x, d)

    def hessian_constrained_batch(self, y: jnp.ndarray, data=None) -> jnp.ndarray:
        d = self.data if data is None else data
        return jax.vmap(jax.hessian(self._lp_constrained_flat), in_axes=(0, None))(y, d)

    def hvp_unconstrained_batch(self, x: jnp.ndarray, v: jnp.ndarray, data=None) -> jnp.ndarray:
        d = self.data if data is None else data
        return jax.vmap(lambda x_, v_: self.hvp_unconstrained(x_, v_, d), in_axes=(0, 0))(x, v)

    def hvp_constrained_batch(self, y: jnp.ndarray, v: jnp.ndarray, data=None) -> jnp.ndarray:
        d = self.data if data is None else data
        return jax.vmap(lambda y_, v_: self.hvp_constrained(y_, v_, d), in_axes=(0, 0))(y, v)

    # -- constrain / unconstrain / flatten passthroughs ----------------------

    def constrain(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        params, _ = self.model_spec.constrain(x)
        return params

    def unconstrain(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        flat, _ = self.model_spec.unconstrain(params)
        return flat

    def to_posteriordb_dict(self, params: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        return self.model_spec.to_posteriordb_dict(params)

    def from_posteriordb_dict(self, flat_dict: Dict[str, float]) -> Dict[str, jnp.ndarray]:
        return self.model_spec.from_posteriordb_dict(flat_dict)
