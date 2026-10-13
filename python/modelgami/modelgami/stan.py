"""Use a Stan program as a `Model`, through BridgeStan (optional extra
`modelgami[stan]`).

    model = from_bridgestan("eight_schools.stan", data)
    f = model.eval_free_log_joint                          # Stan's log density
    jax.grad(f), jax.hessian(f), mg.make_hvp(f)            # from Stan's own derivatives

A Stan program accumulates a single `target`, and its data are fixed when
the model is loaded. So, compared with a model written in JAX:

- Only the joint density of the *free* parameters is available
  (`eval_free_log_joint`, with `jacobian=False` to leave out the log
  Jacobian). Prior and likelihood are not separated and there are no
  single-observation terms.
- Derivatives come from Stan: gradients exactly; Hessians and
  Hessian-vector products from BridgeStan (finite differences of gradients
  unless BridgeStan was built with autodiff Hessians). There are no
  derivatives with respect to data or hyperparameters.
- For new data, call `from_bridgestan` again with the compiled library
  (`model.bridgestan.lib_path`); nothing is recompiled.
- `model.params_pattern` uses Stan's own transforms, with folded values
  keyed by Stan's parameter names.

It is mainly a reference to compare a JAX translation against
(`modelgami.testing.assert_models_match`), and a way to run a method on a
Stan model as-is.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import paragami as pg

from .dgp import DGP
from .model import Model

try:
    import bridgestan
except ImportError:  # optional extra modelgami[stan]
    bridgestan = None

_NAME = re.compile(r"^([^.]+)((?:\.[0-9]+)*)$")


def from_bridgestan(stan_file_or_lib, data: Optional[dict] = None, *, name: Optional[str] = None,
                    seed: int = 1234, **stan_kwargs) -> "StanModel":
    """Load a Stan program (or a compiled library) with its data. For new
    data, call it again with the library path (`model.bridgestan.lib_path`);
    nothing is recompiled."""
    if bridgestan is None:
        raise ImportError("from_bridgestan needs BridgeStan: pip install 'modelgami[stan]'")
    bs = bridgestan.StanModel(str(stan_file_or_lib), _jsonable(data or {}), seed=seed, **stan_kwargs)
    return StanModel(bs=bs, data=data, name=name or bs.name())


class StanModel(Model):
    """A `Model` whose log density is Stan's. Only `eval_free_log_joint`
    exists; `prior` and `likelihood` are empty DGPs, so the other methods
    raise NotImplementedError."""

    def __init__(self, *, bs, data, name):
        super().__init__(params_pattern=StanParamsPattern(bs), likelihood=DGP(), prior=DGP(), data=data, name=name)
        self.bridgestan = bs
        self._log_density = _differentiable_density(bs)

    def eval_free_log_joint(self, free_params, data=None, hyper=None, jacobian: bool = True):
        """Stan's log density of the free parameters, with the log Jacobian
        unless `jacobian=False`. The data and hyperparameters are fixed."""
        if data is not None or hyper is not None:
            raise NotImplementedError(f"{self.name} has fixed data; call from_bridgestan with new data")
        return self._log_density(free_params) if jacobian else _plain(bs=self.bridgestan, x=free_params)


def _plain(*, bs, x):
    """Log density without the Jacobian; gradient via Stan too."""
    @jax.custom_jvp
    def f(x):
        return _cb(lambda x: bs.log_density(x, propto=False, jacobian=False), (), x)

    @f.defjvp
    def f_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        g = _cb(lambda x: bs.log_density_gradient(x, propto=False, jacobian=False)[1], x.shape, x)
        return f(x), jnp.dot(g, t)

    return f(x)


def _differentiable_density(bs):
    """Stan's log density (with Jacobian) as a JAX function with Stan's
    gradient and Hessian-vector products as its derivatives."""

    def grad_np(x):
        return bs.log_density_gradient(x, propto=False, jacobian=True)[1]

    def hvp_np(x, v):
        return bs.log_density_hessian_vector_product(x, v, propto=False, jacobian=True)[1]

    @jax.custom_jvp
    def grad(x):
        return _cb(grad_np, x.shape, x)

    @grad.defjvp
    def grad_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        return grad(x), _cb(hvp_np, x.shape, x, t)

    @jax.custom_jvp
    def lp(x):
        return _cb(lambda x: bs.log_density(x, propto=False, jacobian=True), (), x)

    @lp.defjvp
    def lp_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        return lp(x), jnp.dot(grad(x), t)

    return lp


def _cb(fn, shape, *args):
    """Call numpy code from JAX (works under jit; vmap runs it sequentially)."""
    out = jax.ShapeDtypeStruct(shape, jnp.float64)
    return jax.pure_callback(lambda *a: np.asarray(fn(*[np.asarray(x, dtype=np.float64) for x in a]),
                                                   dtype=np.float64),
                             out, *args, vmap_method="sequential")


class StanParamsPattern(pg.Pattern):
    """Stan's own constrain/unconstrain, as a paragami pattern over a dict
    keyed by Stan parameter names. Folding is not differentiable (it calls
    Stan); log densities should be taken through `StanModel.eval_free_log_joint`."""

    def __init__(self, bs):
        self._bs = bs
        self._stan_names = list(bs.param_names())
        self._shapes = _shapes_from_names(self._stan_names)
        super().__init__(flat_length=len(self._stan_names), free_length=bs.param_unc_num())

    def to_dict(self):
        return {"stan_model": self._bs.name(), "shapes": self._shapes}

    def generate_code(self, *, module: str = "pg", indent: int = 4) -> str:
        raise NotImplementedError("a StanParamsPattern is built from a BridgeStan model: "
                                  "use modelgami.from_bridgestan")

    def _flat_to_folded(self, flat):
        return _fold_by_names(names=self._stan_names, shapes=self._shapes, flat=flat)

    def _folded_to_flat(self, folded):
        return jnp.asarray([jnp.asarray(folded[b])[i] for b, i in map(_parse, self._stan_names)])

    def _free_to_folded(self, free):
        flat = _cb(lambda x: self._bs.param_constrain(x), (len(self._stan_names),), free)
        return self._flat_to_folded(flat)

    def _free_to_folded_log_det(self, free):
        # Stan does not expose the log det alone; it is part of eval_free_log_joint.
        return jnp.full((), jnp.nan)

    def _folded_to_free(self, folded):
        flat = self._folded_to_flat(folded)
        return _cb(lambda y: self._bs.param_unconstrain(y), (self._free_length,), flat)

    def _validate(self, folded, validate_value):
        if not isinstance(folded, dict) or set(folded) != set(self._shapes):
            return f"expected a dict with keys {sorted(self._shapes)}"
        for k, s in self._shapes.items():
            if tuple(np.shape(folded[k])) != s:
                return f"[{k!r}] expected shape {s}, got {np.shape(folded[k])}"
        return None

    def _names(self, free):
        return list(self._bs.param_unc_names()) if free else list(self._stan_names)


def _parse(name: str):
    m = _NAME.match(name)
    idx = tuple(int(i) - 1 for i in m.group(2).strip(".").split(".")) if m.group(2) else ()
    return m.group(1), idx


def _shapes_from_names(names: Sequence[str]) -> Dict[str, tuple]:
    shapes = {}
    for b, idx in map(_parse, names):
        cur = shapes.get(b, tuple(0 for _ in idx))
        shapes[b] = tuple(max(c, i + 1) for c, i in zip(cur, idx))
    return shapes


def _fold_by_names(*, names, shapes, flat):
    out = {b: jnp.zeros(s) for b, s in shapes.items()}
    for j, (b, idx) in enumerate(map(_parse, names)):
        out[b] = out[b].at[idx].set(flat[j])
    return out


def _jsonable(data):
    return {k: (np.asarray(v).tolist() if hasattr(v, "tolist") or isinstance(v, (list, tuple)) else v)
            for k, v in data.items()}
