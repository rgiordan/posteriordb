"""`Model`: write the statistics once, get the rest generated.

A model is defined by a few user functions ("hooks"), written against
intuitive structures (dicts of arrays):

    params_pattern               a paragami pattern for the parameters
    log_prior(params, hyper)     optional
    log_lik(params, data, hyper)                    -- or, for exchangeable data,
    log_lik_one(params, datum, hyper)               -- one observation, or
    obs_dists_one(params, datum, hyper) -> {field: distribution}
                                                    -- one observation, as
                                                       distribution objects; gives
                                                       both log_lik_one and sample_one
    log_lik_pointwise(params, data, hyper) -> (N,)  -- all observations at once
    sample_prior(key, hyper) -> params              optional
    prior_dists(hyper) -> {param: distribution or f(earlier params)}
                                                    gives both log_prior and sample_prior
    sample_one(key, params, datum, hyper) -> datum  optional (exchangeable)
    sample_data(key, params, data, hyper) -> data   optional (general)
    check_params(params), check_data(data)          optional extra validation

Pass them to `Model(...)` as keyword arguments, or subclass `Model` and
define methods with the same names (plus `params_pattern` as an attribute or
a method of no arguments). Everything else is derived:

- `log_density_fn(which, params_as, data_as)`: the joint, likelihood or prior
  as a JAX function of parameters (and optionally data) in folded, flat or
  free form, ready for `jax.grad`, `jax.hessian`, `jax.vmap`, `jax.jit`.
- For exchangeable models: pointwise log likelihoods, a likelihood with
  data weights, data subsets for minibatching and the bootstrap.
- Folding/flattening helpers, random parameters, validation, simulation,
  and ArviZ export.

The hooks are re-exposed on the model with the model's own data and
hyperparameters as defaults, e.g. `model.log_lik(params)`.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import paragami as pg

from .dgp import DGP
from .validation import ValidationError, run_check

_REPRESENTATIONS = ("folded", "flat", "free")
_HOOKS = (
    "log_prior", "log_lik", "log_lik_one", "log_lik_pointwise",
    "sample_prior", "sample_one", "sample_data", "check_params", "check_data", "obs_dists_one",
    "prior_dists",
)


class Model:
    def __init__(
        self,
        *,
        params_pattern: Optional[pg.Pattern] = None,
        data: Any = None,
        shared_data: Optional[Dict[str, Any]] = None,
        hyper: Optional[Dict[str, Any]] = None,
        data_pattern: Optional[pg.Pattern] = None,
        name: Optional[str] = None,
        validate: bool = True,
        **hooks: Optional[Callable],
    ):
        unknown = set(hooks) - set(_HOOKS)
        if unknown:
            raise TypeError(f"unknown arguments {sorted(unknown)}; hooks are {_HOOKS}")
        # User functions: keyword arguments win over subclass methods.
        self._user = {}
        for h in _HOOKS:
            fn = hooks.get(h) or getattr(self, h, None)
            if fn is not None:
                self._user[h] = fn

        self.name = name or type(self).__name__
        self.hyper = {} if hyper is None else dict(hyper)

        if params_pattern is None:
            params_pattern = getattr(self, "params_pattern", None)
            if callable(params_pattern) and not isinstance(params_pattern, pg.Pattern):
                self.data, self.shared_data = data, shared_data  # available to the method
                params_pattern = params_pattern()
        if params_pattern is not None and not isinstance(params_pattern, pg.Pattern):
            raise TypeError("params_pattern must be a paragami Pattern")
        self.params_pattern = params_pattern

        u = self._user
        if "obs_dists_one" in u:
            if "log_lik_one" in u:
                raise TypeError("give either obs_dists_one or log_lik_one, not both")
            u["log_lik_one"], one_sampler = _from_obs_dists(u["obs_dists_one"])
            u.setdefault("sample_one", one_sampler)
        if "prior_dists" in u:
            if "log_prior" in u:
                raise TypeError("give either prior_dists or log_prior, not both")
            u["log_prior"], prior_sampler = _from_prior_dists(u["prior_dists"])
            u.setdefault("sample_prior", prior_sampler)
        if "log_lik" in u and ("log_lik_one" in u or "log_lik_pointwise" in u):
            raise TypeError("give either log_lik, or log_lik_one / log_lik_pointwise for exchangeable data, not both")
        if "log_lik" not in u and "log_lik_one" not in u and "log_lik_pointwise" not in u:
            raise TypeError("a model needs log_lik, log_lik_one or log_lik_pointwise")
        self.exchangeable = "log_lik_one" in u or "log_lik_pointwise" in u

        self._set_data(data, shared_data)
        self.data_pattern = data_pattern if data_pattern is not None else _infer_data_pattern(self.data)
        self._expose_hooks()
        if validate:
            self.validate_data()

    # -- data -------------------------------------------------------------------

    def _set_data(self, data, shared_data):
        shared = {k: _to_array(v) for k, v in (shared_data or {}).items()}
        if self.exchangeable:
            if isinstance(data, (list, tuple)):
                if not data:
                    raise ValidationError("data is an empty list")
                data = jax.tree_util.tree_map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *data)
            if not isinstance(data, dict):
                raise ValidationError("exchangeable data must be a list of dicts or a dict of stacked arrays")
            obs = {k: jnp.asarray(v) for k, v in data.items()}
            clash = set(obs) & set(shared)
            if clash:
                raise ValidationError(f"keys {sorted(clash)} are in both data and shared_data")
            sizes = {k: (v.shape[0] if v.ndim else None) for k, v in obs.items()}
            if any(s is None for s in sizes.values()) or len(set(sizes.values())) != 1:
                raise ValidationError(f"every per-observation field needs the same leading length N, got {sizes}")
            self.obs_keys = tuple(obs)
            self.n_obs = next(iter(sizes.values()))
            self.data = {**obs, **shared}
        else:
            self.obs_keys, self.n_obs = (), None
            if isinstance(data, dict):
                data = {k: _to_array(v) for k, v in data.items()}
                clash = set(data) & set(shared)
                if clash:
                    raise ValidationError(f"keys {sorted(clash)} are in both data and shared_data")
                data = {**data, **shared}
            elif shared:
                raise ValidationError("shared_data needs data to be a dict")
            self.data = data
        self.shared_data = shared

    def with_data(self, data=None, shared_data=None, hyper=None) -> "Model":
        """A copy of this model with new data (and/or hyperparameters). No
        recompilation; the parameter pattern is kept."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        if hyper is not None:
            new.hyper = dict(hyper)
        if data is not None or shared_data is not None:
            if data is None:
                data = {k: self.data[k] for k in self.obs_keys} if self.exchangeable else self.data
            new._set_data(data, shared_data if shared_data is not None else self.shared_data)
            new.data_pattern = _infer_data_pattern(new.data)
        new._expose_hooks()  # rebind the derived functions to the copy
        new.validate_data()
        return new

    def _datum_axes(self, data):
        return {k: (0 if k in self.obs_keys else None) for k in data}

    def _fold_data(self, d, data_as):
        if data_as == "folded":
            return d
        if self.data_pattern is None:
            raise ValueError(f"data_as={data_as!r} needs a data_pattern")
        folded = self.data_pattern.fold(d, free=(data_as == "free"), validate_value=False)
        return {**self.data, **folded} if isinstance(self.data, dict) else folded

    # -- hooks with defaults -------------------------------------------------------

    def _expose_hooks(self):
        u = self._user
        if self.exchangeable:
            if "log_lik_pointwise" in u:
                pointwise = u["log_lik_pointwise"]
            else:
                one = u["log_lik_one"]

                def pointwise(params, data, hyper):
                    return jax.vmap(one, in_axes=(None, self._datum_axes(data), None))(params, data, hyper)

            def log_lik(params, data, hyper, weights=None):
                lp = pointwise(params, data, hyper)
                return jnp.sum(lp) if weights is None else jnp.sum(weights * lp)

            self._pointwise = pointwise
            self.log_lik_pointwise = self._defaults(pointwise)
        else:
            log_lik = u["log_lik"]
        self._log_lik = log_lik
        self.log_lik = self._defaults(log_lik)
        self._log_prior = u.get("log_prior")
        if self._log_prior is not None:
            self.log_prior = lambda params, hyper=None: self._log_prior(params, self._h(hyper))
        if "sample_prior" in u:
            self.sample_prior = lambda key, hyper=None: u["sample_prior"](key, self._h(hyper))
        sample_data = u.get("sample_data")
        if sample_data is None and self.exchangeable and "sample_one" in u:
            sample_data = self._sample_data_from_one
        if sample_data is not None:
            self.sample_data = lambda key, params, data=None, hyper=None: sample_data(
                key, params, self.data if data is None else data, self._h(hyper))

    def _h(self, hyper):
        return self.hyper if hyper is None else hyper

    def _defaults(self, fn):
        def wrapped(params, data=None, hyper=None, **kwargs):
            return fn(params, self.data if data is None else data, self._h(hyper), **kwargs)
        return wrapped

    def _sample_data_from_one(self, key, params, data, hyper):
        one = self._user["sample_one"]
        keys = jax.random.split(key, self.n_obs)

        def draw(k, datum):
            out = one(k, params, datum, hyper)
            return {o: out.get(o, datum[o]) for o in self.obs_keys}

        new = jax.vmap(draw, in_axes=(0, self._datum_axes(data)))(keys, data)
        return {**data, **new}

    # -- the main derived function ---------------------------------------------------

    def log_density_fn(self, which: str = "joint", params_as: str = "free", data_as: str = "folded",
                       jacobian: Optional[bool] = None) -> Callable:
        """Return `f(p, data=None, hyper=None, weights=None) -> scalar`.

        which:     "joint" (log prior + log lik), "lik", or "prior".
        params_as: "folded" (dict of arrays), "flat" (constrained vector) or
                   "free" (unconstrained vector).
        data_as:   the same choices, for the `data` argument (defaults to the
                   model's data when `data` is None).
        jacobian:  add the log Jacobian of the free -> constrained map when
                   params_as="free". Defaults to True for "joint" and
                   "prior" (a density over free parameters, as in Stan) and
                   False for "lik". Data never get a Jacobian term.
        weights:   exchangeable models only; data weights for the likelihood.
        """
        if which not in ("joint", "lik", "prior"):
            raise ValueError("which must be 'joint', 'lik' or 'prior'")
        for a in (params_as, data_as):
            if a not in _REPRESENTATIONS:
                raise ValueError(f"representations are {_REPRESENTATIONS}, got {a!r}")
        if params_as != "folded" and self.params_pattern is None:
            raise ValueError(f"params_as={params_as!r} needs a params_pattern")
        if which == "prior" and self._log_prior is None:
            raise ValueError("this model has no log_prior")
        if jacobian is None:
            jacobian = which != "lik"
        pattern = self.params_pattern

        def f(p, data=None, hyper=None, weights=None):
            hyper = self._h(hyper)
            logdet = 0.0
            if params_as == "free":
                params, logdet = pattern.fold_with_log_det(p)
            elif params_as == "flat":
                params = pattern.fold(p, free=False, validate_value=False)
            else:
                params = p
            d = self.data if data is None else self._fold_data(data, data_as)
            lp = 0.0
            if which in ("joint", "lik"):
                if weights is not None:
                    if not self.exchangeable:
                        raise ValueError("weights need an exchangeable model (log_lik_one or log_lik_pointwise)")
                    lp = lp + self._log_lik(params, d, hyper, weights=weights)
                else:
                    lp = lp + self._log_lik(params, d, hyper)
            if which in ("joint", "prior") and self._log_prior is not None:
                lp = lp + self._log_prior(params, hyper)
            if jacobian and params_as == "free":
                lp = lp + logdet
            return lp

        return f

    # -- exchangeable extras ------------------------------------------------------------

    def _need_exchangeable(self):
        if not self.exchangeable:
            raise ValueError(f"{self.name} is not exchangeable (define log_lik_one or log_lik_pointwise)")

    def subset(self, idx, data=None):
        """The data restricted to observations `idx` (shared fields kept)."""
        self._need_exchangeable()
        data = self.data if data is None else data
        idx = jnp.asarray(idx)
        return {k: (v[idx] if k in self.obs_keys else v) for k, v in data.items()}

    def log_lik_subset(self, params, idx, data=None, hyper=None, scale: Optional[float] = None):
        """Log likelihood of observations `idx`, times `scale` (default
        N / len(idx), an unbiased estimate of the full log likelihood)."""
        self._need_exchangeable()
        scale = self.n_obs / len(idx) if scale is None else scale
        return scale * jnp.sum(self._pointwise(params, self.subset(idx, data), self._h(hyper)))

    def bootstrap_weights(self, key):
        """Multinomial(N, 1/N) counts, for the nonparametric bootstrap as
        `log_lik(params, weights=...)`."""
        self._need_exchangeable()
        idx = jax.random.randint(key, (self.n_obs,), 0, self.n_obs)
        return jnp.zeros(self.n_obs).at[idx].add(1.0)

    # -- parameters ---------------------------------------------------------------------

    @property
    def free_dim(self) -> int:
        return self.params_pattern.flat_length(True)

    @property
    def flat_dim(self) -> int:
        return self.params_pattern.flat_length(False)

    def fold_params(self, x, free: bool = True):
        return self.params_pattern.fold(x, free=free)

    def flatten_params(self, params, free: bool = True):
        return self.params_pattern.flatten(params, free=free)

    def random_params(self, key, scale: float = 1.0):
        """A random valid parameter dict (a normal draw in free space; not a
        prior draw)."""
        return self.params_pattern.random(key, scale=scale)

    # -- validation --------------------------------------------------------------------

    def validate_params(self, params):
        if self.params_pattern is not None:
            ok, msg = self.params_pattern.validate_folded(params)
            if not ok:
                raise ValidationError(f"{self.name} params: {msg}")
        run_check(self._user.get("check_params"), params, f"{self.name} params")

    def validate_data(self, data=None):
        data = self.data if data is None else data
        if self.data_pattern is not None and isinstance(data, dict):
            sub = {k: data[k] for k in self.data_pattern.keys()} if isinstance(self.data_pattern, pg.PatternDict) else data
            ok, msg = self.data_pattern.validate_folded(sub)
            if not ok:
                raise ValidationError(f"{self.name} data: {msg}")
        run_check(self._user.get("check_data"), data, f"{self.name} data")

    # -- the pieces, as data-generating processes ----------------------------------

    @property
    def prior(self) -> Optional[DGP]:
        """log_density(params, hyper=None); sample(key, hyper=None)."""
        if self._log_prior is None:
            return None
        return DGP(self.log_prior, getattr(self, "sample_prior", None))

    @property
    def likelihood(self) -> DGP:
        """log_density(params, data=None, hyper=None[, weights]);
        sample(key, params, data=None, hyper=None)."""
        return DGP(self.log_lik, getattr(self, "sample_data", None),
                   self.log_lik_pointwise if self.exchangeable else None)

    # -- metadata and export ------------------------------------------------------------

    @property
    def capabilities(self) -> Dict[str, Any]:
        """What this model supports, so a method can filter models."""
        return {
            "name": self.name,
            "free_dim": self.free_dim if self.params_pattern is not None else None,
            "exchangeable": self.exchangeable,
            "n_obs": self.n_obs,
            "has_prior": self._log_prior is not None,
            "can_sample_prior": "sample_prior" in self._user,
            "can_sample_data": hasattr(self, "sample_data"),
            "pattern_types": sorted(_pattern_types(self.params_pattern)) if self.params_pattern is not None else [],
        }

    def to_arviz(self, draws, **kwargs):
        """See `modelgami.arviz_io.to_arviz`."""
        from .arviz_io import to_arviz

        return to_arviz(self, draws, **kwargs)

    def __repr__(self):
        c = self.capabilities
        return f"<Model {self.name}: free_dim={c['free_dim']}, exchangeable={c['exchangeable']}, n_obs={c['n_obs']}>"


def hvp(f, argnum: int = 0):
    """Hessian-vector product of a scalar function: `hvp(f)(x, v, *rest)`,
    forward-over-reverse (never forms the Hessian)."""
    def hv(x, v, *rest):
        g = lambda x_: jax.grad(f, argnums=0)(x_, *rest)
        return jax.jvp(g, (x,), (v,))[1]
    if argnum != 0:
        raise NotImplementedError("hvp currently differentiates the first argument")
    return hv


def _from_obs_dists(obs_dists_one):
    """log_lik_one and sample_one from a function returning, for one
    observation, a dict {field: distribution} of the observed fields. Any
    distribution object with `log_prob(value)` and `sample(key)` works, e.g.
    from numpyro.distributions or tfp.substrates.jax.distributions."""
    def log_lik_one(params, datum, hyper):
        return sum(jnp.sum(d.log_prob(datum[k])) for k, d in obs_dists_one(params, datum, hyper).items())

    def sample_one(key, params, datum, hyper):
        ds = obs_dists_one(params, datum, hyper)
        keys = jax.random.split(key, len(ds))
        return {k: _draw(d, kk).astype(jnp.asarray(datum[k]).dtype) for (k, d), kk in zip(ds.items(), keys)}

    return log_lik_one, sample_one


def _from_prior_dists(prior_dists):
    """log_prior and sample_prior from `prior_dists(hyper)`, which returns
    a dict {param: distribution, or function(earlier params) -> distribution},
    in an order where each entry depends only on earlier ones."""
    def resolve(entry, params):
        return entry(params) if callable(entry) and not hasattr(entry, "log_prob") else entry

    def log_prior(params, hyper):
        total = 0.0
        for k, entry in prior_dists(hyper).items():
            total = total + jnp.sum(resolve(entry, params).log_prob(params[k]))
        return total

    def sample_prior(key, hyper):
        entries = prior_dists(hyper)
        keys = jax.random.split(key, len(entries))
        params = {}
        for (k, entry), kk in zip(entries.items(), keys):
            params[k] = _draw(resolve(entry, params), kk)
        return params

    return log_prior, sample_prior


def _draw(d, key):
    try:
        return d.sample(key)  # numpyro
    except TypeError:
        return d.sample(seed=key)  # tfp


def _to_array(v):
    if isinstance(v, (int, bool, str)) or v is None:
        return v
    return jnp.asarray(v)


def _infer_data_pattern(data):
    """An unconstrained pattern over the floating-point fields of dict data
    (integer fields stay fixed). None for non-dict data."""
    if not isinstance(data, dict):
        return None
    pats = {}
    for k, v in data.items():
        if isinstance(v, jnp.ndarray) and jnp.issubdtype(v.dtype, jnp.floating):
            pats[k] = pg.NumericArrayPattern(v.shape)
    return pg.PatternDict(pats) if pats else None


def _pattern_types(p):
    if isinstance(p, pg.PatternDict):
        return set().union(*(_pattern_types(v) for _, v in p.items())) if len(p) else set()
    if isinstance(p, pg.PatternArray):
        return _pattern_types(p.base_pattern) | {"PatternArray"}
    name = type(p).__name__
    if isinstance(p, pg.NumericArrayPattern):
        if p.has_lb or p.has_ub:
            name += "(bounded)"
    return {name}
