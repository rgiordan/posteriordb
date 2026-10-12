"""`Model` and `ExchangeableModel`.

A model is a paragami pattern for the parameters, a prior and a likelihood
(both `DGP`s), the data and the hyperparameters:

    model = Model(params_pattern=pattern, prior=DGP(log_prior, sample_prior),
                  likelihood=DGP(log_lik, sample_data), data=data, hyper=hyper)

From these it builds log densities of the parameters in folded, flat or free
form (`log_density_fn`), ready for `jax.grad`, `jax.hessian`, `jax.vmap` and
`jax.jit`. The data and hyperparameters are arguments of every function,
with the model's own as defaults.

`ExchangeableModel` takes an `ExchangeableDGP` likelihood and adds what
only makes sense for exchangeable data: the number of observations,
pointwise and weighted log likelihoods, and bootstrap weights.

Move parameters between representations with `model.params_pattern`
directly (paragami).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import paragami as pg

from .arviz_io import to_arviz
from .dgp import DGP, ExchangeableDGP, n_obs

_REPRESENTATIONS = ("folded", "flat", "free")


class Model:
    def __init__(self, *, params_pattern: pg.Pattern, likelihood: DGP, prior: Optional[DGP] = None,
                 data: Any = None, hyper: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        """`prior=None` is a flat (improper) prior: log density zero, no sampler."""
        if not isinstance(params_pattern, pg.Pattern):
            raise TypeError("params_pattern must be a paragami Pattern")
        if not isinstance(likelihood, DGP):
            raise TypeError("likelihood must be a DGP")
        if prior is not None and not isinstance(prior, DGP):
            raise TypeError("prior must be a DGP")
        self.params_pattern = params_pattern
        self.likelihood = likelihood
        self.prior = DGP(log_density=_flat_log_density) if prior is None else prior
        self.data = data
        self.hyper = {} if hyper is None else hyper
        self.name = name or type(self).__name__

    # -- the prior and likelihood, with the model's data and hyper as defaults ----

    def log_prior(self, params, hyper=None):
        return self.prior.log_density(params, self.hyper if hyper is None else hyper)

    def log_lik(self, params, data=None, hyper=None, **kwargs):
        return self.likelihood.log_density(params, self.data if data is None else data,
                                           self.hyper if hyper is None else hyper, **kwargs)

    def sample_prior(self, key, hyper=None):
        return self.prior.sample(key, self.hyper if hyper is None else hyper)

    def sample_data(self, key, params, data=None, hyper=None):
        return self.likelihood.sample(key, params, self.data if data is None else data,
                                      self.hyper if hyper is None else hyper)

    # -- log densities of the parameters ------------------------------------------

    def log_density_fn(self, which: str = "joint", *, params_as: str = "free",
                       jacobian: Optional[bool] = None) -> Callable:
        """Return `f(p, data=None, hyper=None, **kwargs) -> scalar`, where `p`
        is the parameters in `params_as` form and `kwargs` go to the
        likelihood (e.g. `weights` for an exchangeable model).

        which:     "joint" (log prior + log likelihood), "lik" or "prior".
        params_as: "folded" (dict of arrays), "flat" (constrained vector)
                   or "free" (unconstrained vector).
        jacobian:  add the log Jacobian of the free -> folded map when
                   params_as="free". By default each function is, up to a
                   constant, the density it names with respect to the
                   parameters you pass: True for "joint" and "prior", False
                   for "lik".
        """
        if which not in ("joint", "lik", "prior"):
            raise ValueError(f"which must be 'joint', 'lik' or 'prior', got {which!r}")
        if params_as not in _REPRESENTATIONS:
            raise ValueError(f"params_as must be one of {_REPRESENTATIONS}, got {params_as!r}")
        if jacobian is None:
            jacobian = which != "lik"
        pattern = self.params_pattern

        def f(p, data=None, hyper=None, **kwargs):
            if params_as == "free":
                params = pattern.free_to_folded(p)
            elif params_as == "flat":
                params = pattern.flat_to_folded(p)
            else:
                params = p
            lp = 0.0
            if which in ("joint", "lik"):
                lp = lp + self.log_lik(params, data, hyper, **kwargs)
            if which in ("joint", "prior"):
                lp = lp + self.log_prior(params, hyper)
            if jacobian and params_as == "free":
                lp = lp + pattern.free_to_folded_log_abs_det(p)
            return lp

        return f

    def to_arviz(self, draws, **kwargs):
        """See `modelgami.arviz_io.to_arviz`."""
        return to_arviz(self, draws, **kwargs)

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}: free dimension {self.params_pattern.free_length()}>"


class ExchangeableModel(Model):
    """A model whose likelihood is an `ExchangeableDGP`: the data are N
    observations (every data leaf has a leading axis of length N),
    independent given the parameters and hyperparameters."""

    def __init__(self, *, params_pattern: pg.Pattern, likelihood: ExchangeableDGP, prior: Optional[DGP] = None,
                 data: Any = None, hyper: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        if not isinstance(likelihood, ExchangeableDGP):
            raise TypeError("an ExchangeableModel needs an ExchangeableDGP likelihood")
        super().__init__(params_pattern=params_pattern, likelihood=likelihood, prior=prior,
                         data=data, hyper=hyper, name=name)

    @property
    def n_obs(self) -> int:
        return n_obs(self.data)

    def log_lik_pointwise(self, params, data=None, hyper=None):
        return self.likelihood.log_density_pointwise(params, self.data if data is None else data,
                                                     self.hyper if hyper is None else hyper)

    def bootstrap_weights(self, key):
        """Multinomial(N, 1/N) counts, for the nonparametric bootstrap as
        `log_lik(params, weights=...)`."""
        idx = jax.random.randint(key, (self.n_obs,), 0, self.n_obs)
        return jnp.zeros(self.n_obs).at[idx].add(1.0)


def hvp(f, argnum: int = 0):
    """Hessian-vector product of a scalar function: `hvp(f)(x, v, *rest)`,
    forward-over-reverse (never forms the Hessian)."""
    if argnum != 0:
        raise NotImplementedError("hvp currently differentiates the first argument")

    def hv(x, v, *rest):
        return jax.jvp(lambda x_: jax.grad(f, argnums=0)(x_, *rest), (x,), (v,))[1]

    return hv


def _flat_log_density(params, hyper):
    return 0.0
