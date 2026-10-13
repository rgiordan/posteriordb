"""Build a `Model` from one NumPyro program, which specifies both how to
simulate and how to evaluate the log density.

    import numpyro, numpyro.distributions as dist

    def regression(data, hyper):
        sigma2 = numpyro.sample("sigma2", dist.InverseGamma(hyper["a0"], hyper["b0"]))
        beta = numpyro.sample("beta", dist.MultivariateNormal(hyper["m0"], sigma2 * hyper["V0"]))
        with numpyro.plate("obs", data["x"].shape[0]):
            numpyro.sample("y", dist.Normal(data["x"] @ beta, jnp.sqrt(sigma2)), obs=data["y"])

    model = mg.from_numpyro(regression, data={"x": X, "y": y}, hyper=hyper, exchangeable=True)

The program's sites become the model's prior and likelihood:

- Latent sites are the parameters. Their supports become a paragami pattern,
  so the unconstrained space uses Stan's transforms, not NumPyro's.
- The prior's log density is the latent sites' log density, and its sampler
  runs the program forward.
- Observed sites (`obs=...`) make up the likelihood. With
  `exchangeable=True` the result is an `ExchangeableModel`, and each
  observed site's log density must have one entry per observation (a plate
  over the leading axis of the data).
- The likelihood's sampler re-runs the program, drawing the observed sites
  instead of conditioning on them. Name each observed site after its field
  in `data` (e.g. `numpyro.sample("y", ..., obs=data["y"])`).

Everything stays a JAX function of params, data and hyper, so derivatives
with respect to data and hyperparameters work as for hand-written hooks.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import paragami as pg

from .dgp import DGP, ExchangeableDGP
from .model import ExchangeableModel, Model

try:
    from numpyro import handlers
    from numpyro.distributions import constraints as c
    from numpyro.primitives import Messenger
except ImportError:  # optional extra modelgami[numpyro]
    handlers = c = None
    Messenger = object


def from_numpyro(model_fn: Callable, data, hyper=None, *, exchangeable: bool = False,
                 params_pattern: Optional[pg.Pattern] = None, name: Optional[str] = None) -> Model:
    """`model_fn(data, hyper)` is a NumPyro program; see the module docstring."""
    if handlers is None:
        raise ImportError("from_numpyro needs NumPyro: pip install 'modelgami[numpyro]'")
    hyper = {} if hyper is None else hyper

    def trace(*, params, d, h, key=None, resample=()):
        fn = model_fn
        if resample:
            fn = _Unobserve(fn, names=set(resample))
        fn = handlers.seed(fn, rng_seed=jax.random.PRNGKey(0) if key is None else key)
        if params is not None:
            fn = handlers.substitute(fn, data=params)
        return handlers.trace(fn).get_trace(d, h)

    tr = trace(params=None, d=data, h=hyper)
    latent = [k for k, s in tr.items() if s["type"] == "sample" and not s["is_observed"]]
    observed = [k for k, s in tr.items() if s["type"] == "sample" and s["is_observed"]]
    if params_pattern is None:
        params_pattern = pg.PatternDict({k: pattern_for_support(tr[k]["fn"].support, shape=jnp.shape(tr[k]["value"]))
                                         for k in latent})

    def sum_sites(t, names, *, singleobs=False):
        total = 0.0
        for k in names:
            lp = t[k]["fn"].log_prob(t[k]["value"])
            total = total + (lp if singleobs else jnp.sum(lp))
        return total

    # The prior runs the program with the model's data (for plate sizes).
    prior = DGP(
        eval_log_density=lambda params, h: sum_sites(trace(params=params, d=data, h=h), latent),
        draw=lambda key, h: {k: v["value"] for k, v in trace(params=None, d=data, h=h, key=key).items()
                             if k in latent},
    )

    def draw_data(key, d, params, h):
        t = trace(params=params, d=d, h=h, key=key, resample=observed)
        return {**d, **{k: t[k]["value"] for k in observed if k in d}}

    name = name or getattr(model_fn, "__name__", None)
    if exchangeable:
        likelihood = _NumPyroLikelihood(
            eval_singleobs=lambda d, params, h: sum_sites(trace(params=params, d=d, h=h), observed, singleobs=True),
            draw=draw_data)
        return ExchangeableModel(params_pattern=params_pattern, prior=prior, likelihood=likelihood, data=data,
                                 hyper=hyper, name=name)
    likelihood = DGP(eval_log_density=lambda d, params, h: sum_sites(trace(params=params, d=d, h=h), observed),
                     draw=draw_data)
    return Model(params_pattern=params_pattern, prior=prior, likelihood=likelihood, data=data, hyper=hyper,
                 name=name)


class _NumPyroLikelihood(ExchangeableDGP):
    """An exchangeable likelihood whose single-observation terms come from
    the program's observed sites, all observations at once."""

    def __init__(self, *, eval_singleobs, draw):
        super().__init__()
        self._eval_singleobs = eval_singleobs
        self.draw = draw

    def eval_log_density_singleobs(self, data, params, hyper):
        return self._eval_singleobs(data, params, hyper)


class _Unobserve(Messenger):
    """A handler that turns the named observed sites back into draws."""

    def __init__(self, fn, *, names):
        self.names = names
        super().__init__(fn)

    def process_message(self, msg):
        if msg["type"] == "sample" and msg["name"] in self.names:
            msg["is_observed"] = False
            msg["value"] = None


def pattern_for_support(support, *, shape) -> pg.Pattern:
    """A paragami pattern for a NumPyro support (constraint) and value shape."""
    shape = tuple(shape)
    if support is c.real or support is c.real_vector:
        return pg.NumericArrayPattern(shape)
    if support is c.positive:
        return pg.NumericArrayPattern(shape, lb=0.0)
    if support is c.unit_interval:
        return pg.NumericArrayPattern(shape, lb=0.0, ub=1.0)
    if isinstance(support, c.greater_than):
        return pg.NumericArrayPattern(shape, lb=float(support.lower_bound))
    if isinstance(support, c.less_than):
        return pg.NumericArrayPattern(shape, ub=float(support.upper_bound))
    if isinstance(support, c.interval):
        return pg.NumericArrayPattern(shape, lb=float(support.lower_bound), ub=float(support.upper_bound))
    if support is c.simplex:
        return pg.SimplexArrayPattern(shape[-1], array_shape=shape[:-1])
    if support is c.ordered_vector and len(shape) == 1:
        return pg.OrderedVectorPattern(shape[0])
    if support is c.positive_ordered_vector and len(shape) == 1:
        return pg.OrderedVectorPattern(shape[0], positive=True)
    if support is c.corr_cholesky and len(shape) == 2:
        return pg.CholeskyCorrPattern(shape[0])
    if support is c.lower_cholesky and len(shape) == 2:
        return pg.CholeskyCovPattern(shape[0])
    if support is c.corr_matrix and len(shape) == 2:
        return pg.CorrMatrixPattern(shape[0])
    if support is c.sphere and len(shape) == 1:
        return pg.UnitVectorPattern(shape[0])
    if support is c.positive_definite and len(shape) == 2:
        return pg.PSDSymmetricMatrixPattern(shape[0])
    raise NotImplementedError(f"no paragami pattern for NumPyro support {support!r}; pass params_pattern=")

