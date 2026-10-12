"""Build a `Model` from one NumPyro program, which specifies both how to
simulate and how to evaluate the log density.

    import numpyro, numpyro.distributions as dist

    def regression(data, hyper):
        sigma2 = numpyro.sample("sigma2", dist.InverseGamma(hyper["a0"], hyper["b0"]))
        beta = numpyro.sample("beta", dist.MultivariateNormal(hyper["m0"], sigma2 * hyper["V0"]))
        with numpyro.plate("obs", data["x"].shape[0]):
            numpyro.sample("y", dist.Normal(data["x"] @ beta, jnp.sqrt(sigma2)), obs=data["y"])

    model = mg.from_numpyro(regression, data={"x": X, "y": y}, hyper=hyper, exchangeable=True)

The program's sites become the usual hooks:

- Latent sites are the parameters. Their supports become a paragami pattern,
  so the unconstrained space uses Stan's transforms, not NumPyro's.
- Latent sites make up `log_prior`.
- Observed sites (`obs=...`) make up the likelihood. With
  `exchangeable=True`, each observed site's log density must have one entry
  per observation (a plate over the leading axis of the data), which gives
  `log_lik_pointwise`, weights and subsets.
- `sample_prior` runs the program forward.
- `sample_data` re-runs it, drawing the observed sites instead of
  conditioning on them. To use it, name each observed site after its field
  in `data` (e.g. `numpyro.sample("y", ..., obs=data["y"])`).

Everything stays a JAX function of params, data and hyper, so derivatives
with respect to data and hyperparameters work as for hand-written hooks.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import paragami as pg

from .model import Model


def from_numpyro(model_fn: Callable, data, hyper=None, *, shared_data=None, exchangeable: bool = False,
                 params_pattern: Optional[pg.Pattern] = None, name: Optional[str] = None, **kwargs) -> Model:
    """`model_fn(data, hyper)` is a NumPyro program; see the module docstring.
    Extra keyword arguments (e.g. `check_data`) are passed to `Model`."""
    from numpyro import handlers

    hyper = {} if hyper is None else hyper
    full_data = _stack(data)
    if shared_data:
        full_data = {**full_data, **shared_data}

    def trace(params, d, h, key=None, resample=()):
        fn = model_fn
        if resample:
            fn = _Unobserve(fn, set(resample))
        fn = handlers.seed(fn, jax.random.PRNGKey(0) if key is None else key)
        if params is not None:
            fn = handlers.substitute(fn, data=params)
        return handlers.trace(fn).get_trace(d, h)

    tr = trace(None, full_data, hyper)
    latent = [k for k, s in tr.items() if s["type"] == "sample" and not s["is_observed"]]
    observed = [k for k, s in tr.items() if s["type"] == "sample" and s["is_observed"]]
    if params_pattern is None:
        params_pattern = pg.PatternDict({k: pattern_for_support(tr[k]["fn"].support, jnp.shape(tr[k]["value"]))
                                         for k in latent})

    def site_sum(t, names, pointwise=False):
        total = 0.0
        for k in names:
            lp = t[k]["fn"].log_prob(t[k]["value"])
            total = total + (lp if pointwise else jnp.sum(lp))
        return total

    def log_prior(params, h):
        return site_sum(trace(params, full_data, h), latent)

    def sample_prior(key, h):
        t = trace(None, full_data, h, key=key)
        return {k: t[k]["value"] for k in latent}

    def sample_data(key, params, d, h):
        t = trace(params, d, h, key=key, resample=observed)
        return {**d, **{k: t[k]["value"] for k in observed if k in d}}

    hooks = dict(log_prior=log_prior if latent else None, sample_prior=sample_prior if latent else None)
    hooks = {k: v for k, v in hooks.items() if v is not None}
    if exchangeable:
        hooks["log_lik_pointwise"] = lambda params, d, h: site_sum(trace(params, d, h), observed, pointwise=True)
    else:
        hooks["log_lik"] = lambda params, d, h: site_sum(trace(params, d, h), observed)
    model = Model(params_pattern=params_pattern, data=data, shared_data=shared_data, hyper=hyper,
                  name=name or getattr(model_fn, "__name__", None), **hooks, **kwargs)
    # Simulation through the program; bound after construction because it
    # simulates whole data sets, not one observation at a time.
    model._user["sample_data"] = sample_data
    model._expose_hooks()
    return model


def _Unobserve(fn, names):
    """A handler that turns the named observed sites back into draws."""
    from numpyro.primitives import Messenger

    class Unobserve(Messenger):
        def process_message(self, msg):
            if msg["type"] == "sample" and msg["name"] in names:
                msg["is_observed"] = False
                msg["value"] = None

    return Unobserve(fn)


def pattern_for_support(support, shape) -> pg.Pattern:
    """A paragami pattern for a NumPyro support (constraint) and value shape."""
    from numpyro.distributions import constraints as c

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
    if support is c.positive_definite and len(shape) == 2:
        return pg.PSDSymmetricMatrixPattern(shape[0])
    raise NotImplementedError(f"no paragami pattern for NumPyro support {support!r}; pass params_pattern=")


def _stack(data):
    if isinstance(data, (list, tuple)):
        return jax.tree_util.tree_map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *data)
    return {k: (jnp.asarray(v) if not isinstance(v, (int, str)) else v) for k, v in data.items()}
