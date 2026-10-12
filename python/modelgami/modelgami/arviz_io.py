"""Export draws from a `Model` to ArviZ. The parameter conversion is
paragami's (`paragami.folded_draws_to_arviz`); this adds the model's
pointwise log likelihood and data."""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import paragami as pg


def to_arviz(model, draws, params_as: str = "folded", *, log_likelihood: bool = True,
             posterior_predictive=None, prior=None, prior_predictive=None,
             dims: Optional[Dict[str, list]] = None, coords: Optional[Dict[str, Any]] = None,
             obs_dim: str = "obs"):
    """Build an ArviZ object from posterior draws.

    draws: parameter draws with leading (chain, draw) axes, either a dict of
        arrays (params_as="folded") or an array of flat/free vectors of shape
        (chain, draw, d) (params_as="flat"/"free").
    log_likelihood: for exchangeable models, add the pointwise log
        likelihood (so `az.loo` / `az.waic` work).
    posterior_predictive, prior, prior_predictive: optional extra groups,
        given in the same (chain, draw, ...) layout as dicts.
    dims, coords: ArviZ dims/coords for parameters and data. Observation
        fields of exchangeable data get the dimension `obs_dim`.
    """
    posterior = _as_folded_draws(model, draws, params_as)
    groups = {}
    dims = dict(dims or {})
    if model.exchangeable:
        for k in model.obs_keys:
            dims.setdefault(k, [obs_dim] + [f"{k}_dim_{i}" for i in range(np.ndim(model.data[k]) - 1)])
    if log_likelihood and model.exchangeable:
        pw = jax.vmap(jax.vmap(lambda p: model.log_lik_pointwise(p)))(posterior)
        groups["log_likelihood"] = {"log_lik": np.asarray(pw)}
        dims.setdefault("log_lik", [obs_dim])
    if isinstance(model.data, dict):
        data = {k: np.asarray(v) for k, v in model.data.items() if not isinstance(v, (str,))}
        obs = {k: v for k, v in data.items() if k in model.obs_keys} if model.exchangeable else data
        const = {k: v for k, v in data.items() if k not in model.obs_keys} if model.exchangeable else {}
        groups["observed_data"] = obs
        if const:
            groups["constant_data"] = const
    if prior is not None:
        groups["prior"] = pg.folded_draws_to_variables(model.params_pattern, prior)
    for name, value in (("posterior_predictive", posterior_predictive), ("prior_predictive", prior_predictive)):
        if value is not None:
            groups[name] = _np(value)

    return pg.folded_draws_to_arviz(model.params_pattern, posterior, dims=dims, coords=coords,
                                    extra_groups=groups)


def _as_folded_draws(model, draws, params_as):
    if params_as == "folded":
        return jax.tree_util.tree_map(jnp.asarray, draws)
    if params_as == "free":
        return pg.free_draws_to_folded(model.params_pattern, jnp.asarray(draws))
    return pg.flat_draws_to_folded(model.params_pattern, jnp.asarray(draws))


def _np(tree):
    return {k: np.asarray(v) for k, v in tree.items()}
