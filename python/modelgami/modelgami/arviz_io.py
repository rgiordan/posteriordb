"""Export draws from a `Model` to ArviZ (works with ArviZ >= 1.0 and the
older 0.x `InferenceData` API)."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


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
    import arviz as az

    posterior = _as_folded_draws(model, draws, params_as)
    groups = {"posterior": _np(posterior)}
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
    for name, value in (("posterior_predictive", posterior_predictive), ("prior", prior), ("prior_predictive", prior_predictive)):
        if value is not None:
            groups[name] = _np(value)

    params = inspect.signature(az.from_dict).parameters
    if "data" in params:  # ArviZ >= 1.0
        return az.from_dict(groups, dims=dims, coords=coords)
    return az.from_dict(**groups, dims=dims, coords=coords)  # ArviZ 0.x


def _as_folded_draws(model, draws, params_as):
    if params_as == "folded":
        return jax.tree_util.tree_map(jnp.asarray, draws)
    draws = jnp.asarray(draws)
    fold = lambda x: model.params_pattern.fold(x, free=(params_as == "free"), validate_value=False)
    return jax.vmap(jax.vmap(fold))(draws)


def _np(tree):
    return {k: np.asarray(v) for k, v in tree.items()}
