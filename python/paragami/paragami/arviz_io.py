"""Convert MCMC draws to and from ArviZ (optional dependency: `pip install paragami[arviz]`).

Draws carry two leading axes, (chain, draw). They can be folded (a dict of
arrays shaped `(chain, draw) + shape`, or one such array), free vectors
`(chain, draw, free_length)` or flat vectors `(chain, draw, flat_length)`.

    idata = pg.free_draws_to_arviz(pattern, free_draws)    # e.g. from a sampler
    free_draws = pg.arviz_to_free_draws(pattern, idata)     # and back

Each entry of a `PatternDict` becomes one ArviZ variable. Nested dicts are
joined with ".", so `{"a": {"b": ...}}` becomes the variable "a.b". A pattern
that is not a `PatternDict` gives a single variable called `name`.
Works with ArviZ >= 1.0 (DataTree) and the older 0.x InferenceData.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import jax
import numpy as np

from .base import Pattern
from .containers import PatternDict


# -- draws to ArviZ ----------------------------------------------------------------

def folded_draws_to_arviz(pattern: Pattern, draws, *, group: str = "posterior", name: str = "x",
                          dims: Optional[Dict[str, list]] = None, coords: Optional[Dict[str, Any]] = None,
                          extra_groups: Optional[Dict[str, Dict[str, Any]]] = None):
    """ArviZ data with the folded `draws` in `group`. `extra_groups` adds
    other groups as {group: {variable: array}} (for example log_likelihood or
    observed_data); `dims` and `coords` are passed to `arviz.from_dict`."""
    groups = {group: folded_draws_to_variables(pattern, draws, name=name)}
    groups.update(extra_groups or {})
    return _from_dict(groups, dims=dims, coords=coords)


def free_draws_to_arviz(pattern: Pattern, draws, **kwargs):
    """Like `folded_draws_to_arviz`, from free vectors of shape (chain, draw, free_length)."""
    return folded_draws_to_arviz(pattern, free_draws_to_folded(pattern, draws), **kwargs)


def flat_draws_to_arviz(pattern: Pattern, draws, **kwargs):
    """Like `folded_draws_to_arviz`, from flat vectors of shape (chain, draw, flat_length)."""
    return folded_draws_to_arviz(pattern, flat_draws_to_folded(pattern, draws), **kwargs)


# -- ArviZ to draws ----------------------------------------------------------------

def arviz_to_folded_draws(pattern: Pattern, data, *, group: str = "posterior", name: str = "x"):
    """Folded draws with leading (chain, draw) axes from ArviZ `data`. The
    group may hold other variables too; only the pattern's are read."""
    dataset = data[group] if not hasattr(data, group) else getattr(data, group)
    get = lambda var: np.asarray(dataset[var].values)
    return variables_to_folded_draws(pattern, get, name=name)


def arviz_to_free_draws(pattern: Pattern, data, **kwargs):
    """Free vectors of shape (chain, draw, free_length) from ArviZ `data`."""
    return folded_draws_to_free(pattern, arviz_to_folded_draws(pattern, data, **kwargs))


def arviz_to_flat_draws(pattern: Pattern, data, **kwargs):
    """Flat vectors of shape (chain, draw, flat_length) from ArviZ `data`."""
    return folded_draws_to_flat(pattern, arviz_to_folded_draws(pattern, data, **kwargs))


# -- batches of draws --------------------------------------------------------------

def _over_draws(f):
    return jax.vmap(jax.vmap(f))


def free_draws_to_folded(pattern: Pattern, draws):
    return _over_draws(pattern.free_to_folded)(draws)


def flat_draws_to_folded(pattern: Pattern, draws):
    return _over_draws(lambda x: pattern.flat_to_folded(x, validate_value=False))(draws)


def folded_draws_to_free(pattern: Pattern, draws):
    return _over_draws(lambda p: pattern.folded_to_free(p, validate_value=False))(draws)


def folded_draws_to_flat(pattern: Pattern, draws):
    return _over_draws(lambda p: pattern.folded_to_flat(p, validate_value=False))(draws)


# -- variables -----------------------------------------------------------------------

def folded_draws_to_variables(pattern: Pattern, draws, name: str = "x") -> Dict[str, np.ndarray]:
    """{variable name: numpy array}, one per leaf pattern, with nested dict keys joined by "."."""
    if not isinstance(pattern, PatternDict):
        if isinstance(draws, dict):  # a custom pattern whose folded values are dicts
            return {k: np.asarray(v) for k, v in draws.items()}
        return {name: np.asarray(draws)}
    out = {}
    for k, p in pattern.items():
        if isinstance(p, PatternDict):
            out.update({f"{k}.{v}": a for v, a in folded_draws_to_variables(p, draws[k]).items()})
        else:
            out[k] = np.asarray(draws[k])
    return out


def variables_to_folded_draws(pattern: Pattern, get, name: str = "x", prefix: str = ""):
    """Inverse of `folded_draws_to_variables`; `get(variable_name)` returns an array."""
    if not isinstance(pattern, PatternDict):
        return get(name)
    return {k: (variables_to_folded_draws(p, get, prefix=f"{prefix}{k}.") if isinstance(p, PatternDict)
                else get(f"{prefix}{k}"))
            for k, p in pattern.items()}


def _from_dict(groups, dims=None, coords=None):
    import arviz as az

    if "data" in inspect.signature(az.from_dict).parameters:  # ArviZ >= 1.0
        return az.from_dict(groups, dims=dims, coords=coords)
    return az.from_dict(**groups, dims=dims, coords=coords)  # ArviZ 0.x
