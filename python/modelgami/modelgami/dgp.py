"""Data-generating processes.

A `DGP` is a distribution of one value given some conditioning arguments,
with a log density and a sampler, either of which may be missing. The value
always comes first:

    eval_log_density(value, *given)
    draw(key, *given) -> value

A model's prior and likelihood are both DGPs:

    prior:       eval_log_density(params, hyper)         draw(key, hyper) -> params
    likelihood:  eval_log_density(data, params, hyper)   draw(key, data, params, hyper) -> data

The likelihood's `draw` takes the current data for its covariates and
shapes, and returns new data.

Pass the functions to `DGP(...)`, or subclass and define the methods. A
method that is not defined raises `NotImplementedError`.

`free_dgp` and `folded_dgp` change a DGP's base measure between a
constrained value and its unconstrained (free) coordinates, with the log
Jacobian in the log density.

`ExchangeableDGP` is a likelihood for observations that are independent
given the parameters, defined one observation at a time.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import paragami as pg


class DGP:
    def __init__(self, eval_log_density: Optional[Callable] = None, draw: Optional[Callable] = None):
        if eval_log_density is not None:
            self.eval_log_density = eval_log_density
        if draw is not None:
            self.draw = draw

    def eval_log_density(self, value, *given):
        raise NotImplementedError(f"{type(self).__name__} has no log density")

    def draw(self, key, *given):
        raise NotImplementedError(f"{type(self).__name__} has no sampler")


def free_dgp(dgp: DGP, pattern: pg.Pattern) -> DGP:
    """The same distribution, of the free vector of `pattern` instead of
    its folded value. The log density adds the log Jacobian of the free ->
    folded map; draws are converted to free vectors."""

    def eval_log_density(free_value, *given):
        folded, log_det = pattern.free_to_folded_with_log_det(free_value)
        return dgp.eval_log_density(folded, *given) + log_det

    def draw(key, *given):
        return pattern.folded_to_free(dgp.draw(key, *given))

    return DGP(eval_log_density, draw)


def folded_dgp(dgp: DGP, pattern: pg.Pattern) -> DGP:
    """The inverse of `free_dgp`: a distribution of free vectors as one of
    their folded values. The log density subtracts the log Jacobian; draws
    are folded."""

    def eval_log_density(folded_value, *given):
        free_value = pattern.folded_to_free(folded_value)
        return dgp.eval_log_density(free_value, *given) - pattern.free_to_folded_log_abs_det(free_value)

    def draw(key, *given):
        return pattern.free_to_folded(dgp.draw(key, *given))

    return DGP(eval_log_density, draw)


class ExchangeableDGP(DGP):
    """A likelihood for N observations that are independent given the
    parameters and hyperparameters.

        eval_log_density_one(datum, params, hyper) -> scalar
        draw_one(key, datum, params, hyper) -> {field: new value}

    The data are an array of N data: a dict (or any pytree) whose every
    leaf has a leading axis of length N, and `datum` is one slice of it.
    Anything shared by all observations belongs in `hyper`. `draw_one`
    returns the fields it draws; the datum's other fields (covariates) are
    kept.

    To vectorize over observations yourself, subclass and override
    `eval_log_density_singleobs(data, params, hyper)`.
    """

    def __init__(self, eval_log_density_one: Optional[Callable] = None, draw_one: Optional[Callable] = None):
        if eval_log_density_one is not None:
            self.eval_log_density_one = eval_log_density_one
        if draw_one is not None:
            self.draw_one = draw_one

    def eval_log_density_one(self, datum, params, hyper):
        raise NotImplementedError(f"{type(self).__name__} has no log density")

    def draw_one(self, key, datum, params, hyper):
        raise NotImplementedError(f"{type(self).__name__} has no sampler")

    def eval_log_density_singleobs(self, data, params, hyper):
        """The (N,) vector of single-observation log densities."""
        return jax.vmap(self.eval_log_density_one, in_axes=(0, None, None))(data, params, hyper)

    def eval_log_density(self, data, params, hyper, weights=None):
        """The total log density, or `sum(weights * singleobs)`."""
        singleobs = self.eval_log_density_singleobs(data, params, hyper)
        return jnp.sum(singleobs) if weights is None else jnp.sum(weights * singleobs)

    def draw(self, key, data, params, hyper):
        """New data: each observation's drawn fields replaced, the rest kept."""
        def one(k, datum):
            return {**datum, **self.draw_one(k, datum, params, hyper)}

        return jax.vmap(one)(jax.random.split(key, count_obs(data)), data)

    def select_subset(self, data, idx):
        """The observations `idx`."""
        idx = jnp.asarray(idx)
        return jax.tree_util.tree_map(lambda v: v[idx], data)

    def eval_log_density_subset(self, data, params, hyper, idx, scale: Optional[float] = None):
        """Log density of observations `idx`, times `scale`. The default
        scale, N / len(idx), makes it an unbiased estimate of the total."""
        scale = count_obs(data) / len(idx) if scale is None else scale
        return scale * jnp.sum(self.eval_log_density_singleobs(self.select_subset(data, idx), params, hyper))


def count_obs(data) -> int:
    """The number of observations in exchangeable data (the leading axis)."""
    return jnp.shape(jax.tree_util.tree_leaves(data)[0])[0]


def draw_np_bootstrap_weights(key, n_obs: int):
    """Multinomial(N, 1/N) counts: nonparametric bootstrap weights for N
    observations, for use as likelihood weights."""
    idx = jax.random.randint(key, (n_obs,), 0, n_obs)
    return jnp.zeros(n_obs).at[idx].add(1.0)


def exchangeable_dgp_from_distributions(make_dists_one: Callable) -> ExchangeableDGP:
    """An `ExchangeableDGP` written once as distribution objects:
    `make_dists_one(datum, params, hyper)` returns {field: distribution} for
    the observed fields of one observation. Any distribution with
    `log_prob(value)` and `sample(key)` works (NumPyro, or TFP, which is
    sampled with `sample(seed=key)`)."""

    def eval_log_density_one(datum, params, hyper):
        return sum(jnp.sum(d.log_prob(datum[k])) for k, d in make_dists_one(datum, params, hyper).items())

    def draw_one(key, datum, params, hyper):
        dists = make_dists_one(datum, params, hyper)
        keys = jax.random.split(key, len(dists))
        return {k: _draw_from(dist=d, key=kk).astype(jnp.asarray(datum[k]).dtype)
                for (k, d), kk in zip(dists.items(), keys)}

    return ExchangeableDGP(eval_log_density_one, draw_one)


def _draw_from(*, dist, key):
    if type(dist).__module__.startswith("tensorflow_probability"):
        return dist.sample(seed=key)
    return dist.sample(key)
