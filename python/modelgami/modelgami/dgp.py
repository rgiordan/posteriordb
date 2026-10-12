"""Data-generating processes.

A `DGP` is a distribution with a log density and a sampler, either of which
may be missing. A model's prior and its likelihood are both DGPs:

    prior:       log_density(params, hyper)          sample(key, hyper) -> params
    likelihood:  log_density(params, data, hyper)    sample(key, params, data, hyper) -> data

The parameters always come first, since they are what a model's log density
is differentiated with respect to. The likelihood's `sample` takes the
current data for its covariates and shapes and returns a new data set.

Pass the functions to `DGP(...)`, or subclass and define the methods. A
method that is not defined raises `NotImplementedError`.

`ExchangeableDGP` is a likelihood for observations that are independent
given the parameters, defined one observation at a time.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp


class DGP:
    def __init__(self, log_density: Optional[Callable] = None, sample: Optional[Callable] = None):
        if log_density is not None:
            self.log_density = log_density
        if sample is not None:
            self.sample = sample

    def log_density(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} has no log density")

    def sample(self, key, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} has no sampler")


class ExchangeableDGP(DGP):
    """A likelihood for N observations that are independent given the
    parameters (and hyperparameters).

        log_density_one(params, datum, hyper) -> scalar
        sample_one(key, params, datum, hyper) -> {field: new value}

    The data are an array of N data: a dict (or any pytree) whose every
    leaf has a leading axis of length N, and `datum` is one slice of it.
    Anything shared by all observations belongs in `hyper`. `sample_one`
    returns the fields it draws; the datum's other fields (covariates) are
    kept.

    To vectorize over observations yourself, subclass and override
    `log_density_pointwise(params, data, hyper)`.
    """

    def __init__(self, log_density_one: Optional[Callable] = None, sample_one: Optional[Callable] = None):
        if log_density_one is not None:
            self.log_density_one = log_density_one
        if sample_one is not None:
            self.sample_one = sample_one

    def log_density_one(self, params, datum, hyper):
        raise NotImplementedError(f"{type(self).__name__} has no log density")

    def sample_one(self, key, params, datum, hyper):
        raise NotImplementedError(f"{type(self).__name__} has no sampler")

    def log_density_pointwise(self, params, data, hyper):
        """The (N,) vector of per-observation log densities."""
        return jax.vmap(self.log_density_one, in_axes=(None, 0, None))(params, data, hyper)

    def log_density(self, params, data, hyper, weights=None):
        """The total log density, or `sum(weights * pointwise)`."""
        pointwise = self.log_density_pointwise(params, data, hyper)
        return jnp.sum(pointwise) if weights is None else jnp.sum(weights * pointwise)

    def sample(self, key, params, data, hyper):
        """New data: each observation's drawn fields replaced, the rest kept."""
        def one(k, datum):
            return {**datum, **self.sample_one(k, params, datum, hyper)}

        return jax.vmap(one)(jax.random.split(key, n_obs(data)), data)

    def subset(self, data, idx):
        """The observations `idx`."""
        idx = jnp.asarray(idx)
        return jax.tree_util.tree_map(lambda v: v[idx], data)

    def log_density_subset(self, params, data, hyper, idx, scale: Optional[float] = None):
        """Log density of observations `idx`, times `scale`. The default
        scale, N / len(idx), makes it an unbiased estimate of the total."""
        scale = n_obs(data) / len(idx) if scale is None else scale
        return scale * jnp.sum(self.log_density_pointwise(params, self.subset(data, idx), hyper))


def n_obs(data) -> int:
    """The number of observations in exchangeable data (the leading axis)."""
    return jnp.shape(jax.tree_util.tree_leaves(data)[0])[0]


def exchangeable_dgp_from_distributions(dists_one: Callable) -> ExchangeableDGP:
    """An `ExchangeableDGP` written once as distribution objects:
    `dists_one(params, datum, hyper)` returns {field: distribution} for the
    observed fields of one observation. Any distribution with
    `log_prob(value)` and `sample(key)` works (NumPyro, or TFP, which is
    sampled with `sample(seed=key)`)."""

    def log_density_one(params, datum, hyper):
        return sum(jnp.sum(d.log_prob(datum[k])) for k, d in dists_one(params, datum, hyper).items())

    def sample_one(key, params, datum, hyper):
        dists = dists_one(params, datum, hyper)
        keys = jax.random.split(key, len(dists))
        return {k: _draw(dist=d, key=kk).astype(jnp.asarray(datum[k]).dtype)
                for (k, d), kk in zip(dists.items(), keys)}

    return ExchangeableDGP(log_density_one, sample_one)


def _draw(*, dist, key):
    if type(dist).__module__.startswith("tensorflow_probability"):
        return dist.sample(seed=key)
    return dist.sample(key)
