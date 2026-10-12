"""Shared code for the many posteriordb models that are a normal linear
regression on per-observation features: each model only says how to build
its feature vector and response from one observation."""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg


def normal_regression(
    *,
    name: str,
    data: dict,
    n_features: int,
    features: Callable,                       # datum -> (n_features,) feature vector
    response: Callable,                       # datum -> response (possibly transformed)
    log_prior: Optional[Callable] = None,     # (params, hyper) -> scalar; None = flat
    response_field: Optional[str] = None,     # field to redraw when simulating
    inverse_response: Optional[Callable] = None,  # response -> field value (e.g. jnp.exp)
    shared_data: Optional[dict] = None,
) -> mg.Model:
    """`response(datum) ~ normal(features(datum) @ beta, sigma)`, with
    params {"beta": (n_features,), "sigma": > 0}."""

    def log_lik_one(params, datum, hyper):
        return stats.norm.logpdf(response(datum), loc=features(datum) @ params["beta"], scale=params["sigma"])

    hooks = {"log_lik_one": log_lik_one}
    if log_prior is not None:
        hooks["log_prior"] = log_prior
    if response_field is not None:
        inv = inverse_response or (lambda r: r)

        def sample_one(key, params, datum, hyper):
            r = features(datum) @ params["beta"] + params["sigma"] * jax.random.normal(key)
            return {response_field: inv(r)}

        hooks["sample_one"] = sample_one
    return mg.Model(
        name=name,
        params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(n_features), sigma=pg.NumericScalarPattern(lb=0.0)),
        data=data,
        shared_data=shared_data,
        **hooks,
    )


def cauchy_sigma_prior(scale: float):
    return lambda params, hyper: stats.cauchy.logpdf(params["sigma"], loc=0.0, scale=scale)


def ones_and(*xs):
    return jnp.stack([jnp.ones_like(xs[0])] + list(xs))


def logistic_regression(
    *,
    name: str,
    data: dict,
    n_features: int,
    features: Callable,                       # datum -> (n_features,) feature vector, no intercept
    response_field: str,                      # 0/1 outcome
    shared_data: Optional[dict] = None,
) -> mg.Model:
    """`response ~ bernoulli_logit_glm(x, alpha, beta)` with flat priors, for
    the posteriordb models written that way, with params {"alpha": scalar,
    "beta": (n_features,)}."""

    def logit(params, datum):
        return params["alpha"] + features(datum) @ params["beta"]

    def log_lik_one(params, datum, hyper):
        eta = logit(params=params, datum=datum)
        return datum[response_field] * eta - jax.nn.softplus(eta)

    def sample_one(key, params, datum, hyper):
        draw = jax.random.bernoulli(key, p=jax.nn.sigmoid(logit(params=params, datum=datum)))
        return {response_field: draw.astype(datum[response_field].dtype)}

    data = dict(data)
    data[response_field] = jnp.asarray(data[response_field], dtype=jnp.int32)
    return mg.Model(
        name=name,
        params_pattern=pg.PatternDict(alpha=pg.NumericScalarPattern(), beta=pg.NumericVectorPattern(n_features)),
        data=data,
        shared_data=shared_data,
        log_lik_one=log_lik_one,
        sample_one=sample_one,
    )
