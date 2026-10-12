"""Shared code for the many posteriordb models that are a normal linear or
logistic regression on per-observation features: each model only says how to
build its feature vector and response from one observation."""

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
    features: Callable,                       # (datum, hyper) -> (n_features,) feature vector
    response: Callable,                       # datum -> response (possibly transformed)
    response_field: str,                      # field to redraw when simulating
    log_prior: Optional[Callable] = None,     # (params, hyper) -> scalar; None = flat
    inverse_response: Optional[Callable] = None,  # response -> field value (e.g. jnp.exp)
    hyper: Optional[dict] = None,
) -> mg.ExchangeableModel:
    """`response(datum) ~ normal(features(datum, hyper) @ beta, sigma)`, with
    params {"beta": (n_features,), "sigma": > 0}."""
    inv = (lambda r: r) if inverse_response is None else inverse_response

    def log_lik_one(params, datum, hyper):
        mean = features(datum, hyper) @ params["beta"]
        return stats.norm.logpdf(response(datum), loc=mean, scale=params["sigma"])

    def sample_one(key, params, datum, hyper):
        r = features(datum, hyper) @ params["beta"] + params["sigma"] * jax.random.normal(key)
        return {response_field: inv(r)}

    return mg.ExchangeableModel(
        name=name,
        params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(n_features), sigma=pg.NumericScalarPattern(lb=0.0)),
        prior=None if log_prior is None else mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data=data,
        hyper=hyper,
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
    features: Callable,                       # (datum, hyper) -> (n_features,) feature vector, no intercept
    response_field: str,                      # 0/1 outcome
    hyper: Optional[dict] = None,
) -> mg.ExchangeableModel:
    """`response ~ bernoulli_logit_glm(x, alpha, beta)` with flat priors, for
    the posteriordb models written that way, with params {"alpha": scalar,
    "beta": (n_features,)}."""

    def logit(params, datum, hyper):
        return params["alpha"] + features(datum, hyper) @ params["beta"]

    def log_lik_one(params, datum, hyper):
        eta = logit(params=params, datum=datum, hyper=hyper)
        return datum[response_field] * eta - jax.nn.softplus(eta)

    def sample_one(key, params, datum, hyper):
        p = jax.nn.sigmoid(logit(params=params, datum=datum, hyper=hyper))
        return {response_field: jax.random.bernoulli(key, p=p).astype(datum[response_field].dtype)}

    data = dict(data)
    data[response_field] = jnp.asarray(data[response_field], dtype=jnp.int32)
    return mg.ExchangeableModel(
        name=name,
        params_pattern=pg.PatternDict(alpha=pg.NumericScalarPattern(), beta=pg.NumericVectorPattern(n_features)),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data=data,
        hyper=hyper,
    )
