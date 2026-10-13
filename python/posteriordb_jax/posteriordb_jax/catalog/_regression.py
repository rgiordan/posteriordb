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


def build_normal_regression(
    *,
    name: str,
    data: dict,
    n_features: int,
    compute_features: Callable,               # (datum, hyper) -> (n_features,) feature vector
    compute_response: Callable,               # datum -> response (possibly transformed)
    response_field: str,                      # field to redraw when simulating
    eval_log_prior: Optional[Callable] = None,     # (params, hyper) -> scalar; None = flat
    invert_response: Optional[Callable] = None,   # response -> field value (e.g. jnp.exp)
    hyper: Optional[dict] = None,
) -> mg.ExchangeableModel:
    """`compute_response(datum) ~ normal(compute_features(datum, hyper) @ beta, sigma)`, with
    params {"beta": (n_features,), "sigma": > 0}."""
    inv = (lambda r: r) if invert_response is None else invert_response

    def eval_log_lik_one(datum, params, hyper):
        mean = compute_features(datum, hyper) @ params["beta"]
        return stats.norm.logpdf(compute_response(datum), loc=mean, scale=params["sigma"])

    def draw_one(key, datum, params, hyper):
        r = compute_features(datum, hyper) @ params["beta"] + params["sigma"] * jax.random.normal(key)
        return {response_field: inv(r)}

    return mg.ExchangeableModel(
        name=name,
        params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(n_features), sigma=pg.NumericScalarPattern(lb=0.0)),
        prior=None if eval_log_prior is None else mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data=data,
        hyper=hyper,
    )


def make_cauchy_sigma_prior(scale: float):
    return lambda params, hyper: stats.cauchy.logpdf(params["sigma"], loc=0.0, scale=scale)


def prepend_ones(*xs):
    return jnp.stack([jnp.ones_like(xs[0])] + list(xs))


def build_logistic_regression(
    *,
    name: str,
    data: dict,
    n_features: int,
    compute_features: Callable,               # (datum, hyper) -> (n_features,) feature vector, no intercept
    response_field: str,                      # 0/1 outcome
    hyper: Optional[dict] = None,
) -> mg.ExchangeableModel:
    """`response ~ bernoulli_logit_glm(x, alpha, beta)` with flat priors, for
    the posteriordb models written that way, with params {"alpha": scalar,
    "beta": (n_features,)}."""

    def logit(params, datum, hyper):
        return params["alpha"] + compute_features(datum, hyper) @ params["beta"]

    def eval_log_lik_one(datum, params, hyper):
        eta = logit(params=params, datum=datum, hyper=hyper)
        return datum[response_field] * eta - jax.nn.softplus(eta)

    def draw_one(key, datum, params, hyper):
        p = jax.nn.sigmoid(logit(params=params, datum=datum, hyper=hyper))
        return {response_field: jax.random.bernoulli(key, p=p).astype(datum[response_field].dtype)}

    data = dict(data)
    data[response_field] = jnp.asarray(data[response_field], dtype=jnp.int32)
    return mg.ExchangeableModel(
        name=name,
        params_pattern=pg.PatternDict(alpha=pg.NumericScalarPattern(), beta=pg.NumericVectorPattern(n_features)),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data=data,
        hyper=hyper,
    )
