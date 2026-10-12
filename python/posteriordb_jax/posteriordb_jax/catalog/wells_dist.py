"""wells_dist.stan (logistic regression, flat prior):

    parameters { vector[2] beta; }
    model { switched ~ bernoulli_logit(beta[1] + beta[2] * dist); }
"""

import jax
import jax.numpy as jnp

import modelgami as mg
import paragami as pg

STAN_MODEL = "wells_dist"
POSTERIORS = ["wells_data-wells_dist"]


def _logit(params, datum):
    return params["beta"][0] + params["beta"][1] * datum["dist"]


def log_lik_one(params, datum, hyper):
    logit = _logit(params=params, datum=datum)
    return datum["switched"] * logit - jax.nn.softplus(logit)


def sample_one(key, params, datum, hyper):
    return {"switched": jax.random.bernoulli(key, p=jax.nn.sigmoid(_logit(params=params, datum=datum))).astype(datum["switched"].dtype)}


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(2)),
        data={"switched": jnp.asarray(raw["switched"], dtype=jnp.int32), "dist": raw["dist"]},
        log_lik_one=log_lik_one,
        sample_one=sample_one,
    )
