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


def _compute_logit(params, datum):
    return params["beta"][0] + params["beta"][1] * datum["dist"]


def eval_log_lik_one(datum, params, hyper):
    logit = _compute_logit(params=params, datum=datum)
    return datum["switched"] * logit - jax.nn.softplus(logit)


def draw_one(key, datum, params, hyper):
    return {"switched": jax.random.bernoulli(key, p=jax.nn.sigmoid(_compute_logit(params=params, datum=datum))).astype(datum["switched"].dtype)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(2)),
        data={"switched": jnp.asarray(raw["switched"], dtype=jnp.int32), "dist": raw["dist"]},
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
    )
