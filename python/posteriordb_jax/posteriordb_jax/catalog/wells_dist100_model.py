"""wells_dist100_model.stan (logistic regression, flat priors):

    transformed data { vector[N] dist100 = dist / 100.0; }
    parameters { real alpha; vector[1] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }   // x = [dist100]
"""

import jax
import jax.numpy as jnp

import modelgami as mg
import paragami as pg

STAN_MODEL = "wells_dist100_model"
POSTERIORS = ["wells_data-wells_dist100_model"]


def _logit(params, datum):
    return params["alpha"] + params["beta"][0] * datum["dist"] / 100.0


def log_lik_one(params, datum, hyper):
    logit = _logit(params, datum)
    return datum["switched"] * logit - jax.nn.softplus(logit)


def sample_one(key, params, datum, hyper):
    return {"switched": jax.random.bernoulli(key, jax.nn.sigmoid(_logit(params, datum))).astype(datum["switched"].dtype)}


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(alpha=pg.NumericScalarPattern(), beta=pg.NumericVectorPattern(1)),
        data={"switched": jnp.asarray(raw["switched"], dtype=jnp.int32), "dist": raw["dist"]},
        log_lik_one=log_lik_one,
        sample_one=sample_one,
    )
