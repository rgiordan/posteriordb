"""logearn_height.stan (flat priors):

    transformed data { vector[N] log_earn = log(earn); }
    parameters { vector[2] beta; real<lower=0> sigma; }
    model { log_earn ~ normal(beta[1] + beta[2] * height, sigma); }
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "logearn_height"
POSTERIORS = ["earnings-logearn_height"]


def log_lik_one(params, datum, hyper):
    mean = params["beta"][0] + params["beta"][1] * datum["height"]
    return stats.norm.logpdf(jnp.log(datum["earn"]), loc=mean, scale=params["sigma"])


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(2), sigma=pg.NumericScalarPattern(lb=0.0)),
        data={"earn": raw["earn"], "height": raw["height"]},
        log_lik_one=log_lik_one,
    )
