"""dogs_hierarchical.stan:

    parameters { real<lower=0, upper=1> a; real<lower=0, upper=1> b; }
    model { y[j, t] ~ bernoulli(a ^ prev_shock[j, t] * b ^ prev_avoid[j, t]); }

a and b have flat priors. Exchangeable: each dog's trial history is one datum.
"""

import jax.numpy as jnp

import modelgami as mg
import paragami as pg

from ._dogs import bernoulli_log_lik, previous_counts

STAN_MODEL = "dogs_hierarchical"
POSTERIORS = ["dogs-dogs_hierarchical"]


def log_lik_one(params, datum, hyper):
    shocks, avoids = previous_counts(datum["y"])
    p = params["a"] ** shocks * params["b"] ** avoids
    return bernoulli_log_lik(y=datum["y"], p=p)


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            a=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            b=pg.NumericScalarPattern(lb=0.0, ub=1.0),
        ),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
    )
