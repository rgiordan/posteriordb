"""dogs.stan:

    parameters { vector[3] beta; }
    transformed parameters {
      n_avoid[j, t], n_shock[j, t] = avoidances and shocks before trial t;
      p[j, t] = beta[1] + beta[2] * n_avoid[j, t] + beta[3] * n_shock[j, t];
    }
    model {
      beta ~ normal(0, 100);
      y[i, j] ~ bernoulli_logit(p[i, j]);
    }

Exchangeable in the modelgami sense: each dog's trial history is one datum.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._dogs import previous_counts

STAN_MODEL = "dogs"
POSTERIORS = ["dogs-dogs"]


def log_prior(params, hyper):
    return jnp.sum(stats.norm.logpdf(params["beta"], loc=0.0, scale=100.0))


def log_lik_one(params, datum, hyper):
    y = datum["y"]
    shocks, avoids = previous_counts(y)
    beta = params["beta"]
    logit_p = beta[0] + beta[1] * avoids + beta[2] * shocks
    return jnp.sum(y * logit_p - jax.nn.softplus(logit_p))


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(3)),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
    )
