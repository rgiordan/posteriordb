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

from ._dogs import count_previous

STAN_MODEL = "dogs"
POSTERIORS = ["dogs-dogs"]


def eval_log_prior(params, hyper):
    return jnp.sum(stats.norm.logpdf(params["beta"], loc=0.0, scale=100.0))


def eval_log_lik_one(datum, params, hyper):
    y = datum["y"]
    shocks, avoids = count_previous(y)
    beta = params["beta"]
    logit_p = beta[0] + beta[1] * avoids + beta[2] * shocks
    return jnp.sum(y * logit_p - jax.nn.softplus(logit_p))


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(3)),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
    )
