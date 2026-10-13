"""dogs_log.stan:

    parameters { vector[2] beta; }
    transformed parameters { p[j, t] = inv_logit(beta[1] * n_avoid[j, t] + beta[2] * n_shock[j, t]); }
    model {
      beta[1] ~ uniform(-100, 0);
      beta[2] ~ uniform(0, 100);
      y[i, j] ~ bernoulli(p[i, j]);
    }

beta is unconstrained in Stan, so the log density is -inf outside the
uniform priors' supports. Exchangeable: each dog's trial history is one datum.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._dogs import count_previous

STAN_MODEL = "dogs_log"
POSTERIORS = ["dogs-dogs_log"]


def eval_log_prior(params, hyper):
    return (stats.uniform.logpdf(params["beta"][0], loc=-100.0, scale=100.0)
            + stats.uniform.logpdf(params["beta"][1], loc=0.0, scale=100.0))


def eval_log_lik_one(datum, params, hyper):
    y = datum["y"]
    shocks, avoids = count_previous(y)
    logit_p = params["beta"][0] * avoids + params["beta"][1] * shocks
    return jnp.sum(y * logit_p - jax.nn.softplus(logit_p))


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(2)),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
    )
