"""blr.stan (Bayesian linear regression):

    parameters { vector[D] beta; real<lower=0> sigma; }
    model {
      target += normal_lpdf(beta | 0, 10);
      target += normal_lpdf(sigma | 0, 10);
      target += normal_lpdf(y | X * beta, sigma);
    }
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "blr"
POSTERIORS = ["sblri-blr", "sblrc-blr"]


def eval_log_prior(params, hyper):
    return jnp.sum(stats.norm.logpdf(params["beta"], loc=0.0, scale=10.0)) + stats.norm.logpdf(params["sigma"], loc=0.0, scale=10.0)


def eval_log_lik_one(datum, params, hyper):
    return stats.norm.logpdf(datum["y"], loc=datum["X"] @ params["beta"], scale=params["sigma"])


def draw_one(key, datum, params, hyper):
    return {"y": datum["X"] @ params["beta"] + params["sigma"] * jax.random.normal(key)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            beta=pg.NumericVectorPattern(int(raw["D"])), sigma=pg.NumericScalarPattern(lb=0.0)),
        data={"y": raw["y"], "X": raw["X"]},
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
    )
