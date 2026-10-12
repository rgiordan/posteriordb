"""kidscore_momiq.stan:

    parameters { vector[2] beta; real<lower=0> sigma; }
    model {
      sigma ~ cauchy(0, 2.5);
      kid_score ~ normal(beta[1] + beta[2] * mom_iq, sigma);
    }

`beta` has an improper flat prior (no term).
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "kidscore_momiq"
POSTERIORS = ["kidiq-kidscore_momiq"]


def log_prior(params, hyper):
    return stats.cauchy.logpdf(params["sigma"], loc=0.0, scale=2.5)


def log_lik_one(params, datum, hyper):
    mean = params["beta"][0] + params["beta"][1] * datum["mom_iq"]
    return stats.norm.logpdf(datum["kid_score"], loc=mean, scale=params["sigma"])


def sample_one(key, params, datum, hyper):
    mean = params["beta"][0] + params["beta"][1] * datum["mom_iq"]
    return {"kid_score": mean + params["sigma"] * jax.random.normal(key)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(2), sigma=pg.NumericScalarPattern(lb=0.0)),
        data={"kid_score": raw["kid_score"], "mom_iq": raw["mom_iq"]},
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
    )
