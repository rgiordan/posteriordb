"""radon_county_intercept.stan:

    parameters {
      vector[J] alpha;
      real beta;
      real<lower=0> sigma_y;
    }
    model {
      vector[N] mu;
      // Prior
      sigma_y ~ normal(0, 1);
      alpha ~ normal(0, 10);
      beta ~ normal(0, 10);
  
      // Likelihood
      for (n in 1 : N) {
        mu[n] = alpha[county_idx[n]] + beta * floor_measure[n];
        target += normal_lpdf(log_radon[n] | mu[n], sigma_y);
      }
    }

Exchangeable in the modelgami sense: each house is one datum, carrying its
county index, and houses are independent given the parameters.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "radon_county_intercept"
POSTERIORS = ["radon_all-radon_county_intercept", "radon_mn-radon_county_intercept"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["sigma_y"], loc=0.0, scale=1.0)
            + jnp.sum(stats.norm.logpdf(params["alpha"], loc=0.0, scale=10.0))
            + stats.norm.logpdf(params["beta"], loc=0.0, scale=10.0))


def _mean(params, datum):
    return params["alpha"][datum["county_idx"] - 1] + params["beta"] * datum["floor_measure"]


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["log_radon"], loc=_mean(params=params, datum=datum), scale=params["sigma_y"])


def sample_one(key, params, datum, hyper):
    return {"log_radon": _mean(params=params, datum=datum) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    j = int(raw["J"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericVectorPattern(j),
            beta=pg.NumericScalarPattern(),
            sigma_y=pg.NumericScalarPattern(lb=0.0),
        ),
        data={
            "county_idx": jnp.asarray(raw["county_idx"], dtype=jnp.int32),
            "floor_measure": raw["floor_measure"],
            "log_radon": raw["log_radon"],
        },
        log_prior=log_prior,
        log_lik_one=log_lik_one,
        sample_one=sample_one,
    )
