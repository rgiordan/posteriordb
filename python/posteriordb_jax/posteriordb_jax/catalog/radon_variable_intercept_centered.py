"""radon_variable_intercept_centered.stan:

    parameters {
      vector[J] alpha;
      real beta;
      real mu_alpha;
      real<lower=0> sigma_alpha;
      real<lower=0> sigma_y;
    }
    model {
      vector[N] mu;
  
      // Prior
      sigma_y ~ normal(0, 1);
      sigma_alpha ~ normal(0, 1);
      mu_alpha ~ normal(0, 10);
      beta ~ normal(0, 10);
  
      alpha ~ normal(mu_alpha, sigma_alpha);
      for (n in 1 : N) {
        mu[n] = alpha[county_idx[n]] + floor_measure[n] * beta;
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

STAN_MODEL = "radon_variable_intercept_centered"
POSTERIORS = ["radon_all-radon_variable_intercept_centered", "radon_mn-radon_variable_intercept_centered"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["sigma_y"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["sigma_alpha"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["mu_alpha"], loc=0.0, scale=10.0)
            + stats.norm.logpdf(params["beta"], loc=0.0, scale=10.0)
            + jnp.sum(stats.norm.logpdf(params["alpha"], loc=params["mu_alpha"], scale=params["sigma_alpha"])))


def _mean(params, datum):
    return params["alpha"][datum["county_idx"] - 1] + datum["floor_measure"] * params["beta"]


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["log_radon"], loc=_mean(params=params, datum=datum), scale=params["sigma_y"])


def sample_one(key, params, datum, hyper):
    return {"log_radon": _mean(params=params, datum=datum) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    j = int(raw["J"])
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericVectorPattern(j),
            beta=pg.NumericScalarPattern(),
            mu_alpha=pg.NumericScalarPattern(),
            sigma_alpha=pg.NumericScalarPattern(lb=0.0),
            sigma_y=pg.NumericScalarPattern(lb=0.0),
        ),
        data={
            "county_idx": jnp.asarray(raw["county_idx"], dtype=jnp.int32),
            "floor_measure": raw["floor_measure"],
            "log_radon": raw["log_radon"],
        },
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
    )
