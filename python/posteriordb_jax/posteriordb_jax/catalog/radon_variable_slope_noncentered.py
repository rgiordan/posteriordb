"""radon_variable_slope_noncentered.stan:

    parameters {
      real alpha;
      vector[J] beta_raw;
      real mu_beta;
      real<lower=0> sigma_beta;
      real<lower=0> sigma_y;
    }
    transformed parameters {
      vector[J] beta;
      // implies: beta ~ normal(mu_beta, sigma_beta);
      beta = mu_beta + sigma_beta * beta_raw;
    }
    model {
      vector[N] mu;
      // Prior
      alpha ~ normal(0, 10);
      sigma_y ~ normal(0, 1);
      sigma_beta ~ normal(0, 1);
      mu_beta ~ normal(0, 10);
      beta_raw ~ normal(0, 1);
  
      for (n in 1 : N) {
        mu[n] = alpha + floor_measure[n] * beta[county_idx[n]];
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

STAN_MODEL = "radon_variable_slope_noncentered"
POSTERIORS = ["radon_all-radon_variable_slope_noncentered", "radon_mn-radon_variable_slope_noncentered"]


def beta(params):
    """The transformed parameter beta = mu_beta + sigma_beta * beta_raw."""
    return params["mu_beta"] + params["sigma_beta"] * params["beta_raw"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["alpha"], loc=0.0, scale=10.0)
            + stats.norm.logpdf(params["sigma_y"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["sigma_beta"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["mu_beta"], loc=0.0, scale=10.0)
            + jnp.sum(stats.norm.logpdf(params["beta_raw"], loc=0.0, scale=1.0)))


def _mean(params, datum):
    return params["alpha"] + datum["floor_measure"] * beta(params)[datum["county_idx"] - 1]


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["log_radon"], loc=_mean(params=params, datum=datum), scale=params["sigma_y"])


def sample_one(key, params, datum, hyper):
    return {"log_radon": _mean(params=params, datum=datum) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    j = int(raw["J"])
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(),
            beta_raw=pg.NumericVectorPattern(j),
            mu_beta=pg.NumericScalarPattern(),
            sigma_beta=pg.NumericScalarPattern(lb=0.0),
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
