"""radon_variable_intercept_slope_noncentered.stan:

    parameters {
      real<lower=0> sigma_y;
      real<lower=0> sigma_alpha;
      real<lower=0> sigma_beta;
      vector[J] alpha_raw;
      vector[J] beta_raw;
      real mu_alpha;
      real mu_beta;
    }
    transformed parameters {
      vector[J] alpha;
      vector[J] beta;
      // implies: alpha ~ normal(mu_alpha, sigma_alpha);
      alpha = mu_alpha + sigma_alpha * alpha_raw;
      // implies: beta ~ normal(mu_beta, sigma_beta);
      beta = mu_beta + sigma_beta * beta_raw;
    }
    model {
      vector[N] mu;
      // Prior
      sigma_y ~ normal(0, 1);
      sigma_beta ~ normal(0, 1);
      sigma_alpha ~ normal(0, 1);
      mu_alpha ~ normal(0, 10);
      mu_beta ~ normal(0, 10);
      alpha_raw ~ normal(0, 1);
      beta_raw ~ normal(0, 1);
  
      // Likelihood
      for (n in 1 : N) {
        mu[n] = alpha[county_idx[n]] + floor_measure[n] * beta[county_idx[n]];
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

STAN_MODEL = "radon_variable_intercept_slope_noncentered"
POSTERIORS = ["radon_all-radon_variable_intercept_slope_noncentered", "radon_mn-radon_variable_intercept_slope_noncentered"]


def alpha(params):
    """The transformed parameter alpha = mu_alpha + sigma_alpha * alpha_raw."""
    return params["mu_alpha"] + params["sigma_alpha"] * params["alpha_raw"]


def beta(params):
    """The transformed parameter beta = mu_beta + sigma_beta * beta_raw."""
    return params["mu_beta"] + params["sigma_beta"] * params["beta_raw"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["sigma_y"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["sigma_beta"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["sigma_alpha"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["mu_alpha"], loc=0.0, scale=10.0)
            + stats.norm.logpdf(params["mu_beta"], loc=0.0, scale=10.0)
            + jnp.sum(stats.norm.logpdf(params["alpha_raw"], loc=0.0, scale=1.0))
            + jnp.sum(stats.norm.logpdf(params["beta_raw"], loc=0.0, scale=1.0)))


def _mean(params, datum):
    c = datum["county_idx"] - 1
    return alpha(params)[c] + datum["floor_measure"] * beta(params)[c]


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["log_radon"], loc=_mean(params=params, datum=datum), scale=params["sigma_y"])


def sample_one(key, params, datum, hyper):
    return {"log_radon": _mean(params=params, datum=datum) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    j = int(raw["J"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            sigma_y=pg.NumericScalarPattern(lb=0.0),
            sigma_alpha=pg.NumericScalarPattern(lb=0.0),
            sigma_beta=pg.NumericScalarPattern(lb=0.0),
            alpha_raw=pg.NumericVectorPattern(j),
            beta_raw=pg.NumericVectorPattern(j),
            mu_alpha=pg.NumericScalarPattern(),
            mu_beta=pg.NumericScalarPattern(),
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
