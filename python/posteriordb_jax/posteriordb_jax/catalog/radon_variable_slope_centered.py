"""radon_variable_slope_centered.stan:

    parameters {
      real alpha;
      vector[J] beta;
      real mu_beta;
      real<lower=0> sigma_beta;
      real<lower=0> sigma_y;
    }
    model {
      vector[N] mu;
      // Prior
      alpha ~ normal(0, 10);
      sigma_y ~ normal(0, 1);
      sigma_beta ~ normal(0, 1);
      mu_beta ~ normal(0, 10);
  
      beta ~ normal(mu_beta, sigma_beta);
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

STAN_MODEL = "radon_variable_slope_centered"
POSTERIORS = ["radon_all-radon_variable_slope_centered", "radon_mn-radon_variable_slope_centered"]


def eval_log_prior(params, hyper):
    return (stats.norm.logpdf(params["alpha"], loc=0.0, scale=10.0)
            + stats.norm.logpdf(params["sigma_y"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["sigma_beta"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["mu_beta"], loc=0.0, scale=10.0)
            + jnp.sum(stats.norm.logpdf(params["beta"], loc=params["mu_beta"], scale=params["sigma_beta"])))


def _compute_mean(params, datum):
    return params["alpha"] + datum["floor_measure"] * params["beta"][datum["county_idx"] - 1]


def eval_log_lik_one(datum, params, hyper):
    return stats.norm.logpdf(datum["log_radon"], loc=_compute_mean(params=params, datum=datum), scale=params["sigma_y"])


def draw_one(key, datum, params, hyper):
    return {"log_radon": _compute_mean(params=params, datum=datum) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    j = int(raw["J"])
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(),
            beta=pg.NumericVectorPattern(j),
            mu_beta=pg.NumericScalarPattern(),
            sigma_beta=pg.NumericScalarPattern(lb=0.0),
            sigma_y=pg.NumericScalarPattern(lb=0.0),
        ),
        data={
            "county_idx": jnp.asarray(raw["county_idx"], dtype=jnp.int32),
            "floor_measure": raw["floor_measure"],
            "log_radon": raw["log_radon"],
        },
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
    )
