"""radon_partially_pooled_centered.stan:

    parameters {
      vector[J] alpha;
      real mu_alpha;
      real<lower=0> sigma_alpha;
      real<lower=0> sigma_y;
    }
    model {
      vector[N] mu;
  
      // priors
      sigma_y ~ normal(0, 1);
      sigma_alpha ~ normal(0, 1);
      mu_alpha ~ normal(0, 10);
  
      // likelihood
      alpha ~ normal(mu_alpha, sigma_alpha);
      for (n in 1 : N) {
        mu[n] = alpha[county_idx[n]];
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

STAN_MODEL = "radon_partially_pooled_centered"
POSTERIORS = ["radon_all-radon_partially_pooled_centered", "radon_mn-radon_partially_pooled_centered"]


def eval_log_prior(params, hyper):
    return (stats.norm.logpdf(params["sigma_y"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["sigma_alpha"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["mu_alpha"], loc=0.0, scale=10.0)
            + jnp.sum(stats.norm.logpdf(params["alpha"], loc=params["mu_alpha"], scale=params["sigma_alpha"])))


def _compute_mean(params, datum):
    return params["alpha"][datum["county_idx"] - 1]


def eval_log_lik_one(datum, params, hyper):
    return stats.norm.logpdf(datum["log_radon"], loc=_compute_mean(params=params, datum=datum), scale=params["sigma_y"])


def draw_one(key, datum, params, hyper):
    return {"log_radon": _compute_mean(params=params, datum=datum) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    j = int(raw["J"])
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericVectorPattern(j),
            mu_alpha=pg.NumericScalarPattern(),
            sigma_alpha=pg.NumericScalarPattern(lb=0.0),
            sigma_y=pg.NumericScalarPattern(lb=0.0),
        ),
        data={
            "county_idx": jnp.asarray(raw["county_idx"], dtype=jnp.int32),
            "log_radon": raw["log_radon"],
        },
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
    )
