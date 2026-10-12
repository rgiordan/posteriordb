"""radon_county.stan:

    parameters {
      vector[J] a;
      real mu_a;
      real<lower=0, upper=100> sigma_a;
      real<lower=0, upper=100> sigma_y;
    }
    model {
      vector[N] y_hat;
      for (i in 1 : N) {
        y_hat[i] = a[county[i]];
      }
  
      mu_a ~ normal(0, 1);
      a ~ normal(mu_a, sigma_a);
      y ~ normal(y_hat, sigma_y);
    }

Exchangeable in the modelgami sense: each house is one datum, carrying its
county index, and houses are independent given the parameters.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "radon_county"
POSTERIORS = ["radon_mod-radon_county"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["mu_a"], loc=0.0, scale=1.0)
            + jnp.sum(stats.norm.logpdf(params["a"], loc=params["mu_a"], scale=params["sigma_a"])))


def _mean(params, datum):
    return params["a"][datum["county"] - 1]


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["y"], loc=_mean(params=params, datum=datum), scale=params["sigma_y"])


def sample_one(key, params, datum, hyper):
    return {"y": _mean(params=params, datum=datum) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    j = int(raw["J"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            a=pg.NumericVectorPattern(j),
            mu_a=pg.NumericScalarPattern(),
            sigma_a=pg.NumericScalarPattern(lb=0.0, ub=100.0),
            sigma_y=pg.NumericScalarPattern(lb=0.0, ub=100.0),
        ),
        data={
            "county": jnp.asarray(raw["county"], dtype=jnp.int32),
            "y": raw["y"],
        },
        log_prior=log_prior,
        log_lik_one=log_lik_one,
        sample_one=sample_one,
    )
