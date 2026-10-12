"""normal_mixture.stan (two-component mixture, unit scales):

    parameters { real<lower=0, upper=1> theta; array[2] real mu; }
    model {
      theta ~ uniform(0, 1);
      mu ~ normal(0, 10);
      for (n in 1:N)
        target += log_mix(theta, normal_lpdf(y[n] | mu[1], 1.0), normal_lpdf(y[n] | mu[2], 1.0));
    }
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "normal_mixture"
POSTERIORS = ["normal_2-normal_mixture"]


def log_prior(params, hyper):
    return stats.uniform.logpdf(params["theta"], loc=0.0, scale=1.0) + jnp.sum(stats.norm.logpdf(params["mu"], loc=0.0, scale=10.0))


def log_lik_one(params, datum, hyper):
    mu, theta = params["mu"], params["theta"]
    return jnp.logaddexp(jnp.log(theta) + stats.norm.logpdf(datum["y"], loc=mu[0], scale=1.0),
                         jnp.log1p(-theta) + stats.norm.logpdf(datum["y"], loc=mu[1], scale=1.0))


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            theta=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            mu=pg.NumericVectorPattern(2),
        ),
        data={"y": raw["y"]},
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one),
    )
