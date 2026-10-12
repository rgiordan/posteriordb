"""low_dim_gauss_mix.stan (two-component normal mixture):

    parameters { ordered[2] mu; array[2] real<lower=0> sigma; real<lower=0, upper=1> theta; }
    model {
      sigma ~ normal(0, 2);
      mu ~ normal(0, 2);
      theta ~ beta(5, 5);
      for (n in 1:N)
        target += log_mix(theta, normal_lpdf(y[n] | mu[1], sigma[1]), normal_lpdf(y[n] | mu[2], sigma[2]));
    }
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "low_dim_gauss_mix"
POSTERIORS = ["low_dim_gauss_mix-low_dim_gauss_mix"]


def log_prior(params, hyper):
    return (jnp.sum(stats.norm.logpdf(params["sigma"], loc=0.0, scale=2.0))
            + jnp.sum(stats.norm.logpdf(params["mu"], loc=0.0, scale=2.0))
            + stats.beta.logpdf(params["theta"], a=5.0, b=5.0))


def log_lik_one(params, datum, hyper):
    mu, sigma, theta = params["mu"], params["sigma"], params["theta"]
    return jnp.logaddexp(jnp.log(theta) + stats.norm.logpdf(datum["y"], loc=mu[0], scale=sigma[0]),
                         jnp.log1p(-theta) + stats.norm.logpdf(datum["y"], loc=mu[1], scale=sigma[1]))


def sample_one(key, params, datum, hyper):
    k1, k2 = jax.random.split(key)
    z = jax.random.bernoulli(k1, p=params["theta"]).astype(int)
    comp = 1 - z  # component 0 with probability theta
    return {"y": params["mu"][comp] + params["sigma"][comp] * jax.random.normal(k2)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            mu=pg.OrderedVectorPattern(2),
            sigma=pg.NumericVectorPattern(2, lb=0.0),
            theta=pg.NumericScalarPattern(lb=0.0, ub=1.0),
        ),
        data={"y": raw["y"]},
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
    )
