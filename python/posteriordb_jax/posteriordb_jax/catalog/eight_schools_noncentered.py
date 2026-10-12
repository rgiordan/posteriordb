"""eight_schools_noncentered.stan:

    parameters { vector[J] theta_trans; real mu; real<lower=0> tau; }
    transformed parameters { vector[J] theta = theta_trans * tau + mu; }
    model {
      theta_trans ~ normal(0, 1);
      y ~ normal(theta, sigma);
      mu ~ normal(0, 5);
      tau ~ cauchy(0, 5);
    }

Not exchangeable in the modelgami sense: each school has its own parameter.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "eight_schools_noncentered"
POSTERIORS = ["eight_schools-eight_schools_noncentered"]


def log_prior(params, hyper):
    return (jnp.sum(stats.norm.logpdf(params["theta_trans"], 0.0, 1.0))
            + stats.norm.logpdf(params["mu"], 0.0, 5.0)
            + stats.cauchy.logpdf(params["tau"], 0.0, 5.0))


def log_lik(params, data, hyper):
    theta = params["theta_trans"] * params["tau"] + params["mu"]
    return jnp.sum(stats.norm.logpdf(data["y"], theta, data["sigma"]))


def build(raw):
    j = int(raw["J"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            theta_trans=pg.NumericVectorPattern(j),
            mu=pg.NumericScalarPattern(),
            tau=pg.NumericScalarPattern(lb=0.0),
        ),
        data={"y": raw["y"], "sigma": raw["sigma"]},
        log_prior=log_prior,
        log_lik=log_lik,
    )
