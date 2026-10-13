"""eight_schools_centered.stan:

    parameters { vector[J] theta; real mu; real<lower=0> tau; }
    model {
      mu ~ normal(0, 5);
      tau ~ cauchy(0, 5);
      theta ~ normal(mu, tau);
      y ~ normal(theta, sigma);
    }
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "eight_schools_centered"
POSTERIORS = ["eight_schools-eight_schools_centered"]


def eval_log_prior(params, hyper):
    return (stats.norm.logpdf(params["mu"], loc=0.0, scale=5.0)
            + stats.cauchy.logpdf(params["tau"], loc=0.0, scale=5.0)
            + jnp.sum(stats.norm.logpdf(params["theta"], loc=params["mu"], scale=params["tau"])))


def eval_log_lik(data, params, hyper):
    return jnp.sum(stats.norm.logpdf(data["y"], loc=params["theta"], scale=data["sigma"]))


def build(raw):
    j = int(raw["J"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            theta=pg.NumericVectorPattern(j),
            mu=pg.NumericScalarPattern(),
            tau=pg.NumericScalarPattern(lb=0.0),
        ),
        data={"y": raw["y"], "sigma": raw["sigma"]},
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
    )
