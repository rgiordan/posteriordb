"""dugongs_model.stan (nonlinear growth curve):

    parameters { real alpha; real beta; real<lower=.5, upper=1> lambda; real<lower=0> tau; }
    transformed parameters { real sigma = 1 / sqrt(tau); real U3 = logit(lambda); }
    model {
      Y ~ normal(alpha - beta * pow(lambda, x), sigma);
      alpha ~ normal(0.0, 1000);
      beta ~ normal(0.0, 1000);
      lambda ~ uniform(.5, 1);
      tau ~ gamma(.0001, .0001);
    }
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "dugongs_model"
POSTERIORS = ["dugongs_data-dugongs_model"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["alpha"], loc=0.0, scale=1000.0)
            + stats.norm.logpdf(params["beta"], loc=0.0, scale=1000.0)
            + stats.uniform.logpdf(params["lambda"], loc=0.5, scale=0.5)
            + stats.gamma.logpdf(params["tau"], a=1.0e-4, scale=1.0e4))


def mean(params, datum):
    return params["alpha"] - params["beta"] * params["lambda"] ** datum["x"]


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["Y"], loc=mean(params=params, datum=datum), scale=1.0 / jnp.sqrt(params["tau"]))


def sample_one(key, params, datum, hyper):
    return {"Y": mean(params=params, datum=datum) + jax.random.normal(key) / jnp.sqrt(params["tau"])}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(),
            beta=pg.NumericScalarPattern(),
            # `lambda` is a Python keyword, so the pattern is built from a dict.
            **{"lambda": pg.NumericScalarPattern(lb=0.5, ub=1.0)},
            tau=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data={"x": raw["x"], "Y": raw["Y"]},
    )
