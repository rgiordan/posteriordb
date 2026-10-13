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


def eval_log_prior(params, hyper):
    return (stats.norm.logpdf(params["alpha"], loc=0.0, scale=1000.0)
            + stats.norm.logpdf(params["beta"], loc=0.0, scale=1000.0)
            + stats.uniform.logpdf(params["lambda"], loc=0.5, scale=0.5)
            + stats.gamma.logpdf(params["tau"], a=1.0e-4, scale=1.0e4))


def compute_mean(params, datum):
    return params["alpha"] - params["beta"] * params["lambda"] ** datum["x"]


def eval_log_lik_one(datum, params, hyper):
    return stats.norm.logpdf(datum["Y"], loc=compute_mean(params=params, datum=datum), scale=1.0 / jnp.sqrt(params["tau"]))


def draw_one(key, datum, params, hyper):
    return {"Y": compute_mean(params=params, datum=datum) + jax.random.normal(key) / jnp.sqrt(params["tau"])}


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
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data={"x": raw["x"], "Y": raw["Y"]},
    )
