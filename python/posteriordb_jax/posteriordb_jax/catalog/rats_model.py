"""rats_model.stan (growth curves with rat-specific intercepts and slopes):

    parameters {
      array[N] real alpha; array[N] real beta; real mu_alpha; real mu_beta;
      real<lower=0> sigma_y; real<lower=0> sigma_alpha; real<lower=0> sigma_beta;
    }
    model {
      mu_alpha ~ normal(0, 100);
      mu_beta ~ normal(0, 100);
      alpha ~ normal(mu_alpha, sigma_alpha);
      beta ~ normal(mu_beta, sigma_beta);
      y[n] ~ normal(alpha[rat[n]] + beta[rat[n]] * (x[n] - xbar), sigma_y);
    }

Exchangeable in the modelgami sense: each measurement is one datum carrying
its rat index. xbar is a hyperparameter.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "rats_model"
POSTERIORS = ["rats_data-rats_model"]


def eval_log_prior(params, hyper):
    return (stats.norm.logpdf(params["mu_alpha"], loc=0.0, scale=100.0)
            + stats.norm.logpdf(params["mu_beta"], loc=0.0, scale=100.0)
            + jnp.sum(stats.norm.logpdf(params["alpha"], loc=params["mu_alpha"], scale=params["sigma_alpha"]))
            + jnp.sum(stats.norm.logpdf(params["beta"], loc=params["mu_beta"], scale=params["sigma_beta"])))


def compute_mean(params, datum, hyper):
    r = datum["rat"] - 1
    return params["alpha"][r] + params["beta"][r] * (datum["x"] - hyper["xbar"])


def eval_log_lik_one(datum, params, hyper):
    return stats.norm.logpdf(datum["y"], loc=compute_mean(params=params, datum=datum, hyper=hyper), scale=params["sigma_y"])


def draw_one(key, datum, params, hyper):
    return {"y": compute_mean(params=params, datum=datum, hyper=hyper) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    n = int(raw["N"])
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericVectorPattern(n),
            beta=pg.NumericVectorPattern(n),
            mu_alpha=pg.NumericScalarPattern(),
            mu_beta=pg.NumericScalarPattern(),
            sigma_y=pg.NumericScalarPattern(lb=0.0),
            sigma_alpha=pg.NumericScalarPattern(lb=0.0),
            sigma_beta=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data={"rat": jnp.asarray(raw["rat"], dtype=jnp.int32), "x": raw["x"], "y": raw["y"]},
        hyper={"xbar": float(raw["xbar"])},
    )
