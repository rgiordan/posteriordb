"""Mb_model.stan (capture-recapture with a behavioral response: detection
probability is p before the first capture and c after a capture on the
previous occasion):

    parameters { real<lower=0, upper=1> omega; real<lower=0, upper=1> p; real<lower=0, upper=1> c; }
    transformed parameters {
      p_eff[i, 1] = p;
      p_eff[i, j] = (1 - y[i, j - 1]) * p + y[i, j - 1] * c;   // j >= 2
    }
    model {
      for (i in 1:M) {
        if (s[i] > 0) target += bernoulli_lpmf(1 | omega) + bernoulli_lpmf(y[i] | p_eff[i]);
        else target += log_sum_exp(bernoulli_lpmf(1 | omega) + bernoulli_lpmf(y[i] | p_eff[i]), bernoulli_lpmf(0 | omega));
      }
    }

Each row of the capture history `y` (M x T) is one exchangeable datum. All
three parameters have flat priors. No data simulator: p_eff depends on the
previous occasion's capture, so occasions are not independent draws.
"""

import jax.numpy as jnp

import modelgami as mg
import paragami as pg

from ._capture import bernoulli_log_lik, log_lik_augmented

STAN_MODEL = "Mb_model"
POSTERIORS = ["Mb_data-Mb_model"]


def p_eff(params, y):
    previous = jnp.concatenate([jnp.zeros(1, dtype=y.dtype), y[:-1]])
    return (1 - previous) * params["p"] + previous * params["c"]


def log_lik_one(params, datum, hyper):
    captures = bernoulli_log_lik(y=datum["y"], p=p_eff(params=params, y=datum["y"]))
    return log_lik_augmented(omega=params["omega"], log_lik_captures=captures, captured=jnp.sum(datum["y"]) > 0)


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            omega=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            p=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            c=pg.NumericScalarPattern(lb=0.0, ub=1.0),
        ),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
    )
