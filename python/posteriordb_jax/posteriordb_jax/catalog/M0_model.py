"""M0_model.stan (capture-recapture, constant detection probability):

    parameters { real<lower=0, upper=1> omega; real<lower=0, upper=1> p; }
    model {
      for (i in 1:M) {
        if (s[i] > 0) target += bernoulli_lpmf(1 | omega) + binomial_lpmf(s[i] | T, p);
        else target += log_sum_exp(bernoulli_lpmf(1 | omega) + binomial_lpmf(0 | T, p), bernoulli_lpmf(0 | omega));
      }
    }

Each row of the capture history `y` (M x T) is one exchangeable datum; T is a
hyperparameter. omega and p have flat priors.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._capture import eval_log_lik_augmented, sample_history

STAN_MODEL = "M0_model"
POSTERIORS = ["M0_data-M0_model"]


def eval_log_lik_one(datum, params, hyper):
    s = jnp.sum(datum["y"])
    captures = stats.binom.logpmf(s, n=hyper["T"], p=params["p"])
    return eval_log_lik_augmented(omega=params["omega"], log_lik_captures=captures, captured=s > 0)


def draw_one(key, datum, params, hyper):
    return {"y": sample_history(key=key, omega=params["omega"], p=jnp.full(datum["y"].shape, params["p"]))}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            omega=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            p=pg.NumericScalarPattern(lb=0.0, ub=1.0),
        ),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
        hyper={"T": int(raw["T"])},
    )
