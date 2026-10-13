"""Mt_model.stan (capture-recapture, detection probability varying by occasion):

    parameters { real<lower=0, upper=1> omega; vector<lower=0, upper=1>[T] p; }
    model {
      for (i in 1:M) {
        if (s[i] > 0) target += bernoulli_lpmf(1 | omega) + bernoulli_lpmf(y[i] | p);
        else target += log_sum_exp(bernoulli_lpmf(1 | omega) + bernoulli_lpmf(y[i] | p), bernoulli_lpmf(0 | omega));
      }
    }

Each row of the capture history `y` (M x T) is one exchangeable datum. omega
and p have flat priors.
"""

import jax.numpy as jnp

import modelgami as mg
import paragami as pg

from ._capture import eval_bernoulli_log_lik, eval_log_lik_augmented, sample_history

STAN_MODEL = "Mt_model"
POSTERIORS = ["Mt_data-Mt_model"]


def eval_log_lik_one(datum, params, hyper):
    captures = eval_bernoulli_log_lik(y=datum["y"], p=params["p"])
    return eval_log_lik_augmented(omega=params["omega"], log_lik_captures=captures, captured=jnp.sum(datum["y"]) > 0)


def draw_one(key, datum, params, hyper):
    return {"y": sample_history(key=key, omega=params["omega"], p=params["p"])}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            omega=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            p=pg.NumericVectorPattern(int(raw["T"]), lb=0.0, ub=1.0),
        ),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
    )
