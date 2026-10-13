"""Mth_model.stan (capture-recapture with occasion effects and individual
heterogeneity):

    parameters {
      real<lower=0, upper=1> omega; array[T] real<lower=0, upper=1> mean_p;
      real<lower=0, upper=5> sigma; vector[M] eps_raw;
    }
    transformed parameters { logit_p[i, j] = logit(mean_p[j]) + sigma * eps_raw[i]; }
    model {
      eps_raw ~ normal(0, 1);
      for (i in 1:M) {
        if (s[i] > 0) target += bernoulli_lpmf(1 | omega) + bernoulli_logit_lpmf(y[i] | logit_p[i]);
        else target += log_sum_exp(bernoulli_lpmf(1 | omega) + bernoulli_logit_lpmf(y[i] | logit_p[i]), bernoulli_lpmf(0 | omega));
      }
    }

Not exchangeable in the modelgami sense: each individual has its own eps_raw.
"""

import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import logit

import modelgami as mg
import paragami as pg

from ._capture import eval_bernoulli_logit_log_lik, eval_log_lik_augmented

STAN_MODEL = "Mth_model"
POSTERIORS = ["Mth_data-Mth_model"]


def eval_log_prior(params, hyper):
    return jnp.sum(stats.norm.logpdf(params["eps_raw"], loc=0.0, scale=1.0))


def eval_log_lik(data, params, hyper):
    y = data["y"]
    logit_p = logit(params["mean_p"])[None, :] + params["sigma"] * params["eps_raw"][:, None]
    captures = eval_bernoulli_logit_log_lik(y=y, logit_p=logit_p)
    return jnp.sum(eval_log_lik_augmented(omega=params["omega"], log_lik_captures=captures, captured=jnp.sum(y, axis=1) > 0))


def build(raw):
    m, t = int(raw["M"]), int(raw["T"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            omega=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            mean_p=pg.NumericVectorPattern(t, lb=0.0, ub=1.0),
            sigma=pg.NumericScalarPattern(lb=0.0, ub=5.0),
            eps_raw=pg.NumericVectorPattern(m),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
    )
