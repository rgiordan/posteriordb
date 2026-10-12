"""Mtbh_model.stan (capture-recapture with occasion effects, a behavioral
response and individual heterogeneity):

    parameters {
      real<lower=0, upper=1> omega; vector<lower=0, upper=1>[T] mean_p; real gamma;
      real<lower=0, upper=3> sigma; vector[M] eps_raw;
    }
    transformed parameters {
      logit_p[i, 1] = logit(mean_p[1]) + sigma * eps_raw[i];
      logit_p[i, j] = logit(mean_p[j]) + sigma * eps_raw[i] + gamma * y[i, j - 1];   // j >= 2
    }
    model {
      gamma ~ normal(0, 10);
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

from ._capture import bernoulli_logit_log_lik, log_lik_augmented

STAN_MODEL = "Mtbh_model"
POSTERIORS = ["Mtbh_data-Mtbh_model"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["gamma"], loc=0.0, scale=10.0)
            + jnp.sum(stats.norm.logpdf(params["eps_raw"], loc=0.0, scale=1.0)))


def log_lik(params, data, hyper):
    y = data["y"]
    previous = jnp.concatenate([jnp.zeros((y.shape[0], 1), dtype=y.dtype), y[:, :-1]], axis=1)
    logit_p = (logit(params["mean_p"])[None, :] + params["sigma"] * params["eps_raw"][:, None]
               + params["gamma"] * previous)
    captures = bernoulli_logit_log_lik(y=y, logit_p=logit_p)
    return jnp.sum(log_lik_augmented(omega=params["omega"], log_lik_captures=captures, captured=jnp.sum(y, axis=1) > 0))


def build(raw):
    m, t = int(raw["M"]), int(raw["T"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            omega=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            mean_p=pg.NumericVectorPattern(t, lb=0.0, ub=1.0),
            gamma=pg.NumericScalarPattern(),
            sigma=pg.NumericScalarPattern(lb=0.0, ub=3.0),
            eps_raw=pg.NumericVectorPattern(m),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
    )
