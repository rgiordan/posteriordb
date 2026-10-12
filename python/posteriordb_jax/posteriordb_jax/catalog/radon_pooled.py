"""radon_pooled.stan:

    parameters { real alpha; real beta; real<lower=0> sigma_y; }
    model {
      sigma_y ~ normal(0, 1);
      alpha ~ normal(0, 10);
      beta ~ normal(0, 10);
      log_radon[n] ~ normal(alpha + beta * floor_measure[n], sigma_y);
    }
"""

import jax
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "radon_pooled"
POSTERIORS = ["radon_all-radon_pooled", "radon_mn-radon_pooled"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["sigma_y"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["alpha"], loc=0.0, scale=10.0)
            + stats.norm.logpdf(params["beta"], loc=0.0, scale=10.0))


def _mean(params, datum):
    return params["alpha"] + params["beta"] * datum["floor_measure"]


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["log_radon"], loc=_mean(params=params, datum=datum), scale=params["sigma_y"])


def sample_one(key, params, datum, hyper):
    return {"log_radon": _mean(params=params, datum=datum) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(), beta=pg.NumericScalarPattern(), sigma_y=pg.NumericScalarPattern(lb=0.0)),
        data={"floor_measure": raw["floor_measure"], "log_radon": raw["log_radon"]},
        log_prior=log_prior,
        log_lik_one=log_lik_one,
        sample_one=sample_one,
    )
