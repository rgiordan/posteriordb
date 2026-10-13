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


def eval_log_prior(params, hyper):
    return (stats.norm.logpdf(params["sigma_y"], loc=0.0, scale=1.0)
            + stats.norm.logpdf(params["alpha"], loc=0.0, scale=10.0)
            + stats.norm.logpdf(params["beta"], loc=0.0, scale=10.0))


def _compute_mean(params, datum):
    return params["alpha"] + params["beta"] * datum["floor_measure"]


def eval_log_lik_one(datum, params, hyper):
    return stats.norm.logpdf(datum["log_radon"], loc=_compute_mean(params=params, datum=datum), scale=params["sigma_y"])


def draw_one(key, datum, params, hyper):
    return {"log_radon": _compute_mean(params=params, datum=datum) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(), beta=pg.NumericScalarPattern(), sigma_y=pg.NumericScalarPattern(lb=0.0)),
        data={"floor_measure": raw["floor_measure"], "log_radon": raw["log_radon"]},
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
    )
