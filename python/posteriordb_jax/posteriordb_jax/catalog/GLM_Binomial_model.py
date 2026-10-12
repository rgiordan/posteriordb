"""GLM_Binomial_model.stan:

    parameters { real alpha; real beta1; real beta2; }
    transformed parameters { logit_p = alpha + beta1 * year + beta2 * year .* year; }
    model {
      alpha ~ normal(0, 100); beta1 ~ normal(0, 100); beta2 ~ normal(0, 100);
      C ~ binomial_logit(N, logit_p);
    }
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._distributions import binomial_logit_logpmf

STAN_MODEL = "GLM_Binomial_model"
POSTERIORS = ["GLM_Binomial_data-GLM_Binomial_model"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["alpha"], loc=0.0, scale=100.0)
            + stats.norm.logpdf(params["beta1"], loc=0.0, scale=100.0)
            + stats.norm.logpdf(params["beta2"], loc=0.0, scale=100.0))


def logit_p(params, datum):
    return params["alpha"] + params["beta1"] * datum["year"] + params["beta2"] * datum["year"] ** 2


def log_lik_one(params, datum, hyper):
    return binomial_logit_logpmf(datum["C"], n=datum["N"], logit_p=logit_p(params=params, datum=datum))


def sample_one(key, params, datum, hyper):
    p = jax.nn.sigmoid(logit_p(params=params, datum=datum))
    return {"C": jax.random.binomial(key, n=datum["N"], p=p).astype(datum["C"].dtype)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(), beta1=pg.NumericScalarPattern(), beta2=pg.NumericScalarPattern()),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data={"C": jnp.asarray(raw["C"], dtype=jnp.int32), "N": jnp.asarray(raw["N"], dtype=jnp.int32), "year": raw["year"]},
    )
