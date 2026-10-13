"""GLMM1_model.stan (Poisson counts with random site effects):

    parameters { vector[nsite] alpha; real mu_alpha; real<lower=0, upper=5> sd_alpha; }
    transformed parameters { log_lambda[year, site] = alpha[site]; }
    model {
      alpha ~ normal(mu_alpha, sd_alpha);
      mu_alpha ~ normal(0, 10);
      obs[i] ~ poisson_log(log_lambda[obsyear[i], obssite[i]]);
    }

log_lambda does not depend on the year. sd_alpha has a flat prior.
Exchangeable: each observed count is one datum carrying its site index.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "GLMM1_model"
POSTERIORS = ["GLMM_data-GLMM1_model"]


def log_prior(params, hyper):
    return (jnp.sum(stats.norm.logpdf(params["alpha"], loc=params["mu_alpha"], scale=params["sd_alpha"]))
            + stats.norm.logpdf(params["mu_alpha"], loc=0.0, scale=10.0))


def log_lik_one(params, datum, hyper):
    return stats.poisson.logpmf(datum["obs"], mu=jnp.exp(params["alpha"][datum["obssite"] - 1]))


def sample_one(key, params, datum, hyper):
    draw = jax.random.poisson(key, lam=jnp.exp(params["alpha"][datum["obssite"] - 1]))
    return {"obs": draw.astype(datum["obs"].dtype)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericVectorPattern(int(raw["nsite"])),
            mu_alpha=pg.NumericScalarPattern(),
            sd_alpha=pg.NumericScalarPattern(lb=0.0, ub=5.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data={"obs": jnp.asarray(raw["obs"], dtype=jnp.int32), "obssite": jnp.asarray(raw["obssite"], dtype=jnp.int32)},
    )
