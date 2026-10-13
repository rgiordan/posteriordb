"""surgical_model.stan (random-effects model for surgical mortality rates):

    parameters { real mu; real<lower=0> sigmasq; array[N] real b; }
    transformed parameters { real<lower=0> sigma = sqrt(sigmasq); p = inv_logit(b); }
    model {
      mu ~ normal(0.0, 1000.0);
      sigmasq ~ inv_gamma(0.001, 0.001);
      b ~ normal(mu, sigma);
      r ~ binomial_logit(n, b);
    }

Not exchangeable in the modelgami sense: each hospital has its own b.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._distributions import eval_binomial_logit_logpmf, eval_inv_gamma_logpdf

STAN_MODEL = "surgical_model"
POSTERIORS = ["surgical_data-surgical_model"]


def eval_log_prior(params, hyper):
    return (stats.norm.logpdf(params["mu"], loc=0.0, scale=1000.0)
            + eval_inv_gamma_logpdf(params["sigmasq"], a=0.001, b=0.001)
            + jnp.sum(stats.norm.logpdf(params["b"], loc=params["mu"], scale=jnp.sqrt(params["sigmasq"]))))


def eval_log_lik(data, params, hyper):
    return jnp.sum(eval_binomial_logit_logpmf(data["r"], n=data["n"], logit_p=params["b"]))


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            mu=pg.NumericScalarPattern(),
            sigmasq=pg.NumericScalarPattern(lb=0.0),
            b=pg.NumericVectorPattern(int(raw["N"])),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"r": jnp.asarray(raw["r"], dtype=jnp.int32), "n": jnp.asarray(raw["n"], dtype=jnp.int32)},
    )
