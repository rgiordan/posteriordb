"""lsat_model.stan (Rasch-type model for 1000 students on 5 LSAT questions;
the data are run-length encoded: culm[i] students gave response pattern i):

    parameters { array[T] real alpha; vector[N] theta; real<lower=0> beta; }
    model {
      alpha ~ normal(0, 100.);
      theta ~ normal(0, 1);
      beta ~ normal(0.0, 100.);
      for (k in 1:T) r[k] ~ bernoulli_logit(beta * theta - alpha[k]);
    }

Not exchangeable in the modelgami sense: each student has its own theta. The
response matrix r (students x questions) is expanded from the encoding here.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "lsat_model"
POSTERIORS = ["lsat_data-lsat_model"]


def log_prior(params, hyper):
    return (jnp.sum(stats.norm.logpdf(params["alpha"], loc=0.0, scale=100.0))
            + jnp.sum(stats.norm.logpdf(params["theta"], loc=0.0, scale=1.0))
            + stats.norm.logpdf(params["beta"], loc=0.0, scale=100.0))


def log_lik(params, data, hyper):
    logit_p = params["beta"] * params["theta"][:, None] - params["alpha"][None, :]
    return jnp.sum(data["r"] * logit_p - jax.nn.softplus(logit_p))


def build(raw):
    culm = np.asarray(raw["culm"], dtype=int)
    counts = np.diff(np.concatenate([[0], culm]))
    r = np.repeat(np.asarray(raw["response"], dtype=int), repeats=counts, axis=0)
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericVectorPattern(int(raw["T"])),
            theta=pg.NumericVectorPattern(int(raw["N"])),
            beta=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"r": jnp.asarray(r, dtype=jnp.int32)},
    )
