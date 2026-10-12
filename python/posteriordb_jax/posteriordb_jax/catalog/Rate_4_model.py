"""Rate_4_model.stan:

    parameters { real<lower=0, upper=1> theta; real<lower=0, upper=1> thetaprior; }
    model { theta ~ beta(1, 1); thetaprior ~ beta(1, 1); k ~ binomial(n, theta); }

The data are a few scalar counts, so this is a plain (not exchangeable) Model.
"""

from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "Rate_4_model"
POSTERIORS = ["Rate_4_data-Rate_4_model"]


def log_prior(params, hyper):
    return (stats.beta.logpdf(params["theta"], a=1.0, b=1.0)
            + stats.beta.logpdf(params["thetaprior"], a=1.0, b=1.0))


def log_lik(params, data, hyper):
    return stats.binom.logpmf(data["k"], n=data["n"], p=params["theta"])


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            theta=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            thetaprior=pg.NumericScalarPattern(lb=0.0, ub=1.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"k": raw["k"], "n": raw["n"]},
    )
