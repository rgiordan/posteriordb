"""Rate_2_model.stan:

    parameters { real<lower=0, upper=1> theta1; real<lower=0, upper=1> theta2; }
    transformed parameters { real delta = theta1 - theta2; }
    model {
      theta1 ~ beta(1, 1); theta2 ~ beta(1, 1);
      k1 ~ binomial(n1, theta1); k2 ~ binomial(n2, theta2);
    }

The data are a few scalar counts, so this is a plain (not exchangeable) Model.
"""

from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "Rate_2_model"
POSTERIORS = ["Rate_2_data-Rate_2_model"]


def eval_log_prior(params, hyper):
    return (stats.beta.logpdf(params["theta1"], a=1.0, b=1.0)
            + stats.beta.logpdf(params["theta2"], a=1.0, b=1.0))


def eval_log_lik(data, params, hyper):
    return (stats.binom.logpmf(data["k1"], n=data["n1"], p=params["theta1"])
            + stats.binom.logpmf(data["k2"], n=data["n2"], p=params["theta2"]))


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            theta1=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            theta2=pg.NumericScalarPattern(lb=0.0, ub=1.0),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"k1": raw["k1"], "n1": raw["n1"], "k2": raw["k2"], "n2": raw["n2"]},
    )
