"""kilpisjarvi.stan (linear trend in summer temperatures):

    parameters { real alpha; real beta; real<lower=0> sigma; }
    model {
      alpha ~ normal(pmualpha, psalpha);
      beta ~ normal(pmubeta, psbeta);
      y ~ normal(alpha + beta * x, sigma);
    }

The prior means and scales arrive as Stan data; here they are
hyperparameters, so prior sensitivity is a gradient with respect to `hyper`.
"""

from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "kilpisjarvi"
POSTERIORS = ["kilpisjarvi_mod-kilpisjarvi"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["alpha"], hyper["pmualpha"], hyper["psalpha"])
            + stats.norm.logpdf(params["beta"], hyper["pmubeta"], hyper["psbeta"]))


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["y"], params["alpha"] + params["beta"] * datum["x"], params["sigma"])


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(),
            beta=pg.NumericScalarPattern(),
            sigma=pg.NumericScalarPattern(lb=0.0),
        ),
        data={"x": raw["x"], "y": raw["y"]},
        hyper={k: float(raw[k]) for k in ("pmualpha", "psalpha", "pmubeta", "psbeta")},
        log_prior=log_prior,
        log_lik_one=log_lik_one,
    )
