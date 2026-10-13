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


def eval_log_prior(params, hyper):
    return (stats.norm.logpdf(params["alpha"], loc=hyper["pmualpha"], scale=hyper["psalpha"])
            + stats.norm.logpdf(params["beta"], loc=hyper["pmubeta"], scale=hyper["psbeta"]))


def eval_log_lik_one(datum, params, hyper):
    return stats.norm.logpdf(datum["y"], loc=params["alpha"] + params["beta"] * datum["x"], scale=params["sigma"])


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(),
            beta=pg.NumericScalarPattern(),
            sigma=pg.NumericScalarPattern(lb=0.0),
        ),
        data={"x": raw["x"], "y": raw["y"]},
        hyper={k: float(raw[k]) for k in ("pmualpha", "psalpha", "pmubeta", "psbeta")},
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one),
    )
