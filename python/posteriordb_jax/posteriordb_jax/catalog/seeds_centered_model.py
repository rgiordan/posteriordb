"""seeds_centered_model.stan (random effects centered to sum to zero):

    parameters { real alpha0; real alpha1; real alpha12; real alpha2; vector[I] c; real<lower=0> sigma; }
    transformed parameters { vector[I] b = c - mean(c); }
    model {
      alpha0 ~ normal(0.0, 1.0); alpha1 ~ normal(0.0, 1.0);
      alpha2 ~ normal(0.0, 1.0); alpha12 ~ normal(0.0, 1.0);
      sigma ~ cauchy(0, 1);
      c ~ normal(0.0, sigma);
      n ~ binomial_logit(N, alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b);
    }

Not exchangeable in the modelgami sense: each plate has its own c.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from . import _seeds

STAN_MODEL = "seeds_centered_model"
POSTERIORS = ["seeds_data-seeds_centered_model"]


def eval_log_prior(params, hyper):
    return (sum(stats.norm.logpdf(params[k], loc=0.0, scale=1.0) for k in ("alpha0", "alpha1", "alpha2", "alpha12"))
            + stats.cauchy.logpdf(params["sigma"], loc=0.0, scale=1.0)
            + jnp.sum(stats.norm.logpdf(params["c"], loc=0.0, scale=params["sigma"])))


def eval_log_lik(data, params, hyper):
    return _seeds.eval_log_lik(params=params, b=params["c"] - jnp.mean(params["c"]), data=data)


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha0=pg.NumericScalarPattern(),
            alpha1=pg.NumericScalarPattern(),
            alpha12=pg.NumericScalarPattern(),
            alpha2=pg.NumericScalarPattern(),
            c=pg.NumericVectorPattern(int(raw["I"])),
            sigma=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data=_seeds.extract_data(raw),
    )
