"""Mh_model.stan (capture-recapture with individual detection heterogeneity;
y[i] is the number of occasions individual i was captured, out of T):

    parameters {
      real<lower=0, upper=1> omega; real<lower=0, upper=1> mean_p;
      real<lower=0, upper=5> sigma; vector[M] eps_raw;
    }
    transformed parameters { vector[M] eps = logit(mean_p) + sigma * eps_raw; }
    model {
      eps_raw ~ normal(0, 1);
      for (i in 1:M) {
        if (y[i] > 0) target += bernoulli_lpmf(1 | omega) + binomial_logit_lpmf(y[i] | T, eps[i]);
        else target += log_sum_exp(bernoulli_lpmf(1 | omega) + binomial_logit_lpmf(0 | T, eps[i]), bernoulli_lpmf(0 | omega));
      }
    }

Not exchangeable in the modelgami sense: each individual has its own eps_raw.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import gammaln, logit

import modelgami as mg
import paragami as pg

from ._capture import log_lik_augmented

STAN_MODEL = "Mh_model"
POSTERIORS = ["Mh_data-Mh_model"]


def log_prior(params, hyper):
    return jnp.sum(stats.norm.logpdf(params["eps_raw"], loc=0.0, scale=1.0))


def log_lik(params, data, hyper):
    y, t = data["y"], hyper["T"]
    eps = logit(params["mean_p"]) + params["sigma"] * params["eps_raw"]
    log_choose = gammaln(t + 1.0) - gammaln(y + 1.0) - gammaln(t - y + 1.0)
    captures = log_choose + y * eps - t * jax.nn.softplus(eps)
    return jnp.sum(log_lik_augmented(omega=params["omega"], log_lik_captures=captures, captured=y > 0))


def build(raw):
    m = int(raw["M"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            omega=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            mean_p=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            sigma=pg.NumericScalarPattern(lb=0.0, ub=5.0),
            eps_raw=pg.NumericVectorPattern(m),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
        hyper={"T": int(raw["T"])},
    )
