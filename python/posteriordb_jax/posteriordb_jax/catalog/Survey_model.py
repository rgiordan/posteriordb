"""Survey_model.stan (unknown number of surveys n, uniform on 1..nmax, with the
same return rate theta; n is summed out):

    parameters { real<lower=0, upper=1> theta; }
    transformed parameters {
      lp_parts[n] = log(1.0 / nmax) + (n < max(k) ? negative_infinity() : binomial_lpmf(k | n, theta));
    }
    model { target += log_sum_exp(lp_parts); }

theta has a flat prior. A plain Model: the marginal likelihood couples all
of k through the shared n. The candidate values 1..nmax are a hyperparameter.
"""

import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import logsumexp

import modelgami as mg
import paragami as pg

STAN_MODEL = "Survey_model"
POSTERIORS = ["Survey_data-Survey_model"]


def eval_log_lik(data, params, hyper):
    n_values, k = hyper["n_values"], data["k"]
    log_lik_given_n = jnp.sum(stats.binom.logpmf(k[None, :], n=n_values[:, None], p=params["theta"]), axis=1)
    possible = n_values >= jnp.max(k)
    lp_parts = -jnp.log(n_values.shape[0]) + jnp.where(possible, log_lik_given_n, -jnp.inf)
    return logsumexp(lp_parts)


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(theta=pg.NumericScalarPattern(lb=0.0, ub=1.0)),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"k": jnp.asarray(raw["k"], dtype=jnp.int32)},
        hyper={"n_values": jnp.arange(1, int(raw["nmax"]) + 1)},
    )
