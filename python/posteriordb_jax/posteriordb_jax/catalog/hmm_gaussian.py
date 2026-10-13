"""hmm_gaussian.stan (Gaussian hidden Markov model with an initial
distribution and K transition rows):

    parameters { simplex[K] pi1; array[K] simplex[K] A; ordered[K] mu; array[K] real<lower=0> sigma; }
    transformed parameters {
      logalpha[1] = log(pi1) + normal_lpdf(y[1] | mu, sigma);
      logalpha[t, j] = log_sum_exp_i(logalpha[t - 1, i] + log(A[i, j]) + normal_lpdf(y[t] | mu[j], sigma[j]));
    }
    model { target += log_sum_exp(logalpha[T]); }

Note that normal_lpdf(y[1] | mu, sigma) with vector arguments is a single
real (the sum over states), added to every entry of log(pi1). There is no
prior beyond the flat ones on the constrained parameters. A plain Model: the
forward recursion couples the observations.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._hmm import forward_log_marginal

STAN_MODEL = "hmm_gaussian"
POSTERIORS = ["hmm_gaussian_simulated-hmm_gaussian"]


def log_lik(params, data, hyper):
    log_emission = stats.norm.logpdf(data["y"][:, None], loc=params["mu"][None, :],
                                     scale=params["sigma"][None, :])   # (T, K)
    log_initial = jnp.log(params["pi1"]) + jnp.sum(log_emission[0]) - log_emission[0]
    return forward_log_marginal(log_transition=jnp.log(params["A"]), log_emission=log_emission,
                                log_initial=log_initial)


def build(raw):
    k = int(raw["K"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            pi1=pg.SimplexArrayPattern(k),
            A=pg.SimplexArrayPattern(k, array_shape=(k,)),
            mu=pg.OrderedVectorPattern(k),
            sigma=pg.NumericVectorPattern(k, lb=0.0),
        ),
        prior=None,
        likelihood=mg.DGP(log_density=log_lik),
        data={"y": raw["y"]},
    )
