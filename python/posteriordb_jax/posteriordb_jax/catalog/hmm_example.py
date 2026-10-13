"""hmm_example.stan (two-state Gaussian hidden Markov model, forward algorithm):

    parameters { simplex[K] theta1; simplex[K] theta2; positive_ordered[K] mu; }
    transformed parameters { theta[1] = theta1; theta[2] = theta2; }   // transition rows
    model {
      target += normal_lpdf(mu[1] | 3, 1);
      target += normal_lpdf(mu[2] | 10, 1);
      gamma[1, k] = normal_lpdf(y[1] | mu[k], 1);
      gamma[t, k] = log_sum_exp_j(gamma[t - 1, j] + log(theta[j, k]) + normal_lpdf(y[t] | mu[k], 1));
      target += log_sum_exp(gamma[N]);
    }

The data have K = 2 (only two transition rows are defined). A plain Model:
the forward recursion couples the observations (a `lax.scan`).
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._hmm import eval_forward_log_marginal

STAN_MODEL = "hmm_example"
POSTERIORS = ["hmm_example-hmm_example"]


def eval_log_prior(params, hyper):
    mu = params["mu"]
    return stats.norm.logpdf(mu[0], loc=3.0, scale=1.0) + stats.norm.logpdf(mu[1], loc=10.0, scale=1.0)


def eval_log_lik(data, params, hyper):
    mu = params["mu"]
    log_transition = jnp.log(jnp.stack([params["theta1"], params["theta2"]]))   # [from, to]
    log_emission = stats.norm.logpdf(data["y"][:, None], loc=mu[None, :], scale=1.0)   # (N, K)
    return eval_forward_log_marginal(log_transition=log_transition, log_emission=log_emission)


def build(raw):
    k = int(raw["K"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            theta1=pg.SimplexArrayPattern(k),
            theta2=pg.SimplexArrayPattern(k),
            mu=pg.OrderedVectorPattern(k, positive=True),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"y": raw["y"]},
    )
