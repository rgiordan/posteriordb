"""iohmm_reg.stan (input-output hidden Markov model with regression
emissions):

    parameters { simplex[K] pi1; array[K] vector[M] w; array[K] vector[M] b; array[K] real<lower=0> sigma; }
    transformed parameters {
      logA[t] = log(softmax(u[t]' * w));                 // for t >= 2
      logoblik[t][j] = normal_lpdf(y[t] | u[t]' * b[j], sigma[j]);
      logalpha[1][j] = log(pi1[j]) + logoblik[1][j];
      logalpha[t, j] = log_sum_exp_i(logalpha[t - 1, i] + logA[t][i] + logoblik[t][j]);
    }
    model { w[j] ~ normal(0, 5); b[j] ~ normal(0, 5); sigma[j] ~ normal(0, 3); target += log_sum_exp(logalpha[T]); }

As written, the transition term logA[t][i] depends only on the previous state
i, not on the current state j, so the recursion factors as
logalpha[t, j] = log_sum_exp(logalpha[t - 1] + logA[t]) + logoblik[t][j];
this reproduces the Stan model as written. A plain Model: the forward
recursion couples the observations.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import logsumexp

import modelgami as mg
import paragami as pg

STAN_MODEL = "iohmm_reg"
POSTERIORS = ["iohmm_reg_simulated-iohmm_reg"]


def log_prior(params, hyper):
    return (jnp.sum(stats.norm.logpdf(params["w"], loc=0.0, scale=5.0))
            + jnp.sum(stats.norm.logpdf(params["b"], loc=0.0, scale=5.0))
            + jnp.sum(stats.norm.logpdf(params["sigma"], loc=0.0, scale=3.0)))


def log_lik(params, data, hyper):
    u = data["u"]
    log_a = jax.nn.log_softmax(u @ params["w"].T, axis=1)   # (T, K); row 1 unused
    log_oblik = stats.norm.logpdf(data["y"][:, None], loc=u @ params["b"].T, scale=params["sigma"][None, :])

    def step(log_alpha_prev, inputs):
        log_a_t, log_oblik_t = inputs
        return logsumexp(log_alpha_prev + log_a_t) + log_oblik_t, None

    log_alpha_last, _ = jax.lax.scan(step, jnp.log(params["pi1"]) + log_oblik[0], (log_a[1:], log_oblik[1:]))
    return logsumexp(log_alpha_last)


def build(raw):
    k, m = int(raw["K"]), int(raw["M"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            pi1=pg.SimplexArrayPattern(k),
            w=pg.NumericArrayPattern((k, m)),
            b=pg.NumericArrayPattern((k, m)),
            sigma=pg.NumericVectorPattern(k, lb=0.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"y": raw["y"], "u": raw["u"]},
    )
