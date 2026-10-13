"""normal_mixture_k.stan (K-component normal mixture):

    parameters { simplex[K] theta; array[K] real mu; array[K] real<lower=0, upper=10> sigma; }
    model {
      mu ~ normal(0, 10);
      for (n in 1:N) target += log_sum_exp(log(theta) + normal_lpdf(y[n] | mu, sigma));
    }

theta and sigma have flat priors on their supports.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import logsumexp

import modelgami as mg
import paragami as pg

STAN_MODEL = "normal_mixture_k"
POSTERIORS = ["normal_5-normal_mixture_k"]


def eval_log_prior(params, hyper):
    return jnp.sum(stats.norm.logpdf(params["mu"], loc=0.0, scale=10.0))


def eval_log_lik_one(datum, params, hyper):
    return logsumexp(jnp.log(params["theta"]) + stats.norm.logpdf(datum["y"], loc=params["mu"], scale=params["sigma"]))


def draw_one(key, datum, params, hyper):
    component_key, noise_key = jax.random.split(key)
    k = jax.random.categorical(component_key, logits=jnp.log(params["theta"]))
    return {"y": params["mu"][k] + params["sigma"][k] * jax.random.normal(noise_key)}


def build(raw):
    k = int(raw["K"])
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            theta=pg.SimplexArrayPattern(k),
            mu=pg.NumericVectorPattern(k),
            sigma=pg.NumericVectorPattern(k, lb=0.0, ub=10.0),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data={"y": raw["y"]},
    )
