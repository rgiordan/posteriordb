"""diamonds.stan (brms-generated normal regression with centered predictors):

    transformed data { Xc = X[:, 2:K] with each column centered at its mean; }
    parameters { vector[K - 1] b; real Intercept; real<lower=0> sigma; }
    model {
      target += normal_lpdf(b | 0, 1);
      target += student_t_lpdf(Intercept | 3, 8, 10);
      target += student_t_lpdf(sigma | 3, 0, 10) - student_t_lccdf(0 | 3, 0, 10);
      target += normal_id_glm_lpdf(Y | Xc, Intercept, b, sigma);
    }

The column means use the whole data set (Stan's transformed data), so they are
computed once here and passed as hyperparameters. student_t_lccdf(0 | 3, 0, 10)
is log(1/2): sigma's prior is a half Student t.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "diamonds"
POSTERIORS = ["diamonds-diamonds"]


def log_prior(params, hyper):
    return (jnp.sum(stats.norm.logpdf(params["b"], loc=0.0, scale=1.0))
            + stats.t.logpdf(params["Intercept"], df=3.0, loc=8.0, scale=10.0)
            + stats.t.logpdf(params["sigma"], df=3.0, loc=0.0, scale=10.0) + jnp.log(2.0))


def mean(params, datum, hyper):
    return params["Intercept"] + (datum["X"][1:] - hyper["means_X"]) @ params["b"]


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["Y"], loc=mean(params=params, datum=datum, hyper=hyper), scale=params["sigma"])


def sample_one(key, params, datum, hyper):
    return {"Y": mean(params=params, datum=datum, hyper=hyper) + params["sigma"] * jax.random.normal(key)}


def build(raw):
    x = np.asarray(raw["X"], dtype=float)
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            b=pg.NumericVectorPattern(int(raw["K"]) - 1),
            Intercept=pg.NumericScalarPattern(),
            sigma=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data={"Y": raw["Y"], "X": raw["X"]},
        hyper={"means_X": jnp.asarray(x[:, 1:].mean(axis=0))},
    )
