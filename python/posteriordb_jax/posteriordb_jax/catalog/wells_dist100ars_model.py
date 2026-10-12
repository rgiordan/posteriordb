"""wells_dist100ars_model.stan (logistic regression, flat priors):

    transformed data {
      vector[N] dist100 = dist / 100.0;
      matrix[N, 2] x = [dist100', arsenic']';
    }
    parameters { real alpha; vector[2] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }
"""

import jax.numpy as jnp

from ._regression import logistic_regression

STAN_MODEL = "wells_dist100ars_model"
POSTERIORS = ["wells_data-wells_dist100ars_model"]


def features(d):
    return jnp.stack([d["dist"] / 100.0, d["arsenic"]])


def build(raw):
    return logistic_regression(
        name=STAN_MODEL,
        data={"switched": raw["switched"], "dist": raw["dist"], "arsenic": raw["arsenic"]},
        n_features=2,
        features=features,
        response_field="switched",
    )
