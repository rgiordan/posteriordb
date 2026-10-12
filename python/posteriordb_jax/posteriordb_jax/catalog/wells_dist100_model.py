"""wells_dist100_model.stan (logistic regression, flat priors):

    transformed data { vector[N] dist100 = dist / 100.0; }
    parameters { real alpha; vector[1] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }   // x = [dist100]
"""

import jax.numpy as jnp

from ._regression import logistic_regression

STAN_MODEL = "wells_dist100_model"
POSTERIORS = ["wells_data-wells_dist100_model"]


def features(d):
    return jnp.stack([d["dist"] / 100.0])


def build(raw):
    return logistic_regression(
        name=STAN_MODEL,
        data={"switched": raw["switched"], "dist": raw["dist"]},
        n_features=1,
        features=features,
        response_field="switched",
    )
