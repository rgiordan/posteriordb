"""wells_interaction_model.stan (logistic regression, flat priors):

    transformed data {
      vector[N] dist100 = dist / 100.0;
      vector[N] inter = dist100 .* arsenic;
      matrix[N, 3] x = [dist100', arsenic', inter']';
    }
    parameters { real alpha; vector[3] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }
"""

import jax.numpy as jnp

from ._regression import build_logistic_regression

STAN_MODEL = "wells_interaction_model"
POSTERIORS = ["wells_data-wells_interaction_model"]


def compute_features(d, hyper):
    dist100 = d["dist"] / 100.0
    return jnp.stack([dist100, d["arsenic"], dist100 * d["arsenic"]])


def build(raw):
    return build_logistic_regression(
        name=STAN_MODEL,
        data={"switched": raw["switched"], "dist": raw["dist"], "arsenic": raw["arsenic"]},
        n_features=3,
        compute_features=compute_features,
        response_field="switched",
    )
