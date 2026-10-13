"""wells_dae_model.stan (logistic regression, flat priors):

    transformed data {
      vector[N] dist100 = dist / 100.0;
      vector[N] educ4 = educ / 4.0;
      matrix[N, 3] x = [dist100', arsenic', educ4']';
    }
    parameters { real alpha; vector[3] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }
"""

import jax.numpy as jnp

from ._regression import build_logistic_regression

STAN_MODEL = "wells_dae_model"
POSTERIORS = ["wells_data-wells_dae_model"]


def compute_features(d, hyper):
    return jnp.stack([d["dist"] / 100.0, d["arsenic"], d["educ"] / 4.0])


def build(raw):
    return build_logistic_regression(
        name=STAN_MODEL,
        data={"switched": raw["switched"], "dist": raw["dist"], "arsenic": raw["arsenic"], "educ": raw["educ"]},
        n_features=3,
        compute_features=compute_features,
        response_field="switched",
    )
