"""nes_logit_model.stan (logistic regression, flat priors):

    transformed data { matrix[N, 1] x = [income']'; }
    parameters { real alpha; vector[1] beta; }
    model { vote ~ bernoulli_logit_glm(x, alpha, beta); }
"""

import jax.numpy as jnp

from ._regression import build_logistic_regression

STAN_MODEL = "nes_logit_model"
POSTERIORS = ["nes_logit_data-nes_logit_model"]


def compute_features(d, hyper):
    return jnp.stack([d["income"]])


def build(raw):
    return build_logistic_regression(
        name=STAN_MODEL,
        data={"vote": raw["vote"], "income": raw["income"]},
        n_features=1,
        compute_features=compute_features,
        response_field="vote",
    )
