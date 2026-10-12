"""earn_height.stan (normal linear regression; flat priors unless noted):

    earn ~ normal(beta[1] + beta[2] * height, sigma)
"""

import jax.numpy as jnp

from ._regression import cauchy_sigma_prior, normal_regression, ones_and  # noqa: F401

STAN_MODEL = "earn_height"
POSTERIORS = ['earnings-earn_height']


def features(d):
    return ones_and(d["height"])


def response(d):
    return d["earn"]


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"earn": raw["earn"], "height": raw["height"]},
        n_features=2,
        features=features,
        response=response,
        log_prior=None,
        response_field="earn",
    )
