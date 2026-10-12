"""logearn_height_male.stan (normal linear regression; flat priors unless noted):

    log(earn) ~ normal(beta[1] + beta[2] * height + beta[3] * male, sigma)
"""

import jax.numpy as jnp

from ._regression import normal_regression, ones_and

STAN_MODEL = "logearn_height_male"
POSTERIORS = ['earnings-logearn_height_male']


def features(d):
    return ones_and(d["height"], d["male"])


def response(d):
    return jnp.log(d["earn"])


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"earn": raw["earn"], "height": raw["height"], "male": raw["male"]},
        n_features=3,
        features=features,
        response=response,
        log_prior=None,
        response_field="earn",
        inverse_response=jnp.exp,
    )
