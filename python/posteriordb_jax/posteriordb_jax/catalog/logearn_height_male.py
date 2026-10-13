"""logearn_height_male.stan (normal linear regression; flat priors unless noted):

    log(earn) ~ normal(beta[1] + beta[2] * height + beta[3] * male, sigma)
"""

import jax.numpy as jnp

from ._regression import build_normal_regression, prepend_ones

STAN_MODEL = "logearn_height_male"
POSTERIORS = ['earnings-logearn_height_male']


def compute_features(d, hyper):
    return prepend_ones(d["height"], d["male"])


def compute_response(d):
    return jnp.log(d["earn"])


def build(raw):
    return build_normal_regression(
        name=STAN_MODEL,
        data={"earn": raw["earn"], "height": raw["height"], "male": raw["male"]},
        n_features=3,
        compute_features=compute_features,
        compute_response=compute_response,
        response_field="earn",
        invert_response=jnp.exp,
    )
