"""logearn_interaction.stan (normal linear regression, flat priors):

    log(earn) ~ normal(beta[1] + beta[2] * height + beta[3] * male + beta[4] * height .* male, sigma)
"""

import jax.numpy as jnp

from ._regression import build_normal_regression, prepend_ones

STAN_MODEL = "logearn_interaction"
POSTERIORS = ["earnings-logearn_interaction"]


def compute_features(d, hyper):
    return prepend_ones(d["height"], d["male"], d["height"] * d["male"])


def build(raw):
    return build_normal_regression(
        name=STAN_MODEL,
        data={"earn": raw["earn"], "height": raw["height"], "male": raw["male"]},
        n_features=4,
        compute_features=compute_features,
        compute_response=lambda d: jnp.log(d["earn"]),
        response_field="earn",
        invert_response=jnp.exp,
    )
