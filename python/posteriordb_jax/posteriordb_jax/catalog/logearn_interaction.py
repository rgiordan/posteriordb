"""logearn_interaction.stan (normal linear regression, flat priors):

    log(earn) ~ normal(beta[1] + beta[2] * height + beta[3] * male + beta[4] * height .* male, sigma)
"""

import jax.numpy as jnp

from ._regression import normal_regression, ones_and

STAN_MODEL = "logearn_interaction"
POSTERIORS = ["earnings-logearn_interaction"]


def features(d):
    return ones_and(d["height"], d["male"], d["height"] * d["male"])


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"earn": raw["earn"], "height": raw["height"], "male": raw["male"]},
        n_features=4,
        features=features,
        response=lambda d: jnp.log(d["earn"]),
        response_field="earn",
        inverse_response=jnp.exp,
    )
