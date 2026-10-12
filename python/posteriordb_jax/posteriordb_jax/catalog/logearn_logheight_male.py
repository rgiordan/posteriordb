"""logearn_logheight_male.stan (normal linear regression, flat priors):

    log(earn) ~ normal(beta[1] + beta[2] * log(height) + beta[3] * male, sigma)
"""

import jax.numpy as jnp

from ._regression import normal_regression, ones_and

STAN_MODEL = "logearn_logheight_male"
POSTERIORS = ["earnings-logearn_logheight_male"]


def features(d):
    return ones_and(jnp.log(d["height"]), d["male"])


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"earn": raw["earn"], "height": raw["height"], "male": raw["male"]},
        n_features=3,
        features=features,
        response=lambda d: jnp.log(d["earn"]),
        response_field="earn",
        inverse_response=jnp.exp,
    )
