"""logmesquite.stan (normal linear regression, flat priors):

    log(weight) ~ normal(beta[1] + beta[2] * log(diam1) + beta[3] * log(diam2) + beta[4] * log(canopy_height)
                       + beta[5] * log(total_height) + beta[6] * log(density) + beta[7] * group, sigma)
"""

import jax.numpy as jnp

from ._regression import normal_regression, ones_and

STAN_MODEL = "logmesquite"
POSTERIORS = ["mesquite-logmesquite"]


def features(d, hyper):
    return ones_and(
        jnp.log(d["diam1"]),
        jnp.log(d["diam2"]),
        jnp.log(d["canopy_height"]),
        jnp.log(d["total_height"]),
        jnp.log(d["density"]),
        d["group"],
    )


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"weight": raw["weight"], "diam1": raw["diam1"], "diam2": raw["diam2"], "canopy_height": raw["canopy_height"], "total_height": raw["total_height"], "density": raw["density"], "group": raw["group"]},
        n_features=7,
        features=features,
        response=lambda d: jnp.log(d["weight"]),
        response_field="weight",
        inverse_response=jnp.exp,
    )
