"""logmesquite_logva.stan (normal linear regression, flat priors):

    log(weight) ~ normal(beta[1] + beta[2] * log(diam1 .* diam2 .* canopy_height)
                       + beta[3] * log(diam1 .* diam2) + beta[4] * group, sigma)
"""

import jax.numpy as jnp

from ._regression import normal_regression, ones_and

STAN_MODEL = "logmesquite_logva"
POSTERIORS = ["mesquite-logmesquite_logva"]


def features(d):
    return ones_and(
        jnp.log(d["diam1"] * d["diam2"] * d["canopy_height"]),
        jnp.log(d["diam1"] * d["diam2"]),
        d["group"],
    )


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"weight": raw["weight"], "diam1": raw["diam1"], "diam2": raw["diam2"], "canopy_height": raw["canopy_height"], "group": raw["group"]},
        n_features=4,
        features=features,
        response=lambda d: jnp.log(d["weight"]),
        response_field="weight",
        inverse_response=jnp.exp,
    )
