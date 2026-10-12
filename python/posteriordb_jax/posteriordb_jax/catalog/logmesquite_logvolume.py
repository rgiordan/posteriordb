"""logmesquite_logvolume.stan (normal linear regression; flat priors unless noted):

    log(weight) ~ normal(beta[1] + beta[2] * log(diam1 .* diam2 .* canopy_height), sigma)
"""

import jax.numpy as jnp

from ._regression import normal_regression, ones_and

STAN_MODEL = "logmesquite_logvolume"
POSTERIORS = ['mesquite-logmesquite_logvolume']


def features(d, hyper):
    return ones_and(jnp.log(d["diam1"] * d["diam2"] * d["canopy_height"]))


def response(d):
    return jnp.log(d["weight"])


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"weight": raw["weight"], "diam1": raw["diam1"], "diam2": raw["diam2"], "canopy_height": raw["canopy_height"]},
        n_features=2,
        features=features,
        response=response,
        response_field="weight",
        inverse_response=jnp.exp,
    )
