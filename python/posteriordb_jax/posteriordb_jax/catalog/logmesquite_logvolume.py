"""logmesquite_logvolume.stan (normal linear regression; flat priors unless noted):

    log(weight) ~ normal(beta[1] + beta[2] * log(diam1 .* diam2 .* canopy_height), sigma)
"""

import jax.numpy as jnp

from ._regression import build_normal_regression, prepend_ones

STAN_MODEL = "logmesquite_logvolume"
POSTERIORS = ['mesquite-logmesquite_logvolume']


def compute_features(d, hyper):
    return prepend_ones(jnp.log(d["diam1"] * d["diam2"] * d["canopy_height"]))


def compute_response(d):
    return jnp.log(d["weight"])


def build(raw):
    return build_normal_regression(
        name=STAN_MODEL,
        data={"weight": raw["weight"], "diam1": raw["diam1"], "diam2": raw["diam2"], "canopy_height": raw["canopy_height"]},
        n_features=2,
        compute_features=compute_features,
        compute_response=compute_response,
        response_field="weight",
        invert_response=jnp.exp,
    )
