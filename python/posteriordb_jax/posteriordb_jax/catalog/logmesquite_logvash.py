"""logmesquite_logvash.stan (normal linear regression, flat priors):

    log(weight) ~ normal(beta[1] + beta[2] * log(diam1 .* diam2 .* canopy_height) + beta[3] * log(diam1 .* diam2)
                       + beta[4] * log(diam1 ./ diam2) + beta[5] * log(total_height) + beta[6] * group, sigma)
"""

import jax.numpy as jnp

from ._regression import build_normal_regression, prepend_ones

STAN_MODEL = "logmesquite_logvash"
POSTERIORS = ["mesquite-logmesquite_logvash"]


def compute_features(d, hyper):
    return prepend_ones(
        jnp.log(d["diam1"] * d["diam2"] * d["canopy_height"]),
        jnp.log(d["diam1"] * d["diam2"]),
        jnp.log(d["diam1"] / d["diam2"]),
        jnp.log(d["total_height"]),
        d["group"],
    )


def build(raw):
    return build_normal_regression(
        name=STAN_MODEL,
        data={"weight": raw["weight"], "diam1": raw["diam1"], "diam2": raw["diam2"], "canopy_height": raw["canopy_height"], "total_height": raw["total_height"], "group": raw["group"]},
        n_features=6,
        compute_features=compute_features,
        compute_response=lambda d: jnp.log(d["weight"]),
        response_field="weight",
        invert_response=jnp.exp,
    )
