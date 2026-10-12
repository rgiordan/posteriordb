"""kidscore_momhs.stan (normal linear regression; flat priors unless noted):

    kid_score ~ normal(beta[1] + beta[2] * mom_hs, sigma);  sigma ~ cauchy(0, 2.5)
"""

import jax.numpy as jnp

from ._regression import cauchy_sigma_prior, normal_regression, ones_and  # noqa: F401

STAN_MODEL = "kidscore_momhs"
POSTERIORS = ['kidiq-kidscore_momhs']


def features(d):
    return ones_and(d["mom_hs"])


def response(d):
    return d["kid_score"]


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"kid_score": raw["kid_score"], "mom_hs": raw["mom_hs"]},
        n_features=2,
        features=features,
        response=response,
        log_prior=cauchy_sigma_prior(2.5),
        response_field="kid_score",
    )
