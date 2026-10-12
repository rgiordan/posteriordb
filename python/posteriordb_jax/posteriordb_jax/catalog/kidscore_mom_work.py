"""kidscore_mom_work.stan (normal linear regression; flat priors unless noted):

    kid_score ~ normal(beta[1] + beta[2] * work2 + beta[3] * work3 + beta[4] * work4, sigma)   (workK = mom_work == K)
"""

import jax.numpy as jnp

from ._regression import normal_regression, ones_and

STAN_MODEL = "kidscore_mom_work"
POSTERIORS = ['kidiq_with_mom_work-kidscore_mom_work']


def features(d, hyper):
    return ones_and(*(jnp.asarray(d["mom_work"] == k, dtype=jnp.float64) for k in (2, 3, 4)))


def response(d):
    return d["kid_score"]


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"kid_score": raw["kid_score"], "mom_work": raw["mom_work"]},
        n_features=4,
        features=features,
        response=response,
        response_field="kid_score",
    )
