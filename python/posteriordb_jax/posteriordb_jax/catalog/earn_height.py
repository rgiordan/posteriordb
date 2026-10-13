"""earn_height.stan (normal linear regression; flat priors unless noted):

    earn ~ normal(beta[1] + beta[2] * height, sigma)
"""

from ._regression import build_normal_regression, prepend_ones

STAN_MODEL = "earn_height"
POSTERIORS = ['earnings-earn_height']


def compute_features(d, hyper):
    return prepend_ones(d["height"])


def compute_response(d):
    return d["earn"]


def build(raw):
    return build_normal_regression(
        name=STAN_MODEL,
        data={"earn": raw["earn"], "height": raw["height"]},
        n_features=2,
        compute_features=compute_features,
        compute_response=compute_response,
        response_field="earn",
    )
