"""mesquite.stan (normal linear regression; flat priors unless noted):

    weight ~ normal(beta[1] + beta[2] * diam1 + ... + beta[7] * group, sigma)
"""

from ._regression import build_normal_regression, prepend_ones

STAN_MODEL = "mesquite"
POSTERIORS = ['mesquite-mesquite']


def compute_features(d, hyper):
    return prepend_ones(d["diam1"], d["diam2"], d["canopy_height"], d["total_height"], d["density"], d["group"])


def compute_response(d):
    return d["weight"]


def build(raw):
    return build_normal_regression(
        name=STAN_MODEL,
        data={"weight": raw["weight"], "diam1": raw["diam1"], "diam2": raw["diam2"], "canopy_height": raw["canopy_height"], "total_height": raw["total_height"], "density": raw["density"], "group": raw["group"]},
        n_features=7,
        compute_features=compute_features,
        compute_response=compute_response,
        response_field="weight",
    )
