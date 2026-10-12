"""kidscore_momhsiq.stan (normal linear regression; flat priors unless noted):

    kid_score ~ normal(beta[1] + beta[2] * mom_hs + beta[3] * mom_iq, sigma);  sigma ~ cauchy(0, 2.5)
"""

from ._regression import cauchy_sigma_prior, normal_regression, ones_and

STAN_MODEL = "kidscore_momhsiq"
POSTERIORS = ["kidiq-kidscore_momhsiq"]


def features(d):
    return ones_and(d["mom_hs"], d["mom_iq"])


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"kid_score": raw["kid_score"], "mom_hs": raw["mom_hs"], "mom_iq": raw["mom_iq"]},
        n_features=3,
        features=features,
        response=lambda d: d["kid_score"],
        log_prior=cauchy_sigma_prior(2.5),
        response_field="kid_score",
    )
