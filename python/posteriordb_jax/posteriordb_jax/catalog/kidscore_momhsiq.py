"""kidscore_momhsiq.stan (normal linear regression; flat priors unless noted):

    kid_score ~ normal(beta[1] + beta[2] * mom_hs + beta[3] * mom_iq, sigma);  sigma ~ cauchy(0, 2.5)
"""

from ._regression import make_cauchy_sigma_prior, build_normal_regression, prepend_ones

STAN_MODEL = "kidscore_momhsiq"
POSTERIORS = ["kidiq-kidscore_momhsiq"]


def compute_features(d, hyper):
    return prepend_ones(d["mom_hs"], d["mom_iq"])


def build(raw):
    return build_normal_regression(
        name=STAN_MODEL,
        data={"kid_score": raw["kid_score"], "mom_hs": raw["mom_hs"], "mom_iq": raw["mom_iq"]},
        n_features=3,
        compute_features=compute_features,
        compute_response=lambda d: d["kid_score"],
        eval_log_prior=make_cauchy_sigma_prior(2.5),
        response_field="kid_score",
    )
