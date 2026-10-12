"""kidscore_interaction_c2.stan (normal linear regression, flat priors):

    transformed data {
      c2_mom_hs = mom_hs - 0.5;
      c2_mom_iq = mom_iq - 100;
      inter = c2_mom_hs .* c2_mom_iq;
    }
    model { kid_score ~ normal(beta[1] + beta[2] * c2_mom_hs + beta[3] * c2_mom_iq + beta[4] * inter, sigma); }
"""

from ._regression import normal_regression, ones_and

STAN_MODEL = "kidscore_interaction_c2"
POSTERIORS = ["kidiq_with_mom_work-kidscore_interaction_c2"]


def features(d, hyper):
    hs = d["mom_hs"] - 0.5
    iq = d["mom_iq"] - 100.0
    return ones_and(hs, iq, hs * iq)


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"kid_score": raw["kid_score"], "mom_hs": raw["mom_hs"], "mom_iq": raw["mom_iq"]},
        n_features=4,
        features=features,
        response=lambda d: d["kid_score"],
        response_field="kid_score",
    )
