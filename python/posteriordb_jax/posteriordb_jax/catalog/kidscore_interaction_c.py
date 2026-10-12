"""kidscore_interaction_c.stan (normal linear regression, flat priors):

    transformed data {
      c_mom_hs = mom_hs - mean(mom_hs);
      c_mom_iq = mom_iq - mean(mom_iq);
      inter = c_mom_hs .* c_mom_iq;
    }
    model { kid_score ~ normal(beta[1] + beta[2] * c_mom_hs + beta[3] * c_mom_iq + beta[4] * inter, sigma); }

The centering uses the whole data set (Stan's transformed data), so the means
are computed once here and passed as hyperparameters. Subsets and reweighting
keep the full-data centering.
"""

import numpy as np

from ._regression import normal_regression, ones_and

STAN_MODEL = "kidscore_interaction_c"
POSTERIORS = ["kidiq_with_mom_work-kidscore_interaction_c"]


def features(d, hyper):
    hs = d["mom_hs"] - hyper["mom_hs_mean"]
    iq = d["mom_iq"] - hyper["mom_iq_mean"]
    return ones_and(hs, iq, hs * iq)


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"kid_score": raw["kid_score"], "mom_hs": raw["mom_hs"], "mom_iq": raw["mom_iq"]},
        hyper={
            "mom_hs_mean": float(np.mean(raw["mom_hs"])),
            "mom_iq_mean": float(np.mean(raw["mom_iq"])),
        },
        n_features=4,
        features=features,
        response=lambda d: d["kid_score"],
        response_field="kid_score",
    )
