"""kidscore_interaction_z.stan (normal linear regression, flat priors):

    transformed data {
      z_mom_hs = (mom_hs - mean(mom_hs)) / (2 * sd(mom_hs));
      z_mom_iq = (mom_iq - mean(mom_iq)) / (2 * sd(mom_iq));
      inter = z_mom_hs .* z_mom_iq;
    }
    model { kid_score ~ normal(beta[1] + beta[2] * z_mom_hs + beta[3] * z_mom_iq + beta[4] * inter, sigma); }

The standardization uses the whole data set (Stan's transformed data), so the
means and sds are computed once here and passed as hyperparameters. Subsets and
reweighting keep the full-data standardization.
"""

import numpy as np

from ._regression import normal_regression, ones_and

STAN_MODEL = "kidscore_interaction_z"
POSTERIORS = ["kidiq_with_mom_work-kidscore_interaction_z"]


def features(d, hyper):
    hs = (d["mom_hs"] - hyper["mom_hs_mean"]) / (2.0 * hyper["mom_hs_sd"])
    iq = (d["mom_iq"] - hyper["mom_iq_mean"]) / (2.0 * hyper["mom_iq_sd"])
    return ones_and(hs, iq, hs * iq)


def build(raw):
    hs = np.asarray(raw["mom_hs"], dtype=float)
    iq = np.asarray(raw["mom_iq"], dtype=float)
    return normal_regression(
        name=STAN_MODEL,
        data={"kid_score": raw["kid_score"], "mom_hs": raw["mom_hs"], "mom_iq": raw["mom_iq"]},
        hyper={
            "mom_hs_mean": float(hs.mean()),
            "mom_hs_sd": float(hs.std(ddof=1)),
            "mom_iq_mean": float(iq.mean()),
            "mom_iq_sd": float(iq.std(ddof=1)),
        },
        n_features=4,
        features=features,
        response=lambda d: d["kid_score"],
        response_field="kid_score",
    )
