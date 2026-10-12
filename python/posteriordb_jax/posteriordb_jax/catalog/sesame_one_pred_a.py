"""sesame_one_pred_a.stan (normal linear regression, flat priors):

    watched ~ normal(beta[1] + beta[2] * encouraged, sigma)
"""

from ._regression import normal_regression, ones_and

STAN_MODEL = "sesame_one_pred_a"
POSTERIORS = ["sesame_data-sesame_one_pred_a"]


def features(d, hyper):
    return ones_and(d["encouraged"])


def build(raw):
    return normal_regression(
        name=STAN_MODEL,
        data={"watched": raw["watched"], "encouraged": raw["encouraged"]},
        n_features=2,
        features=features,
        response=lambda d: d["watched"],
        response_field="watched",
    )
