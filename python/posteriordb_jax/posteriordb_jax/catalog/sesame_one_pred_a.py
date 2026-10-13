"""sesame_one_pred_a.stan (normal linear regression, flat priors):

    watched ~ normal(beta[1] + beta[2] * encouraged, sigma)
"""

from ._regression import build_normal_regression, prepend_ones

STAN_MODEL = "sesame_one_pred_a"
POSTERIORS = ["sesame_data-sesame_one_pred_a"]


def compute_features(d, hyper):
    return prepend_ones(d["encouraged"])


def build(raw):
    return build_normal_regression(
        name=STAN_MODEL,
        data={"watched": raw["watched"], "encouraged": raw["encouraged"]},
        n_features=2,
        compute_features=compute_features,
        compute_response=lambda d: d["watched"],
        response_field="watched",
    )
