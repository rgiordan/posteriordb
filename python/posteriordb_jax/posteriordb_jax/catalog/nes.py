"""nes.stan (normal linear regression, flat priors):

    transformed data { age30_44 = age_discrete == 2; age45_64 = age_discrete == 3; age65up = age_discrete == 4; }
    model {
      partyid7 ~ normal(beta[1] + beta[2] * real_ideo + beta[3] * race_adj
                        + beta[4] * age30_44 + beta[5] * age45_64 + beta[6] * age65up
                        + beta[7] * educ1 + beta[8] * gender + beta[9] * income, sigma);
    }
"""

import jax.numpy as jnp

from ._regression import build_normal_regression, prepend_ones

STAN_MODEL = "nes"
POSTERIORS = [f"nes{year}-nes" for year in range(1972, 2001, 4)]

FIELDS = ["partyid7", "real_ideo", "race_adj", "educ1", "gender", "income", "age_discrete"]


def compute_features(d, hyper):
    age = d["age_discrete"]
    return prepend_ones(
        d["real_ideo"],
        d["race_adj"],
        (age == 2).astype(d["real_ideo"].dtype),
        (age == 3).astype(d["real_ideo"].dtype),
        (age == 4).astype(d["real_ideo"].dtype),
        d["educ1"],
        d["gender"],
        d["income"],
    )


def build(raw):
    data = {f: raw[f] for f in FIELDS}
    data["age_discrete"] = jnp.asarray(raw["age_discrete"], dtype=jnp.int32)
    return build_normal_regression(
        name=STAN_MODEL,
        data=data,
        n_features=9,
        compute_features=compute_features,
        compute_response=lambda d: d["partyid7"],
        response_field="partyid7",
    )
