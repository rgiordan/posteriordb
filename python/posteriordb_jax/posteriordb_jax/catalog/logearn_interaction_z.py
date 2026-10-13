"""logearn_interaction_z.stan:

    transformed data {
      log_earn = log(earn);
      z_height = (height - mean(height)) / sd(height);
      inter = z_height .* male;
    }
    model { log_earn ~ normal(beta[1] + beta[2] * z_height + beta[3] * male + beta[4] * inter, sigma); }

The standardization uses the whole data set (Stan's transformed data), so
the mean and sd are computed once here and passed as hyperparameters. Subsets
and reweighting keep the full-data standardization.
"""

import jax.numpy as jnp
import numpy as np

from ._regression import build_normal_regression, prepend_ones

STAN_MODEL = "logearn_interaction_z"
POSTERIORS = ["earnings-logearn_interaction_z"]


def compute_features(d, hyper):
    z = (d["height"] - hyper["height_mean"]) / hyper["height_sd"]
    return prepend_ones(z, d["male"], z * d["male"])


def build(raw):
    height = np.asarray(raw["height"], dtype=float)
    return build_normal_regression(
        name=STAN_MODEL,
        data={"earn": raw["earn"], "height": raw["height"], "male": raw["male"]},
        hyper={"height_mean": float(height.mean()), "height_sd": float(height.std(ddof=1))},
        n_features=4,
        compute_features=compute_features,
        compute_response=lambda d: jnp.log(d["earn"]),
        response_field="earn",
        invert_response=jnp.exp,
    )
