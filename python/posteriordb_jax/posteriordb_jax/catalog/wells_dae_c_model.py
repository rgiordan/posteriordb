"""wells_dae_c_model.stan (logistic regression, flat priors):

    transformed data {
      vector[N] c_dist100 = (dist - mean(dist)) / 100.0;
      vector[N] c_arsenic = arsenic - mean(arsenic);
      vector[N] da_inter = c_dist100 .* c_arsenic;
      vector[N] educ4 = educ / 4.0;
      matrix[N, 4] x = [c_dist100', c_arsenic', da_inter', educ4']';
    }
    parameters { real alpha; vector[4] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }

The centering uses the whole data set (Stan's transformed data), so the means
are computed once here and passed as hyperparameters. Subsets and reweighting
keep the full-data centering.
"""

import jax.numpy as jnp
import numpy as np

from ._regression import build_logistic_regression

STAN_MODEL = "wells_dae_c_model"
POSTERIORS = ["wells_data-wells_dae_c_model"]


def compute_features(d, hyper):
    c_dist100 = (d["dist"] - hyper["dist_mean"]) / 100.0
    c_arsenic = d["arsenic"] - hyper["arsenic_mean"]
    return jnp.stack([c_dist100, c_arsenic, c_dist100 * c_arsenic, d["educ"] / 4.0])


def build(raw):
    return build_logistic_regression(
        name=STAN_MODEL,
        data={"switched": raw["switched"], "dist": raw["dist"], "arsenic": raw["arsenic"], "educ": raw["educ"]},
        hyper={"dist_mean": float(np.mean(raw["dist"])), "arsenic_mean": float(np.mean(raw["arsenic"]))},
        n_features=4,
        compute_features=compute_features,
        response_field="switched",
    )
