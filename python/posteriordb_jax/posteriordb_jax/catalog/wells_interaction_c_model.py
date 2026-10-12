"""wells_interaction_c_model.stan (logistic regression, flat priors):

    transformed data {
      vector[N] c_dist100 = (dist - mean(dist)) / 100.0;
      vector[N] c_arsenic = arsenic - mean(arsenic);
      vector[N] inter = c_dist100 .* c_arsenic;
      matrix[N, 3] x = [c_dist100', c_arsenic', inter']';
    }
    parameters { real alpha; vector[3] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }

The centering uses the whole data set (Stan's transformed data), so the means
are computed once here and passed as hyperparameters. Subsets and reweighting
keep the full-data centering.
"""

import jax.numpy as jnp
import numpy as np

from ._regression import logistic_regression

STAN_MODEL = "wells_interaction_c_model"
POSTERIORS = ["wells_data-wells_interaction_c_model"]


def features(d, hyper):
    c_dist100 = (d["dist"] - hyper["dist_mean"]) / 100.0
    c_arsenic = d["arsenic"] - hyper["arsenic_mean"]
    return jnp.stack([c_dist100, c_arsenic, c_dist100 * c_arsenic])


def build(raw):
    return logistic_regression(
        name=STAN_MODEL,
        data={"switched": raw["switched"], "dist": raw["dist"], "arsenic": raw["arsenic"]},
        hyper={"dist_mean": float(np.mean(raw["dist"])), "arsenic_mean": float(np.mean(raw["arsenic"]))},
        n_features=3,
        features=features,
        response_field="switched",
    )
