"""wells_dae_inter_model.stan (logistic regression, flat priors):

    transformed data {
      vector[N] c_dist100 = (dist - mean(dist)) / 100.0;
      vector[N] c_arsenic = arsenic - mean(arsenic);
      vector[N] c_educ4 = (educ - mean(educ)) / 4.0;
      vector[N] da_inter = c_dist100 .* c_arsenic;
      vector[N] de_inter = c_dist100 .* c_educ4;
      vector[N] ae_inter = c_arsenic .* c_educ4;
      matrix[N, 6] x = [c_dist100', c_arsenic', c_educ4', da_inter', de_inter', ae_inter']';
    }
    parameters { real alpha; vector[6] beta; }
    model { switched ~ bernoulli_logit_glm(x, alpha, beta); }

The centering uses the whole data set (Stan's transformed data), so the means
are computed once here and passed as shared data. Subsets and reweighting
keep the full-data centering.
"""

import jax.numpy as jnp
import numpy as np

from ._regression import logistic_regression

STAN_MODEL = "wells_dae_inter_model"
POSTERIORS = ["wells_data-wells_dae_inter_model"]


def features(d):
    c_dist100 = (d["dist"] - d["dist_mean"]) / 100.0
    c_arsenic = d["arsenic"] - d["arsenic_mean"]
    c_educ4 = (d["educ"] - d["educ_mean"]) / 4.0
    return jnp.stack([c_dist100, c_arsenic, c_educ4, c_dist100 * c_arsenic, c_dist100 * c_educ4, c_arsenic * c_educ4])


def build(raw):
    return logistic_regression(
        name=STAN_MODEL,
        data={"switched": raw["switched"], "dist": raw["dist"], "arsenic": raw["arsenic"], "educ": raw["educ"]},
        shared_data={"dist_mean": float(np.mean(raw["dist"])), "arsenic_mean": float(np.mean(raw["arsenic"])), "educ_mean": float(np.mean(raw["educ"]))},
        n_features=6,
        features=features,
        response_field="switched",
    )
