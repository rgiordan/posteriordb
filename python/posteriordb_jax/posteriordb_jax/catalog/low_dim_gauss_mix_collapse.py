"""low_dim_gauss_mix_collapse.stan: low_dim_gauss_mix.stan with an unordered
mu (vector[2] instead of ordered[2]), so the two components can swap labels.
The prior and likelihood are the same functions.
"""

import modelgami as mg
import paragami as pg

from .low_dim_gauss_mix import log_lik_one, log_prior, sample_one

STAN_MODEL = "low_dim_gauss_mix_collapse"
POSTERIORS = ["low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse"]


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            mu=pg.NumericVectorPattern(2),
            sigma=pg.NumericVectorPattern(2, lb=0.0),
            theta=pg.NumericScalarPattern(lb=0.0, ub=1.0),
        ),
        data={"y": raw["y"]},
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
    )
