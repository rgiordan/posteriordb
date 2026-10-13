"""GLM_Poisson_model.stan (flat priors inside the parameter bounds):

    parameters {
      real<lower=-20, upper=20> alpha;
      real<lower=-10, upper=10> beta1;
      real<lower=-10, upper=10> beta2;
      real<lower=-10, upper=10> beta3;
    }
    transformed parameters { log_lambda = alpha + beta1 * year + beta2 * year^2 + beta3 * year^3; }
    model { C ~ poisson_log(log_lambda); }
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "GLM_Poisson_model"
POSTERIORS = ["GLM_Poisson_Data-GLM_Poisson_model"]


def compute_log_lambda(params, datum):
    year = datum["year"]
    return params["alpha"] + params["beta1"] * year + params["beta2"] * year**2 + params["beta3"] * year**3


def eval_log_lik_one(datum, params, hyper):
    return stats.poisson.logpmf(datum["C"], mu=jnp.exp(compute_log_lambda(params=params, datum=datum)))


def draw_one(key, datum, params, hyper):
    draw = jax.random.poisson(key, lam=jnp.exp(compute_log_lambda(params=params, datum=datum)))
    return {"C": draw.astype(datum["C"].dtype)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(lb=-20.0, ub=20.0),
            beta1=pg.NumericScalarPattern(lb=-10.0, ub=10.0),
            beta2=pg.NumericScalarPattern(lb=-10.0, ub=10.0),
            beta3=pg.NumericScalarPattern(lb=-10.0, ub=10.0),
        ),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data={"C": jnp.asarray(raw["C"], dtype=jnp.int32), "year": raw["year"]},
    )
