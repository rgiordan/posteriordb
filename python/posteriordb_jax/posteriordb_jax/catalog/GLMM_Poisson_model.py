"""GLMM_Poisson_model.stan (cubic Poisson trend with year effects):

    parameters {
      real<lower=-20, upper=20> alpha; real<lower=-10, upper=10> beta1;
      real<lower=-10, upper=20> beta2; real<lower=-10, upper=10> beta3;
      vector[n] eps; real<lower=0, upper=5> sigma;
    }
    transformed parameters { log_lambda = alpha + beta1 * year + beta2 * year^2 + beta3 * year^3 + eps; }
    model {
      alpha ~ uniform(-20, 20); beta1 ~ uniform(-10, 10); beta2 ~ uniform(-10, 10);
      beta3 ~ uniform(-10, 10); sigma ~ uniform(0, 5);
      C ~ poisson_log(log_lambda);
      eps ~ normal(0, sigma);
    }

beta2's bounds (-10, 20) are wider than its uniform(-10, 10) prior, as in the
Stan code, so the log density is -inf for beta2 > 10. A plain Model: each
year has its own eps.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "GLMM_Poisson_model"
POSTERIORS = ["GLMM_Poisson_data-GLMM_Poisson_model"]


def log_prior(params, hyper):
    return (stats.uniform.logpdf(params["alpha"], loc=-20.0, scale=40.0)
            + stats.uniform.logpdf(params["beta1"], loc=-10.0, scale=20.0)
            + stats.uniform.logpdf(params["beta2"], loc=-10.0, scale=20.0)
            + stats.uniform.logpdf(params["beta3"], loc=-10.0, scale=20.0)
            + stats.uniform.logpdf(params["sigma"], loc=0.0, scale=5.0)
            + jnp.sum(stats.norm.logpdf(params["eps"], loc=0.0, scale=params["sigma"])))


def log_lik(params, data, hyper):
    year = data["year"]
    log_lambda = (params["alpha"] + params["beta1"] * year + params["beta2"] * year**2
                  + params["beta3"] * year**3 + params["eps"])
    return jnp.sum(stats.poisson.logpmf(data["C"], mu=jnp.exp(log_lambda)))


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(lb=-20.0, ub=20.0),
            beta1=pg.NumericScalarPattern(lb=-10.0, ub=10.0),
            beta2=pg.NumericScalarPattern(lb=-10.0, ub=20.0),
            beta3=pg.NumericScalarPattern(lb=-10.0, ub=10.0),
            eps=pg.NumericVectorPattern(int(raw["n"])),
            sigma=pg.NumericScalarPattern(lb=0.0, ub=5.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"C": jnp.asarray(raw["C"], dtype=jnp.int32), "year": raw["year"]},
    )
