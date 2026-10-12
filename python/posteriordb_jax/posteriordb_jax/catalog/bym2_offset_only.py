"""bym2_offset_only.stan (BYM2 spatial Poisson model with an exposure offset):

    parameters {
      real beta0; real<lower=0> sigma; real<lower=0, upper=1> rho; vector[N] theta; vector[N] phi;
    }
    transformed parameters {
      convolved_re = sqrt(1 - rho) * theta + sqrt(rho / scaling_factor) * phi;
    }
    model {
      y ~ poisson_log(log(E) + beta0 + convolved_re * sigma);
      target += -0.5 * dot_self(phi[node1] - phi[node2]);
      beta0 ~ normal(0, 1);
      theta ~ normal(0, 1);
      sigma ~ normal(0, 1);
      rho ~ beta(0.5, 0.5);
      sum(phi) ~ normal(0, 0.001 * N);
    }

The ICAR term on phi is part of the prior. The graph edges and scaling factor
are hyperparameters. A plain Model: each area has its own theta and phi.
"""

import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "bym2_offset_only"
POSTERIORS = ["traffic_accident_nyc-bym2_offset_only"]


def log_prior(params, hyper):
    phi = params["phi"]
    n = phi.shape[0]
    icar = -0.5 * jnp.sum((phi[hyper["node1"] - 1] - phi[hyper["node2"] - 1]) ** 2)
    return (icar
            + stats.norm.logpdf(params["beta0"], loc=0.0, scale=1.0)
            + jnp.sum(stats.norm.logpdf(params["theta"], loc=0.0, scale=1.0))
            + stats.norm.logpdf(params["sigma"], loc=0.0, scale=1.0)
            + stats.beta.logpdf(params["rho"], a=0.5, b=0.5)
            + stats.norm.logpdf(jnp.sum(phi), loc=0.0, scale=0.001 * n))


def log_lik(params, data, hyper):
    rho = params["rho"]
    convolved_re = jnp.sqrt(1.0 - rho) * params["theta"] + jnp.sqrt(rho / hyper["scaling_factor"]) * params["phi"]
    log_rate = jnp.log(data["E"]) + params["beta0"] + convolved_re * params["sigma"]
    return jnp.sum(stats.poisson.logpmf(data["y"], mu=jnp.exp(log_rate)))


def build(raw):
    n = int(raw["N"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            beta0=pg.NumericScalarPattern(),
            sigma=pg.NumericScalarPattern(lb=0.0),
            rho=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            theta=pg.NumericVectorPattern(n),
            phi=pg.NumericVectorPattern(n),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32), "E": raw["E"]},
        hyper={
            "node1": jnp.asarray(raw["node1"], dtype=jnp.int32),
            "node2": jnp.asarray(raw["node2"], dtype=jnp.int32),
            "scaling_factor": float(raw["scaling_factor"]),
        },
    )
