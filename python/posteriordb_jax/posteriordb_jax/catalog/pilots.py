"""pilots.stan (crossed random effects for pilot group and scenario):

    parameters {
      vector[n_groups] a; vector[n_scenarios] b; real mu_a; real mu_b;
      real<lower=0, upper=100> sigma_a; real<lower=0, upper=100> sigma_b; real<lower=0, upper=100> sigma_y;
    }
    model {
      mu_a ~ normal(0, 1);
      a ~ normal(10 * mu_a, sigma_a);
      mu_b ~ normal(0, 1);
      b ~ normal(10 * mu_b, sigma_b);
      y ~ normal(a[group_id] + b[scenario_id], sigma_y);
    }

Exchangeable in the modelgami sense: each trial is one datum carrying its
group and scenario indices.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "pilots"
POSTERIORS = ["pilots-pilots"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["mu_a"], loc=0.0, scale=1.0)
            + jnp.sum(stats.norm.logpdf(params["a"], loc=10.0 * params["mu_a"], scale=params["sigma_a"]))
            + stats.norm.logpdf(params["mu_b"], loc=0.0, scale=1.0)
            + jnp.sum(stats.norm.logpdf(params["b"], loc=10.0 * params["mu_b"], scale=params["sigma_b"])))


def mean(params, datum):
    return params["a"][datum["group_id"] - 1] + params["b"][datum["scenario_id"] - 1]


def log_lik_one(params, datum, hyper):
    return stats.norm.logpdf(datum["y"], loc=mean(params=params, datum=datum), scale=params["sigma_y"])


def sample_one(key, params, datum, hyper):
    return {"y": mean(params=params, datum=datum) + params["sigma_y"] * jax.random.normal(key)}


def build(raw):
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            a=pg.NumericVectorPattern(int(raw["n_groups"])),
            b=pg.NumericVectorPattern(int(raw["n_scenarios"])),
            mu_a=pg.NumericScalarPattern(),
            mu_b=pg.NumericScalarPattern(),
            sigma_a=pg.NumericScalarPattern(lb=0.0, ub=100.0),
            sigma_b=pg.NumericScalarPattern(lb=0.0, ub=100.0),
            sigma_y=pg.NumericScalarPattern(lb=0.0, ub=100.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one, sample_one=sample_one),
        data={
            "group_id": jnp.asarray(raw["group_id"], dtype=jnp.int32),
            "scenario_id": jnp.asarray(raw["scenario_id"], dtype=jnp.int32),
            "y": raw["y"],
        },
    )
