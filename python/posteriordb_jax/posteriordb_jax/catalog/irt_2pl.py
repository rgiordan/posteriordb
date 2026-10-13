"""irt_2pl.stan (two-parameter logistic item response model; y is items x students):

    parameters {
      real<lower=0> sigma_theta; vector[J] theta; real<lower=0> sigma_a; vector<lower=0>[I] a;
      real mu_b; real<lower=0> sigma_b; vector[I] b;
    }
    model {
      sigma_theta ~ cauchy(0, 2);
      theta ~ normal(0, sigma_theta);
      sigma_a ~ cauchy(0, 2);
      a ~ lognormal(0, sigma_a);
      mu_b ~ normal(0, 5);
      sigma_b ~ cauchy(0, 2);
      b ~ normal(mu_b, sigma_b);
      for (i in 1:I) y[i] ~ bernoulli_logit(a[i] * (theta - b[i]));
    }

Not exchangeable in the modelgami sense: each student and item has its own parameters.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "irt_2pl"
POSTERIORS = ["irt_2pl-irt_2pl"]


def eval_log_prior(params, hyper):
    a = params["a"]
    lognormal_a = stats.norm.logpdf(jnp.log(a), loc=0.0, scale=params["sigma_a"]) - jnp.log(a)
    return (stats.cauchy.logpdf(params["sigma_theta"], loc=0.0, scale=2.0)
            + jnp.sum(stats.norm.logpdf(params["theta"], loc=0.0, scale=params["sigma_theta"]))
            + stats.cauchy.logpdf(params["sigma_a"], loc=0.0, scale=2.0)
            + jnp.sum(lognormal_a)
            + stats.norm.logpdf(params["mu_b"], loc=0.0, scale=5.0)
            + stats.cauchy.logpdf(params["sigma_b"], loc=0.0, scale=2.0)
            + jnp.sum(stats.norm.logpdf(params["b"], loc=params["mu_b"], scale=params["sigma_b"])))


def eval_log_lik(data, params, hyper):
    logit_p = params["a"][:, None] * (params["theta"][None, :] - params["b"][:, None])
    return jnp.sum(data["y"] * logit_p - jax.nn.softplus(logit_p))


def build(raw):
    i, j = int(raw["I"]), int(raw["J"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            sigma_theta=pg.NumericScalarPattern(lb=0.0),
            theta=pg.NumericVectorPattern(j),
            sigma_a=pg.NumericScalarPattern(lb=0.0),
            a=pg.NumericVectorPattern(i, lb=0.0),
            mu_b=pg.NumericScalarPattern(),
            sigma_b=pg.NumericScalarPattern(lb=0.0),
            b=pg.NumericVectorPattern(i),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"y": jnp.asarray(raw["y"], dtype=jnp.int32)},
    )
