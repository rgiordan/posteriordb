"""election88_full.stan (multilevel logistic regression of 1988 vote choice):

    parameters {
      vector[n_age] a; vector[n_edu] b; vector[n_age_edu] c; vector[n_state] d; vector[n_region_full] e;
      vector[5] beta; real<lower=0, upper=100> sigma_a; ... sigma_e;
    }
    transformed parameters {
      y_hat = beta[1] + beta[2] * black + beta[3] * female + beta[5] * female .* black
              + beta[4] * v_prev_full + a[age] + b[edu] + c[age_edu] + d[state] + e[region_full];
    }
    model {
      a ~ normal(0, sigma_a); b ~ normal(0, sigma_b); c ~ normal(0, sigma_c);
      d ~ normal(0, sigma_d); e ~ normal(0, sigma_e);
      beta ~ normal(0, 100);
      y ~ bernoulli_logit(y_hat);
    }

Exchangeable in the modelgami sense: each respondent is one datum carrying
their group indices.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "election88_full"
POSTERIORS = ["election88-election88_full"]

GROUPS = [("a", "age", "n_age"), ("b", "edu", "n_edu"), ("c", "age_edu", "n_age_edu"),
          ("d", "state", "n_state"), ("e", "region_full", "n_region_full")]


def eval_log_prior(params, hyper):
    lp = jnp.sum(stats.norm.logpdf(params["beta"], loc=0.0, scale=100.0))
    for effect, _, _ in GROUPS:
        lp = lp + jnp.sum(stats.norm.logpdf(params[effect], loc=0.0, scale=params[f"sigma_{effect}"]))
    return lp


def compute_y_hat(params, datum):
    beta = params["beta"]
    out = (beta[0] + beta[1] * datum["black"] + beta[2] * datum["female"]
           + beta[4] * datum["female"] * datum["black"] + beta[3] * datum["v_prev_full"])
    for effect, index, _ in GROUPS:
        out = out + params[effect][datum[index] - 1]
    return out


def eval_log_lik_one(datum, params, hyper):
    eta = compute_y_hat(params=params, datum=datum)
    return datum["y"] * eta - jax.nn.softplus(eta)


def draw_one(key, datum, params, hyper):
    p = jax.nn.sigmoid(compute_y_hat(params=params, datum=datum))
    return {"y": jax.random.bernoulli(key, p=p).astype(datum["y"].dtype)}


def build(raw):
    patterns = {effect: pg.NumericVectorPattern(int(raw[size])) for effect, _, size in GROUPS}
    patterns["beta"] = pg.NumericVectorPattern(5)
    for effect, _, _ in GROUPS:
        patterns[f"sigma_{effect}"] = pg.NumericScalarPattern(lb=0.0, ub=100.0)
    data = {index: jnp.asarray(raw[index], dtype=jnp.int32) for _, index, _ in GROUPS}
    data.update({"black": raw["black"], "female": raw["female"], "v_prev_full": raw["v_prev_full"],
                 "y": jnp.asarray(raw["y"], dtype=jnp.int32)})
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(**patterns),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data=data,
    )
