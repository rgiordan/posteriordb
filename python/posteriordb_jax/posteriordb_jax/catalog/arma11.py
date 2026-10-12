"""arma11.stan (ARMA(1, 1) model):

    parameters { real mu; real phi; real theta; real<lower=0> sigma; }
    model {
      mu ~ normal(0, 10); phi ~ normal(0, 2); theta ~ normal(0, 2); sigma ~ cauchy(0, 2.5);
      nu[1] = mu + phi * mu;                       // assume err[0] == 0
      err[1] = y[1] - nu[1];
      for (t in 2:T) {
        nu[t] = mu + phi * y[t - 1] + theta * err[t - 1];
        err[t] = y[t] - nu[t];
      }
      err ~ normal(0, sigma);
    }

Not exchangeable: the errors are a recursion over time (a `lax.scan`).
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "arma11"
POSTERIORS = ["arma-arma11"]


def log_prior(params, hyper):
    return (stats.norm.logpdf(params["mu"], loc=0.0, scale=10.0)
            + stats.norm.logpdf(params["phi"], loc=0.0, scale=2.0)
            + stats.norm.logpdf(params["theta"], loc=0.0, scale=2.0)
            + stats.cauchy.logpdf(params["sigma"], loc=0.0, scale=2.5))


def errors(params, y):
    mu, phi, theta = params["mu"], params["phi"], params["theta"]
    err_1 = y[0] - (mu + phi * mu)

    def step(err_prev, y_pair):
        y_prev, y_t = y_pair
        err_t = y_t - (mu + phi * y_prev + theta * err_prev)
        return err_t, err_t

    _, err_rest = jax.lax.scan(step, err_1, (y[:-1], y[1:]))
    return jnp.concatenate([err_1[None], err_rest])


def log_lik(params, data, hyper):
    err = errors(params=params, y=data["y"])
    return jnp.sum(stats.norm.logpdf(err, loc=0.0, scale=params["sigma"]))


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            mu=pg.NumericScalarPattern(),
            phi=pg.NumericScalarPattern(),
            theta=pg.NumericScalarPattern(),
            sigma=pg.NumericScalarPattern(lb=0.0),
        ),
        prior=mg.DGP(log_density=log_prior),
        likelihood=mg.DGP(log_density=log_lik),
        data={"y": raw["y"]},
    )
