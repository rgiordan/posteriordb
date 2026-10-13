"""multi_occupancy.stan (multispecies site-occupancy model with data
augmentation: n observed species plus S - n never-observed ones, each with
correlated occupancy and detection effects):

    parameters {
      real alpha; real beta; real<lower=0, upper=1> Omega; real<lower=-1, upper=1> rho_uv;
      vector<lower=0>[2] sigma_uv; vector[S] uv1; vector[S] uv2;
    }
    transformed parameters { logit_psi = uv1 + alpha; logit_theta = uv2 + beta; }
    model {
      alpha ~ cauchy(0, 2.5); beta ~ cauchy(0, 2.5); sigma_uv ~ cauchy(0, 2.5);
      (rho_uv + 1) / 2 ~ beta(2, 2);           // no Jacobian adjustment, as in Stan
      [uv1[i], uv2[i]] ~ multi_normal(0, [[sigma_uv[1]^2, rho sigma_uv[1] sigma_uv[2]], [., sigma_uv[2]^2]]);
      Omega ~ beta(2, 2);
      for observed species i: 1 ~ bernoulli(Omega);
        X[i, j] > 0:  log_inv_logit(logit_psi[i]) + binomial_logit_lpmf(X[i, j] | K, logit_theta[i]);
        X[i, j] == 0: log_sum_exp(that with X = 0, log1m_inv_logit(logit_psi[i]));
      for never-observed species i:
        log_sum_exp(log(1 - Omega), log(Omega) + J * lp_unobserved(K, logit_psi[i], logit_theta[i]));
    }

A plain Model: each species has its own parameters, and the never-observed
species have no data of their own.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._distributions import eval_binomial_logit_logpmf

STAN_MODEL = "multi_occupancy"
POSTERIORS = ["butterfly-multi_occupancy"]


def eval_bivariate_normal_logpdf(*, u, v, sigma, rho):
    """multi_normal_lpdf([u, v] | 0, cov_matrix_2d(sigma, rho)), elementwise."""
    zu, zv = u / sigma[0], v / sigma[1]
    one_minus_rho2 = 1.0 - rho**2
    return (-jnp.log(2.0 * jnp.pi) - jnp.log(sigma[0]) - jnp.log(sigma[1]) - 0.5 * jnp.log(one_minus_rho2)
            - 0.5 * (zu**2 - 2.0 * rho * zu * zv + zv**2) / one_minus_rho2)


def eval_log_prior(params, hyper):
    rho = params["rho_uv"]
    return (stats.cauchy.logpdf(params["alpha"], loc=0.0, scale=2.5)
            + stats.cauchy.logpdf(params["beta"], loc=0.0, scale=2.5)
            + jnp.sum(stats.cauchy.logpdf(params["sigma_uv"], loc=0.0, scale=2.5))
            + stats.beta.logpdf((rho + 1.0) / 2.0, a=2.0, b=2.0)
            + jnp.sum(eval_bivariate_normal_logpdf(u=params["uv1"], v=params["uv2"], sigma=params["sigma_uv"], rho=rho))
            + stats.beta.logpdf(params["Omega"], a=2.0, b=2.0))


def eval_log_lik(data, params, hyper):
    x, n_visits = data["X"], hyper["K"]
    n_observed, n_sites = x.shape
    logit_psi = params["uv1"] + params["alpha"]
    logit_theta = params["uv2"] + params["beta"]
    log_psi, log1m_psi = -jax.nn.softplus(-logit_psi), -jax.nn.softplus(logit_psi)
    log_never_detected = log_psi + eval_binomial_logit_logpmf(0.0, n=n_visits, logit_p=logit_theta)
    lp_unobserved = jnp.logaddexp(log_never_detected, log1m_psi)   # per species

    observed_psi, observed_theta = log_psi[:n_observed, None], logit_theta[:n_observed, None]
    lp_observed = observed_psi + eval_binomial_logit_logpmf(x, n=n_visits, logit_p=observed_theta)
    observed = (n_observed * jnp.log(params["Omega"])
                + jnp.sum(jnp.where(x > 0, lp_observed, lp_unobserved[:n_observed, None])))
    never_observed = jnp.logaddexp(jnp.log1p(-params["Omega"]),
                                   jnp.log(params["Omega"]) + n_sites * lp_unobserved[n_observed:])
    return observed + jnp.sum(never_observed)


def build(raw):
    n_species = int(raw["S"])
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            alpha=pg.NumericScalarPattern(),
            beta=pg.NumericScalarPattern(),
            Omega=pg.NumericScalarPattern(lb=0.0, ub=1.0),
            rho_uv=pg.NumericScalarPattern(lb=-1.0, ub=1.0),
            sigma_uv=pg.NumericVectorPattern(2, lb=0.0),
            uv1=pg.NumericVectorPattern(n_species),
            uv2=pg.NumericVectorPattern(n_species),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"X": jnp.asarray(raw["X"], dtype=jnp.int32)},
        hyper={"K": float(raw["K"])},
    )
