"""Example, written from scratch: Bayesian linear regression with a
conjugate normal-inverse-gamma prior.

    y_i | beta, sigma2  ~  Normal(x_i' beta, sigma2)          i = 1..N, independent
    beta | sigma2       ~  MultivariateNormal(m0, sigma2 * V0)
    sigma2              ~  InverseGamma(a0, b0)

The first half of this file is everything a user writes to define the model.
The second half is the closed-form posterior and marginal likelihood, which
this conjugate model happens to have; the tests use them to check modelgami.

Usage:

    from modelgami.examples import conjugate_regression as cr
    X, y = cr.simulate_data(jax.random.PRNGKey(0), n=100, p=3)
    model = cr.make_model(X, y)

    model.eval_free_log_joint(x)                     # for HMC, VI, Laplace, ...
    model.eval_log_lik_singleobs(params)             # single-observation terms
    model.eval_log_joint(params, weights=w)          # weighted
    model.draw_prior(key), model.draw_joint(key)     # simulation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import gammaln

import modelgami as mg
import paragami as pg

# ----------------------------------------------------------------------------
# The model: this is all a user writes.
# ----------------------------------------------------------------------------


def make_params_pattern(p: int) -> pg.PatternDict:
    return pg.PatternDict(
        beta=pg.NumericVectorPattern(p),
        sigma2=pg.NumericScalarPattern(lb=0.0),
    )


def make_default_hyper(p: int):
    return {"m0": jnp.zeros(p), "V0": 10.0 * jnp.eye(p), "a0": 2.0, "b0": 2.0}


def eval_log_inv_gamma(x, a, b):
    return a * jnp.log(b) - gammaln(a) - (a + 1) * jnp.log(x) - b / x


def eval_log_prior(params, hyper):
    beta, sigma2 = params["beta"], params["sigma2"]
    return (stats.multivariate_normal.logpdf(beta, hyper["m0"], sigma2 * hyper["V0"])
            + eval_log_inv_gamma(sigma2, hyper["a0"], hyper["b0"]))


def draw_prior(key, hyper):
    k1, k2 = jax.random.split(key)
    sigma2 = hyper["b0"] / jax.random.gamma(k1, hyper["a0"])
    chol = jnp.linalg.cholesky(hyper["V0"])
    beta = hyper["m0"] + jnp.sqrt(sigma2) * chol @ jax.random.normal(k2, hyper["m0"].shape)
    return {"beta": beta, "sigma2": sigma2}


def eval_log_lik_one(datum, params, hyper):
    """One observation: datum = {"x": (p,), "y": ()}."""
    mean = datum["x"] @ params["beta"]
    return stats.norm.logpdf(datum["y"], mean, jnp.sqrt(params["sigma2"]))


def draw_one(key, datum, params, hyper):
    """Draw a new y for one observation; its covariates x are kept."""
    mean = datum["x"] @ params["beta"]
    return {"y": mean + jnp.sqrt(params["sigma2"]) * jax.random.normal(key)}


def make_model(X, y, hyper=None) -> mg.ExchangeableModel:
    p = X.shape[1]
    return mg.ExchangeableModel(
        name="conjugate_regression",
        params_pattern=make_params_pattern(p),
        prior=mg.DGP(eval_log_density=eval_log_prior, draw=draw_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data={"x": X, "y": y},  # N observations: every field has a leading N axis
        hyper=make_default_hyper(p) if hyper is None else hyper,
    )


# The same prior and likelihood as classes. Equivalent; use whichever reads better.
class NormalInverseGammaPrior(mg.DGP):
    def eval_log_density(self, params, hyper):
        return eval_log_prior(params, hyper)

    def draw(self, key, hyper):
        return draw_prior(key, hyper)


class NormalRegressionLikelihood(mg.ExchangeableDGP):
    def eval_log_density_one(self, datum, params, hyper):
        return eval_log_lik_one(datum, params, hyper)

    def draw_one(self, key, datum, params, hyper):
        return draw_one(key, datum, params, hyper)


# ----------------------------------------------------------------------------
# Closed-form results (conjugacy), for checking and benchmarking.
# ----------------------------------------------------------------------------


def simulate_data(key, n: int = 100, p: int = 3, sigma: float = 1.0):
    k1, k2, k3 = jax.random.split(key, 3)
    X = jnp.concatenate([jnp.ones((n, 1)), jax.random.normal(k1, (n, p - 1))], axis=1)
    beta = jax.random.normal(k2, (p,))
    y = X @ beta + sigma * jax.random.normal(k3, (n,))
    return X, y


def compute_exact_posterior(X, y, hyper):
    """Posterior hyperparameters: beta | sigma2, y ~ N(mn, sigma2 Vn),
    sigma2 | y ~ InvGamma(an, bn)."""
    V0_inv = jnp.linalg.inv(hyper["V0"])
    Vn_inv = V0_inv + X.T @ X
    Vn = jnp.linalg.inv(Vn_inv)
    mn = Vn @ (V0_inv @ hyper["m0"] + X.T @ y)
    an = hyper["a0"] + 0.5 * X.shape[0]
    bn = hyper["b0"] + 0.5 * (y @ y + hyper["m0"] @ V0_inv @ hyper["m0"] - mn @ Vn_inv @ mn)
    return {"m0": mn, "V0": Vn, "a0": an, "b0": bn}


def eval_exact_log_posterior(params, X, y, hyper):
    return eval_log_prior(params, compute_exact_posterior(X, y, hyper))


def eval_log_marginal_likelihood(X, y, hyper):
    post = compute_exact_posterior(X, y, hyper)
    n = X.shape[0]
    return (-0.5 * n * jnp.log(2 * jnp.pi)
            + 0.5 * jnp.linalg.slogdet(post["V0"])[1] - 0.5 * jnp.linalg.slogdet(hyper["V0"])[1]
            + hyper["a0"] * jnp.log(hyper["b0"]) - post["a0"] * jnp.log(post["b0"])
            + gammaln(post["a0"]) - gammaln(hyper["a0"]))


def draw_exact_posterior(key, X, y, hyper, n_chains: int = 4, n_draws: int = 1000):
    """Exact posterior draws with leading (chain, draw) axes."""
    post = compute_exact_posterior(X, y, hyper)
    keys = jax.random.split(key, n_chains * n_draws)
    draws = jax.vmap(lambda k: draw_prior(k, post))(keys)
    return jax.tree_util.tree_map(lambda a: a.reshape((n_chains, n_draws) + a.shape[1:]), draws)
