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

    f = model.log_density_fn("joint", params_as="free")   # for HMC, VI, Laplace, ...
    model.log_lik_pointwise(params)                        # per-observation terms
    model.log_lik(params, weights=w)                        # weighted likelihood
    model.sample_prior(key), model.sample_data(key, params) # simulation
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


def params_pattern(p: int) -> pg.PatternDict:
    return pg.PatternDict(
        beta=pg.NumericVectorPattern(p),
        sigma2=pg.NumericScalarPattern(lb=0.0),
    )


def default_hyper(p: int):
    return {"m0": jnp.zeros(p), "V0": 10.0 * jnp.eye(p), "a0": 2.0, "b0": 2.0}


def log_inv_gamma(x, a, b):
    return a * jnp.log(b) - gammaln(a) - (a + 1) * jnp.log(x) - b / x


def log_prior(params, hyper):
    beta, sigma2 = params["beta"], params["sigma2"]
    return (stats.multivariate_normal.logpdf(beta, hyper["m0"], sigma2 * hyper["V0"])
            + log_inv_gamma(sigma2, hyper["a0"], hyper["b0"]))


def log_lik_one(params, datum, hyper):
    """One observation: datum = {"x": (p,), "y": ()}."""
    mean = datum["x"] @ params["beta"]
    return stats.norm.logpdf(datum["y"], mean, jnp.sqrt(params["sigma2"]))


def sample_prior(key, hyper):
    k1, k2 = jax.random.split(key)
    sigma2 = hyper["b0"] / jax.random.gamma(k1, hyper["a0"])
    chol = jnp.linalg.cholesky(hyper["V0"])
    beta = hyper["m0"] + jnp.sqrt(sigma2) * chol @ jax.random.normal(k2, hyper["m0"].shape)
    return {"beta": beta, "sigma2": sigma2}


def sample_one(key, params, datum, hyper):
    """Draw a new y for one observation, keeping its covariates."""
    mean = datum["x"] @ params["beta"]
    return {"y": mean + jnp.sqrt(params["sigma2"]) * jax.random.normal(key)}


def check_data(data):
    if data["x"].ndim != 2:
        return "x must be (N, p)"


def make_model(X, y, hyper=None) -> mg.Model:
    p = X.shape[1]
    return mg.Model(
        name="conjugate_regression",
        params_pattern=params_pattern(p),
        data={"x": X, "y": y},  # exchangeable: every field has a leading N axis
        hyper=default_hyper(p) if hyper is None else hyper,
        log_prior=log_prior,
        log_lik_one=log_lik_one,
        sample_prior=sample_prior,
        sample_one=sample_one,
        check_data=check_data,
    )


# The same model in class style. Equivalent; use whichever reads better.
class ConjugateRegression(mg.Model):
    def params_pattern(self):
        return params_pattern(self.data["x"].shape[1])

    def log_prior(self, params, hyper):
        return log_prior(params, hyper)

    def log_lik_one(self, params, datum, hyper):
        return log_lik_one(params, datum, hyper)

    def sample_prior(self, key, hyper):
        return sample_prior(key, hyper)

    def sample_one(self, key, params, datum, hyper):
        return sample_one(key, params, datum, hyper)


# ----------------------------------------------------------------------------
# Closed-form results (conjugacy), for checking and benchmarking.
# ----------------------------------------------------------------------------


def simulate_data(key, n: int = 100, p: int = 3, sigma: float = 1.0):
    k1, k2, k3 = jax.random.split(key, 3)
    X = jnp.concatenate([jnp.ones((n, 1)), jax.random.normal(k1, (n, p - 1))], axis=1)
    beta = jax.random.normal(k2, (p,))
    y = X @ beta + sigma * jax.random.normal(k3, (n,))
    return X, y


def exact_posterior(X, y, hyper):
    """Posterior hyperparameters: beta | sigma2, y ~ N(mn, sigma2 Vn),
    sigma2 | y ~ InvGamma(an, bn)."""
    V0_inv = jnp.linalg.inv(hyper["V0"])
    Vn_inv = V0_inv + X.T @ X
    Vn = jnp.linalg.inv(Vn_inv)
    mn = Vn @ (V0_inv @ hyper["m0"] + X.T @ y)
    an = hyper["a0"] + 0.5 * X.shape[0]
    bn = hyper["b0"] + 0.5 * (y @ y + hyper["m0"] @ V0_inv @ hyper["m0"] - mn @ Vn_inv @ mn)
    return {"m0": mn, "V0": Vn, "a0": an, "b0": bn}


def exact_log_posterior(params, X, y, hyper):
    return log_prior(params, exact_posterior(X, y, hyper))


def log_marginal_likelihood(X, y, hyper):
    post = exact_posterior(X, y, hyper)
    n = X.shape[0]
    return (-0.5 * n * jnp.log(2 * jnp.pi)
            + 0.5 * jnp.linalg.slogdet(post["V0"])[1] - 0.5 * jnp.linalg.slogdet(hyper["V0"])[1]
            + hyper["a0"] * jnp.log(hyper["b0"]) - post["a0"] * jnp.log(post["b0"])
            + gammaln(post["a0"]) - gammaln(hyper["a0"]))


def sample_exact_posterior(key, X, y, hyper, n_chains: int = 4, n_draws: int = 1000):
    """Exact posterior draws with leading (chain, draw) axes."""
    post = exact_posterior(X, y, hyper)
    keys = jax.random.split(key, n_chains * n_draws)
    draws = jax.vmap(lambda k: sample_prior(k, post))(keys)
    return jax.tree_util.tree_map(lambda a: a.reshape((n_chains, n_draws) + a.shape[1:]), draws)
