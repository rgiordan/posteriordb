"""kronecker_gp.stan (Gaussian process on a grid with a Kronecker-structured
covariance: an RBF kernel over the n1 locations times a free n2 x n2
correlation matrix, plus white noise):

    parameters { real<lower=0> var1; real<lower=0> bw1; cholesky_factor_corr[n2] L; real<lower=0.00001> sigma1; }
    transformed parameters {
      Sigma1 = var1 * exp(-bw1 * (x1[i] - x1[j])^2) + 1e-5 I;    (Q1, R1) = eigen(Sigma1);
      Lambda = L * L';                                          (Q2, R2) = eigen(Lambda);
      eigenvalues[i, j] = R2[i] * R1[j] + sigma1;
    }
    model {
      var1 ~ lognormal(0, 1); bw1 ~ cauchy(0, 2.5); sigma1 ~ lognormal(0, 1); L ~ lkj_corr_cholesky(2);
      target += -0.5 * sum(y .* (Q2 * ((Q2' * y * Q1) ./ eigenvalues) * Q1')) - 0.5 * sum(log(eigenvalues));
    }

That is the multivariate normal log density of vec(y) with covariance
Lambda (x) Sigma1 + sigma1 I, up to its constant. The data have n1 = n2 = 30.
A plain Model: the grid's values are jointly Gaussian.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._distributions import eval_lkj_corr_cholesky_logpdf, eval_lognormal_logpdf

STAN_MODEL = "kronecker_gp"
POSTERIORS = ["synthetic_grid_RBF_kernels-kronecker_gp"]


def eval_log_prior(params, hyper):
    return (eval_lognormal_logpdf(params["var1"], mu=0.0, sigma=1.0)
            + stats.cauchy.logpdf(params["bw1"], loc=0.0, scale=2.5)
            + eval_lognormal_logpdf(params["sigma1"], mu=0.0, sigma=1.0)
            + eval_lkj_corr_cholesky_logpdf(params["L"], eta=2.0))


def eval_log_lik(data, params, hyper):
    """The multivariate normal density of vec(y) evaluated with a Cholesky
    factor of the full Lambda (x) Sigma1 + sigma1 I. Stan's eigendecomposition
    route is the same quantity, but Sigma1's eigenvalues are nearly repeated
    (many sit at the 1e-5 jitter), which costs Stan's log density about 1e-8
    relative accuracy and makes its eigenvector derivatives unreliable."""
    x1, y = data["x1"], data["y"]
    sigma1 = (params["var1"] * jnp.exp(-params["bw1"] * (x1[:, None] - x1[None, :]) ** 2)
              + 1e-5 * jnp.eye(x1.shape[0]))
    cov = jnp.kron(params["L"] @ params["L"].T, sigma1) + params["sigma1"] * jnp.eye(y.size)
    chol = jnp.linalg.cholesky(cov)
    z = jax.scipy.linalg.solve_triangular(chol, jnp.reshape(y, (-1,)), lower=True)
    return -0.5 * jnp.sum(z**2) - jnp.sum(jnp.log(jnp.diag(chol)))


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            var1=pg.NumericScalarPattern(lb=0.0),
            bw1=pg.NumericScalarPattern(lb=0.0),
            L=pg.CholeskyCorrPattern(int(raw["n2"])),
            sigma1=pg.NumericScalarPattern(lb=0.00001),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"x1": raw["x1"], "y": raw["y"]},
    )
