"""Stan distributions that jax.scipy.stats lacks in the parameterization the
posteriordb models use."""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def binomial_logit_logpmf(k, *, n, logit_p):
    """Stan's binomial_logit_lpmf(k | n, logit_p), elementwise, without forming p."""
    log_choose = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
    return log_choose + k * logit_p - n * jax.nn.softplus(logit_p)


def inv_gamma_logpdf(x, *, a, b):
    """Stan's inv_gamma_lpdf(x | a, b) (shape a, scale b)."""
    return a * jnp.log(b) - gammaln(a) - (a + 1.0) * jnp.log(x) - b / x


def lkj_corr_cholesky_logpdf(cholesky, *, eta):
    """Stan's lkj_corr_cholesky_lpdf(L | eta), with its normalizing constant
    (Lewandowski, Kurowicka and Joe 2009)."""
    k = cholesky.shape[-1]
    i = jnp.arange(k)
    unnormalized = jnp.sum((k - 1 - i + 2.0 * (eta - 1.0)) * jnp.log(jnp.diagonal(cholesky)))
    j = jnp.arange(1, k)
    effective_eta = eta + (k - 1 - j) / 2.0
    log_normalizer = (jnp.sum(gammaln(effective_eta) - gammaln(j / 2.0 + effective_eta))
                      + jnp.log(jnp.pi) * k * (k - 1) / 4.0)
    return unnormalized - log_normalizer
