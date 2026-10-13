"""ldaK2.stan (latent Dirichlet allocation with K = 2 topics and flat Dirichlet priors):

    parameters { array[M] simplex[2] theta; array[2] simplex[V] phi; }
    model {
      theta[m] ~ dirichlet(rep_vector(1, 2));
      phi[k] ~ dirichlet(rep_vector(1, V));
      target += log_sum_exp_k(log(theta[doc[n], k]) + log(phi[k, w[n]]));
    }

Exchangeable: each word token is one datum carrying its word and document.
"""

import jax.numpy as jnp

from ._lda import build_lda_model

STAN_MODEL = "ldaK2"
POSTERIORS = ["three_docs1200-ldaK2", "three_men1-ldaK2", "three_men2-ldaK2", "three_men3-ldaK2"]


def build(raw):
    return build_lda_model(name=STAN_MODEL, raw=raw, n_topics=2, alpha=jnp.ones(2), beta=jnp.ones(int(raw["V"])))
