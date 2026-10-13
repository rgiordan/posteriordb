"""ldaK5.stan (latent Dirichlet allocation with K = 5 topics; the Dirichlet
parameters alpha and beta come with the data):

    parameters { array[M] simplex[5] theta; array[5] simplex[V] phi; }
    model {
      theta[m] ~ dirichlet(alpha);
      phi[k] ~ dirichlet(beta);
      target += log_sum_exp_k(log(theta[doc[n], k]) + log(phi[k, w[n]]));
    }

Exchangeable: each word token is one datum carrying its word and document.
"""

from ._lda import build_lda_model

STAN_MODEL = "ldaK5"
POSTERIORS = ["prideprejudice_chapter-ldaK5", "prideprejudice_paragraph-ldaK5"]


def build(raw):
    return build_lda_model(name=STAN_MODEL, raw=raw, n_topics=5, alpha=raw["alpha"], beta=raw["beta"])
