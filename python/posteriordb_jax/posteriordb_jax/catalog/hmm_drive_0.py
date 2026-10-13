"""hmm_drive_0.stan (basketball drive hidden Markov model, exponential
emissions):

    parameters { simplex[K] theta1; simplex[K] theta2; positive_ordered[K] phi; positive_ordered[K] lambda; }
    model {
      theta[k] ~ dirichlet(alpha[k]');
      phi[1] ~ normal(0, 1); phi[2] ~ normal(3, 1); lambda[1] ~ normal(0, 1); lambda[2] ~ normal(3, 1);
      gamma[t, k] = log_sum_exp_j(gamma[t - 1, j] + log(theta[j, k])
                                  + exponential_lpdf(u[t] | phi[k]) + exponential_lpdf(v[t] | lambda[k]));
      target += log_sum_exp(gamma[N]);
    }

A plain Model: the forward recursion couples the observations.
"""

from jax.scipy import stats

from ._hmm_drive import drive_model, log_lik_from_emission

STAN_MODEL = "hmm_drive_0"
POSTERIORS = ["bball_drive_event_0-hmm_drive_0"]


def log_lik(params, data, hyper):
    log_emission = (stats.expon.logpdf(data["u"][:, None], scale=1.0 / params["phi"][None, :])
                    + stats.expon.logpdf(data["v"][:, None], scale=1.0 / params["lambda"][None, :]))
    return log_lik_from_emission(params=params, log_emission=log_emission)


def build(raw):
    return drive_model(name=STAN_MODEL, raw=raw, log_lik=log_lik, positive=True, hyper={})
