"""hmm_drive_1.stan (basketball drive hidden Markov model, normal emissions
with known standard deviations tau and rho):

    parameters { simplex[K] theta1; simplex[K] theta2; ordered[K] phi; ordered[K] lambda; }
    model {
      theta[k] ~ dirichlet(alpha[k]');
      phi[1] ~ normal(0, 1); phi[2] ~ normal(3, 1); lambda[1] ~ normal(0, 1); lambda[2] ~ normal(3, 1);
      gamma[t, k] = log_sum_exp_j(gamma[t - 1, j] + log(theta[j, k])
                                  + normal_lpdf(u[t] | phi[k], tau) + normal_lpdf(v[t] | lambda[k], rho));
      target += log_sum_exp(gamma[N]);
    }

A plain Model: the forward recursion couples the observations.
"""

from jax.scipy import stats

from ._hmm_drive import build_drive_model, eval_log_lik_from_emission

STAN_MODEL = "hmm_drive_1"
POSTERIORS = ["bball_drive_event_1-hmm_drive_1"]


def eval_log_lik(data, params, hyper):
    log_emission = (stats.norm.logpdf(data["u"][:, None], loc=params["phi"][None, :], scale=hyper["tau"])
                    + stats.norm.logpdf(data["v"][:, None], loc=params["lambda"][None, :], scale=hyper["rho"]))
    return eval_log_lik_from_emission(params=params, log_emission=log_emission)


def build(raw):
    return build_drive_model(name=STAN_MODEL, raw=raw, eval_log_lik=eval_log_lik, positive=False,
                       hyper={"tau": raw["tau"], "rho": raw["rho"]})
