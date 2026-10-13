"""covid19imperial_v2.stan and covid19imperial_v3.stan (Imperial College's
early COVID-19 model of deaths in European countries; the two Stan files
are identical):

    parameters {
      array[M] real<lower=0> mu; array[P] real<lower=0> alpha_hier; real<lower=0> kappa;
      array[M] real<lower=0> y; real<lower=0> phi; real<lower=0> tau; array[M] real<lower=0> ifr_noise;
    }
    transformed parameters {
      alpha = alpha_hier - log(1.05) / 6;
      prediction[1:N0, m] = y[m];                       // seeded infections
      Rt[:, m] = mu[m] * exp(-X[m] * alpha);
      for i > N0:                                       // renewal equation with depletion
        prediction[i, m] = (pop[m] - sum(prediction[1:i-1, m])) / pop[m] * Rt[i, m]
                           * sum_{j<i} prediction[j, m] * SI[i - j];
      E_deaths[1, m] = 1e-15 * prediction[1, m];
      E_deaths[i, m] = ifr_noise[m] * sum_{j<i} prediction[j, m] * f[i - j, m];
    }
    model {
      tau ~ exponential(0.03); y[m] ~ exponential(1 / tau); phi ~ normal(0, 5); kappa ~ normal(0, 0.5);
      mu ~ normal(3.28, kappa); alpha_hier ~ gamma(.1667, 1); ifr_noise ~ normal(1, 0.1);
      deaths[EpidemicStart[m]:N[m], m] ~ neg_binomial_2(E_deaths[EpidemicStart[m]:N[m], m], phi);
    }

(For i > N0 Stan's running cumm_sum equals sum(prediction[1:i-1, m]).) A
plain Model: the renewal equation couples each country's days.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import stats
from jax.scipy.special import gammaln

import modelgami as mg
import paragami as pg


def eval_neg_binomial_2_logpmf(k, *, mu, phi):
    """Stan's neg_binomial_2_lpmf(k | mu, phi), elementwise."""
    return (gammaln(k + phi) - gammaln(k + 1.0) - gammaln(phi)
            + phi * (jnp.log(phi) - jnp.log(mu + phi)) + k * (jnp.log(mu) - jnp.log(mu + phi)))


def build_lag_matrix(kernel):
    """W[i, j] = kernel[i - j - 1] for j < i, else 0 (0-based), so that
    (W @ x)[i] = sum_{j<i} x[j] * kernel[i - j] in Stan's 1-based indexing."""
    n = kernel.shape[0]
    lag = np.arange(n)[:, None] - np.arange(n)[None, :] - 1
    return np.where(lag >= 0, np.asarray(kernel)[np.clip(lag, 0, None)], 0.0)


def eval_log_prior(params, hyper):
    tau = params["tau"]
    return (stats.expon.logpdf(tau, scale=1.0 / 0.03)
            + jnp.sum(stats.expon.logpdf(params["y"], scale=tau))
            + stats.norm.logpdf(params["phi"], loc=0.0, scale=5.0)
            + stats.norm.logpdf(params["kappa"], loc=0.0, scale=0.5)
            + jnp.sum(stats.norm.logpdf(params["mu"], loc=3.28, scale=params["kappa"]))
            + jnp.sum(stats.gamma.logpdf(params["alpha_hier"], a=0.1667, scale=1.0))
            + jnp.sum(stats.norm.logpdf(params["ifr_noise"], loc=1.0, scale=0.1)))


def compute_expected_deaths(*, params, data, hyper):
    """E_deaths, (M, N2)."""
    alpha = params["alpha_hier"] - jnp.log(1.05) / 6.0
    si_lags = hyper["si_lags"]
    n_days = si_lags.shape[0]
    n_seeded = hyper["seeded"].shape[0]

    def one_country(mu, y, ifr_noise, x, pop, f_lags):
        rt = mu * jnp.exp(-x @ alpha)
        seeded = jnp.where(jnp.arange(n_days) < n_seeded, y, 0.0)

        def step(prediction, i):
            convolution = si_lags[i] @ prediction
            new = (pop - jnp.sum(prediction)) / pop * rt[i] * convolution
            return prediction.at[i].set(new), None

        prediction, _ = jax.lax.scan(step, seeded, jnp.arange(n_seeded, n_days))
        e_deaths = ifr_noise * (f_lags @ prediction)
        return e_deaths.at[0].set(1e-15 * prediction[0])

    return jax.vmap(one_country)(params["mu"], params["y"], params["ifr_noise"], data["X"], hyper["pop"],
                                 hyper["f_lags"])


def eval_log_lik(data, params, hyper):
    e_deaths = compute_expected_deaths(params=params, data=data, hyper=hyper)
    terms = eval_neg_binomial_2_logpmf(data["deaths"], mu=jnp.where(data["observed"], e_deaths, 1.0), phi=params["phi"])
    return jnp.sum(jnp.where(data["observed"], terms, 0.0))


def build_covid_model(*, name, raw):
    n_countries, n_days, n_covariates = int(raw["M"]), int(raw["N2"]), int(raw["P"])
    deaths = np.asarray(raw["deaths"], dtype=float).T                         # (M, N2)
    day = np.arange(1, n_days + 1)
    observed = ((day[None, :] >= np.asarray(raw["EpidemicStart"])[:, None])
                & (day[None, :] <= np.asarray(raw["N"])[:, None]))
    return mg.Model(
        name=name,
        params_pattern=pg.PatternDict(
            mu=pg.NumericVectorPattern(n_countries, lb=0.0),
            alpha_hier=pg.NumericVectorPattern(n_covariates, lb=0.0),
            kappa=pg.NumericScalarPattern(lb=0.0),
            y=pg.NumericVectorPattern(n_countries, lb=0.0),
            phi=pg.NumericScalarPattern(lb=0.0),
            tau=pg.NumericScalarPattern(lb=0.0),
            ifr_noise=pg.NumericVectorPattern(n_countries, lb=0.0),
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={
            "deaths": jnp.asarray(np.where(observed, deaths, 0.0)),   # unobserved days hold -1 in posteriordb
            "observed": jnp.asarray(observed),
            "X": jnp.asarray(raw["X"]),
        },
        hyper={
            "pop": jnp.asarray(raw["pop"]),
            "si_lags": jnp.asarray(build_lag_matrix(np.asarray(raw["SI"]))),
            "f_lags": jnp.asarray(np.stack([build_lag_matrix(column) for column in np.asarray(raw["f"]).T])),
            "seeded": jnp.zeros(int(raw["N0"])),
        },
    )
