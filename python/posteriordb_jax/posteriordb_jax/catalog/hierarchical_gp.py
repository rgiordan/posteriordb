"""hierarchical_gp.stan (state-level presidential vote shares with year,
region and state effects plus long- and short-range Gaussian processes over
years for each region and state):

    parameters {
      matrix[N_years, N_regions] GP_region_std; matrix[N_years, N_states] GP_state_std;
      vector[N_years_obs] year_std; vector[N_states] state_std; vector[N_regions] region_std;
      real<lower=0> tot_var; simplex[17] prop_var; real mu;
      real<lower=0> length_GP_region_long, length_GP_state_long, length_GP_region_short, length_GP_state_short;
    }
    transformed parameters {
      vars = 17 * prop_var * tot_var;          // variances of the 17 components
      year_re = sqrt(vars[1]) * year_std;  region_re = sqrt(vars[2]) * region_std;
      state_re = sqrt(vars[3:12])[state_region_ind] .* state_std;
      GP_region = cholesky(K(sqrt(vars[13]), length_GP_region_long) + K(sqrt(vars[15]), length_GP_region_short) + 1e-6 I) * GP_region_std;
      GP_state  = cholesky(K(sqrt(vars[14]), length_GP_state_long)  + K(sqrt(vars[16]), length_GP_state_short)  + 1e-6 I) * GP_state_std;
    }
    model {
      y ~ normal(mu + year_re[year_ind] + state_re[state_ind] + region_re[region_ind]
                 + GP_region[year_ind, region_ind] + GP_state[year_ind, state_ind], sqrt(vars[17]));
      GP_region_std, GP_state_std, year_std, state_std, region_std ~ normal(0, 1);
      mu ~ normal(.5, .5);  tot_var ~ gamma(3, 3);  prop_var ~ dirichlet(rep_vector(2, 17));
      length_GP_*_long ~ weibull(30, 8);  length_GP_*_short ~ weibull(30, 3);
    }

K(s, l) is gp_exp_quad_cov over years 1..N_years. Exchangeable: each
observation carries its state, region and year indices.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

import modelgami as mg
import paragami as pg

from ._gp import compute_exp_quad_cov

STAN_MODEL = "hierarchical_gp"
POSTERIORS = ["state_wide_presidential_votes-hierarchical_gp"]

LENGTH_SCALES = ("length_GP_region_long", "length_GP_state_long", "length_GP_region_short", "length_GP_state_short")


def eval_weibull_logpdf(x, *, alpha, sigma):
    return jnp.log(alpha / sigma) + (alpha - 1.0) * jnp.log(x / sigma) - (x / sigma) ** alpha


def eval_log_prior(params, hyper):
    standard_normals = sum(jnp.sum(stats.norm.logpdf(params[field], loc=0.0, scale=1.0))
                           for field in ("GP_region_std", "GP_state_std", "year_std", "state_std", "region_std"))
    lengths = sum(eval_weibull_logpdf(params[field], alpha=30.0, sigma=8.0 if field.endswith("long") else 3.0)
                  for field in LENGTH_SCALES)
    return (standard_normals
            + stats.norm.logpdf(params["mu"], loc=0.5, scale=0.5)
            + stats.gamma.logpdf(params["tot_var"], a=3.0, scale=1.0 / 3.0)
            + stats.dirichlet.logpdf(params["prop_var"], jnp.full(17, 2.0))
            + lengths)


def compute_gp(*, sd_long, length_long, sd_short, length_short, std):
    years = jnp.arange(1.0, std.shape[0] + 1.0)
    cov = (compute_exp_quad_cov(x=years, alpha=sd_long, rho=length_long)
           + compute_exp_quad_cov(x=years, alpha=sd_short, rho=length_short)
           + 1e-6 * jnp.eye(std.shape[0]))
    return jnp.linalg.cholesky(cov) @ std


def compute_mean_and_sd(*, params, datum, hyper):
    sds = jnp.sqrt(17.0 * params["prop_var"] * params["tot_var"])
    year, state, region = datum["year_ind"], datum["state_ind"], datum["region_ind"]
    gp_region = compute_gp(sd_long=sds[12], length_long=params["length_GP_region_long"], sd_short=sds[14],
                   length_short=params["length_GP_region_short"], std=params["GP_region_std"][:, region])
    gp_state = compute_gp(sd_long=sds[13], length_long=params["length_GP_state_long"], sd_short=sds[15],
                  length_short=params["length_GP_state_short"], std=params["GP_state_std"][:, state])
    mean = (params["mu"] + sds[0] * params["year_std"][year]
            + sds[2 + hyper["state_region_ind"][state]] * params["state_std"][state]
            + sds[1] * params["region_std"][region]
            + gp_region[year] + gp_state[year])
    return mean, sds[16]


def eval_log_lik_one(datum, params, hyper):
    mean, sd = compute_mean_and_sd(params=params, datum=datum, hyper=hyper)
    return stats.norm.logpdf(datum["y"], loc=mean, scale=sd)


def draw_one(key, datum, params, hyper):
    mean, sd = compute_mean_and_sd(params=params, datum=datum, hyper=hyper)
    return {"y": mean + sd * jax.random.normal(key)}


def build(raw):
    n_years, n_states, n_regions = int(raw["N_years"]), int(raw["N_states"]), int(raw["N_regions"])
    return mg.ExchangeableModel(
        name=STAN_MODEL,
        params_pattern=pg.PatternDict(
            GP_region_std=pg.NumericArrayPattern((n_years, n_regions), order="F"),
            GP_state_std=pg.NumericArrayPattern((n_years, n_states), order="F"),
            year_std=pg.NumericVectorPattern(int(raw["N_years_obs"])),
            state_std=pg.NumericVectorPattern(n_states),
            region_std=pg.NumericVectorPattern(n_regions),
            tot_var=pg.NumericScalarPattern(lb=0.0),
            prop_var=pg.SimplexArrayPattern(17),
            mu=pg.NumericScalarPattern(),
            **{field: pg.NumericScalarPattern(lb=0.0) for field in LENGTH_SCALES},
        ),
        prior=mg.DGP(eval_log_density=eval_log_prior),
        likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
        data={
            "y": raw["y"],
            "state_ind": jnp.asarray(raw["state_ind"], dtype=jnp.int32) - 1,
            "region_ind": jnp.asarray(raw["region_ind"], dtype=jnp.int32) - 1,
            "year_ind": jnp.asarray(raw["year_ind"], dtype=jnp.int32) - 1,
        },
        hyper={"state_region_ind": jnp.asarray(raw["state_region_ind"], dtype=jnp.int32) - 1},
    )
