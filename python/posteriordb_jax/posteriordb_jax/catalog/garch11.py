"""garch11.stan (GARCH(1,1) volatility model):

    parameters { real mu; real<lower=0> alpha0; real<lower=0, upper=1> alpha1; real<lower=0, upper=(1 - alpha1)> beta1; }
    model {
      sigma[1] = sigma1;
      sigma[t] = sqrt(alpha0 + alpha1 * square(y[t - 1] - mu) + beta1 * square(sigma[t - 1]));
      y ~ normal(mu, sigma);
    }

beta1's upper bound depends on alpha1, the only parameter-dependent bound in
posteriordb, so the parameters get their own pattern (`Garch11Pattern`)
rather than a PatternDict. Flat priors. A plain Model: the volatility
recursion couples the observations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import stats

import modelgami as mg
import paragami as pg

STAN_MODEL = "garch11"
POSTERIORS = ["garch-garch11"]

NAMES = ("mu", "alpha0", "alpha1", "beta1")


class Garch11Pattern(pg.Pattern):
    """{mu, alpha0 > 0, 0 < alpha1 < 1, 0 < beta1 < 1 - alpha1}, with Stan's
    transforms in Stan's order: mu = u0, alpha0 = exp(u1),
    alpha1 = inv_logit(u2), beta1 = (1 - alpha1) * inv_logit(u3)."""

    def __init__(self):
        super().__init__(flat_length=4, free_length=4)

    def to_dict(self):
        return {}

    def generate_code(self, *, module="pg", indent=4):
        raise NotImplementedError("Garch11Pattern is defined in posteriordb_jax.catalog.garch11")

    def _names(self, free):
        return list(NAMES)

    def _flat_to_folded(self, flat_val):
        return {name: flat_val[i] for i, name in enumerate(NAMES)}

    def _folded_to_flat(self, folded_val):
        return jnp.stack([jnp.asarray(folded_val[name]) for name in NAMES])

    def _free_to_folded(self, free_val):
        alpha1 = jax.nn.sigmoid(free_val[2])
        return {"mu": free_val[0], "alpha0": jnp.exp(free_val[1]), "alpha1": alpha1,
                "beta1": (1.0 - alpha1) * jax.nn.sigmoid(free_val[3])}

    def _free_to_folded_log_det(self, free_val):
        u1, u2, u3 = free_val[1], free_val[2], free_val[3]
        # d alpha1 / d u2 = alpha1 (1 - alpha1); d beta1 / d u3 = (1 - alpha1) beta1' with
        # beta1' = inv_logit(u3) (1 - inv_logit(u3)); 1 - alpha1 = inv_logit(-u2). The
        # Jacobian is triangular, so its log det is the sum of these.
        return (u1 + jax.nn.log_sigmoid(u2) + jax.nn.log_sigmoid(-u2)
                + jax.nn.log_sigmoid(-u2) + jax.nn.log_sigmoid(u3) + jax.nn.log_sigmoid(-u3))

    def _folded_to_free(self, folded_val):
        alpha1 = jnp.asarray(folded_val["alpha1"])
        beta1_share = folded_val["beta1"] / (1.0 - alpha1)
        return jnp.stack([jnp.asarray(folded_val["mu"]), jnp.log(folded_val["alpha0"]),
                          jnp.log(alpha1) - jnp.log1p(-alpha1), jnp.log(beta1_share) - jnp.log1p(-beta1_share)])

    def _validate(self, folded_val, validate_value):
        if not isinstance(folded_val, dict) or set(folded_val) != set(NAMES):
            return f"expected a dict with keys {NAMES}"
        if not validate_value:
            return None
        values = {name: float(np.asarray(folded_val[name])) for name in NAMES}
        if not values["alpha0"] > 0.0:
            return "alpha0 must be > 0"
        if not 0.0 < values["alpha1"] < 1.0:
            return "alpha1 must be in (0, 1)"
        if not 0.0 < values["beta1"] < 1.0 - values["alpha1"]:
            return "beta1 must be in (0, 1 - alpha1)"
        return None


def eval_log_lik(data, params, hyper):
    y, mu = data["y"], params["mu"]

    def step(sigma_prev, y_prev):
        sigma = jnp.sqrt(params["alpha0"] + params["alpha1"] * (y_prev - mu) ** 2 + params["beta1"] * sigma_prev**2)
        return sigma, sigma

    _, sigma_rest = jax.lax.scan(step, hyper["sigma1"], y[:-1])
    sigma = jnp.concatenate([jnp.reshape(hyper["sigma1"], (1,)), sigma_rest])
    return jnp.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))


def build(raw):
    return mg.Model(
        name=STAN_MODEL,
        params_pattern=Garch11Pattern(),
        prior=None,
        likelihood=mg.DGP(eval_log_density=eval_log_lik),
        data={"y": raw["y"]},
        hyper={"sigma1": raw["sigma1"]},
    )
