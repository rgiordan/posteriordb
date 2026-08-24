"""JAX/TFP translation of posterior_database/models/stan/bones_model.stan:

    data {
      array[nInd, 4] real gamma;   // -1 if missing/unused
      array[nInd] real delta;
      array[nInd] int<lower=0> ncat;
      array[nChild, nInd] int grade;  // -1 if missing
    }
    parameters { array[nChild] real theta; }
    model {
      theta ~ normal(0.0, 36);
      for (i in 1:nChild) {
        for (j in 1:nInd) {
          for (k in 1:(ncat[j]-1)) Q[i,j,k] = inv_logit(delta[j] * (theta[i] - gamma[j,k]));
          p[i,j,1] = 1 - Q[i,j,1];
          for (k in 2:(ncat[j]-1)) p[i,j,k] = Q[i,j,k-1] - Q[i,j,k];
          p[i,j,ncat[j]] = Q[i,j,ncat[j]-1];
          if (grade[i,j] != -1) target += log(p[i,j,grade[i,j]]);
        }
      }
    }

A ragged ordinal/graded-response model: each indicator `j` has its own
number of valid categories `ncat[j]` (2..5 in this dataset) and its own
cutpoints `gamma[j, 1:ncat[j]-1]` (`gamma[j,k] == -1` for `k >= ncat[j]-1`
is an unused sentinel slot, not data); `grade[i,j] == -1` marks a missing
response, contributing nothing to the log density (not `log(0)`).

`ncat`/`gamma`/`grade` are pure data (never touched by `grad_wrt_data` in
practice -- `ncat`/`grade` are integer-dtype, and even `gamma`'s -1
sentinels aren't differentiated against anything), but the *shape* of the
per-(i,j) computation is still built entirely with `jnp` array ops (masks
via `jnp.where` over the actual `ncat`/`grade` values), not Python-level
branching on traced values -- consistent with every other model here.

The `Q[i,j,k]` -> `p[i,j,k]` cumulative-probability construction is exactly
the standard "graded response" decomposition `p[k] = Q[k-1] - Q[k]` with
boundary values `Q[0] = 1` and `Q[ncat[j]] = 0` (Stan's `p[i,j,1] = 1 -
Q[i,j,1]` and `p[i,j,ncat[j]] = Q[i,j,ncat[j]-1]` are exactly this
boundary rule spelled out, since Stan's fixed-size `Q` array has no slot to
literally hold `Q[0]` or `Q[ncat[j]]`). Reproduced here as one vectorized
`Qext` array (indices `0..5`, covering `ncat[j] in 2..5`) so `p` at any
valid `(j, k)` is a uniform `Qext[k-1] - Qext[k]` -- no per-`ncat`
special-casing needed. The final `log(p[i,j,grade[i,j]])` term gathers `p`
at `grade[i,j]` (clamped to a safe in-range index when missing) *before*
`log`, then masks the *output* with `grade[i,j] != -1` -- gathering is
always in-range here (unlike `Survey_model`'s genuinely invalid `n < k`
domain), so there is no analogous invalid-input-to-`log` risk to guard
against with an input-side `jnp.where`.
"""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..constraints import ConstraintType, ParameterSpec
from ..model import StatisticalModel

tfd = tfp.distributions

_MAX_CAT = 5  # matches Stan's fixed-size `array[nChild, nInd, 5] real p;`


class BonesModel(StatisticalModel):
    def param_specs(self, data) -> List[ParameterSpec]:
        n_child = int(data["nChild"])
        return [ParameterSpec("theta", (n_child,), ConstraintType.UNCONSTRAINED)]

    def log_density(self, params: Dict[str, jnp.ndarray], data) -> jnp.ndarray:
        theta = params["theta"]  # (nChild,)

        gamma = jnp.asarray(data["gamma"], dtype=jnp.float64)  # (nInd, 4)
        delta = jnp.asarray(data["delta"], dtype=jnp.float64)  # (nInd,)
        ncat = jnp.asarray(data["ncat"])  # (nInd,) int
        grade = jnp.asarray(data["grade"])  # (nChild, nInd) int

        # Q[i,j,k] for k = 1..4 (Stan's 1-indexed cutpoints), computed for
        # every slot -- including unused (`gamma[j,k] == -1`) ones, which
        # are simply never selected downstream. inv_logit is smooth
        # everywhere, so no NaN-gradient risk from evaluating it there.
        q = jax.nn.sigmoid(delta[None, :, None] * (theta[:, None, None] - gamma[None, :, :]))  # (nChild, nInd, 4)

        # Qext[..., k] for k = 0..5: Qext[0] = 1, Qext[k] = Q[k] for
        # 1 <= k <= ncat[j]-1, Qext[k] = 0 for k >= ncat[j] (boundary,
        # including k == ncat[j]). This makes p[k] = Qext[k-1] - Qext[k]
        # (k = 1..ncat[j]) exactly Stan's piecewise construction.
        n_child, n_ind = grade.shape
        dummy0 = jnp.ones((n_child, n_ind, 1), dtype=q.dtype)
        dummy5 = jnp.zeros((n_child, n_ind, 1), dtype=q.dtype)
        q_padded = jnp.concatenate([dummy0, q, dummy5], axis=-1)  # (nChild, nInd, 6): slots 0..5
        idx = jnp.arange(6)
        qext = jnp.where(
            idx[None, None, :] == 0,
            1.0,
            jnp.where(idx[None, None, :] < ncat[None, :, None], q_padded, 0.0),
        )  # (nChild, nInd, 6)

        p_full = qext[..., :-1] - qext[..., 1:]  # (nChild, nInd, 5): p_full[...,m] == Stan's p[...,m+1]

        mask = grade >= 1  # (nChild, nInd) -- False where grade == -1 (missing)
        safe_idx = jnp.where(mask, grade - 1, 0)  # in-range gather index
        p_sel = jnp.take_along_axis(p_full, safe_idx[..., None], axis=-1)[..., 0]  # (nChild, nInd)

        lp_lik = jnp.sum(jnp.where(mask, jnp.log(p_sel), 0.0))
        lp = jnp.sum(tfd.Normal(loc=0.0, scale=36.0).log_prob(theta)) + lp_lik
        return lp
