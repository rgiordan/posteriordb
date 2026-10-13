"""NUTS with BlackJAX's window adaptation, on a modelgami model's free
(unconstrained) parameters, optionally with weights on an exchangeable
model's observations (for the nonparametric bootstrap). Needs blackjax."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import paragami as pg

import modelgami as mg


@dataclass
class NutsResult:
    draws: Dict[str, Any]       # folded draws, each with leading (chain, draw) axes
    free_draws: np.ndarray      # (chain, draw, free dimension)
    n_divergent: int
    step_size: np.ndarray       # adapted step size, per chain
    seconds: float              # sampling time, including compilation on a sampler's first call


def build_nuts_sampler(*, model: mg.Model, n_chains: int = 4, n_warmup: int = 1000, n_draws: int = 1000,
                 weighted: bool = False, target_acceptance_rate: float = 0.8) -> Callable:
    """A compiled `draw_samples(key, weights) -> NutsResult` that runs `n_chains`
    independent NUTS chains, each initialized at a random draw from the
    parameters' pattern. With `weighted`, `weights` (one per observation of
    an `ExchangeableModel`) multiply the pointwise log likelihood; otherwise
    pass weights=None. Build it once and call it for each bootstrap
    replicate, so the sampler compiles only once."""
    if weighted and not isinstance(model, mg.ExchangeableModel):
        raise ValueError("weights need an ExchangeableModel")
    def run_chain(chain_key, weights):
        if weighted:
            logdensity = lambda x: model.eval_free_log_joint(x, weights=weights)
        else:
            logdensity = model.eval_free_log_joint
        init_key, warmup_key, sample_key = jax.random.split(chain_key, 3)
        x_init = model.params_pattern.folded_to_free(model.params_pattern.draw_random(init_key))
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, target_acceptance_rate=target_acceptance_rate)
        (state, tuned), _ = warmup.run(warmup_key, x_init, num_steps=n_warmup)
        step = blackjax.nuts(logdensity, **tuned).step

        def take_step(state, step_key):
            state, info = step(step_key, state)
            return state, (state.position, info.is_divergent)

        _, (positions, divergent) = jax.lax.scan(take_step, state, jax.random.split(sample_key, n_draws))
        return positions, divergent, tuned["step_size"]

    run_chains = jax.jit(jax.vmap(run_chain, in_axes=(0, None)))

    def draw_samples(key, weights=None) -> NutsResult:
        start = time.perf_counter()
        free_draws, divergent, step_size = run_chains(jax.random.split(key, n_chains), weights)
        free_draws = np.asarray(free_draws)
        seconds = time.perf_counter() - start
        draws = jax.tree_util.tree_map(np.asarray, pg.free_draws_to_folded(pattern=model.params_pattern,
                                                                           draws=free_draws))
        return NutsResult(draws=draws, free_draws=free_draws, n_divergent=int(np.sum(divergent)),
                          step_size=np.asarray(step_size), seconds=seconds)

    return draw_samples


def run_nuts(*, model: mg.Model, key, n_chains: int = 4, n_warmup: int = 1000, n_draws: int = 1000,
         weights: Optional[jnp.ndarray] = None, target_acceptance_rate: float = 0.8) -> NutsResult:
    """One run of `build_nuts_sampler`."""
    draw_samples = build_nuts_sampler(model=model, n_chains=n_chains, n_warmup=n_warmup, n_draws=n_draws,
                          weighted=weights is not None, target_acceptance_rate=target_acceptance_rate)
    return draw_samples(key, weights)
