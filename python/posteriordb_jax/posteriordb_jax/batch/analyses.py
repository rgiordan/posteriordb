"""Ready-made analyses, each `analysis(entry, out_dir)` once its options are
bound with `functools.partial`:

- `run_mcmc`: NUTS on the posterior.
- `run_bootstrap_mcmc`: NUTS on bootstrap-weighted posteriors, with the
  classic nonparametric bootstrap's weights or ones from a function you
  supply.

Both write, per run, `draws.npz` (folded draws with leading (chain, draw)
axes, one array per parameter, nested names joined with "/") and
`summary.json` (posterior means and sds, divergences, step sizes, run time,
and, for `run_mcmc` when posteriordb has reference draws of the parameters,
the largest |mean - reference mean| / reference sd). Needs blackjax.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

import modelgami as mg

from .entry import Entry
from .mcmc import NutsResult, build_nuts_sampler, run_nuts
from .runner import NotApplicable


def flatten_draws(draws, prefix: str = "") -> Dict[str, np.ndarray]:
    """{"a": {"b": x}} -> {"a/b": x}."""
    if isinstance(draws, dict):
        flat = {}
        for key, value in draws.items():
            flat.update(flatten_draws(value, prefix=f"{prefix}{key}/"))
        return flat
    return {prefix.rstrip("/"): np.asarray(draws)}


def compute_reference_discrepancy(*, draws: Dict[str, np.ndarray], reference: Dict[str, np.ndarray]) -> float:
    """max over parameters and entries of |mean - reference mean| / reference sd."""
    worst = 0.0
    for name, ref in reference.items():
        if name not in draws:
            continue
        ref_sd = ref.std(axis=(0, 1))
        diff = np.abs(draws[name].mean(axis=(0, 1)) - ref.mean(axis=(0, 1)))
        worst = max(worst, float(np.max(np.where(ref_sd > 0, diff / np.where(ref_sd > 0, ref_sd, 1.0), 0.0))))
    return worst


def write_mcmc_output(*, out_dir: Path, result: NutsResult, extra: Optional[dict] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    draws = flatten_draws(result.draws)
    np.savez_compressed(out_dir / "draws.npz", **draws)
    summary = {
        "mean": {name: value.mean(axis=(0, 1)).tolist() for name, value in draws.items()},
        "sd": {name: value.std(axis=(0, 1)).tolist() for name, value in draws.items()},
        "n_chains": int(result.free_draws.shape[0]),
        "n_draws": int(result.free_draws.shape[1]),
        "n_divergent": result.n_divergent,
        "step_size": result.step_size.tolist(),
        "seconds": round(result.seconds, 3),
        **(extra or {}),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def run_mcmc(entry: Entry, out_dir: Path, *, seed: int = 0, n_chains: int = 4, n_warmup: int = 1000,
         n_draws: int = 1000) -> None:
    """NUTS on the posterior; writes draws.npz and summary.json to out_dir."""
    result = run_nuts(model=entry.model, key=jax.random.PRNGKey(seed), n_chains=n_chains, n_warmup=n_warmup,
                  n_draws=n_draws)
    extra = {}
    if entry.has_reference_draws:
        try:
            reference = entry.load_reference_draws()
        except ValueError as error:
            # Some reference draws hold only transformed parameters (e.g.
            # eight_schools_noncentered's theta, not theta_trans).
            extra["reference_comparison_skipped"] = str(error)
        else:
            extra["max_z_vs_reference"] = compute_reference_discrepancy(
                draws=flatten_draws(result.draws), reference=flatten_draws(reference))
    write_mcmc_output(out_dir=Path(out_dir), result=result, extra=extra)


def run_bootstrap_mcmc(entry: Entry, out_dir: Path, *, draw_weights: Optional[Callable[[Entry], np.ndarray]] = None,
                       n_replicates: int = 1, seed: int = 0, n_chains: int = 4, n_warmup: int = 1000,
                       n_draws: int = 1000) -> None:
    """NUTS on the posterior with each observation's log likelihood
    multiplied by a weight, once per bootstrap replicate. The weights are
    `draw_weights(entry)` (shape (n_obs,) or (n_replicates, n_obs)) or, by
    default, `n_replicates` draws of `mg.draw_np_bootstrap_weights` (the
    classic nonparametric bootstrap). Replicate r goes to
    out_dir/replicate_<r>/ with its weights in weights.npy. Not applicable
    to models whose data are not exchangeable."""
    if not entry.is_exchangeable:
        raise NotApplicable(f"{entry.name}: the data are not exchangeable observations, so there is nothing to "
                            "weight")
    n_obs = entry.model.n_obs
    if draw_weights is None:
        keys = jax.random.split(jax.random.PRNGKey(seed), n_replicates)
        all_weights = np.stack([np.asarray(mg.draw_np_bootstrap_weights(key, n_obs)) for key in keys])
    else:
        all_weights = np.atleast_2d(np.asarray(draw_weights(entry), dtype=float))
    if all_weights.ndim != 2 or all_weights.shape[1] != n_obs:
        raise ValueError(f"{entry.name}: weights must have shape (n_obs,) or (n_replicates, n_obs) with "
                         f"n_obs = {n_obs}, got {all_weights.shape}")
    draw_samples = build_nuts_sampler(model=entry.model, n_chains=n_chains, n_warmup=n_warmup, n_draws=n_draws, weighted=True)
    for replicate, replicate_weights in enumerate(all_weights):
        result = draw_samples(jax.random.fold_in(jax.random.PRNGKey(seed), replicate), jnp.asarray(replicate_weights))
        replicate_dir = Path(out_dir) / f"replicate_{replicate:03d}"
        write_mcmc_output(out_dir=replicate_dir, result=result)
        np.save(replicate_dir / "weights.npy", replicate_weights)
