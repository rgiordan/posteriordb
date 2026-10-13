"""Example analyses for `python -m posteriordb_jax.batch`.

An analysis is a function `analysis(entry, out_dir)`; this file binds the
options of the ready-made ones with `functools.partial` and defines a custom
one. Run, for example:

    # MCMC on two posteriors, here
    python -m posteriordb_jax.batch local --analysis examples/batch_analyses.py:quick_mcmc --out results/mcmc \\
        --posteriors kidiq-kidscore_momiq eight_schools-eight_schools_noncentered

    # MCMC on every posterior, on Slurm
    python -m posteriordb_jax.batch submit --analysis examples/batch_analyses.py:full_mcmc --out results/mcmc \\
        --time 04:00:00 --mem 16G --setup "source ~/venvs/posteriordb/bin/activate"

    # 20 bootstrap replicates of every exchangeable model, on Slurm
    python -m posteriordb_jax.batch submit --analysis examples/batch_analyses.py:bootstrap_20 --out results/bootstrap \\
        --exchangeable --time 12:00:00

    # what finished, what failed (with tracebacks); then rerun only the rest
    python -m posteriordb_jax.batch status --out results/mcmc --verbose
    python -m posteriordb_jax.batch submit --analysis examples/batch_analyses.py:full_mcmc --out results/mcmc --only-unfinished
"""

import json
from functools import partial

import jax
import numpy as np

from posteriordb_jax.batch import analyses

quick_mcmc = partial(analyses.mcmc, n_chains=2, n_warmup=200, n_draws=200)
full_mcmc = partial(analyses.mcmc, n_chains=4, n_warmup=1000, n_draws=1000)

# The nonparametric bootstrap: Multinomial(n, 1/n) weights, 20 replicates.
bootstrap_20 = partial(analyses.bootstrap_mcmc,
                       weights=partial(analyses.multinomial_weights, seed=1, n_replicates=20))


def bayesian_bootstrap_weights(entry, *, seed=0, n_replicates=10):
    """Any weights work: here the Bayesian bootstrap's n * Dirichlet(1, ..., 1).
    Weights could equally be read from a file per posterior, e.g.
    np.load(f"weights/{entry.name}.npy")."""
    rng = np.random.default_rng(seed)
    n_obs = entry.model.n_obs
    return n_obs * rng.dirichlet(np.ones(n_obs), size=n_replicates)


bayesian_bootstrap = partial(analyses.bootstrap_mcmc, weights=bayesian_bootstrap_weights, n_draws=500)


def prior_draw_check(entry, out_dir):
    """A custom analysis: anything that takes an entry and writes to out_dir.
    Here, the log prior at a random point of each model."""
    model = entry.model
    params = model.params_pattern.random(jax.random.PRNGKey(0))
    with open(out_dir / "log_prior.json", "w") as f:
        json.dump({"log_prior": float(model.log_prior(params))}, f)
