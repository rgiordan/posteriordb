"""Run one analysis on many posteriordb entries, locally or as a Slurm array
job.

An analysis is any Python function `analysis(entry, out_dir)` that takes an
`Entry` (one posteriordb posterior, with its modelgami model) and writes its
results into `out_dir`. Name it to the command line as "module:function" or
"path/to/file.py:function"; configure it with `functools.partial`:

    # my_analyses.py
    from functools import partial
    from posteriordb_jax.batch import analyses

    short_mcmc = partial(analyses.run_mcmc, n_warmup=500, n_draws=500)

    python -m posteriordb_jax.batch local  --analysis my_analyses.py:short_mcmc --out results --posteriors eight_schools-eight_schools_noncentered
    python -m posteriordb_jax.batch submit --analysis my_analyses.py:short_mcmc --out results --time 02:00:00

Each entry's results go to `<out>/<posterior name>/`, next to a
`status.json` that records whether the analysis finished, was not applicable
(it raised `NotApplicable`) or failed (with the traceback). Finished entries
are skipped on later runs unless `--overwrite` is given, so resubmitting the
same command reruns only what failed or never ran.

Modules:
- `entry`:    `Entry` and `entries` (selecting posteriors).
- `runner`:   running an analysis on entries and recording each one's status.
- `slurm`:    writing and submitting the Slurm array job.
- `run_mcmc`:     NUTS with BlackJAX, optionally with bootstrap weights (needs blackjax).
- `analyses`: ready-made analyses: `run_mcmc` and `run_bootstrap_mcmc` (need blackjax).
"""

from .entry import Entry, select_entries
from .runner import NotApplicable, load_analysis, read_status, run_entry, run_local, read_statuses
from .slurm import SlurmOptions, submit, write_job

__all__ = ["Entry", "select_entries", "NotApplicable", "load_analysis", "read_status", "run_entry", "run_local",
           "read_statuses", "SlurmOptions", "submit", "write_job"]
