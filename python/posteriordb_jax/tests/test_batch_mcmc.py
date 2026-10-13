"""The ready-made MCMC analyses of the batch runner (need blackjax)."""

import json

import numpy as np
import pytest

pytest.importorskip("blackjax")

from posteriordb_jax.batch import Entry, NotApplicable, analyses  # noqa: E402  (needs blackjax)

EXCHANGEABLE = "kidiq-kidscore_momiq"
NOT_EXCHANGEABLE = "eight_schools-eight_schools_noncentered"


def test_mcmc_and_bootstrap_mcmc(tmp_path):
    entry = Entry(EXCHANGEABLE)
    analyses.mcmc(entry, tmp_path / "mcmc", n_chains=2, n_warmup=200, n_draws=200)
    summary = json.loads((tmp_path / "mcmc" / "summary.json").read_text())
    assert summary["max_z_vs_reference"] < 0.5
    assert np.load(tmp_path / "mcmc" / "draws.npz")["beta"].shape == (2, 200, 2)

    weights = lambda entry: analyses.multinomial_weights(entry, seed=0, n_replicates=2)
    analyses.bootstrap_mcmc(entry, tmp_path / "boot", weights=weights, n_chains=2, n_warmup=200, n_draws=200)
    for replicate in range(2):
        replicate_dir = tmp_path / "boot" / f"replicate_{replicate:03d}"
        assert np.load(replicate_dir / "weights.npy").sum() == entry.model.n_obs
        assert (replicate_dir / "draws.npz").exists()

    with pytest.raises(NotApplicable):
        analyses.bootstrap_mcmc(Entry(NOT_EXCHANGEABLE), tmp_path / "none", weights=weights)
    with pytest.raises(ValueError):
        analyses.bootstrap_mcmc(entry, tmp_path / "bad", weights=lambda entry: np.ones(3))
