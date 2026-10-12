"""New-style (modelgami) models: the contract every model must meet, plus
agreement with the old `StatisticalModel` translations and with Stan."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import posteriordb_jax  # noqa: F401  (enables float64)
from posteriordb_jax import catalog
from posteriordb_jax.io import load_data, load_posterior_info, stan_file_path
from posteriordb_jax.models import MODEL_REGISTRY
from posteriordb_jax.testing import StanComparisonHarness, assert_model_matches_stan
from modelgami.testing import check_model_contract

NAMES = catalog.list_models()


@pytest.mark.parametrize("name", NAMES)
def test_contract(name):
    check_model_contract(catalog.load(name))


@pytest.mark.parametrize("name", NAMES)
def test_matches_old_translation(name):
    """Same free vector, same log density and gradient as the existing
    (BridgeStan-verified) StatisticalModel version. The tolerance allows
    TFP's ~1e-7 constant offset in some log densities (e.g. Cauchy), which
    the new versions don't have: they match Stan to ~1e-14."""
    model = catalog.load(name)
    mod = catalog.module_for(name)
    raw = load_data(load_posterior_info(name)["data_name"])
    data = {k: (jnp.asarray(v, dtype=jnp.float64) if isinstance(v, list) else v) for k, v in raw.items()}
    if mod.STAN_MODEL == "wells_dist":
        data["switched"] = raw["switched"]
    old = MODEL_REGISTRY[mod.STAN_MODEL](data)
    f = model.log_density_fn("joint", "free")
    for k in jax.random.split(jax.random.PRNGKey(0), 5):
        x = jax.random.normal(k, (model.free_dim,))
        np.testing.assert_allclose(f(x), old.log_prob_unconstrained(x), rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(jax.grad(f)(x), old.grad_unconstrained(x), rtol=1e-9, atol=1e-8)


@pytest.mark.stan
@pytest.mark.parametrize("name", NAMES)
def test_matches_stan(name):
    model = catalog.load(name)
    raw = load_data(load_posterior_info(name)["data_name"])
    harness = StanComparisonHarness(stan_file_path(catalog.module_for(name).STAN_MODEL), raw)
    report = assert_model_matches_stan(model, harness, n_synthetic=10, check_hessian=True)
    report.assert_within(atol=1e-6, atol_hessian=1e-2, atol_log_density=1e-5)


def test_list_and_filter():
    assert "kidiq-kidscore_momiq" in catalog.list_models(exchangeable=True)
    assert "eight_schools-eight_schools_noncentered" not in catalog.list_models(exchangeable=True)
    with pytest.raises(KeyError):
        catalog.load("not-a-posterior")


def test_reference_draws():
    draws = catalog.reference_draws("kidiq-kidscore_momiq")
    assert draws["beta"].shape[-1] == 2 and draws["beta"].shape[:2] == draws["sigma"].shape
    idata = catalog.reference_draws("kidiq-kidscore_momiq", as_arviz=True)
    assert "log_lik" in idata["log_likelihood"]
