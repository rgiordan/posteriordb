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

# Per-posterior exceptions to the BridgeStan comparison:
# - Stan's bernoulli_logit_lpmf and bernoulli_logit_glm_lpmf switch to cutoff
#   approximations for |logit| > 20 and, above the cutoff, return the partial
#   -exp(-logit) instead of the exact exp(-logit) / (1 + exp(-logit)), an error
#   of up to ~4e-9 per term, so their gradients get a looser bound. The log
#   densities still agree to 1e-11.
# - For the largest models the Hessian is skipped: jax.hessian is slow at
#   thousands of free parameters, and BridgeStan's Hessian is finite
#   differences of gradients, which is too noisy there to compare.
STAN_GRAD_ATOL = {
    "irt_2pl-irt_2pl": 1e-7,
    "sat-hier_2pl": 1e-7,
    "ovarian-logistic_regression_rhs": 1e-7,
    "prostate-logistic_regression_rhs": 1e-7,
}
# The old translations of these give nan gradients: p is exactly 1 on each
# dog's first trial, and their Bernoulli log density differentiates log(1 - p)
# there. The new versions' gradients match Stan (test_matches_stan).
OLD_GRADIENT_NAN = {"dogs-dogs_hierarchical", "dogs-dogs_nonhierarchical"}
NO_HESSIAN = {"ovarian-logistic_regression_rhs", "prostate-logistic_regression_rhs",
              "traffic_accident_nyc-bym2_offset_only", "prideprejudice_chapter-ldaK5",
              "prideprejudice_paragraph-ldaK5", "three_docs1200-ldaK2", "three_men1-ldaK2",
              "three_men2-ldaK2", "three_men3-ldaK2"}


@pytest.mark.parametrize("name", NAMES)
def test_contract(name):
    check_model_contract(catalog.load(name))


@pytest.mark.parametrize("name", NAMES)
def test_matches_old_translation(name):
    """Same free vector, same log density and gradient as the existing
    (BridgeStan-verified) StatisticalModel version. The tolerance allows
    TFP's small constant offsets in some log densities (e.g. Cauchy, Beta, or
    a normal summed over the ~12,000 radon houses) and in gradients of
    the vague gamma and inverse-gamma priors (e.g. gamma(1e-4, 1e-4)), which
    the new versions don't have: they match Stan to ~1e-14."""
    model = catalog.load(name)
    mod = catalog.module_for(name)
    raw = load_data(load_posterior_info(name)["data_name"])
    # Lists become float arrays, except integer fields the new model either
    # casts to int32 (indices and 0/1 outcomes) or doesn't take at all
    # (raw encodings it expands itself, e.g. lsat's run lengths).
    data_fields = model.data if isinstance(model.data, dict) else {}
    int_fields = {k for k, v in raw.items()
                  if isinstance(v, list) and (jnp.asarray(data_fields[k]).dtype == jnp.int32 if k in data_fields
                                              else jnp.issubdtype(jnp.asarray(v).dtype, jnp.integer))}
    data = {k: (jnp.asarray(v, dtype=jnp.int32 if k in int_fields else jnp.float64) if isinstance(v, list) else v)
            for k, v in raw.items()}
    old = MODEL_REGISTRY[mod.STAN_MODEL](data)
    f = model.log_density_fn("joint", params_as="free")
    for k in jax.random.split(jax.random.PRNGKey(0), 5):
        x = jax.random.normal(k, (model.params_pattern.free_length(),))
        np.testing.assert_allclose(f(x), old.log_prob_unconstrained(x), rtol=1e-8, atol=1e-5)
        if name not in OLD_GRADIENT_NAN:
            np.testing.assert_allclose(jax.grad(f)(x), old.grad_unconstrained(x), rtol=1e-7, atol=1e-6)


@pytest.mark.stan
@pytest.mark.parametrize("name", NAMES)
def test_matches_stan(name):
    model = catalog.load(name)
    raw = load_data(load_posterior_info(name)["data_name"])
    harness = StanComparisonHarness(stan_file_path(catalog.module_for(name).STAN_MODEL), raw)
    report = assert_model_matches_stan(model, harness, n_synthetic=10, check_hessian=name not in NO_HESSIAN)
    # Scaled errors (see assert_model_matches_stan). BridgeStan's Hessian is
    # finite differences of gradients, hence the looser Hessian bound.
    report.assert_within(atol=STAN_GRAD_ATOL.get(name, 1e-9), atol_hessian=1e-4, atol_log_density=1e-11)


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
