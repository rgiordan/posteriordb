import jax.numpy as jnp

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.nes import Nes
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("nes1976")
    data = {
        "N": raw["N"],
        "partyid7": jnp.asarray(raw["partyid7"], dtype=jnp.float64),
        "real_ideo": jnp.asarray(raw["real_ideo"], dtype=jnp.float64),
        "race_adj": jnp.asarray(raw["race_adj"], dtype=jnp.float64),
        "educ1": jnp.asarray(raw["educ1"], dtype=jnp.float64),
        "gender": jnp.asarray(raw["gender"], dtype=jnp.float64),
        "income": jnp.asarray(raw["income"], dtype=jnp.float64),
        "age_discrete": raw["age_discrete"],
    }
    return Nes(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("nes"), data)
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == [f"beta.{i}" for i in range(1, 10)] + ["sigma"]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("nes"), data)
    report = assert_matches_stan(model, harness, n_synthetic=20, seed=0, check_hessian=True, scale=0.3)
    report.assert_within(atol=1e-4, atol_hessian=1.0)
