import jax.numpy as jnp
import numpy as np

import posteriordb_jax  # noqa: F401
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models.eight_schools_noncentered import EightSchoolsNoncentered
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan


def _load_model():
    raw = load_data("eight_schools")
    data = {
        "J": raw["J"],
        "y": jnp.asarray(raw["y"], dtype=jnp.float64),
        "sigma": jnp.asarray(raw["sigma"], dtype=jnp.float64),
    }
    return EightSchoolsNoncentered(data), data


def test_param_specs_order_matches_stan():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("eight_schools_noncentered"), data)
    # Free-parameter order/count must line up index-for-index between the
    # JAX model's flat unconstrained vector and Stan's, since
    # assert_matches_stan feeds the same raw vector to both.
    assert harness.param_unc_num() == model.model_spec.free_dim
    assert harness.param_unc_names() == [
        "theta_trans.1",
        "theta_trans.2",
        "theta_trans.3",
        "theta_trans.4",
        "theta_trans.5",
        "theta_trans.6",
        "theta_trans.7",
        "theta_trans.8",
        "mu",
        "tau",
    ]


def test_matches_bridgestan_log_density_grad_hessian():
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("eight_schools_noncentered"), data)
    report = assert_matches_stan(model, harness, n_synthetic=25, seed=0, check_hessian=True)
    report.assert_within(atol=1e-6)


def test_theta_reconstruction_matches_transformed_parameter():
    """Extra check: our computed `theta = theta_trans * tau + mu` matches
    Stan's own `transformed parameters` block output exactly -- this is the
    quantity posteriordb's reference draws are actually keyed by (not
    theta_trans), so getting this right matters for any future use of the
    stored reference draws as an alternate input source."""
    model, data = _load_model()
    harness = StanComparisonHarness(stan_file_path("eight_schools_noncentered"), data)
    rng = np.random.default_rng(1)
    for _ in range(10):
        x_np = rng.normal(size=harness.param_unc_num())
        c_stan_with_tp = harness.param_constrain(x_np, include_tp=True)
        theta_stan = c_stan_with_tp[10:18]  # after theta_trans(8), mu, tau (indices 8,9) -> theta(8)

        params = model.constrain(jnp.asarray(x_np))
        theta_jax = params["theta_trans"] * params["tau"] + params["mu"]
        np.testing.assert_allclose(np.asarray(theta_jax), theta_stan, atol=1e-8)
