# posteriordb_jax

JAX/TensorFlow-Probability translations of [posteriordb](../../README.md)
Stan models, behind a `StatisticalModel` interface that mirrors Stan's own
(data → model, constrained/unconstrained parameters) but adds what Stan
can't do: no recompilation for new data, gradients with respect to data
itself (not just parameters), and easy vectorization across draws.

Every constraining transform used here (`constraints.py`/`stan_bijectors.py`)
replicates **Stan's own transform exactly**, not just a support-equivalent
substitute -- so unconstrained-space log density, gradients, and Hessians
match Stan bit-for-bit, not just the constrained-space log density. This is
verified for every constraint type actually used, per-model, against a live
compiled Stan model via [BridgeStan](https://roualdes.us/bridgestan/) (see
`testing.py`).

## Status

Converted so far, each with a passing BridgeStan-comparison test
(`tests/test_<name>.py`) checking log density, gradient, and Hessian in
unconstrained space, plus the constrained-space flat vector:

| Model | Constraint types exercised |
|---|---|
| `eight_schools_noncentered` | unconstrained, `lower` |
| `hmm_example` | `simplex`, `positive_ordered` |
| `dogs_nonhierarchical` | `lower`, `cholesky_factor_corr`, matrix-shaped unconstrained param |

`cov_matrix`/`cholesky_factor_cov` are implemented and unit-tested against
BridgeStan directly (`tests/test_stan_bijectors.py`) but not yet exercised
by a converted model (no model in this repo has one as a *free* parameter).
`unit_vector` and a fully general `corr_matrix` are not implemented (see
"Known gaps" below).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install jax tensorflow-probability bridgestan numpy pytest
.venv/bin/pytest tests/
```

BridgeStan will bootstrap its own bundled Stan source on first use (one-time
network access, then cached under `~/.bridgestan`). No `tensorflow` or
`cmdstanpy` needed -- `tensorflow-probability`'s JAX substrate
(`tensorflow_probability.substrates.jax`) has no `tensorflow` dependency.

```python
import jax.numpy as jnp
from posteriordb_jax.io import load_data, stan_file_path
from posteriordb_jax.models import MODEL_REGISTRY
from posteriordb_jax.testing import StanComparisonHarness, assert_matches_stan

raw = load_data("eight_schools")
data = {"J": raw["J"], "y": jnp.asarray(raw["y"]), "sigma": jnp.asarray(raw["sigma"])}
model = MODEL_REGISTRY["eight_schools_noncentered"](data)

x = jnp.zeros(model.model_spec.free_dim)          # unconstrained parameter vector
model.log_prob_unconstrained(x)                   # scalar
model.grad_unconstrained(x)                        # (free_dim,)
model.hessian_unconstrained(x)                     # (free_dim, free_dim)
model.hvp_unconstrained(x, x)                       # (free_dim,) -- forward-over-reverse, no full Hessian
model.grad_wrt_data(x)                              # gradient wrt the *data* pytree -- Stan can't do this
model.constrain(x)                                  # -> dict of named, shaped constrained arrays

# batched over a stack of draws:
xs = jnp.zeros((10, model.model_spec.free_dim))
model.log_prob_unconstrained_batch(xs)              # (10,)
model.hvp_unconstrained_batch(xs, xs)                # (10, free_dim) -- O(n*free_dim), not O(n*free_dim^2)

# verify against Stan directly:
harness = StanComparisonHarness(stan_file_path("eight_schools_noncentered"), data)
report = assert_matches_stan(model, harness, n_synthetic=20)
report.assert_within(atol=1e-6)
```

## Architecture

```
posteriordb_jax/
  __init__.py          # enables jax_enable_x64 (required -- JAX defaults to
                        #   float32, Stan/BridgeStan are always float64)
  stan_bijectors.py     # custom tfp.bijectors.Bijector subclasses replicating
                        #   Stan's exact transforms where no off-the-shelf
                        #   TFP bijector matches (StanSimplex,
                        #   StanCholeskyFactorCorr, StanCholeskyFactorCov)
  constraints.py         # ParameterSpec + ConstraintType: the declarative,
                        #   per-model constraint vocabulary, dispatching to
                        #   either an off-the-shelf TFP bijector/Chain or one
                        #   of the custom classes above
  flatten.py             # ModelSpec: flatten/unflatten between a dict of
                        #   named shaped arrays and (a) a flat vector per
                        #   draw, (b) posteriordb's flat per-element naming
  model.py                # StatisticalModel base class -- everything
                        #   (log-prob in both spaces, grad/hessian/hvp,
                        #   batching, grad wrt data) derived generically from
                        #   one user-written `log_density(params, data)`
  io.py                   # loaders for posteriordb's Stan/data/draws files
  testing.py               # StanComparisonHarness (wraps one BridgeStan
                        #   model) + assert_matches_stan
  models/                  # one hand-converted StatisticalModel subclass per
                        #   Stan model, plus MODEL_REGISTRY
tests/
  test_stan_bijectors.py   # per-bijector: round trip, finite-diff Jacobian,
                        #   and direct BridgeStan cross-checks
  test_constraints.py, test_flatten.py, test_model_base.py  # Stan-free
  test_<model_name>.py      # one per converted model, via assert_matches_stan
  stan_fixtures/*.stan       # minimal single-constraint-type Stan files used
                        #   to isolate a bijector from a full model when
                        #   validating it
```

### Converting a new Stan model

1. Add a `ParameterSpec` per `parameters`-block variable in `param_specs()`
   (shapes may depend on `data`, e.g. `J = len(data["y"])` -- never trust
   posteriordb's `posteriors/*.json` `"dimensions"` field for shapes; it's
   the *printed*, not free, element count, and has been observed to drift
   from the actual data, see "Known gaps" below).
2. Translate the Stan `model`/`transformed parameters` block logic by hand
   into `log_density(params, data)`, using
   `tensorflow_probability.substrates.jax.distributions` for standard
   log-prob terms. `data` must be read from the explicit argument, never
   `self.data`, so `grad_wrt_data` can differentiate through it.
3. If the Stan model reparameterizes a reported quantity away from its raw
   sampled parameter (e.g. a noncentered `theta = theta_raw * tau + mu`),
   your `param_specs`/`log_density` should use the *raw* Stan parameter name
   directly (as `eight_schools_noncentered.py` does with `theta_trans`) --
   `assert_matches_stan`'s default synthetic-draw comparison needs the free
   parameter order/names to line up index-for-index with Stan's own
   `parameters` block.
4. Write `tests/test_<name>.py`: build a `StanComparisonHarness` against the
   real `.stan`/data files (`io.py`), call `assert_matches_stan`, and inspect
   the resulting `ComparisonReport`. Start with `check_hessian=False` and a
   small `n_synthetic` to get log-density/gradient right first -- the
   Hessian check is the most compute- and precision-sensitive.

## Real discoveries from converting the first three models

These corrected assumptions made during initial design/planning -- worth
knowing before touching `stan_bijectors.py` or adding a new constraint type:

- **Stan's simplex transform changed in 2.37.** The "stick-breaking"
  logistic transform documented in older references is no longer the
  default -- current Stan (this repo's BridgeStan pulls in 2.39) uses a
  "sum-to-zero vector" linear transform followed by `softmax`. `StanSimplex`
  targets the current transform; its forward map and Jacobian were both
  confirmed against a live compiled model, not just derived from docs (a
  first attempt using the old stick-breaking formula reproduced neither).
- **`tfb.FillScaleTriL` does not fill in Stan's element order.** It's built
  on `tfb.FillTriangular`, which uses an internal "zigzag" element order
  (for compute efficiency) rather than Stan's plain row-major fill. Using it
  for `cholesky_factor_cov`/`cov_matrix` produces a *different, still valid*
  Cholesky parameterization -- support-equivalent but numerically wrong for
  an exact-match comparison. `StanCholeskyFactorCov` fills in Stan's actual
  order instead (confirmed against BridgeStan).
- **`corr_matrix[K]` is not `cholesky_factor_corr` + `L @ L.T`, for K>=4.**
  Despite using the same number of unconstrained inputs and
  `cholesky_factor_corr`'s own transform composing correctly with
  `tfb.CholeskyOuterProduct()` for exactly this purpose, Stan's
  `corr_matrix[K]` transform is a genuinely different construction for
  K>=4 (the two agree, apparently coincidentally, at K=2 and K=3). See
  `tests/test_stan_bijectors.py::test_corr_matrix_cholesky_outer_product_boundary`.
- **`posteriors/*.json`'s `"dimensions"` field is not a reliable shape
  source.** It's the printed/constrained element count in some cases and
  doesn't match either the free or printed count in others (e.g.
  `dogs-dogs_nonhierarchical.json` lists `"z": 62` where the actual data
  gives `n_dogs=30 -> 60`). Always derive shapes from `data` at runtime.
- **Not every reference-draws key matches the raw Stan parameter it came
  from.** `eight_schools_noncentered` samples `theta_trans` but posteriordb's
  stored draws (and BridgeStan's default constrained output) are keyed by
  the derived `theta`. `assert_matches_stan`'s synthetic-draw default
  sidesteps this by construction; only the extra sanity check in
  `test_eight_schools_noncentered.py` needs the explicit
  `theta = theta_trans * tau + mu` relationship.
- A model's Hessian in unconstrained space can be **numerically noisier
  than its gradient** when a `SIMPLEX` parameter is involved: the
  bijector's own Jacobian is computed via autodiff (`jax.jacfwd` +
  `slogdet`, deliberately, to avoid a hand-derived-formula transcription
  risk -- see `stan_bijectors.py`), so the model's Hessian differentiates
  *through* that a second time. Observed ~1e-7 *relative* error, larger in
  absolute terms for large Hessian entries (e.g. from a long `lax.scan`
  recursion in `hmm_example`) -- not a formula bug; see
  `ComparisonReport.assert_within`'s docstring and its `atol_hessian` param.
- TFP's `CholeskyLKJ` and Stan's `lkj_corr_cholesky_lpdf` agree to within a
  tiny (~1.4e-7), **L-independent** constant (gradient-invariant) -- almost
  certainly just differing floating-point paths through the log-gamma
  normalizing constant, not a formula difference.

## Known gaps

- `unit_vector[K]`: no off-the-shelf TFP bijector and no custom
  implementation yet. Not required by any model in this repo.
- `corr_matrix[K]`: only correct for K<=3 via the
  `cholesky_factor_corr`-based composition (see above); a fully general
  Stan-exact bijector is unimplemented. Not required by any model in this
  repo as a *free* parameter (only ever appears as a derived
  `transformed parameters` quantity, always at K=2, in `dogs_nonhierarchical`).
- Non-square `cholesky_factor_cov[M, N]` (M != N): unimplemented; not
  required by any model in this repo.
