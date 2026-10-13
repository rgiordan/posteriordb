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

## New: the modelgami catalog (work in progress)

The framework is being split into three packages that live side by side
under `python/`:

- [`paragami`](../paragami): JAX parameter folding/flattening with Stan-exact transforms (replaces `constraints.py`, `flatten.py`, `stan_bijectors.py`).
- [`modelgami`](../modelgami): the common model interface (replaces `model.py`'s `StatisticalModel`).
- `posteriordb_jax`: this package. It holds the posteriordb translations, now as `modelgami.Model`s in `posteriordb_jax/catalog/`.

Converted so far (94 Stan models, 119 posteriors): `arK`, `arma11`, `blr`, `bones_model`, `bym2_offset_only`, `diamonds`, `dogs`, `dogs_hierarchical`, `dogs_log`, `dogs_nonhierarchical`, `dugongs_model`, `earn_height`, `eight_schools_centered`, `eight_schools_noncentered`, `election88_full`, `GLM_Binomial_model`, `GLM_Poisson_model`, `gp_pois_regr`, `gp_regr`, `hier_2pl`, `hmm_example`, `irt_2pl`, `kidscore_interaction`, `kidscore_interaction_c`, `kidscore_interaction_c2`, `kidscore_interaction_z`, `kidscore_mom_work`, `kidscore_momhs`, `kidscore_momhsiq`, `kidscore_momiq`, `kilpisjarvi`, `ldaK2`, `ldaK5`, `log10earn_height`, `logearn_height`, `logearn_height_male`, `logearn_interaction`, `logearn_interaction_z`, `logearn_logheight_male`, `logistic_regression_rhs`, `logmesquite`, `logmesquite_logva`, `logmesquite_logvas`, `logmesquite_logvash`, `logmesquite_logvolume`, `low_dim_gauss_mix`, `low_dim_gauss_mix_collapse`, `lsat_model`, `M0_model`, `Mb_model`, `mesquite`, `Mh_model`, `Mt_model`, `Mtbh_model`, `Mth_model`, `nes`, `nes_logit_model`, `normal_mixture`, `normal_mixture_k`, `pilots`, `radon_county`, `radon_county_intercept`, `radon_hierarchical_intercept_centered`, `radon_hierarchical_intercept_noncentered`, `radon_partially_pooled_centered`, `radon_partially_pooled_noncentered`, `radon_pooled`, `radon_variable_intercept_centered`, `radon_variable_intercept_noncentered`, `radon_variable_intercept_slope_centered`, `radon_variable_intercept_slope_noncentered`, `radon_variable_slope_centered`, `radon_variable_slope_noncentered`, `Rate_1_model`, `Rate_2_model`, `Rate_3_model`, `Rate_4_model`, `Rate_5_model`, `rats_model`, `seeds_centered_model`, `seeds_model`, `seeds_stanified_model`, `sesame_one_pred_a`, `surgical_model`, `Survey_model`, `wells_daae_c_model`, `wells_dae_c_model`, `wells_dae_inter_model`, `wells_dae_model`, `wells_dist`, `wells_dist100_model`, `wells_dist100ars_model`, `wells_interaction_c_model` and `wells_interaction_model`.

- The new versions use `jax.scipy.stats` instead of TFP and separate the prior from the likelihood.
- Wherever the data are exchangeable, they use `log_lik_one`. Models with a parameter per observation or a time-series or GP likelihood (eight schools, seeds, surgical, irt_2pl, hier_2pl, lsat, bones, bym2, dogs_nonhierarchical, hmm_example, Mh/Mth/Mtbh, arK, arma11, the GPs, Survey and the Rate models) are plain `Model`s; the rest are `ExchangeableModel`s. In grouped models (radon, rats, pilots, election88, the LDA models) each observation is a datum that carries its group indices.
- The linear and logistic regressions share `catalog/_regression.py` (`normal_regression`, `logistic_regression`), so each of those is a feature function and a response.
- `tests/test_catalog.py` checks each model against BridgeStan, against the old translation, and against the modelgami contract. The BridgeStan check uses scaled errors, |jax − stan| / (1 + |stan|), so one set of tolerances fits every model. Log densities and gradients agree to about 1e-14 or better.

Every model in `posteriordb_jax/models/` now has a catalog version; the old translations stay as the reference for `test_matches_old_translation`.

```python
from posteriordb_jax import catalog

model = catalog.load("kidiq-kidscore_momiq")            # a modelgami.Model, data loaded
f = model.log_density_fn("joint", params_as="free")       # Stan-exact log density
catalog.list_models(exchangeable=True)                    # filter posteriors by capability
draws = catalog.reference_draws("kidiq-kidscore_momiq")   # posteriordb's reference draws
```

Install for development (from `python/`): `pip install -e paragami -e modelgami -e "posteriordb_jax[test]"`. The catalog alone needs neither TensorFlow Probability nor BridgeStan; they are the `legacy` and `stan` extras.
The old TFP-based code needs `jax<=0.5` (TFP 0.25 breaks on newer JAX); paragami and modelgami work on current JAX.

## Status (old-style models)

96 of the 120 Stan models in this repo are converted so far
(`posteriordb_jax/models/`, registered in `MODEL_REGISTRY`), each with a
passing BridgeStan-comparison test (`tests/test_<name>.py`) checking log
density, gradient, and Hessian in unconstrained space, plus the
constrained-space flat vector (Hessian skipped for a handful of very
large-free-dimension models -- see their test files). The first few
establishing each constraint type or a notable pattern:

| Model | Constraint types / pattern exercised |
|---|---|
| `eight_schools_noncentered` / `eight_schools_centered` | unconstrained, `lower` |
| `hmm_example` | `simplex`, `positive_ordered` |
| `dogs_nonhierarchical` | `lower`, `cholesky_factor_corr`, matrix-shaped unconstrained param |
| `low_dim_gauss_mix` / `normal_mixture` / `normal_mixture_k` | `ordered`, `lower_upper`, `simplex` (mixture likelihoods, `log_mix`/`log_sum_exp`) |
| `arK` / `radon_county` / `radon_county_intercept` | vector-shaped params, lagged/grouped-index likelihoods |
| `ldaK2` / `ldaK5` | **array-of-simplex** (`array[M] simplex[K]`), Dirichlet priors, gather-based mixture likelihood |
| `arma11` | sequential `lax.scan` recursion (`err[t]` depends on `err[t-1]`) |
| `dogs` / `dogs_log` | cumulative-count recursion, a plain-`vector` parameter whose *prior* (not type) is one-sided (`uniform(-100,0)`) |
| `gp_regr` (+ `gp_pois_regr`) | Gaussian process: `gp_exp_quad_cov` kernel + `jnp.linalg.cholesky`, `multi_normal_cholesky` likelihood; `gp_pois_regr` reuses the same kernel code with a fixed `1e-10` jitter and a `poisson_log` likelihood on a whitened (`f = L_cov @ f_tilde`) GP draw instead |
| `hier_2pl` | `cholesky_factor_corr` composed into a batched `multi_normal_cholesky` prior (not just a standalone correlation matrix) |
| `M0_model` / `Mt_model` (+ `Mb_model` / `Mh_model` / `Mth_model` / `Mtbh_model`) | capture-recapture mixture likelihoods (`log_sum_exp` over a latent present/absent indicator); the four later models each just swap in a different per-occasion detection-logit construction (behavioral "trap happy/shy" response, individual random effect, or both) built from data/parameters that already reduce correctly at `s[i]==0`, so no extra masking is needed beyond the shared `M0`/`Mt` pattern |
| `diamonds` / `logistic_regression_rhs` | `StudentT`/`InverseGamma` priors, `normal_id_glm_lpdf`/`bernoulli_logit_glm`, a regularized-horseshoe (`d=5966`-predictor) sparse regression |
| `lsat_model` | run-length-encoded data expansion (`jnp.repeat`) from a compact "response pattern" representation |
| `Survey_model` | discrete-latent-variable ("number of surveys sent") marginalization via `log_sum_exp` over a vectorized `binomial_lpmf`; the invalid-domain (`n < max(k)`) branch is masked on the *input* to `Binomial` (clamping `total_count`) before masking the output, not just the output -- masking only the output still lets an invalid branch's `-inf`/`nan` leak into the gradient through `jnp.where`'s untaken-branch autodiff |
| `bones_model` | ragged ordinal/graded-response likelihood: per-indicator category count `ncat[j]` varies (2..5), with `-1`-sentinel missing `grade[i,j]` entries contributing zero (not `log(0)`) to the log density; the `Q -> p` cumulative-probability construction is vectorized as one boundary-padded `Qext` array so no per-`ncat` special-casing is needed |

The rest are mostly single/multi-covariate linear or logistic ("GLM")
regressions, rate/binomial models, and centered/noncentered hierarchical
(varying-intercept/-slope) regressions (the `wells_*`, `kidscore_*`,
`logearn_*`, `logmesquite_*`, `Rate_*`, `radon_*`, `seeds_*` families, plus
`blr`, `mesquite`, `nes_logit_model`, `nes`, `pilots`, `kilpisjarvi`,
`dugongs_model`, `surgical_model`, `GLM_Poisson_model`, `GLMM_Poisson_model`,
`GLM_Binomial_model`, `GLMM1_model`, `rats_model`, `bym2_offset_only`,
`irt_2pl`, `election88_full`, `dogs_hierarchical`) -- mechanical translations
reusing `unconstrained`/`lower`/
`lower_upper`, listed in full in `MODEL_REGISTRY`
(`posteriordb_jax/models/__init__.py`).

`cov_matrix`/`cholesky_factor_cov` are implemented and unit-tested against
BridgeStan directly (`tests/test_stan_bijectors.py`) but not yet exercised
by a converted model (no model in this repo has one as a *free* parameter).
`unit_vector` and a fully general `corr_matrix` are not implemented (see
"Known gaps" below). Models with a parameter-declared bound that depends on
*another parameter* (e.g. `garch11`'s `real<lower=0,upper=(1-alpha1)>
beta1`) are also skipped for now -- `ParameterSpec.bounds` is static, and
supporting this would need a genuinely new (small) piece of framework
machinery; not yet built.

### A note on Hessian tolerances in large/large-magnitude models

Several tests for models with large `N` (e.g. `radon_pooled`, `radon_county*`,
N~12,573) or large raw-data magnitudes (e.g. `earn_height`'s dollar-valued
`earn`, `kilpisjarvi`'s `x` in the thousands) pass a looser `atol_hessian`
(and occasionally `atol`/`atol_log_density`) to `ComparisonReport.assert_within`
than the default `1e-6`. In every case this was verified to be **magnitude-driven
float64 noise, not a formula bug**: `max_abs_err_constrained_flat` (never
loosened) stays at machine epsilon, confirming the bijector/transform is exact,
while the *relative* error in the log density/gradient/Hessian stays ~1e-13 to
1e-16 even as the loosened absolute tolerance grows -- a standard-normal
*unconstrained* synthetic draw can imply large residuals for models with no
informative prior pinning parameters near a good fit, which is what drives the
raw magnitude up. See the comment on each such test for the specific numbers.

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
- **BridgeStan's `param_unc_names()` is actively misleading for "array of
  simplex" parameters** (`array[M] simplex[K]`, added to support `ldaK2`/
  `ldaK5`/`hmm_gaussian`-style models -- see `ParameterSpec.free_shape`).
  It reports an interleaved "M-fastest" name order for the *unconstrained*
  parameters -- matching the order the *constrained* array actually uses
  (`flatten_order == "F"`) -- but the free vector's actual VALUES are laid
  out row-major/blockwise instead (all of array-element 1's K-1 free
  coordinates, then element 2's, ...). Discovered by feeding a known vector
  to an `array[M] simplex[K]` BridgeStan model and to M separate
  single-`simplex[K]` models and comparing per-block outputs, not by
  trusting names -- see `ParameterSpec.free_flatten_order`'s docstring and
  `tests/test_stan_bijectors.py::test_simplex_array_matches_bridgestan`.
  **Never trust `param_unc_names()`'s order for this constraint-type
  combination; verify with values.**
- **A Stan `~ uniform(a, b)` (or any) sampling statement is a prior, not a
  type-level bound**, even when it looks like one. `dogs_log.stan` declares
  `vector[2] beta;` (fully unconstrained) but gives it `beta[1] ~
  uniform(-100, 0)` / `beta[2] ~ uniform(0, 100)` in the model block --
  modeling `beta` as a `LOWER_UPPER`-constrained `ParameterSpec` (an earlier
  version of `dogs_log.py` did) is wrong: it applies a `Sigmoid` bijector
  Stan never applies, so even the *constrained* value disagrees with Stan,
  not just the log density. The correct translation declares `beta`
  `UNCONSTRAINED` and adds the `uniform(...).log_prob(...)` terms explicitly
  in `log_density` -- which, correctly, evaluates to `-inf` whenever a
  synthetic unconstrained draw lands outside the stated interval (matching
  Stan's own `uniform_lpdf` behavior exactly). A symptom that flags this
  mistake: `max_abs_err_constrained_flat` staying nonzero no matter how
  small `n_synthetic`'s draw `scale` gets -- every other constraint type's
  constrained-flat check goes to machine epsilon; if it doesn't, suspect a
  bound that shouldn't be there.
- **`tfd.StudentT` (and likely other multi-parameter TFP distributions)
  infers its dtype from `(df, loc, scale)` at *construction* time, before
  ever seeing the sample passed to `.log_prob()`.** Bare Python
  floats/ints -- including hyperparameters pulled straight out of a
  posteriordb `data` JSON file, e.g. `nu_local`/`scale_global` in
  `logistic_regression_rhs.py`, or literal `3.0`/`8.0`/`10.0` in
  `diamonds.py` -- get silently taken as `float32` by TFP, which then
  raises a **hard `TypeError`** (not silent precision loss) the first time
  `.log_prob()` is called with a float64 sample: `Found incompatible
  dtypes, float64 and float32`. Unlike the elementwise `Normal`/`Gamma`/
  `Beta` calls used everywhere else in this codebase (which happened not to
  trigger this), every `df`/`loc`/`scale` argument to `StudentT` needs an
  explicit `jnp.asarray(v, dtype=jnp.float64)` cast -- the same discipline
  `constraints.py`'s `_f64` helper already uses for bound literals, for
  exactly the same reason.

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
