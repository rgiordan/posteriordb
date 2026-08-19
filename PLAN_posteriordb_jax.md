# Plan: `posteriordb_jax` — JAX/TFP translation library for posteriordb Stan models

## Status: M0–M7 implemented and passing (M8 partially done)

The library described below has been built at `python/posteriordb_jax/` and
all 71 tests pass (`.venv/bin/pytest python/posteriordb_jax/tests/`),
including full BridgeStan comparisons (log density, gradient, and Hessian in
unconstrained space) for all three seed models. **See
`python/posteriordb_jax/README.md` for the as-built architecture, usage, and
a list of real discoveries made during implementation that corrected several
assumptions below** (Stan's simplex transform changed in 2.37 and is no
longer stick-breaking; `tfb.FillScaleTriL` doesn't match Stan's fill order
for Cholesky factors; `corr_matrix` is not `cholesky_factor_corr` +
`CholeskyOuterProduct` for K>=4). The plan text below is the original
pre-implementation design and is kept for historical context — where it
conflicts with the README, the README reflects what was actually built and
verified.

Remaining from the original M8 (stretch/polish): registering a
`"posteriordb_jax"` key in the three converted models' `info.json` files
(mirroring the existing `pymc3` precedent) has not been done.

## Context

`CLAUDE_TASK.md` asks for a Python library that reimplements posteriordb's Stan
models in JAX/TensorFlow Probability behind a `StatisticalModel` interface that
mirrors Stan's own (data → model, constrained/unconstrained parameters) but adds
things Stan can't do: no recompilation for new data, differentiation with respect
to data/hyperparameters, and easy vectorization across draws. A shared
constraint/transform library and a Stan-comparison testing framework are required
so that future Stan→JAX model conversions (by a human or an AI agent) are
mechanical and self-verifying.

This repo (`posteriordb`) is the *database* itself — Stan files, data, posterior
linkage metadata, and reference posterior draws — not a Python client. There is no
`python/` directory yet and no existing constrain/unconstrain or log-density-vs-Stan
comparison tooling anywhere in this checkout or its sibling R package. This plan
scopes a new package, `python/posteriordb_jax/`, built from scratch, and converts
3 seed Stan models by hand to prove out the framework end to end.

**Confirmed decisions (from user Q&A this session):**
- Testing oracle: **BridgeStan**, not cmdstanpy/MCMC. BridgeStan compiles a Stan
  model to a shared library and exposes `log_density()`, `log_density_gradient()`,
  `log_density_hessian()`, `param_constrain()`, `param_unconstrain()` in-process —
  the purpose-built tool for exactly this comparison, and a better fit than
  cmdstanpy's CLI-based `log_prob` mechanism.
- Bijectors: **replicate Stan's own constraining/unconstraining transforms
  exactly**, rather than settling for generic off-the-shelf TFP bijectors that
  are merely support-equivalent. Where an existing
  `tensorflow_probability.substrates.jax` bijector (or a `Chain` of them) is
  already formula-identical to Stan's transform (bounds, `cholesky_factor_cov`,
  `cov_matrix`, `ordered`, `positive_ordered`), use it directly. Where none
  matches (`simplex`, `cholesky_factor_corr`, `corr_matrix`, `unit_vector`),
  implement a **custom bijector class** that subclasses
  `tfp.bijectors.Bijector` (implementing `_forward`, `_inverse`,
  `_forward_log_det_jacobian`, `forward_min_event_ndims`, etc., per Stan's
  exact formulas) so it composes freely inside `tfb.Chain([...])` alongside
  built-ins — e.g. `corr_matrix = Chain([CholeskyOuterProduct(),
  StanCholeskyFactorCorr()])`. This means unconstrained-space log density,
  gradients, and Hessians should match Stan bit-for-bit for every constraint
  type used by the seed models, and the testing framework compares
  unconstrained-space derivatives universally rather than skipping mismatched
  blocks (see updated table and testing section below).
- Dependencies installed in a **venv** (`python3 -m venv`), minimal set only:
  `jax`, `tensorflow-probability` (JAX substrate needs no `tensorflow` install —
  confirmed), `bridgestan`, `numpy`, `pytest`. No `tensorflow`, no `cmdstanpy`.
- Package location: `python/posteriordb_jax/`.
- Seed models (span constraint types, chosen after checking actual repo content):
  `eight_schools_noncentered`, `hmm_example`, `dogs_nonhierarchical`.

## Key findings that shape the design

- **`posteriors/*.json` `"dimensions"` is not a reliable free-parameter-dimension
  source** — e.g. `hmm_example`'s `simplex[2]` is listed as dimension 2 (the
  constrained count), and `dogs`'s `z` dimension in the posteriors file doesn't
  match `n_dogs*2` in the current data file. **Shapes must always be derived from
  (data + Stan model) at runtime, never trusted from `posteriors/*.json`.**
- **Not every Stan `parameters`-block variable matches posteriordb's stored
  reference-draw names.** `eight_schools_noncentered.stan` samples `theta_trans`
  but stores/reports `theta = theta_trans*tau + mu`. Feeding stored draws to
  BridgeStan's `param_unconstrain()` requires inverting this per model. The
  testing framework's default/primary draw source is therefore **synthetic**:
  sample a random unconstrained vector and use BridgeStan's own
  `param_constrain(..., include_tp=True)` to get everything in Stan's own
  naming — sidesteps the mismatch entirely and is the only option for
  `dogs_nonhierarchical`, which has **no stored reference posterior at all**
  (`reference_posterior_name: null`, no file under `reference_posteriors/draws/`).
  This is independent of the bijector-fidelity decision above and still applies.
- **Gradient/Hessian comparison is organized per parameter block**, but — after
  this revision — no longer to permanently skip mismatched types. Since every
  constraint type's bijector is now built (or chained) to replicate Stan's own
  transform exactly, `assert_matches_stan` compares unconstrained-space log
  density, gradient, and Hessian for **every** block by default. The per-block
  structure is kept purely for debuggability: if a custom bijector's Jacobian
  formula turns out to be wrong, a per-block failure report pinpoints exactly
  which parameter is at fault, rather than one opaque failed assert over the
  whole flat vector. `ParameterSpec.matches_stan_unconstrained` remains in the
  code as a documented escape hatch (e.g. for a future constraint type added
  without yet writing/verifying its Stan-exact bijector) but defaults to `True`
  for every constraint type actually implemented in this plan.
- **Per-constraint-type implementation table** (every formula must be verified
  numerically — round-trip + Jacobian — before being trusted; see M1/M4):

  | Constraint | Implementation | Matches Stan exactly? |
  |---|---|---|
  | none | `Identity()` — TFP built-in | yes |
  | `lower=a` | `Chain([Shift(a), Exp()])` — TFP built-in | yes |
  | `upper=b` | `Chain([Shift(b), Scale(-1), Exp()])` — TFP built-in | yes |
  | `lower=a,upper=b` | `Sigmoid(low=a, high=b)` — TFP built-in | yes |
  | `ordered[K]` | `Chain([Cumsum(), Blockwise([Identity(), Exp()], [1,K-1])])` — TFP built-ins composed | yes — verify numerically |
  | `positive_ordered[K]` | `Chain([Cumsum(), Exp()])` — TFP built-ins composed | yes — verify numerically |
  | `cholesky_factor_cov[K]` (square) | `FillScaleTriL(diag_bijector=Exp())` — TFP built-in | yes |
  | `cov_matrix[K]` | `Chain([CholeskyOuterProduct(), FillScaleTriL(diag_bijector=Exp())])` — TFP built-in | yes |
  | `simplex[K]` | **custom `StanSimplex` bijector** (stick-breaking transform, per Stan Reference Manual) | yes, by construction — verify numerically |
  | `cholesky_factor_corr[K]` | **custom `StanCholeskyFactorCorr` bijector** (canonical-partial-correlation / tanh construction, per Stan Reference Manual) | yes, by construction — verify numerically |
  | `corr_matrix[K]` | `Chain([CholeskyOuterProduct(), StanCholeskyFactorCorr()])` — custom bijector chained with a TFP built-in | yes, by construction — verify numerically |
  | `unit_vector[K]` | deferred — needs its own custom bijector too; not required by any seed model | future work |

  "Verify numerically" means: (a) in M1, test each bijector's forward/inverse
  round trip and its analytic `forward_log_det_jacobian` against an
  independent finite-difference Jacobian, with no Stan dependency; (b) in M4,
  cross-check the same bijectors' `forward`/`inverse`/Jacobian against
  BridgeStan's own `param_constrain`/`param_unconstrain`/
  `log_density_gradient(jacobian=True)` output on a minimal single-constraint
  Stan file, before relying on them inside any full model conversion. This is
  materially more implementation work than reusing off-the-shelf bijectors,
  concentrated in `StanSimplex` and `StanCholeskyFactorCorr` — both transforms
  have Jacobian terms that are easy to get subtly wrong, so budget real time
  for M1/M4 rather than treating them as routine.
- **Matrix-shaped parameters print in column-major order in Stan/posteriordb**
  (`z[1,1], z[2,1], ..., z[1,2], ...`) vs. NumPy/JAX's default row-major reshape
  — the flatten/unflatten utility must use `order="F"` for any `ndim>=2` spec,
  a classic silent-bug source. Test directly against `dogs_nonhierarchical`'s
  `z` and BridgeStan's own `param_names()` ordering.
- None of the 120 Stan models in this repo has a *free* `corr_matrix`,
  `cov_matrix`, or `unit_vector` parameter (only as derived `transformed
  parameters`), so those bijectors are implemented for completeness but aren't
  end-to-end-tested against a real posteriordb model in the seed set —
  acceptable, not a blocker (their unit/M4 tests use minimal hand-written Stan
  files instead).
- `jax` defaults to float32; must call `jax.config.update("jax_enable_x64",
  True)` at package-import time or comparisons against Stan's float64 output
  will spuriously fail tolerance checks.
- `gcc`/`g++`/`make` are present in this environment (needed for BridgeStan's
  compile step). No `CMDSTAN`/`BRIDGESTAN` env var is set, so BridgeStan will
  bootstrap its own bundled Stan source on first use — needs network access
  once, then caches locally.
- Existing precedent for registering a non-Stan backend per model:
  `models/info/eight_schools_noncentered.info.json` has a `model_implementations`
  object with a `"pymc3"` key pointing at `models/pymc3/eight_schools_noncentered.py`.
  Worth mirroring with a `"posteriordb_jax"` key later, as a documentation nicety
  — not a runtime dependency of the library.

## Design

### Directory layout

```
python/posteriordb_jax/
  pyproject.toml
  README.md                       # deliverable: describes this design for future contributors
  .gitignore                      # .venv/, bridgestan cache if colocated
  posteriordb_jax/
    __init__.py                   # sets jax_enable_x64 = True
    constraints.py                # ParameterSpec, ConstraintType enum, bijector dispatch
    stan_bijectors.py             # custom tfp.bijectors.Bijector subclasses replicating Stan's
                                   #   exact transforms (StanSimplex, StanCholeskyFactorCorr, ...)
    flatten.py                    # ModelSpec, flatten/unflatten, posteriordb-dict conversion
    model.py                      # StatisticalModel base class
    io.py                         # load .stan / data.json.zip / draws.json.zip / posteriors.json
    testing.py                    # StanComparisonHarness, assert_matches_stan
    models/
      eight_schools_noncentered.py
      hmm_example.py
      dogs_nonhierarchical.py
  tests/
    conftest.py
    test_stan_bijectors.py        # round-trip + finite-diff Jacobian for each custom bijector,
                                   #   no Stan dependency
    test_constraints.py           # per-constraint-type unit tests (dispatch, shapes), no Stan dependency
    test_flatten.py               # round-trip + column-major matrix test
    test_model_base.py            # toy (non-Stan) model: grad/hessian/hvp vs finite diff,
                                   #   batching == loop, grad-wrt-data vs finite diff
    test_eight_schools_noncentered.py
    test_hmm_example.py
    test_dogs_nonhierarchical.py
```

### Custom Stan bijectors (`stan_bijectors.py`)

For each constraint type without a formula-identical TFP built-in
(`simplex`, `cholesky_factor_corr`, and later `unit_vector`), implement a
`tfp.bijectors.Bijector` subclass following Stan's exact transform as
documented in the Stan Reference Manual's constrained-parameter-transforms
chapter:
- `StanSimplex(K)`: the stick-breaking transform (`K-1` unconstrained inputs
  → a `K`-simplex), with its analytic log-det-Jacobian.
- `StanCholeskyFactorCorr(K)`: the canonical-partial-correlation (tanh-based)
  construction of a `K×K` correlation Cholesky factor from `K(K-1)/2`
  unconstrained inputs, with its analytic log-det-Jacobian.

Each subclass implements `_forward`, `_inverse`, `_forward_log_det_jacobian`
(or `_inverse_log_det_jacobian`), and declares `forward_min_event_ndims`/
`inverse_min_event_ndims` correctly (1 for simplex — vector event; 1 for the
Cholesky-factor case, since it's parameterized by a flat vector of the
lower-triangular free entries, with reshape to/from the `K×K` matrix handled
at the `ParameterSpec`/`flatten.py` layer, not inside the bijector itself —
keeps the bijector's own event shape simple and keeps it composable in a
`Chain`). Because these are real `Bijector` subclasses, `corr_matrix` is
literally `tfb.Chain([tfb.CholeskyOuterProduct(), StanCholeskyFactorCorr(K)])`
— no special-casing needed elsewhere in the codebase for the composed types.

### Constraint library (`constraints.py`)

`ParameterSpec` (frozen dataclass): `name`, `shape` (declared Stan shape),
`constraint` (the `ConstraintType` enum), `bounds` (for lower/upper types).
Derived/cached: `constrained_size`, `free_size`, `bijector` — built lazily,
dispatching to either a `tensorflow_probability.substrates.jax.bijectors`
construction or a `stan_bijectors.py` custom class per the table above —
`matches_stan_unconstrained` (expected `True` everywhere implemented),
`flatten_order` (`"F"` for `ndim>=2`, else irrelevant). Handle `K=1`
degenerate cases explicitly (`free_size==0` for e.g. `simplex[1]`).

### Flatten/unflatten (`flatten.py`)

A `ModelSpec` wraps `list[ParameterSpec]` with cached offsets. Provides, all
working uniformly with or without a leading batch dimension:
- `flatten_constrained`/`unflatten_constrained`, `flatten_unconstrained`/`unflatten_unconstrained`
- `constrain(unconstrained_flat) -> (dict, logdet)` / `unconstrain(dict) -> (flat, logdet)`
  via each spec's bijector `forward`/`inverse` + `forward_log_det_jacobian`
- `to_posteriordb_dict`/`from_posteriordb_dict`: convert to/from posteriordb's
  flat per-element naming (`"theta[1]"`, `"z[3,2]"`, 1-indexed, column-major)
  for interop with reference draws and BridgeStan's `param_names()`.

This is entirely constraint-type-agnostic; a per-model subclass never writes
shape or flattening code.

### `StatisticalModel` base class (`model.py`)

Exactly one piece of hand-written code per model: a scalar log-density function
of a *structured* constrained-parameter dict plus an **explicit data pytree
argument** (not read from `self.data` internally) — this explicitness is what
lets `jax.grad`/`jax.jacfwd` differentiate w.r.t. data.

```python
class StatisticalModel(abc.ABC):
    def __init__(self, data): ...
    @abstractmethod
    def param_specs(self, data) -> list[ParameterSpec]: ...
    @abstractmethod
    def log_prob_constrained(self, params: dict, data) -> Array: ...  # user-written
```

Everything else is derived generically from two flat-scalar core functions
(`_lp_unconstrained_flat` = constrain + user fn + logdet; `_lp_constrained_flat`
= unflatten + user fn, no Jacobian — matches BridgeStan's `jacobian=False`):
`log_prob_{constrained,unconstrained}`, `grad_*` (`jax.grad`), `hessian_*`
(`jax.hessian`), `hvp_*` (**forward-over-reverse** via `jax.jvp` of `jax.grad`,
never materializing a full Hessian), `grad_wrt_data(..., space=...)`
(`jax.grad(..., argnums=1)`), `constrain`/`unconstrain`,
`to_posteriordb_dict`/`from_posteriordb_dict`. Every method has a `_batch`
variant (`jax.vmap`, `in_axes=(0, None)` by default). Batched HVP
(`vmap` over the JVP-of-grad trick) is a first-class method so users aren't
forced into `O(n_draws * free_dim^2)` batched-Hessian cost when they only need
Hessian-vector products — documented as a real tradeoff, not hidden.

`jax.config.update("jax_enable_x64", True)` set once in `__init__.py`.

### Testing framework (`testing.py`)

**`StanComparisonHarness`**: wraps one `bridgestan.StanModel`, compiled once
(module/session-scoped in pytest — compilation is expensive). Exposes
`param_names`, `param_unc_names`, `param_constrain`, `param_unconstrain`,
`log_density`, `log_density_gradient`, `log_density_hessian` (confirm exact
kwarg names against the installed BridgeStan version during implementation).

**`assert_matches_stan(model, harness, block_map, draws=None, n_synthetic=20,
rtol, atol, check_hessian=True, stan_param_adapter=None)`**:
- `block_map`: correspondence between each `ParameterSpec` and index range(s)
  in Stan's unconstrained vector — auto-built by name match, overridable.
- `stan_param_adapter`: optional `dict -> dict` mapping the JAX model's
  constrained dict to Stan's *raw* `parameters`-block constrained values —
  needed for `eight_schools_noncentered` (`theta -> theta_trans`), identity
  for the other two seed models. (Unrelated to bijector fidelity — this
  handles Stan-side reparameterization/naming, not transform math.)
- Draw source: **synthetic by default** (sample unconstrained `x`, call
  `param_constrain(x, include_tp=True)`); posteriordb reference draws as an
  opt-in secondary source (loaded via `io.py`, requires `stan_param_adapter`
  when names diverge).
- Checks per draw, per parameter block:
  1. Constrained-space log-density **value** — always run, every constraint type.
  2. Unconstrained-space **value, gradient, and (if `check_hessian`) Hessian** —
     run for **every** block by default, since every implemented constraint
     type's bijector now replicates Stan's own transform exactly. A block is
     only exempted if `ParameterSpec.matches_stan_unconstrained` is explicitly
     `False` (the escape hatch for a not-yet-verified future constraint type,
     not expected to trigger for any type in this plan).
- Returns a structured per-draw/per-block report (pass/fail/exempt), not a
  bare assert, so pytest failures point at the exact block/draw/check — this
  granularity is what makes a bad Jacobian formula in `StanSimplex` or
  `StanCholeskyFactorCorr` easy to localize instead of just "the model's
  gradient is wrong somewhere."

### Model conversions (`models/*.py`)

Each is a `StatisticalModel` subclass: `param_specs()` derives shapes from
`data` (e.g. `J = len(data["y"])`, `n_dogs = data["n_dogs"]` — never from
`posteriors/*.json`), `log_prob_constrained()` hand-translates the Stan
`model`/`transformed parameters` block logic into JAX/`jax.numpy` +
`tensorflow_probability.substrates.jax.distributions` log-prob calls.

## Sequencing

1. **M0 — Environment.** `python3 -m venv python/posteriordb_jax/.venv`;
   `pip install jax tensorflow-probability bridgestan numpy pytest` inside it.
   Compile one trivial hand-written Stan file via `bridgestan` to confirm the
   C++ toolchain + one-time network bootstrap work before building anything on
   top. Add `.venv/` to `.gitignore`.
2. **M1 — Custom Stan bijectors + constraint library + unit tests.** Implement
   `stan_bijectors.py` (`StanSimplex`, `StanCholeskyFactorCorr`) and
   `constraints.py`'s bijector dispatch. No Stan dependency yet: verify
   support membership, `inverse(forward(x))≈x` round trips, `K=1` degenerate
   cases, and — critically — each bijector's analytic
   `forward_log_det_jacobian` against an independent finite-difference
   Jacobian for every constraint type, with extra scrutiny on `StanSimplex`
   and `StanCholeskyFactorCorr` where the formulas are least routine.
3. **M2 — Flatten/unflatten + unit tests.** Round-trip a synthetic
   multi-parameter dict including a matrix-shaped parameter; test column-major
   convention explicitly.
4. **M3 — `StatisticalModel` base + generic tests**, using a hand-rolled toy
   model (no Stan): grad/hessian/hvp vs. finite differences, `_batch` == loop
   over single draws, `grad_wrt_data` vs. finite differences.
5. **M4 — BridgeStan harness + bijector cross-verification.** Build
   `StanComparisonHarness`/`assert_matches_stan` against the real installed
   BridgeStan API; validate on a minimal 2-line hand-written Stan model first.
   Then, **before moving to full model conversions**, cross-check every custom
   bijector (`StanSimplex`, `StanCholeskyFactorCorr`) and every hand-composed
   chain (`ordered`, `positive_ordered`) against BridgeStan's
   `param_constrain`/`param_unconstrain`/`log_density_gradient(jacobian=True)`
   on a minimal single-constraint-type Stan file each — a bijector bug caught
   here is far cheaper to fix than one discovered inside a full model's
   combined gradient.
6. **M5 — `eight_schools_noncentered`.** Simplest constraint mix (unconstrained
   + one lower bound) → full unconstrained-space validation, and the model
   that establishes the `stan_param_adapter` pattern (`theta`↔`theta_trans`)
   for future reparameterized models.
7. **M6 — `hmm_example`.** Adds `simplex` (now via the verified
   `StanSimplex` bijector) and `positive_ordered` — full unconstrained-space
   gradient/Hessian validation for both, no longer a constrained-value-only
   check for the simplex parameters. Uses posteriordb reference draws
   directly, since names map 1:1 here.
8. **M7 — `dogs_nonhierarchical`.** Adds `cholesky_factor_corr` (now via the
   verified `StanCholeskyFactorCorr` bijector, fully validated rather than
   skipped), a matrix-shaped free parameter `z` (real column-major flattening
   test), and data-dependent shapes. Has no stored reference draws, so depends
   on M4's synthetic path being solid first.
9. **M8 (stretch) — polish.** Register `"posteriordb_jax"` in the three
   converted models' `info.json` (mirroring the `pymc3` precedent), add a
   `models/__init__.py` registry, and write the package `README.md`.

## Verification

- Every module gets pytest unit tests as listed above; run via the venv's
  `pytest python/posteriordb_jax/tests/`.
- Each seed model's correctness is verified end-to-end by
  `assert_matches_stan` against BridgeStan — constrained-space log-density
  value **and** unconstrained-space log-density/gradient/Hessian are checked
  for every parameter block, for every seed model.
- `test_model_base.py`'s toy-model tests give an independent, Stan-free check
  that grad/hessian/hvp/batching/data-differentiation are implemented
  correctly, decoupled from any BridgeStan/compilation flakiness.
- `test_stan_bijectors.py` gives an independent, Stan-free check (finite
  differences) of each custom bijector's Jacobian, decoupled from BridgeStan
  — a bug that happens to fool both the finite-difference check and BridgeStan
  simultaneously is extremely unlikely, so agreement across both is strong
  evidence the formula is right.
- After M5–M7, manually run each seed model's test file and confirm **every**
  block passes unconstrained-space comparison — since every constraint type
  now has a Stan-exact transform, there should be no expected skips left; any
  block that fails or has to fall back to the `matches_stan_unconstrained`
  exemption is a real bug (in the custom bijector's formula or in the model's
  `log_prob_constrained`), not a documented limitation.
