# paragami (JAX)

"Parameter origami": fold and flatten structured, constrained parameters to
and from flat vectors and unconstrained ("free") vectors. This is a JAX
rewrite of [rgiordan/paragami](https://github.com/rgiordan/paragami), which
was built on autograd.

A value has three representations: **folded** (the value itself, such as a
dict of arrays), **flat** (its entries as one vector, still constrained) and
**free** (an unconstrained vector). Conversions are named
`<from>_to_<to>`, and keyword arguments say which side they describe with a
`to_` or `from_` prefix, so there is no doubt about direction.

```python
import jax, jax.numpy as jnp
import paragami as pg

pattern = pg.PatternDict(
    beta=pg.NumericVectorPattern(3),
    sigma=pg.NumericScalarPattern(lb=0.0),
    w=pg.SimplexArrayPattern(4),
)

params = pattern.draw_random(jax.random.PRNGKey(0))     # {"beta": (3,), "sigma": (), "w": (4,)}
x = pattern.folded_to_free(params)                 # unconstrained vector, length 3 + 1 + 3
y = pattern.folded_to_flat(params)                 # constrained vector, length 3 + 1 + 4
params = pattern.free_to_folded(x)                 # no log determinant computed
log_det = pattern.free_to_folded_log_abs_det(x)    # log |Jacobian| of free -> folded, on request
params, log_det = pattern.free_to_folded_with_log_det(x)

# Or pick the vector representation with a required keyword:
x = pattern.flatten(params, to_free=True)
params = pattern.fold(x, from_free=True)

def objective(params): ...
f = pg.function_input_folded_to_free(objective, patterns=pattern)   # now a function of x
jax.grad(f)(x)
```

- Everything is pure JAX, so it works with `grad`, `jit` and `vmap`. A pattern acts on one value; use `jax.vmap` for batches.
- By default the unconstraining transforms are Stan's, including the newer (>= 2.37) simplex and sum-to-zero transforms and Stan's fill orders for Cholesky and correlation factors. Free vectors and log Jacobians therefore match Stan's unconstrained space exactly. Where TensorFlow Probability uses a different transform, `bijector="tfp"` selects TFP's instead: softplus for one-sided bounds, `SoftmaxCentered` for simplexes, `FillScaleTriL` and `CholeskyOuterProduct` for covariances, and `CorrelationCholesky` for correlations. Bounded, ordered and affine transforms are the same in both, so they have no option. `tests/test_against_stan_and_tfp.py` checks every transform against BridgeStan and TFP.
- Validation is on by default. `pattern.disable_validation()` skips all input checks in conversions, for speed in an inner loop; on a `PatternDict` or `PatternArray` it applies to every child. `enable_validation()` turns checks back on. `pattern.validate_folded(value)` always checks keys, shapes and constraint values. Value checks are skipped automatically under `jit`; shape checks are not.
- `make_unconstrained_pattern_like(value, default_name="x")` builds an unconstrained pattern matching a nested structure of dicts, lists and tuples with arrays or numbers at the leaves. Dict keys become the names; list and tuple entries are named from `default_name` with an index.
- `pattern.generate_code()` returns Python code that rebuilds any pattern, so `make_unconstrained_pattern_like(value).generate_code()` gives code for the inferred pattern, ready to paste and edit its constraints.
- Draws and ArviZ (optional, `pip install paragami[arviz]`): `free_draws_to_arviz(pattern=pattern, draws=draws)` turns sampler output of shape (chain, draw, free length) into ArviZ data with one constrained variable per pattern entry, and `arviz_to_free_draws(pattern=pattern, data=idata)` reads it back. `folded_draws_to_arviz` and `arviz_to_folded_draws` do the same for folded draws, and `free_draws_to_folded`, `flat_draws_to_folded`, `folded_draws_to_free` and `folded_draws_to_flat` convert draw arrays without ArviZ.
- Function wrappers change the representation a function takes or returns: `function_input_<from>_to_<to>` (for example `function_input_folded_to_free`, as above) and `function_output_<from>_to_<to>`.
- Sizes and labels: `get_flat_length()`, `get_free_length()`, `get_flat_names()`, `get_free_names()`.

Patterns, with the Stan type each one matches:

| pattern | Stan type |
|---|---|
| `NumericArrayPattern`, `NumericVectorPattern`, `NumericScalarPattern` | `real`, `vector`, `array`, with optional `lb`/`ub` or `offset`/`multiplier`; `order="F"` for Stan's column-major layout |
| `OrderedVectorPattern` | `ordered`, or `positive_ordered` with `positive=True` |
| `UnitVectorPattern` | `unit_vector` |
| `SimplexArrayPattern` | `simplex`, or an array of simplexes |
| `StochasticMatrixPattern` | `column_stochastic_matrix` (`kind="column"`) and `row_stochastic_matrix` (`kind="row"`) |
| `SumToZeroVectorPattern`, `SumToZeroMatrixPattern` | `sum_to_zero_vector`, `sum_to_zero_matrix` |
| `CholeskyCovPattern`, `PSDSymmetricMatrixPattern` | `cholesky_factor_cov`, `cov_matrix` |
| `CholeskyCorrPattern`, `CorrMatrixPattern` | `cholesky_factor_corr`, `corr_matrix` |
| `PatternDict`, `PatternArray` | containers: named entries, or an array of one pattern |

Not yet ported from the original: JSON save/load, sparse Jacobians, `optimization_lib`, and the sparse preconditioners.

Tests: `pytest tests`. The Stan and TFP comparisons need BridgeStan (with a Stan toolchain) and TFP, which needs jax <= 0.5; they are skipped otherwise.
