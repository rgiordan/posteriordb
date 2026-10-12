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

params = pattern.random(jax.random.PRNGKey(0))     # {"beta": (3,), "sigma": (), "w": (4,)}
x = pattern.folded_to_free(params)                 # unconstrained vector, length 3 + 1 + 3
y = pattern.folded_to_flat(params)                 # constrained vector, length 3 + 1 + 4
params = pattern.free_to_folded(x)
params, log_det = pattern.free_to_folded_with_log_det(x)   # log |Jacobian| of free -> folded

# Or pick the vector representation with a keyword:
x = pattern.flatten(params, to_free=True)
params = pattern.fold(x, from_free=True)

def objective(params): ...
f = pg.function_input_folded_to_free(objective, pattern)   # now a function of x
jax.grad(f)(x)
```

- Everything is pure JAX, so it works with `grad`, `jit` and `vmap`. A pattern acts on one value; use `jax.vmap` for batches.
- The unconstraining transforms are Stan's, including the newer (>= 2.37) simplex transform and Stan's fill order for Cholesky factors. Free vectors and log Jacobians therefore match Stan's unconstrained space exactly. `posteriordb_jax/tests/test_paragami_matches_constraints.py` checks this against the BridgeStan-verified transforms.
- Function wrappers change the representation a function takes or returns: `function_input_<from>_to_<to>` (for example `function_input_folded_to_free`, as above) and `function_output_<from>_to_<to>`.
- Sizes and labels: `flat_length()`, `free_length()`, `flat_names()`, `free_names()`.
- Validation: `pattern.validate_folded(value)` checks keys, shapes and constraint values. Value checks are skipped automatically under `jit`; shape checks are not.

Patterns: `NumericArrayPattern` (with `NumericVectorPattern` and `NumericScalarPattern` shorthands; optional `lb`/`ub`; `order="F"` for Stan's column-major layout), `OrderedVectorPattern` (`positive=True` for `positive_ordered`), `SimplexArrayPattern`, `CholeskyCovPattern`, `CholeskyCorrPattern`, `PSDSymmetricMatrixPattern`, and the containers `PatternDict` and `PatternArray`.

Not yet ported from the original: JSON save/load, sparse Jacobians, `optimization_lib`, and the sparse preconditioners. Not yet written: Stan's `unit_vector` and a general `corr_matrix`.

Tests: `pytest tests`.
