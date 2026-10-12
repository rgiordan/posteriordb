"""paragami: parameter origami in JAX.

Fold and flatten structured, constrained parameters to and from flat
vectors and unconstrained ("free") vectors, with log Jacobians, all
differentiable and jit-able. Unconstraining transforms match Stan's, or
TensorFlow Probability's with `bijector="tfp"`.

A JAX rewrite of https://github.com/rgiordan/paragami (originally autograd).
"""

from .arviz_io import (
    arviz_to_folded_draws,
    arviz_to_free_draws,
    flat_draws_to_folded,
    folded_draws_to_arviz,
    folded_draws_to_flat,
    folded_draws_to_free,
    folded_draws_to_variables,
    free_draws_to_arviz,
    free_draws_to_folded,
    variables_to_folded_draws,
)
from .base import BIJECTORS, Pattern, ValidationError, is_concrete
from .codegen import unconstrained_pattern_like
from .containers import PatternArray, PatternDict
from .functions import (
    function_input_flat_to_folded,
    function_input_folded_to_flat,
    function_input_folded_to_free,
    function_input_free_to_folded,
    function_output_flat_to_folded,
    function_output_folded_to_flat,
    function_output_folded_to_free,
    function_output_free_to_folded,
)
from .matrix import CholeskyCorrPattern, CholeskyCovPattern, CorrMatrixPattern, PSDSymmetricMatrixPattern
from .numeric import (
    NumericArrayPattern,
    NumericScalarPattern,
    NumericVectorPattern,
    OrderedVectorPattern,
    UnitVectorPattern,
)
from .simplex import SimplexArrayPattern, StochasticMatrixPattern, SumToZeroMatrixPattern, SumToZeroVectorPattern

__version__ = "1.0.0.dev0"
