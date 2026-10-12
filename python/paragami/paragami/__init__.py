"""paragami: parameter origami in JAX.

Fold and flatten structured, constrained parameters to and from flat
vectors and unconstrained ("free") vectors, with log Jacobians, all
differentiable and jit-able. Unconstraining transforms match Stan's.

A JAX rewrite of https://github.com/rgiordan/paragami (originally autograd).
"""

from .base import Pattern, ValidationError, is_concrete
from .codegen import pattern_to_code
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
from .infer import unconstrained_pattern_like
from .matrix import CholeskyCorrPattern, CholeskyCovPattern, PSDSymmetricMatrixPattern
from .numeric import (
    NumericArrayPattern,
    NumericScalarPattern,
    NumericVectorPattern,
    OrderedVectorPattern,
)
from .simplex import SimplexArrayPattern

__version__ = "1.0.0.dev0"
