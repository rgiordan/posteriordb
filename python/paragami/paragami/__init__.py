"""paragami: parameter origami in JAX.

Fold and flatten structured, constrained parameters to and from flat
vectors and unconstrained ("free") vectors, with log Jacobians, all
differentiable and jit-able. Unconstraining transforms match Stan's.

A JAX rewrite of https://github.com/rgiordan/paragami (originally autograd).
"""

from .base import Pattern, ValidationError, is_concrete
from .containers import PatternArray, PatternDict
from .functions import (
    FlattenFunctionInput,
    FlattenFunctionOutput,
    FoldFunctionInput,
    FoldFunctionOutput,
    TransformFunctionInput,
)
from .matrix import CholeskyCorrPattern, CholeskyCovPattern, PSDSymmetricMatrixPattern
from .numeric import (
    NumericArrayPattern,
    NumericScalarPattern,
    NumericVectorPattern,
    OrderedVectorPattern,
)
from .simplex import SimplexArrayPattern

__version__ = "1.0.0.dev0"
