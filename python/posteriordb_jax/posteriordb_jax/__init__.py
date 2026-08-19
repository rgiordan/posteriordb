"""posteriordb_jax: JAX/TensorFlow-Probability translations of posteriordb Stan models.

Importing this package enables 64-bit precision in JAX. This is required for
log-density/gradient comparisons against Stan (which always operates in
float64) to pass at tight tolerances -- JAX defaults to float32.
"""

import jax

jax.config.update("jax_enable_x64", True)
