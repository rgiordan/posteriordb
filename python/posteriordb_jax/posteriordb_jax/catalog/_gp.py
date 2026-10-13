"""Shared code for the Gaussian-process models."""

import jax.numpy as jnp


def compute_exp_quad_cov(*, x, alpha, rho):
    """Stan's gp_exp_quad_cov(x, alpha, rho) for scalar inputs x."""
    sq_dist = (x[:, None] - x[None, :]) ** 2
    return alpha**2 * jnp.exp(-0.5 * sq_dist / rho**2)
