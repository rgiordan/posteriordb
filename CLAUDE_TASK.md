# Goal

The goal is to translate the Stan model code in this repository into Python using
jax and tensorflow probability,
but with a similar user interface to the existing Stan model creation.  Specifically, create
a Python library that

- Defines a StatisticalModel class that can be instantiated from Stan data with a similar
  user interface that provides methods for
    - log probability, log probability gradients, Hessians, and Hessian-vector products,
      as a function of model parameters
        - all as functions of both constrained and unconstrained parameters
        - vectorized across parameter draws or at a single parameter draw
        - Unlike Stan, it should be possible to evaluate these as functions of the
          data allowing, e.g., automatic differentiation of the log probability with respect
          to hyperparameters or dataset values
    - functions to convert parameters to unconstrained parameters and back
    - functions to convert structured parameter representations (lists of dictionaries)
      to matrices and back

# Design

As much as possible code should be shared across models.  For example, a standalone
library for constraining and unconstraining parameter representations should be
created and used.  A common interface for declaring parameter constrains should
be used.  If there are existing probabilistic programming language conventions
they should be used.

Ideally, it should be as easy as possible for AI to convert future Stan model
code into the new framework and test it using the testing framework below.

Unlike Stan, the StatisticalModel class should not require recompilation for new
data.  This will allow, for example, the use of data subsampling which would
currently require a Stan model to recompile.


# Testing

To ensure that the Python models are correct, there should be a testing framework
that checks that the log probability and derivatives of the Python code matches
the existing Stan code.

This testing framework should be easy to use
for a given model and set of posterior draws, allowing dynamic testing of future
conversion of Stan models.

