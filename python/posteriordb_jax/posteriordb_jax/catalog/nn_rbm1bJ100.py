"""nn_rbm1bJ100.stan: the MNIST network in _nn_rbm.py with J = 100 hidden units."""

from ._nn_rbm import nn_rbm_model

STAN_MODEL = "nn_rbm1bJ100"
POSTERIORS = ["mnist-nn_rbm1bJ100"]


def build(raw):
    return nn_rbm_model(name=STAN_MODEL, raw=raw, n_hidden=100)
