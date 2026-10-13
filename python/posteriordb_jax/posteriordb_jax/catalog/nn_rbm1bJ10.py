"""nn_rbm1bJ10.stan: the MNIST network in _nn_rbm.py with J = 10 hidden units."""

from ._nn_rbm import build_nn_rbm_model

STAN_MODEL = "nn_rbm1bJ10"
POSTERIORS = ["mnist_100-nn_rbm1bJ10"]


def build(raw):
    return build_nn_rbm_model(name=STAN_MODEL, raw=raw, n_hidden=10)
