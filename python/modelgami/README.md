# modelgami

A common, JAX-based interface for statistical models, for methodological
research. A model is a [paragami](../paragami) pattern for its parameters,
a prior and a likelihood, its data and its hyperparameters. The prior and
the likelihood are the same kind of object, a data-generating process
(`DGP`): a log density and a sampler, either of which may be missing. A
method written against this interface then runs on any model.

## Writing a model

```python
import jax.numpy as jnp
from jax.scipy import stats
import paragami as pg
import modelgami as mg

def log_prior(params, hyper):
    return stats.norm.logpdf(params["beta"], 0.0, hyper["scale"]).sum() + stats.expon.logpdf(params["sigma"])

def log_lik_one(params, datum, hyper):            # one observation: {"x": (p,), "y": ()}
    return stats.norm.logpdf(datum["y"], datum["x"] @ params["beta"], params["sigma"])

model = mg.ExchangeableModel(
    params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(3), sigma=pg.NumericScalarPattern(lb=0.0)),
    prior=mg.DGP(log_density=log_prior),
    likelihood=mg.ExchangeableDGP(log_density_one=log_lik_one),
    data={"x": X, "y": y},          # N observations: every field has a leading N axis
    hyper={"scale": 10.0},
)
```

The functions each piece takes, with the parameters always first:

| | log density | sampler |
|---|---|---|
| prior, a `DGP` | `log_density(params, hyper)` | `sample(key, hyper) -> params` |
| likelihood, a `DGP` | `log_density(params, data, hyper)` | `sample(key, params, data, hyper) -> data` |
| exchangeable likelihood, an `ExchangeableDGP` | `log_density_one(params, datum, hyper)` | `sample_one(key, params, datum, hyper) -> {drawn fields}` |

Pass them to the constructor or subclass and define them as methods; see [`examples/conjugate_regression.py`](modelgami/examples/conjugate_regression.py) for both. A missing one raises `NotImplementedError` when called. `prior=None` is a flat prior.

Exchangeable data are an array of N data: a dict (or any pytree) whose every leaf has a leading axis of length N. Anything shared by all observations goes in `hyper`. For a non-exchangeable model, use `mg.Model` with a `DGP` likelihood and any data.

### Exchangeable likelihoods from distributions

`mg.exchangeable_dgp_from_distributions(dists_one)` writes an exchangeable likelihood once, as distribution objects (NumPyro's or TFP's), and gives both its density and its sampler:

```python
import numpyro.distributions as dist

likelihood = mg.exchangeable_dgp_from_distributions(
    lambda params, datum, hyper: {"y": dist.Normal(datum["x"] @ params["beta"], params["sigma"])})
```

### One program for simulation and density (NumPyro)

`mg.from_numpyro(program, data, hyper, exchangeable=True)` turns a NumPyro program `program(data, hyper)` into a model. Latent sites become the parameters (with a paragami pattern built from their supports, so the unconstrained space uses Stan's transforms) and the prior; observed sites become the likelihood. Running the program forward gives the samplers. `tests/test_numpyro_adapter.py` checks it against the hand-written conjugate regression.

### Stan programs as models (BridgeStan)

`mg.from_bridgestan("model.stan", data)` returns a `StanModel` (optional extra `modelgami[stan]`). Its `log_density_fn("joint", params_as="free")` is Stan's log density, and works with `jax.grad`, `jax.hessian`, `mg.hvp`, `jit` and `vmap`, with derivatives from Stan itself. Its `params_pattern` uses Stan's own transforms. A Stan program has a single `target` and fixed data, so there is no prior/likelihood split. For new data, call `from_bridgestan` again with `model.bridgestan.lib_path`; nothing is recompiled. Its main use is as a reference for a JAX translation (`assert_models_match`).

## What you get

| | |
|---|---|
| `model.log_density_fn(which, params_as=..., jacobian=...)` | `"joint"`, `"lik"` or `"prior"` as a JAX function `f(p, data=None, hyper=None, **likelihood_kwargs)` of the parameters in `"folded"` (dict), `"flat"` (constrained vector) or `"free"` (unconstrained vector) form. Use it with `jax.grad`, `jax.hessian`, `jax.vmap`, `jax.jit`, or `mg.hvp`; the data and hyperparameters are arguments too. By default each function is, up to a constant, the density it names with respect to the parameters you pass, so in free form the log Jacobian is included for `joint` and `prior` but not for `lik`. `jacobian=True` or `False` overrides this. |
| `model.log_prior(params, hyper=None)`, `model.log_lik(params, data=None, hyper=None)` | The prior's and likelihood's log densities, with the model's data and hyperparameters as defaults. |
| `model.sample_prior(key)`, `model.sample_data(key, params)` | Simulation. |
| `model.params_pattern` | paragami, to move parameters between representations. |
| `model.to_arviz(draws)` | ArviZ export: posterior, pointwise log likelihood for `az.loo` (exchangeable models), observed data. |

An `ExchangeableModel` also has:

| | |
|---|---|
| `model.n_obs`, `model.log_lik_pointwise(params)` | The number of observations and the per-observation log likelihoods. |
| `model.log_lik(params, weights=w)` | Weighted likelihood; `jax.grad` with respect to `w` gives the influence of each observation. |
| `model.bootstrap_weights(key)` | Multinomial weights for the bootstrap. |
| `model.likelihood.subset(data, idx)`, `model.likelihood.log_density_subset(params, data, hyper, idx)` | Data subsets and minibatch estimates (scaled to be unbiased). |

`modelgami.testing.check_model_contract(model)` checks that the folded, flat and free densities and the exchangeable extras agree with each other. `assert_models_match(a, b)` compares two models.

Tests: `pytest tests`.
