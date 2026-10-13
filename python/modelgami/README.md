# modelgami

A common, JAX-based interface for statistical models, for methodological
research. A model is a prior and a likelihood, a [paragami](../paragami)
pattern for its parameters, its data and its hyperparameters. The prior and
the likelihood are the same kind of object, a data-generating process
(`DGP`): a log density and a sampler of one value given some conditioning
arguments, either of which may be missing. A method written against this
interface then runs on any model.

## Writing a model

```python
import jax.numpy as jnp
from jax.scipy import stats
import paragami as pg
import modelgami as mg

def eval_log_prior(params, hyper):
    return stats.norm.logpdf(params["beta"], 0.0, hyper["scale"]).sum() + stats.expon.logpdf(params["sigma"])

def eval_log_lik_one(datum, params, hyper):       # one observation: {"x": (p,), "y": ()}
    return stats.norm.logpdf(datum["y"], datum["x"] @ params["beta"], params["sigma"])

model = mg.ExchangeableModel(
    prior=mg.DGP(eval_log_density=eval_log_prior),
    likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one),
    data={"x": X, "y": y},          # N observations: every field has a leading N axis
    hyper={"scale": 10.0},
    params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(3), sigma=pg.NumericScalarPattern(lb=0.0)),
)
```

A DGP's functions take the value they describe first, then what it is conditioned on:

| | log density | sampler |
|---|---|---|
| prior, a `DGP` | `eval_log_density(params, hyper)` | `draw(key, hyper) -> params` |
| likelihood, a `DGP` | `eval_log_density(data, params, hyper)` | `draw(key, data, params, hyper) -> data` |
| exchangeable likelihood, an `ExchangeableDGP` | `eval_log_density_one(datum, params, hyper)` | `draw_one(key, datum, params, hyper) -> {drawn fields}` |

The likelihood's sampler takes the current data for their covariates and shapes. Pass the functions to the constructor, or subclass and define them as methods; see [`examples/conjugate_regression.py`](modelgami/examples/conjugate_regression.py) for both. A missing one raises `NotImplementedError` when called. `prior=None` is a flat prior. Without `params_pattern`, the model infers an unconstrained pattern from one prior draw.

Exchangeable data are an array of N data: a dict (or any pytree) whose every leaf has a leading axis of length N. Anything shared by all observations goes in `hyper`. For a non-exchangeable model, use `mg.Model` with a `DGP` likelihood and any data.

## Constrained and unconstrained parameters

`mg.free_dgp(dgp, pattern)` turns a distribution of a constrained value into the same distribution of its free (unconstrained) vector: the log density adds the log Jacobian of the free-to-folded map, and draws are converted to free vectors. `mg.folded_dgp` goes the other way. Only the prior is a density over the parameters, so the unconstrained target for HMC or VI is the free prior plus the likelihood, which is what `model.eval_free_log_joint(free_params)` computes. For a function of the free vector without the Jacobian (for example a posterior mode), compose with paragami: `model.eval_log_joint(pattern.free_to_folded(u))`.

## What a model does

Every method takes `data=None, hyper=None`, defaulting to the model's own.

| | |
|---|---|
| `eval_log_prior(params)`, `eval_log_lik(params)`, `eval_log_joint(params)` | Log densities of the folded parameters. |
| `eval_free_log_joint(free_params)` | The log joint as a density over the free parameters. Use it with `jax.grad`, `jax.hessian`, `jax.vmap`, `jax.jit`, or `mg.make_hvp`. |
| `draw_prior(key)`, `draw_data(key, params)` | Draws from the prior and the likelihood. |
| `draw_joint(key)`, `draw_marginal_data(key)` | `(params, data)` from the joint distribution, and data from their marginal (prior predictive) distribution. |
| `params_pattern` | paragami, to move parameters between representations. |

An `ExchangeableModel` also has:

| | |
|---|---|
| `n_obs`, `eval_log_lik_singleobs(params)` | The number of observations and the (N,) single-observation log likelihoods. |
| `eval_log_joint(params, weights=w)` | Weighted log joint; `jax.grad` with respect to `w` gives the influence of each observation. |
| `eval_log_joint_subset(params, idx)` | The log joint with the likelihood of observations `idx`, scaled to be unbiased for minibatches. |
| `draw_np_bootstrap_weights(key)` | Nonparametric bootstrap weights; the standalone `mg.draw_np_bootstrap_weights(key, n_obs)` works for any N. |
| `likelihood.select_subset(data, idx)` | The observations `idx`. |

For ArviZ, use paragami's `pg.free_draws_to_arviz`, passing the single-observation log likelihoods as `extra_groups={"log_likelihood": {...}}` for `az.loo`.

### Exchangeable likelihoods from distributions

`mg.exchangeable_dgp_from_distributions(make_dists_one)` writes an exchangeable likelihood once, as distribution objects (NumPyro's or TFP's), and gives both its density and its sampler:

```python
import numpyro.distributions as dist

likelihood = mg.exchangeable_dgp_from_distributions(
    lambda datum, params, hyper: {"y": dist.Normal(datum["x"] @ params["beta"], params["sigma"])})
```

### One program for simulation and density (NumPyro)

`mg.from_numpyro(program, data, hyper, exchangeable=True)` turns a NumPyro program `program(data, hyper)` into a model. Latent sites become the parameters (with a paragami pattern built from their supports, so the unconstrained space uses Stan's transforms) and the prior; observed sites become the likelihood. Running the program forward gives the samplers. `tests/test_numpyro_adapter.py` checks it against the hand-written conjugate regression.

### Stan programs as models (BridgeStan)

`mg.from_bridgestan("model.stan", data)` returns a `StanModel` (optional extra `modelgami[stan]`). Its `eval_free_log_joint` is Stan's log density (`jacobian=False` leaves out the log Jacobian), and works with `jax.grad`, `jax.hessian`, `mg.make_hvp`, `jit` and `vmap`, with derivatives from Stan itself. Its `params_pattern` uses Stan's own transforms. A Stan program has a single `target` and fixed data, so there is no prior/likelihood split. For new data, call `from_bridgestan` again with `model.bridgestan.lib_path`; nothing is recompiled. Its main use is as a reference for a JAX translation (`assert_models_match`).

`modelgami.testing.check_model_contract(model)` checks that the free and folded densities and the exchangeable extras agree with each other. `assert_models_match(a, b)` compares two models' free log joints.

Tests: `pytest tests`.
