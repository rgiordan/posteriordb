# modelgami

A common, JAX-based interface for statistical models, for methodological
research. You write a model's statistics once, in terms of dicts of arrays:
a [paragami](../paragami) pattern for the parameters, a log prior and a log
likelihood. Everything else is generated on request. A method written
against this interface then runs on any model.

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

model = mg.Model(
    params_pattern=pg.PatternDict(beta=pg.NumericVectorPattern(3), sigma=pg.NumericScalarPattern(lb=0.0)),
    data={"x": X, "y": y},          # or a list of per-observation dicts
    hyper={"scale": 10.0},
    log_prior=log_prior,
    log_lik_one=log_lik_one,        # exchangeable data; use log_lik=... otherwise
)
```

Optional hooks: `sample_prior(key, hyper)`, `sample_one(key, params, datum, hyper)` (or `sample_data` for non-exchangeable data), `log_lik_pointwise` (a vectorized alternative to `log_lik_one`), and `check_params` / `check_data` for extra validation. The same model can also be written as a subclass of `mg.Model`, with these as methods and `params_pattern` as a method or attribute. See [`examples/conjugate_regression.py`](modelgami/examples/conjugate_regression.py) for both styles.

### Write the likelihood once, get the density and the sampler

Instead of writing `log_lik_one` and `sample_one` separately, you can return the observation's distribution, using any distribution library (NumPyro's or TFP's distributions). It is an ordinary JAX function:

```python
import numpyro.distributions as dist

def obs_dists_one(params, datum, hyper):
    return {"y": dist.Normal(datum["x"] @ params["beta"], params["sigma"])}

model = mg.Model(..., obs_dists_one=obs_dists_one)   # gives log_lik_one and sample_one
```

The prior works the same way, including parameters that depend on earlier ones:

```python
def prior_dists(hyper):
    return {
        "sigma2": dist.InverseGamma(hyper["a0"], hyper["b0"]),
        "beta": lambda p: dist.MultivariateNormal(hyper["m0"], p["sigma2"] * hyper["V0"]),
    }

model = mg.Model(..., prior_dists=prior_dists)        # gives log_prior and sample_prior
```

These are optional; hand-written `log_*` and `sample_*` functions always work.

### One program for simulation and density (NumPyro)

Probabilistic programming languages specify a model once and use the same program both to simulate and to evaluate the log density. `mg.from_numpyro(program, data, hyper, exchangeable=True)` turns a NumPyro program `program(data, hyper)` into a `Model`:

- Latent sites become the parameters, with a paragami pattern built from their supports, so the unconstrained space uses Stan's transforms.
- Latent sites make up the prior, and observed sites make up the likelihood. With `exchangeable=True`, a plate over observations gives the pointwise likelihood.
- Running the program forward gives `sample_prior` and `sample_data`.

The hand-written hooks remain the interface methods rely on; the adapter just generates them. See `tests/test_numpyro_adapter.py`, which checks it against the hand-written conjugate regression.

Data shared across observations (sizes, design constants) go in `shared_data=`. `log_lik_one` sees them as part of `datum`.

## What you get

| | |
|---|---|
| `model.log_density_fn(which, params_as, data_as)` | `"joint"`, `"lik"` or `"prior"` as a JAX function of parameters (and data) in `"folded"` (dict), `"flat"` (constrained vector) or `"free"` (unconstrained vector) form. Use it with `jax.grad`, `jax.hessian`, `jax.vmap`, `jax.jit`, or `mg.hvp`. In free form the log Jacobian is included for `joint`/`prior` (`jacobian=False` turns it off). |
| `model.log_lik(params)`, `model.log_prior(params)` | Your hooks, with the model's data and hyperparameters as defaults. |
| `model.log_lik_pointwise(params)` | Per-observation log likelihoods (exchangeable models). |
| `model.log_lik(params, weights=w)` | Weighted likelihood; `jax.grad` with respect to `w` gives the influence of each observation. |
| `model.log_lik_subset(params, idx)`, `model.subset(idx)`, `model.bootstrap_weights(key)` | Minibatches (scaled to be unbiased) and the bootstrap. |
| `model.fold_params`, `model.flatten_params`, `model.random_params`, `model.free_dim` | Moving between representations. |
| `model.sample_prior(key)`, `model.sample_data(key, params)` | Simulation, when the hooks are given. |
| `model.validate_params(p)`, `model.validate_data(d)` | Pattern checks plus your own checks. Data are checked when the model is built. |
| `model.with_data(new_data)` | Same model, new data, no recompilation. |
| `model.prior`, `model.likelihood` | The two pieces as `DGP` objects (log density + sampler). |
| `model.capabilities` | What the model supports, for filtering models in a study. |
| `model.to_arviz(draws)` | ArviZ export (posterior, pointwise log likelihood for `az.loo`, observed and constant data). |

`modelgami.testing.check_model_contract(model)` checks that all of the above agree with each other. `assert_models_match(a, b)` compares two models.

Tests: `pytest tests`.
