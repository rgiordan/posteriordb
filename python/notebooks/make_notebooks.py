"""Generate the demo notebooks in python/notebooks (cells only; execute separately)."""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])

KERNEL = {
    "kernelspec": {"name": "modelgami", "display_name": "Python (modelgami)", "language": "python"},
    "language_info": {"name": "python"},
}


def notebook(name, cells):
    nb = nbf.v4.new_notebook(metadata=KERNEL)
    for kind, src in cells:
        src = src.strip("\n")
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nbf.write(nb, OUT / name)


md = lambda s: ("md", s)
code = lambda s: ("code", s)

SETUP_NOTE = md("""
Run `./setup_venv.sh` in this folder once and register the kernel it prints; this notebook uses that kernel ("Python (modelgami)").
""")

# ---------------------------------------------------------------- 1. paragami
notebook("01_paragami.ipynb", [
    md("""
# paragami: folding and flattening parameters

paragami describes the *shape and constraints* of a parameter with a **pattern**. A pattern moves a value between three representations:

| representation | what it is | example for a covariance matrix |
|---|---|---|
| **folded** | the value as you'd write it in code (an array, or a dict of arrays) | a $K \\times K$ positive definite matrix |
| **flat** | the folded value's entries as one vector, constraints and all | its $K^2$ entries |
| **free** | an unconstrained vector of the right dimension | the $K(K+1)/2$ entries of the log-Cholesky factor |

Conversions are named `<from>_to_<to>`: `folded_to_free`, `free_to_folded`, `folded_to_flat` and `flat_to_folded`. The generic `flatten(value, to_free=...)` and `fold(vector, from_free=...)` choose the vector representation with a keyword whose prefix says which side it describes.

The free representation is what optimizers and samplers want. By default paragami uses Stan's unconstraining transforms, so free vectors and log Jacobians match Stan's unconstrained space exactly. Where TensorFlow Probability uses a different transform, `bijector="tfp"` selects TFP's instead. Everything is pure JAX, so it works under `grad`, `jit` and `vmap`.
"""),
    SETUP_NOTE,
    code("""
import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import paragami as pg

jax.config.update("jax_enable_x64", True)
"""),
    md("""
## A single pattern

A positive scalar, and a vector bounded to $(0, 1)$.
"""),
    code("""
sigma_pattern = pg.NumericScalarPattern(lb=0.0)
x = sigma_pattern.folded_to_free(2.0)
print("free value:", x, "  (log 2 =", np.log(2.0), ")")
print("folded back:", sigma_pattern.free_to_folded(x))
print("flat value:", sigma_pattern.folded_to_flat(2.0))

p_pattern = pg.NumericVectorPattern(3, lb=0.0, ub=1.0)
p = jnp.array([0.1, 0.5, 0.9])
print("free value of p:", p_pattern.folded_to_free(p), " (the logit)")

# The same conversions with the generic methods:
print(p_pattern.flatten(p, to_free=True), p_pattern.fold(p_pattern.flatten(p, to_free=True), from_free=True))
"""),
    md("""
## Structured constraints

Some patterns have a free dimension smaller than their flat dimension. A simplex in $K$ dimensions has $K - 1$ free parameters, and a $K \\times K$ covariance matrix has $K(K+1)/2$. There is a pattern for each of Stan's constrained types.
"""),
    code("""
patterns = {
    "simplex(4)": pg.SimplexArrayPattern(4),
    "ordered(3)": pg.OrderedVectorPattern(3),
    "positive_ordered(3)": pg.OrderedVectorPattern(3, positive=True),
    "cov_matrix(3)": pg.PSDSymmetricMatrixPattern(3),
    "cholesky_factor_cov(3)": pg.CholeskyCovPattern(3),
    "cholesky_factor_corr(3)": pg.CholeskyCorrPattern(3),
    "corr_matrix(3)": pg.CorrMatrixPattern(3),
    "unit_vector(3)": pg.UnitVectorPattern(3),
    "sum_to_zero_vector(4)": pg.SumToZeroVectorPattern(4),
    "sum_to_zero_matrix(3, 4)": pg.SumToZeroMatrixPattern(3, 4),
    "column_stochastic(3, 2)": pg.StochasticMatrixPattern(3, 2, kind="column"),
    "vector<offset=1, multiplier=2>": pg.NumericVectorPattern(3, offset=1.0, multiplier=2.0),
}
for name, pat in patterns.items():
    print(f"{name:25s} flat length {pat.get_flat_length():2d}   free length {pat.get_free_length():2d}")
"""),
    code("""
cov_pattern = pg.PSDSymmetricMatrixPattern(3)
u = jax.random.normal(jax.random.PRNGKey(0), (cov_pattern.get_free_length(),))
cov = cov_pattern.free_to_folded(u)    # any free vector folds to a valid covariance matrix
print(cov)
print("eigenvalues:", jnp.linalg.eigvalsh(cov))
print("round trip error:", jnp.max(jnp.abs(cov_pattern.folded_to_free(cov) - u)))
"""),
    md("""
Patterns where Stan and TFP use different transforms take `bijector="stan"` (the default) or `bijector="tfp"`: one-sided bounds (TFP uses softplus instead of exp), simplexes, and the covariance and correlation patterns. The folded value is the same kind of object either way; only the free coordinates differ.
"""),
    code("""
for bijector in ("stan", "tfp"):
    pat = pg.NumericScalarPattern(lb=0.0, bijector=bijector)
    print(f"{bijector}: free value of 2.0 is {float(pat.folded_to_free(2.0)[0]):.4f}")
"""),
    md("""
## Dictionaries of parameters

`PatternDict` combines patterns, and its folded values are dicts. This is how a model's parameters are described in modelgami.
"""),
    code("""
pattern = pg.PatternDict(
    mu=pg.NumericVectorPattern(3),
    sigma=pg.NumericScalarPattern(lb=0.0),
    w=pg.SimplexArrayPattern(4),
    cov=pg.PSDSymmetricMatrixPattern(2),
)
params = pattern.draw_random(jax.random.PRNGKey(1))
params
"""),
    code("""
x_free = pattern.folded_to_free(params)
x_flat = pattern.folded_to_flat(params)
print("free length:", x_free.shape[0], "  flat length:", x_flat.shape[0])
print(pattern.get_free_names())
"""),
    md("""
### Nested dictionaries

A `PatternDict` can contain other `PatternDict`s, to any depth. The folded value is then a nested dict, and the flat and free vectors concatenate everything in order. Entry names show the path.
"""),
    code("""
nested = pg.PatternDict(
    likelihood=pg.PatternDict(beta=pg.NumericVectorPattern(2), sigma=pg.NumericScalarPattern(lb=0.0)),
    mixture=pg.PatternDict(
        weights=pg.SimplexArrayPattern(3),
        components=pg.PatternDict(means=pg.OrderedVectorPattern(3), cov=pg.PSDSymmetricMatrixPattern(2)),
    ),
)
nested_params = nested.draw_random(jax.random.PRNGKey(5))
print(jax.tree_util.tree_map(jnp.shape, nested_params))

u = nested.folded_to_free(nested_params)
print("free length:", nested.get_free_length(), "  flat length:", nested.get_flat_length())
print(nested.get_free_names())
print("round trip error:", jax.tree_util.tree_reduce(
    max, jax.tree_util.tree_map(lambda a, b: float(jnp.max(jnp.abs(a - b))), nested.free_to_folded(u), nested_params)))
"""),
    md("""
### Writing a pattern for a structure you already have

If you already have your parameters as a nested structure of dicts, lists and tuples of arrays, you don't need to write the pattern by hand. `make_unconstrained_pattern_like` builds an unconstrained pattern of the same shape. Every pattern can print the code that builds it with `generate_code()`, so together they give code ready to paste into your own code and edit.

Dict keys become the pattern's names. List and tuple entries have no names, so they are named from `default_name` (or the enclosing dict key) with an index, and they fold to dicts.
"""),
    code("""
my_params = {
    "beta": jnp.zeros(3),
    "sigma": 1.0,
    "groups": [jnp.zeros(2), jnp.zeros(2)],
    "nested": {"cov": jnp.eye(2)},
}
print(pg.make_unconstrained_pattern_like(my_params).generate_code())
"""),
    md("""
Paste the printed code and edit it to add constraints. Here `sigma` becomes positive and `cov` a covariance matrix:
"""),
    code("""
my_pattern = pg.PatternDict(
    beta=pg.NumericVectorPattern(3),
    sigma=pg.NumericScalarPattern(lb=0.0),
    groups=pg.PatternDict(
        groups_0=pg.NumericVectorPattern(2),
        groups_1=pg.NumericVectorPattern(2),
    ),
    nested=pg.PatternDict(
        cov=pg.PSDSymmetricMatrixPattern(2),
    ),
)
inferred = pg.make_unconstrained_pattern_like(my_params)
print("free length:", my_pattern.get_free_length(), "  (unconstrained version:", inferred.get_free_length(), ")")
print("generated code rebuilds the inferred pattern exactly:", eval(inferred.generate_code()) == inferred)
"""),
    code("""
# For a tuple or list at the top level, names come from default_name.
print(pg.make_unconstrained_pattern_like((jnp.zeros(2), 0.0), default_name="theta").get_free_names())
"""),
    md("""
## Log Jacobians

A density over the constrained parameters becomes a density over the free vector by adding the log absolute determinant of the free-to-constrained Jacobian. `free_to_folded` does not compute it; `free_to_folded_log_abs_det` computes only the log determinant, and `free_to_folded_with_log_det` returns both. Here we check it against autodiff for the simplex, where the free and flat dimensions differ, so we use the flat value without its last (redundant) entry.
"""),
    code("""
simplex = pg.SimplexArrayPattern(4)
u = jnp.array([0.3, -1.0, 0.5])
w, log_det = simplex.free_to_folded_with_log_det(u)

J = jax.jacfwd(lambda v: simplex.fold(v, from_free=True)[:-1])(u)
print("paragami log det:", log_det)
print("autodiff log det:", jnp.linalg.slogdet(J)[1])
"""),
    md("""
## Validation

`validate_folded` checks keys, shapes and constraint values and returns `(ok, message)`. The conversions validate their inputs; inside `jit`, value checks are skipped automatically and shape checks remain.
"""),
    code("""
bad = dict(params, sigma=-1.0)
print(pattern.validate_folded(bad))
print(pattern.validate_folded(dict(params, w=jnp.ones(3) / 3)))
try:
    pattern.flatten(bad, to_free=True)
except Exception as e:
    print(type(e).__name__, ":", e)
"""),
    md("""
Conversions in an inner loop can skip every check. `disable_validation()` turns them off for a pattern and, for a `PatternDict` or `PatternArray`, for everything inside it; `enable_validation()` turns them back on.
"""),
    code("""
pattern.disable_validation()
print("no check, no error:", pattern.flatten(bad, to_free=True)[:4])
pattern.enable_validation()
print("children re-enabled too:", pattern["sigma"].validate)
"""),
    md("""
## Arrays of patterns

`PatternArray` repeats a pattern over an array of shape `array_shape`, for example one simplex per group.
"""),
    code("""
group_weights = pg.PatternArray((5,), pg.SimplexArrayPattern(3))
print("free length:", group_weights.get_free_length())
W = group_weights.draw_random(jax.random.PRNGKey(2))
print(W.shape, W.sum(axis=-1))
"""),
    md("""
## Using patterns with JAX transformations

A function of folded parameters becomes a function of the free vector with `function_input_folded_to_free`. You can then differentiate, compile and vectorize it.
"""),
    code("""
def objective(params):
    return jnp.sum(params["mu"] ** 2) + params["sigma"] + jnp.sum(params["w"] * jnp.arange(4.0)) + jnp.trace(params["cov"])

f = pg.function_input_folded_to_free(objective, patterns=pattern)
print("f(x):", f(x_free))
print("grad:", jax.jit(jax.grad(f))(x_free))

xs = jax.random.normal(jax.random.PRNGKey(3), (10, x_free.shape[0]))
print("vmap over 10 free vectors:", jax.vmap(f)(xs).shape)
"""),
    md("""
A pattern acts on a single value, so batches of draws are folded with `jax.vmap`. The result is a dict of arrays with a leading batch axis. Notebook 3 uses this to turn MCMC output into structured draws.
"""),
    code("""
folded_batch = jax.vmap(lambda v: pattern.fold(v, from_free=True))(xs)
{k: v.shape for k, v in folded_batch.items()}
"""),
    md("""
## Draws and ArviZ

MCMC output usually comes as an array of free vectors with shape (chain, draw, free length). paragami converts such arrays to [ArviZ](https://python.arviz.org) and back (install ArviZ, or `paragami[arviz]`):

- `free_draws_to_arviz` and `folded_draws_to_arviz` make ArviZ data with one variable per pattern entry, in constrained form. Nested dict entries are named with dots, such as `"inner.w"`.
- `arviz_to_free_draws` and `arviz_to_folded_draws` read them back.
- `free_draws_to_folded`, `flat_draws_to_folded`, `folded_draws_to_free` and `folded_draws_to_flat` convert between representations without ArviZ.

Here the "draws" are just random free vectors standing in for sampler output.
"""),
    code("""
free_draws = jax.random.normal(jax.random.PRNGKey(6), (4, 500, pattern.get_free_length()))
idata = pg.free_draws_to_arviz(pattern=pattern, draws=free_draws, coords={"component": ["a", "b", "c", "d"]},
                               dims={"w": ["component"]})
az.summary(idata, var_names=["sigma", "w"])
"""),
    code("""
back = pg.arviz_to_free_draws(pattern=pattern, data=idata)
print("round trip error:", float(jnp.max(jnp.abs(back - free_draws))))

folded = pg.arviz_to_folded_draws(pattern=pattern, data=idata)
print({k: v.shape for k, v in folded.items()})
"""),
])

# ---------------------------------------------------------------- 2. defining a model
notebook("02_defining_a_model.ipynb", [
    md("""
# Defining a Bayesian model with modelgami

A modelgami model is a **prior** and a **likelihood**, a paragami pattern for the parameters, the data and the hyperparameters. The prior and the likelihood are the same kind of object, a data-generating process (`DGP`): a log density and a sampler of one value given some conditioning arguments, either of which may be missing. Their functions take the value they describe first. They are ordinary JAX functions on **dicts of arrays**; there is no modelling language.

| | log density | sampler |
|---|---|---|
| prior, a `DGP` | `eval_log_density(params, hyper)` | `draw(key, hyper) -> params` |
| likelihood, a `DGP` | `eval_log_density(data, params, hyper)` | `draw(key, data, params, hyper) -> data` |
| exchangeable likelihood, an `ExchangeableDGP` | `eval_log_density_one(datum, params, hyper)` | `draw_one(key, datum, params, hyper)` |

The example is a robust regression: $y_n \\\\sim \\\\text{Student-}t_\\\\nu(x_n^\\\\top \\\\beta, \\\\sigma)$, with priors $\\\\beta \\\\sim N(0, s^2 I)$, $\\\\sigma \\\\sim \\\\text{Exponential}(1)$ and $\\\\nu \\\\sim \\\\text{Gamma}(2, 0.1)$. The observations are exchangeable, so it is an `ExchangeableModel`.
"""),
    SETUP_NOTE,
    code("""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from jax.scipy import stats

import paragami as pg
import modelgami as mg
from modelgami.testing import check_model_contract

jax.config.update("jax_enable_x64", True)
"""),
    md("""
## Simulated data

Exchangeable data are an array of N data: a dict of arrays that all share a leading observation axis. Anything shared by every observation goes in the hyperparameters.
"""),
    code("""
n, p = 200, 3
k1, k2 = jax.random.split(jax.random.PRNGKey(0))
X = jnp.concatenate([jnp.ones((n, 1)), jax.random.normal(k1, (n, p - 1))], axis=1)
beta_true = jnp.array([1.0, -2.0, 0.5])
y = X @ beta_true + 0.7 * jax.random.t(k2, 4.0, (n,))
data = {"x": X, "y": y}
"""),
    md("""
## The model
"""),
    code("""
params_pattern = pg.PatternDict(
    beta=pg.NumericVectorPattern(p),
    sigma=pg.NumericScalarPattern(lb=0.0),
    nu=pg.NumericScalarPattern(lb=0.0),
)
hyper = {"beta_scale": 5.0}

def eval_log_prior(params, hyper):
    return (stats.norm.logpdf(params["beta"], 0.0, hyper["beta_scale"]).sum()
            + stats.expon.logpdf(params["sigma"])
            + stats.gamma.logpdf(params["nu"], 2.0, scale=10.0))

def draw_prior(key, hyper):
    kb, ks, kn = jax.random.split(key, 3)
    return {
        "beta": hyper["beta_scale"] * jax.random.normal(kb, (p,)),
        "sigma": jax.random.exponential(ks),
        "nu": 10.0 * jax.random.gamma(kn, 2.0),
    }

def eval_log_lik_one(datum, params, hyper):
    # One observation: datum = {"x": (p,), "y": ()}
    z = (datum["y"] - datum["x"] @ params["beta"]) / params["sigma"]
    return stats.t.logpdf(z, params["nu"]) - jnp.log(params["sigma"])

def draw_one(key, datum, params, hyper):
    # A new y for one observation; its covariates are kept.
    return {"y": datum["x"] @ params["beta"] + params["sigma"] * jax.random.t(key, params["nu"])}

model = mg.ExchangeableModel(
    name="robust_regression",
    params_pattern=params_pattern,
    prior=mg.DGP(eval_log_density=eval_log_prior, draw=draw_prior),
    likelihood=mg.ExchangeableDGP(eval_log_density_one=eval_log_lik_one, draw_one=draw_one),
    data=data,
    hyper=hyper,
)
model
"""),
    md("""
## The same likelihood as a class

A DGP can also be a subclass with the functions as methods. The two forms are interchangeable. A function that isn't defined raises `NotImplementedError`; here the class has no sampler.
"""),
    code("""
class StudentTRegression(mg.ExchangeableDGP):
    def eval_log_density_one(self, datum, params, hyper):
        return eval_log_lik_one(datum, params, hyper)

model_cls = mg.ExchangeableModel(params_pattern=params_pattern, prior=model.prior,
                                 likelihood=StudentTRegression(), data=data, hyper=hyper)
params = params_pattern.draw_random(jax.random.PRNGKey(1))
print(model.eval_log_lik(params), model_cls.eval_log_lik(params))
try:
    model_cls.draw_data(jax.random.PRNGKey(0), params)
except NotImplementedError as e:
    print("NotImplementedError:", e)
"""),
    md("""
## Log densities

The model's job is to combine the prior and the likelihood: `eval_log_prior`, `eval_log_lik` and `eval_log_joint` take the folded parameters (a dict). The data and hyperparameters are arguments too, defaulting to the model's own, so you can evaluate the model on other data or differentiate with respect to the hyperparameters.
"""),
    code("""
print("log joint:", model.eval_log_joint(params))
print("first 50 observations:", model.eval_log_joint(params, data={"x": X[:50], "y": y[:50]}))
print("d log joint / d beta_scale:", jax.grad(lambda h: model.eval_log_joint(params, hyper=h))(hyper)["beta_scale"])
"""),
    md("""
## Constrained and unconstrained parameters

Samplers and optimizers work on $\\mathbb{R}^d$. `mg.free_dgp(dgp, pattern)` turns a distribution of a constrained value into the same distribution of its free vector, adding the log Jacobian to the log density (and `mg.folded_dgp` goes back). Only the prior is a density over the parameters, so the unconstrained target is the free prior plus the likelihood: that is `eval_free_log_joint`.
"""),
    code("""
free_prior = mg.free_dgp(model.prior, params_pattern)
x = params_pattern.folded_to_free(params)
print("free dimension:", params_pattern.get_free_length())
print("free prior:", free_prior.eval_log_density(x, hyper),
      " = prior + log Jacobian:", model.eval_log_prior(params) + params_pattern.free_to_folded_log_abs_det(x))
print("free log joint:", model.eval_free_log_joint(x))
print("a free prior draw:", free_prior.draw(jax.random.PRNGKey(2), hyper))
"""),
    code("""
grad = jax.jit(jax.grad(model.eval_free_log_joint))
hess = jax.jit(jax.hessian(model.eval_free_log_joint))
print("gradient:", grad(x))
print("Hessian-vector product:", mg.make_hvp(model.eval_free_log_joint)(x, jnp.ones_like(x)))
print("Hessian diagonal:", jnp.diag(hess(x)))
"""),
    md("""
## Exchangeable data: single-observation, weighted and subset log densities

Because the likelihood is defined one observation at a time, an `ExchangeableModel` can evaluate it observation by observation, with data weights, or on a minibatch.
"""),
    code("""
singleobs = model.eval_log_lik_singleobs(params)
print("single-observation shape:", singleobs.shape, "  sum matches eval_log_lik:",
      jnp.allclose(singleobs.sum(), model.eval_log_lik(params)))

# Weighted log joint; the gradient in the weights is each observation's influence.
w = jnp.ones(model.n_obs)
influence = jax.grad(lambda w: model.eval_log_joint(params, weights=w))(w)
print("d log joint / d w equals the single-observation values:", jnp.allclose(influence, singleobs))

# A minibatch, with the likelihood scaled by N / batch size to be unbiased.
idx = jnp.arange(20)
print("subset estimate:", model.eval_log_joint_subset(params, idx), "  full:", model.eval_log_joint(params))

# Nonparametric bootstrap weights.
print("bootstrap log joint:", model.eval_log_joint(params, weights=model.draw_np_bootstrap_weights(jax.random.PRNGKey(2))))
"""),
    md("""
## Simulation

`draw_prior` draws parameters, and `draw_data` draws a full data set from `draw_one`. `draw_joint` draws both, and `draw_marginal_data` draws data from their prior predictive distribution.
"""),
    code("""
keys = jax.random.split(jax.random.PRNGKey(3), 500)
prior_draws = jax.vmap(model.draw_prior)(keys)
print({k: v.shape for k, v in prior_draws.items()})

sim_params, sim_data = model.draw_joint(jax.random.PRNGKey(4))
print("simulated data keys:", list(sim_data), " y shape:", sim_data["y"].shape)
"""),
    code("""
y_rep = jax.vmap(lambda k: model.draw_marginal_data(k)["y"])(jax.random.split(jax.random.PRNGKey(5), 200))
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(np.clip(np.asarray(y_rep).ravel(), -60, 60), bins=120, density=True, alpha=0.5, label="prior predictive")
ax.hist(np.asarray(y), bins=40, density=True, alpha=0.7, label="observed")
ax.set_xlabel("y"); ax.legend(); plt.show()
"""),
    md("""
## Checking the model

`check_model_contract` checks that the free and folded log densities and the exchangeable extras agree with each other, and that the samplers run. Parameter values are checked by paragami.
"""),
    code("""
ok, message = params_pattern.validate_folded(dict(params, sigma=-1.0))
print(ok, "...", message[-45:])
check_model_contract(model)
"""),
    md("""
## Optional: an exchangeable likelihood from distributions

To write an exchangeable likelihood once and get both its density and its sampler, return the observation's distributions (NumPyro or TFP). A NumPyro program can also be turned into a whole model with `mg.from_numpyro`.
"""),
    code("""
likelihood = mg.exchangeable_dgp_from_distributions(
    lambda datum, params, hyper: {"y": dist.StudentT(params["nu"], datum["x"] @ params["beta"], params["sigma"])})
model_dists = mg.ExchangeableModel(params_pattern=params_pattern, prior=model.prior, likelihood=likelihood,
                                   data=data, hyper=hyper)
print(model_dists.eval_log_lik(params), model.eval_log_lik(params))
print("simulated y:", model_dists.draw_data(jax.random.PRNGKey(6), params)["y"][:3])
"""),
])

# ---------------------------------------------------------------- 3. blackjax + arviz
notebook("03_sampling_with_blackjax.ipynb", [
    md("""
# From a modelgami model to MCMC draws in ArviZ

This notebook takes a model, turns it into an unconstrained log density with its gradient, samples it with BlackJAX's NUTS, and converts the array of unconstrained MCMC samples into structured draws in ArviZ.

The model is the conjugate Bayesian regression from `modelgami.examples`: $y \\sim N(X\\beta, \\sigma^2 I)$ with $\\beta \\mid \\sigma^2 \\sim N(m_0, \\sigma^2 V_0)$ and $\\sigma^2 \\sim \\text{InvGamma}(a_0, b_0)$. Its posterior is known exactly, so we can check the sampler.
"""),
    SETUP_NOTE,
    code("""
import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import paragami as pg
import modelgami as mg
from modelgami.examples import conjugate_regression as cr

jax.config.update("jax_enable_x64", True)

X, y = cr.simulate_data(jax.random.PRNGKey(0), n=100, p=3)
model = cr.make_model(X, y)
model.params_pattern
"""),
    md("""
## 1. The unconstrained log density and its gradient

BlackJAX samples on $\\mathbb{R}^d$, so the target is the log joint as a density over the free parameters, `eval_free_log_joint`: the prior over the free parameters (which includes the log Jacobian) plus the likelihood. BlackJAX computes the gradient itself with `jax.grad`; we show it here only to make the point.
"""),
    code("""
logdensity = model.eval_free_log_joint
grad_logdensity = jax.grad(logdensity)

x0 = model.params_pattern.folded_to_free(model.params_pattern.draw_random(jax.random.PRNGKey(1)))
print("dimension:", model.params_pattern.get_free_length())
print("log density:", logdensity(x0))
print("gradient:", grad_logdensity(x0))
"""),
    md("""
## 2. NUTS with BlackJAX

We run window adaptation (step size and diagonal mass matrix) separately for each of four chains, then sample. Each chain is one `vmap`ped call.
"""),
    code("""
n_chains, n_warmup, n_draws = 4, 1000, 1000

def run_chain(key, x_init):
    warmup_key, sample_key = jax.random.split(key)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, tuned), _ = warmup.run(warmup_key, x_init, num_steps=n_warmup)
    step = blackjax.nuts(logdensity, **tuned).step

    def one_step(state, key):
        state, info = step(key, state)
        return state, (state.position, info.is_divergent)

    _, (positions, divergent) = jax.lax.scan(one_step, state, jax.random.split(sample_key, n_draws))
    return positions, divergent

init_keys = jax.random.split(jax.random.PRNGKey(2), n_chains)
x_inits = jax.vmap(lambda k: model.params_pattern.folded_to_free(model.params_pattern.draw_random(k)))(init_keys)
chain_keys = jax.random.split(jax.random.PRNGKey(3), n_chains)

free_draws, divergent = jax.jit(jax.vmap(run_chain))(chain_keys, x_inits)
print("free draws:", free_draws.shape, "(chain, draw, free dimension)")
print("divergences:", int(divergent.sum()))
"""),
    md("""
## 3. From unconstrained arrays to structured draws in ArviZ

paragami converts the (chain, draw, free dimension) array into ArviZ data with one variable per parameter, in constrained form. `pg.arviz_to_free_draws` goes back the other way.
"""),
    code("""
pattern = model.params_pattern
idata = pg.free_draws_to_arviz(pattern=pattern, draws=free_draws, coords={"coef": ["intercept", "x1", "x2"]},
                               dims={"beta": ["coef"]})
print(idata["posterior"])
print("round trip error:", float(jnp.max(jnp.abs(pg.arviz_to_free_draws(pattern=pattern, data=idata) - free_draws))))

draws = pg.arviz_to_folded_draws(pattern=pattern, data=idata)   # or pg.free_draws_to_folded(pattern=pattern, draws=free_draws)
{k: v.shape for k, v in draws.items()}
"""),
    md("""
## 4. ArviZ with the log likelihood and data

For LOO, add the single-observation log likelihoods of each draw as the `log_likelihood` group, and the data as `observed_data`.
"""),
    code("""
singleobs = jax.vmap(jax.vmap(model.eval_log_lik_singleobs))(draws)
idata = pg.free_draws_to_arviz(
    pattern=pattern, draws=free_draws, coords={"coef": ["intercept", "x1", "x2"]},
    dims={"beta": ["coef"], "y": ["obs"], "x": ["obs", "coef"]},
    extra_groups={"log_likelihood": {"y": np.asarray(singleobs)},
                  "observed_data": {"y": np.asarray(y)}, "constant_data": {"x": np.asarray(X)}})
idata
"""),
    code("""
az.summary(idata, var_names=["beta", "sigma2"])
"""),
    code("""
az.loo(idata)
"""),
    md("""
## 5. Checking against the exact posterior
"""),
    code("""
post = cr.compute_exact_posterior(X, y, model.hyper)
an, bn = post["a0"], post["b0"]
print("posterior mean of beta   exact:", np.round(post["m0"], 4), "  NUTS:", np.round(draws["beta"].mean(axis=(0, 1)), 4))
print("posterior mean of sigma2 exact:", round(float(bn / (an - 1)), 4), "  NUTS:", round(float(draws["sigma2"].mean()), 4))
"""),
    code("""
exact = cr.draw_exact_posterior(jax.random.PRNGKey(4), X, y, model.hyper, n_chains=1, n_draws=20000)
fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))
for i in range(3):
    axes[i].hist(np.asarray(draws["beta"][..., i]).ravel(), bins=50, density=True, alpha=0.6, label="NUTS")
    axes[i].hist(np.asarray(exact["beta"][..., i]).ravel(), bins=50, density=True, histtype="step", lw=2, label="exact")
    axes[i].set_title(f"beta[{i}]")
axes[3].hist(np.asarray(draws["sigma2"]).ravel(), bins=50, density=True, alpha=0.6, label="NUTS")
grid = np.linspace(float(draws["sigma2"].min()), float(draws["sigma2"].max()), 200)
axes[3].plot(grid, np.exp(cr.eval_log_inv_gamma(grid, an, bn)), lw=2, label="exact")
axes[3].set_title("sigma2"); axes[3].legend()
plt.tight_layout(); plt.show()
"""),
])

# ---------------------------------------------------------------- 4. posteriordb_jax + blackjax
notebook("04_posteriordb_with_blackjax.ipynb", [
    md("""
# Fitting a posteriordb model with BlackJAX

`posteriordb_jax.catalog` holds posteriordb's Stan models translated into modelgami models, with their data loaded. Because every model has the same interface, one sampling function runs on all of them. posteriordb also ships reference posterior draws, which we use to check the fit.
"""),
    SETUP_NOTE,
    code("""
import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import paragami as pg
from posteriordb_jax import catalog
from posteriordb_jax.io import load_reference_draws
from posteriordb_jax.stan_names import folded_draws_from_posteriordb

jax.config.update("jax_enable_x64", True)

catalog.list_models()
"""),
    md("""
## Load a model

The non-centered eight schools model: `theta_trans` are the standardized school effects, `mu` and `tau` the population mean and scale.
"""),
    code("""
model = catalog.load("eight_schools-eight_schools_noncentered")
print(model.params_pattern)
print("free dimension:", model.params_pattern.get_free_length())
model.data
"""),
    md("""
## A sampler for any model

This function only uses the common interface: `eval_free_log_joint` (the log density over the free parameters) and the parameter pattern's `draw_random` and `folded_to_free`. It returns free draws with shape (chain, draw, dimension).
"""),
    code("""
def nuts(model, key, n_chains=4, n_warmup=1000, n_draws=1000):
    logdensity = model.eval_free_log_joint

    def run_chain(key):
        init_key, warmup_key, sample_key = jax.random.split(key, 3)
        x_init = model.params_pattern.folded_to_free(model.params_pattern.draw_random(init_key))
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
        (state, tuned), _ = warmup.run(warmup_key, x_init, num_steps=n_warmup)
        step = blackjax.nuts(logdensity, **tuned).step

        def take_step(state, key):
            state, info = step(key, state)
            return state, (state.position, info.is_divergent)

        _, (positions, divergent) = jax.lax.scan(take_step, state, jax.random.split(sample_key, n_draws))
        return positions, divergent

    free_draws, divergent = jax.jit(jax.vmap(run_chain))(jax.random.split(key, n_chains))
    return free_draws, int(divergent.sum())
"""),
    code("""
free_draws, n_divergent = nuts(model, jax.random.PRNGKey(0))
print("free draws:", free_draws.shape, "  divergences:", n_divergent)

idata = pg.free_draws_to_arviz(pattern=model.params_pattern, draws=free_draws)
az.summary(idata)
"""),
    md("""
## Compare with posteriordb's reference draws

posteriordb stores reference draws by Stan name. For this model they are of `mu`, `tau` and the transformed parameter `theta` (the school effects), so we compute `theta` from our draws too. (For models whose reference draws cover all the parameters, `catalog.load_reference_draws` returns them already folded, as in the next section.)
"""),
    code("""
draws = pg.free_draws_to_folded(pattern=model.params_pattern, draws=free_draws)
draws["theta"] = draws["theta_trans"] * draws["tau"][..., None] + draws["mu"][..., None]

reference_pattern = pg.PatternDict(theta=pg.NumericVectorPattern(8), mu=pg.NumericScalarPattern(),
                                   tau=pg.NumericScalarPattern(lb=0.0))
reference = folded_draws_from_posteriordb(pattern=reference_pattern,
                                          chains=load_reference_draws("eight_schools-eight_schools_noncentered"))

for name in ["mu", "tau", "theta"]:
    ref, ours = np.asarray(reference[name]), np.asarray(draws[name])
    print(f"{name:12s} mean  reference {np.round(ref.mean(axis=(0, 1)), 2)}\\n{'':12s}       NUTS      {np.round(ours.mean(axis=(0, 1)), 2)}")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
for ax, name in zip(axes, ["mu", "tau"]):
    ax.hist(np.asarray(reference[name]).ravel(), bins=60, density=True, alpha=0.5, label="posteriordb reference")
    ax.hist(np.asarray(draws[name]).ravel(), bins=60, density=True, histtype="step", lw=2, label="BlackJAX NUTS")
    ax.set_title(name)
axes[0].legend(); plt.tight_layout(); plt.show()
"""),
    md("""
## The same code on several models

Because the interface is shared, a method can be run across the catalog. Here we fit a few models and compare posterior means to the reference draws, measured in reference standard deviations.
"""),
    code("""
names = ["kidiq-kidscore_momiq", "earnings-logearn_height", "mesquite-logmesquite_logvolume", "kilpisjarvi_mod-kilpisjarvi"]
for name in names:
    m = catalog.load(name)
    free_draws, n_div = nuts(m, jax.random.PRNGKey(1), n_warmup=500, n_draws=500)
    ours = pg.free_draws_to_folded(pattern=m.params_pattern, draws=free_draws)
    ref = catalog.load_reference_draws(name)
    z = max(float(np.max(np.abs(np.asarray(ours[k]).mean(axis=(0, 1)) - np.asarray(ref[k]).mean(axis=(0, 1)))
                         / np.asarray(ref[k]).std(axis=(0, 1)))) for k in ref)
    print(f"{name:35s} free dim {m.params_pattern.get_free_length():2d}   divergences {n_div:3d}   max |mean difference| / sd = {z:.3f}")
"""),
    md("""
For exchangeable models, adding the single-observation log likelihoods to the ArviZ data makes LOO one call.
"""),
    code("""
m = catalog.load("kidiq-kidscore_momiq")
free_draws, _ = nuts(m, jax.random.PRNGKey(2), n_warmup=500, n_draws=500)
folded = pg.free_draws_to_folded(pattern=m.params_pattern, draws=free_draws)
singleobs = jax.vmap(jax.vmap(m.eval_log_lik_singleobs))(folded)
idata = pg.folded_draws_to_arviz(m.params_pattern, folded, extra_groups={"log_likelihood": {"kid_score": np.asarray(singleobs)}})
az.loo(idata)
"""),
])
print("wrote notebooks to", OUT)
