# Ryan's coding preferences

Python and JAX code for statistics and methodology research. Each rule comes from Ryan's own review comments; the examples in parentheses are the cases that prompted it.

## Readability

- Put all imports at the top of scripts and modules.
- When a call passes more than one argument, use keyword arguments, including in internal calls within functions.
- Use snake_case for functions, methods and arguments, and CamelCase for class names (PEP 8).
- Name functions and methods as verbs: `draw_np_bootstrap_weights`, not `bootstrap_weights`.
  - Functions that evaluate a density start with `eval_` (`eval_log_density`, `eval_log_lik_singleobs`).
  - Functions that generate random values start with `draw_`.
- Names should not be ambiguous about direction or meaning.
  - For conversions between named states, use `x_to_y` (`folded_to_flat`, `flat_to_folded`), and prefix the matching keyword arguments with `to_` or `from_` (`to_free=True`, not `free=True`).
  - Pick the clearest term even when a common one exists (`log_lik_singleobs` instead of `log_lik_pointwise`).
- Don't give an argument a default when the caller should decide its meaning. For example, whether a transform is to the free or the flat space should always be stated explicitly.

## Simplicity

- Avoid low-value wrappers. A private method that only forwards to a library call adds code, room for error and ambiguity. Let users call the underlying function directly, or write their own small wrapper.
- Avoid unnecessary complexity in general. Don't add machinery just to allow elaborate overrides of defaults.
- Limit inheritance while still avoiding redundant code. Put functionality that applies only to a special case in a subclass (an `ExchangeableModel` subclass of `Model`), not in the base class.
- Don't duplicate code or keep parallel copies of the same information (for example, a second section that re-declares parameters or re-implements validation).
- Don't track capabilities separately (no `can_sample`-style flags). An unsupported method should raise `NotImplementedError`.
- Prefer an optional argument that defaults to stored state over a helper method. For example, a log likelihood takes an optional `data` argument that defaults to the object's data, instead of a `with_data` method.
- When code switches on object type, move the behavior into a method on each type. For example, each pattern has a `code_to_generate` method instead of one function with a case block over pattern types.
- Merge modules that belong together instead of splitting them artificially.

## Structure

- Generic operations that don't depend on a particular class should be standalone functions that the class calls with its own values. For example, `draw_np_bootstrap_weights(key, n_obs)` is a function, and the model calls it with its own `n_obs`.
- A function should do one thing. If a method does one class-specific thing plus several generic steps, split the generic steps into separate functions outside the class.
- Code that has the same mathematical meaning should share an abstraction. A prior and a likelihood are both data generating processes, so they are the same kind of object.

## Defaults should match the math

- Defaults should match the mathematical meaning of what is being computed, with an option to turn them off. For example, a log density by default evaluates, up to a constant, the density of the object being called, with respect to the parameters being passed. So the Jacobian is included by default for a prior or joint density but not for a likelihood, and either can be toggled.
- Represent operations by their mathematical meaning. For example, unconstraining a parameter is converting a density from one base measure to another, so it is a function from one DGP to another with the Jacobian included.

## Performance

- Operations that run many times inside an algorithm should be able to run as fast as possible.
  - Make input validation optional, with a switch that can be turned off. Setting it on a container should set it on all children.
  - Don't compute expensive extras unless they are asked for. For example, folding should not compute a log determinant unless it is requested.
- Provide optional validation functions so that data and parameters can be checked when they are passed in.

## Library and API design

- Users should be able to write models as ordinary JAX Python code. Don't define anything that looks like a custom modeling language.
- Users write the parts that matter in intuitive structures (dictionaries of arrays, with constraint specifications), and the library generates the boilerplate on request.
- Don't force one style on users. Offer an optional hook for a framework (for example NumPyro), but let users write the log density and the simulation code separately.
- Reduce boilerplate. For example, infer a parameter specification from a data structure the user already has, and print code that recreates a specification so it can be pasted and edited.
- Make optional pieces optional. A data generating process may support sampling, log density evaluation, or both.
- Work with the standard tools in the ecosystem (for example ArviZ).
- Research tooling should be readable, modular and flexible. It should be easy to swap in a custom analysis and run it across many models (for example, on a Slurm cluster).
- The aim is to implement a model once, in a common interface, and then reuse it to try new methods on many models quickly.

## Repository and project hygiene

- Add Python cruft (`.ipynb_checkpoints`, `__pycache__` and similar) to `.gitignore`.
- Demonstrate each component in its own Jupyter notebook in a `notebooks/` folder. Provide a small setup script that creates a venv, installs a `requirements.txt` and prints the command to register a Jupyter kernel, and have the notebooks use that kernel by default.
- Related packages can be developed together in one repository and split out once they're ready.

## Working with Ryan

- Ryan often reviews code by leaving a batch of notes. When he says he is making notes, don't start changing code until he says he's done, but do answer questions as he asks them.
- He sometimes clarifies a request in several short follow-up messages. Read them together before acting.
