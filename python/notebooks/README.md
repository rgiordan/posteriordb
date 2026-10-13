# Notebooks

Demonstrations of the three packages in this directory's parent:

| notebook | what it shows |
|---|---|
| `01_paragami.ipynb` | paragami alone: patterns, folded / flat / free representations, log Jacobians, validation, and use with `grad`, `jit` and `vmap` |
| `02_defining_a_model.ipynb` | Defining a Bayesian model in modelgami with parameter dictionaries, and what the model gives you |
| `03_sampling_with_blackjax.ipynb` | A model's unconstrained log density in BlackJAX's NUTS, and the MCMC draws turned back into structured draws in ArviZ, checked against the exact posterior |
| `04_posteriordb_with_blackjax.ipynb` | Loading posteriordb models from `posteriordb_jax.catalog`, fitting them with BlackJAX, and comparing with posteriordb's reference draws |

## Setup

```bash
./setup_venv.sh
```

This creates `.venv` here, installs `requirements.txt` (the three local packages in editable mode, plus BlackJAX, ArviZ, NumPyro, matplotlib and Jupyter), and prints the command that registers the Jupyter kernel. Run that command once. The notebooks use the kernel it creates, "Python (modelgami)", by default. Set `PYTHON=/path/to/python` to build the venv with a different interpreter.

To run every notebook from the command line:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```
