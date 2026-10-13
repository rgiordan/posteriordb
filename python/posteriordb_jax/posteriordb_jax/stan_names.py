"""Convert between paragami folded values and Stan's / posteriordb's
per-element names ("beta[2]" in posteriordb draws, "beta.2" in BridgeStan;
1-indexed). Element lookup is by name, so Stan's print order (column-major
for matrices) never has to be reproduced."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

import jax.numpy as jnp
import numpy as np

_NAME = re.compile(r"^([^.\[]+)(?:\[([0-9,]+)\]|((?:\.[0-9]+)+))?$")


def parse_name(name: str):
    """'beta[2,1]' or 'beta.2.1' -> ('beta', (1, 0)) (0-indexed)."""
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"can't parse parameter name {name!r}")
    base, brackets, dots = m.groups()
    if brackets:
        idx = tuple(int(i) - 1 for i in brackets.split(","))
    elif dots:
        idx = tuple(int(i) - 1 for i in dots.strip(".").split("."))
    else:
        idx = ()
    return base, idx


def _shapes(pattern) -> Dict[str, tuple]:
    empty = pattern.make_empty(valid=True)
    return {k: np.shape(v) for k, v in empty.items()}


def folded_from_names(pattern, names: Sequence[str], values) -> Dict[str, jnp.ndarray]:
    """Fill a folded dict from values listed by Stan element names.
    `values` has shape `batch + (len(names),)`."""
    values = np.asarray(values)
    batch = values.shape[:-1]
    out = {k: np.full(batch + s, np.nan) for k, s in _shapes(pattern).items()}
    for i, name in enumerate(names):
        base, idx = parse_name(name)
        if base in out:
            out[base][(Ellipsis,) + idx] = values[..., i]
    missing = [k for k, v in out.items() if np.isnan(v).any()]
    if missing:
        raise ValueError(f"no values for {missing} among the given names")
    return {k: jnp.asarray(v) for k, v in out.items()}


def values_by_names(params: Dict, names: Sequence[str]) -> np.ndarray:
    """The inverse: a vector of values in the order of `names`."""
    out = []
    for name in names:
        base, idx = parse_name(name)
        out.append(np.asarray(params[base])[idx])
    return np.asarray(out)


def folded_draws_from_posteriordb(pattern, chains: List[Dict[str, List[float]]]) -> Dict[str, jnp.ndarray]:
    """posteriordb reference draws (a list of per-chain dicts keyed by
    element name) -> folded dict with leading (chain, draw) axes."""
    names = [n for n in chains[0] if parse_name(n)[0] in _shapes(pattern)]
    values = np.stack([np.stack([np.asarray(c[n]) for n in names], axis=-1) for c in chains])
    return folded_from_names(pattern, names, values)
