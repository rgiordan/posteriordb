"""Minimal loaders for posteriordb's on-disk layout: Stan model files,
zipped data JSON, and zipped reference-posterior-draws JSON. Reads directly
from `posterior_database/` in the repo -- no disk writes, no dependency on
the separate `posteriordb-python` package (which isn't vendored in this
repo)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
POSTERIOR_DB_ROOT = REPO_ROOT / "posterior_database"


def stan_file_path(model_name: str) -> Path:
    return POSTERIOR_DB_ROOT / "models" / "stan" / f"{model_name}.stan"


def _load_zipped_json(zip_path: Path) -> Any:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert len(names) == 1, f"expected exactly one file in {zip_path}, got {names}"
        with zf.open(names[0]) as f:
            return json.load(f)


def load_data(data_name: str) -> Dict[str, Any]:
    zip_path = POSTERIOR_DB_ROOT / "data" / "data" / f"{data_name}.json.zip"
    return _load_zipped_json(zip_path)


def load_reference_draws(posterior_name: str) -> List[Dict[str, List[float]]]:
    """A list of per-chain dicts, each keyed by parameter *element* name
    (e.g. "theta[1]") mapping to a list of draw values. Raises
    FileNotFoundError if this posterior has no stored reference draws (e.g.
    dogs_nonhierarchical)."""
    zip_path = POSTERIOR_DB_ROOT / "reference_posteriors" / "draws" / "draws" / f"{posterior_name}.json.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"no reference draws for posterior '{posterior_name}' at {zip_path}")
    return _load_zipped_json(zip_path)


def load_posterior_info(posterior_name: str) -> Dict[str, Any]:
    path = POSTERIOR_DB_ROOT / "posteriors" / f"{posterior_name}.json"
    with open(path) as f:
        return json.load(f)
