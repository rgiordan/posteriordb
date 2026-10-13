"""Running an analysis on entries, one output directory and status file per
entry."""

from __future__ import annotations

import datetime
import importlib
import importlib.util
import json
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from .entry import Entry

STATUS_FILE = "status.json"
FINISHED = ("done", "not_applicable")


class NotApplicable(Exception):
    """Raise from an analysis to skip an entry it does not apply to (for
    example, a bootstrap of a model whose data are not exchangeable). The
    entry is recorded as "not_applicable", not as a failure."""


def load_analysis(spec: str) -> Callable:
    """The analysis named by "package.module:attribute" or
    "path/to/file.py:attribute"."""
    module_name, sep, attribute = spec.rpartition(":")
    if not sep or not module_name or not attribute:
        raise ValueError(f"an analysis is named 'module:function' or 'file.py:function', got {spec!r}")
    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, attribute)


def read_status(out_dir: Path) -> Optional[Dict]:
    """An entry's status record, or None if it has never run."""
    path = Path(out_dir) / STATUS_FILE
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def write_status(*, out_dir: Path, status: Dict) -> None:
    path = Path(out_dir) / STATUS_FILE
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    tmp.replace(path)


def run_entry(*, analysis: Callable, entry: Entry, out_root: Path, overwrite: bool = False) -> str:
    """Run `analysis(entry, out_dir)` with out_dir = out_root / entry.name and
    record the outcome in out_dir / status.json. Returns the state: "done",
    "not_applicable", "failed" or, when an earlier run finished and
    `overwrite` is False, that run's state (the analysis is not rerun)."""
    out_dir = Path(out_root) / entry.name
    previous = read_status(out_dir)
    if previous is not None and previous["state"] in FINISHED and not overwrite:
        return previous["state"]
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "posterior": entry.name,
        "state": "running",
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    write_status(out_dir=out_dir, status=status)
    start = time.perf_counter()
    try:
        analysis(entry, out_dir)
        status["state"] = "done"
    except NotApplicable as error:
        status.update(state="not_applicable", message=str(error))
    except Exception:
        status.update(state="failed", message=traceback.format_exc())
    status["seconds"] = round(time.perf_counter() - start, 3)
    write_status(out_dir=out_dir, status=status)
    return status["state"]


def run_local(*, analysis: Callable, entries: Iterable[Entry], out_root: Path, overwrite: bool = False,
              verbose: bool = True) -> Dict[str, str]:
    """Run the analysis on each entry in turn, in this process. Returns each
    entry's state."""
    states = {}
    for entry in entries:
        states[entry.name] = run_entry(analysis=analysis, entry=entry, out_root=out_root, overwrite=overwrite)
        if verbose:
            print(f"{states[entry.name]:15s} {entry.name}", flush=True)
    return states


def statuses(out_root: Path) -> Dict[str, Dict]:
    """The status record of every entry under `out_root`."""
    records = {}
    for path in sorted(Path(out_root).glob(f"*/{STATUS_FILE}")):
        records[path.parent.name] = read_status(path.parent)
    return records
