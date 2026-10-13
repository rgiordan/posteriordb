"""Writing and submitting a Slurm array job with one task per entry.

The job directory `<out_root>/_slurm/<timestamp>/` holds the list of
posterior names (`names.txt`), the batch script (`job.sbatch`) and the
tasks' logs (`logs/`). Task i runs

    python -m posteriordb_jax.batch run-one --analysis SPEC --names-file names.txt --index i --out OUT_ROOT
"""

from __future__ import annotations

import datetime
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .entry import Entry


@dataclass
class SlurmOptions:
    """Resources for each array task, and the shell lines that set up its
    environment (e.g. "module load python", "source ~/venv/bin/activate")."""

    job_name: str = "posteriordb"
    time: str = "01:00:00"
    mem: str = "8G"
    cpus_per_task: int = 1
    partition: Optional[str] = None
    account: Optional[str] = None
    gpus: Optional[int] = None
    max_concurrent: Optional[int] = None      # at most this many tasks at once (sbatch --array=...%N)
    setup: Sequence[str] = ()
    python: str = "python"
    extra_sbatch: Sequence[str] = field(default_factory=tuple)   # e.g. ["--qos=long"]

    def format_sbatch_lines(self, *, n_tasks: int, log_dir: Path) -> List[str]:
        array = f"0-{n_tasks - 1}" + (f"%{self.max_concurrent}" if self.max_concurrent else "")
        options = [
            f"--job-name={self.job_name}",
            f"--array={array}",
            f"--time={self.time}",
            f"--mem={self.mem}",
            f"--cpus-per-task={self.cpus_per_task}",
            f"--output={log_dir}/%A_%a.out",
        ]
        if self.partition:
            options.append(f"--partition={self.partition}")
        if self.account:
            options.append(f"--account={self.account}")
        if self.gpus:
            options.append(f"--gpus={self.gpus}")
        options.extend(self.extra_sbatch)
        return [f"#SBATCH {option}" for option in options]


def make_analysis_spec_portable(spec: str) -> str:
    """The spec with a file path made absolute, so tasks find it from any
    working directory."""
    module_name, _, attribute = spec.rpartition(":")
    if module_name.endswith(".py"):
        return f"{Path(module_name).resolve()}:{attribute}"
    return spec


def write_job(*, analysis: str, entries: Iterable[Entry], out_root: Path, options: SlurmOptions,
              overwrite: bool = False) -> Path:
    """Write the job directory and return the path of its batch script."""
    names = [entry.name for entry in entries]
    if not names:
        raise ValueError("no posteriors selected")
    out_root = Path(out_root).resolve()
    job_dir = out_root / "_slurm" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_dir = job_dir / "logs"
    log_dir.mkdir(parents=True)
    names_file = job_dir / "names.txt"
    names_file.write_text("\n".join(names) + "\n")
    command = [options.python, "-m", "posteriordb_jax.batch", "run-one",
               "--analysis", make_analysis_spec_portable(analysis), "--names-file", str(names_file),
               "--index", "$SLURM_ARRAY_TASK_ID", "--out", str(out_root)]
    if overwrite:
        command.append("--overwrite")
    script = "\n".join([
        "#!/bin/bash",
        *options.format_sbatch_lines(n_tasks=len(names), log_dir=log_dir),
        "",
        "set -euo pipefail",
        *options.setup,
        " ".join(part if part.startswith("$") else shlex.quote(part) for part in command),
        "",
    ])
    script_path = job_dir / "job.sbatch"
    script_path.write_text(script)
    return script_path


def submit(*, analysis: str, entries: Iterable[Entry], out_root: Path, options: SlurmOptions,
           overwrite: bool = False, dry_run: bool = False) -> Tuple[Path, Optional[str]]:
    """Write the job and submit it with sbatch. Returns the batch script's
    path and Slurm's job id (None for a dry run, which only writes)."""
    script_path = write_job(analysis=analysis, entries=entries, out_root=out_root, options=options,
                            overwrite=overwrite)
    if dry_run:
        return script_path, None
    result = subprocess.run(["sbatch", "--parsable", str(script_path)], check=True, capture_output=True, text=True)
    return script_path, result.stdout.strip().split(";")[0]
