"""The batch runner: entries, status records, the Slurm job script and the
command line. The MCMC analyses are in test_batch_mcmc.py."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from posteriordb_jax.batch import Entry, NotApplicable, load_analysis, read_status, read_statuses, run_local, select_entries
from posteriordb_jax.batch.__main__ import main
from posteriordb_jax.batch.slurm import SlurmOptions, submit

EXCHANGEABLE = "kidiq-kidscore_momiq"
NOT_EXCHANGEABLE = "eight_schools-eight_schools_noncentered"

ANALYSES = '''
import json

from posteriordb_jax.batch import NotApplicable


def write_free_length(entry, out_dir):
    (out_dir / "result.json").write_text(json.dumps({"free_length": entry.model.params_pattern.get_free_length()}))


def write_n_obs_if_exchangeable(entry, out_dir):
    if not entry.is_exchangeable:
        raise NotApplicable("not exchangeable")
    (out_dir / "n_obs.txt").write_text(str(entry.model.n_obs))


def raise_error(entry, out_dir):
    raise RuntimeError("boom")
'''


@pytest.fixture
def analyses_file(tmp_path):
    path = tmp_path / "my_analyses.py"
    path.write_text(ANALYSES)
    return path


def test_entry():
    entry = Entry(EXCHANGEABLE)
    assert entry.stan_model == "kidscore_momiq" and entry.data_name == "kidiq"
    assert entry.is_exchangeable and not Entry(NOT_EXCHANGEABLE).is_exchangeable
    assert entry.load_reference_draws()["beta"].shape[-1] == 2
    assert [e.name for e in select_entries([EXCHANGEABLE, NOT_EXCHANGEABLE], exchangeable=True)] == [EXCHANGEABLE]


def test_run_local_records_each_outcome(analyses_file, tmp_path):
    out = tmp_path / "out"
    names = [Entry(EXCHANGEABLE), Entry(NOT_EXCHANGEABLE)]
    states = run_local(analysis=load_analysis(f"{analyses_file}:write_n_obs_if_exchangeable"), entries=names, out_root=out,
                       verbose=False)
    assert states == {EXCHANGEABLE: "done", NOT_EXCHANGEABLE: "not_applicable"}
    assert (out / EXCHANGEABLE / "n_obs.txt").read_text() == "434"
    assert read_status(out / NOT_EXCHANGEABLE)["message"] == "not exchangeable"

    failed = run_local(analysis=load_analysis(f"{analyses_file}:raise_error"), entries=[Entry(EXCHANGEABLE)],
                       out_root=tmp_path / "raise_error", verbose=False)
    assert failed == {EXCHANGEABLE: "failed"}
    assert "RuntimeError: boom" in read_statuses(tmp_path / "raise_error")[EXCHANGEABLE]["message"]


def test_finished_entries_are_skipped_unless_overwritten(analyses_file, tmp_path):
    analysis = load_analysis(f"{analyses_file}:write_free_length")
    run_local(analysis=analysis, entries=[Entry(EXCHANGEABLE)], out_root=tmp_path, verbose=False)
    result = tmp_path / EXCHANGEABLE / "result.json"
    result.unlink()
    run_local(analysis=analysis, entries=[Entry(EXCHANGEABLE)], out_root=tmp_path, verbose=False)
    assert not result.exists()
    run_local(analysis=analysis, entries=[Entry(EXCHANGEABLE)], out_root=tmp_path, overwrite=True, verbose=False)
    assert json.loads(result.read_text()) == {"free_length": 3}


def test_load_analysis_from_module():
    from posteriordb_jax.batch import runner
    assert load_analysis("posteriordb_jax.batch.runner:run_entry") is runner.run_entry
    with pytest.raises(ValueError):
        load_analysis("no_colon")


def fake_sbatch(tmp_path):
    """An `sbatch` that records its arguments and prints a job id."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "sbatch"
    script.write_text(f'#!/bin/sh\necho "$@" > {tmp_path}/sbatch_args\necho 12345\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def test_submit_writes_an_array_job_and_each_task_runs(analyses_file, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", f"{fake_sbatch(tmp_path)}{os.pathsep}{os.environ['PATH']}")
    options = SlurmOptions(time="00:10:00", mem="2G", partition="short", max_concurrent=5,
                           setup=["echo setting up"], python=sys.executable, extra_sbatch=["--qos=low"])
    script, job_id = submit(analysis=f"{analyses_file}:write_free_length", entries=[Entry(EXCHANGEABLE), Entry(NOT_EXCHANGEABLE)],
                            out_root=tmp_path / "out", options=options)
    assert job_id == "12345"
    assert (tmp_path / "sbatch_args").read_text().split() == ["--parsable", str(script)]
    text = script.read_text()
    for line in ("#SBATCH --array=0-1%5", "#SBATCH --time=00:10:00", "#SBATCH --mem=2G", "#SBATCH --partition=short",
                 "#SBATCH --qos=low", "echo setting up", '--index $SLURM_ARRAY_TASK_ID'):
        assert line in text
    assert (script.parent / "names.txt").read_text().split() == [EXCHANGEABLE, NOT_EXCHANGEABLE]

    # What Slurm would do: run the script once per array index.
    for index in range(2):
        subprocess.run(["bash", str(script)], check=True, env={**os.environ, "SLURM_ARRAY_TASK_ID": str(index)},
                       cwd="/", capture_output=True)
    records = read_statuses(tmp_path / "out")
    assert {name: record["state"] for name, record in records.items()} == {EXCHANGEABLE: "done", NOT_EXCHANGEABLE: "done"}


def test_command_line(analyses_file, tmp_path, capsys):
    out = tmp_path / "out"
    main(["local", "--analysis", f"{analyses_file}:write_n_obs_if_exchangeable", "--out", str(out),
          "--posteriors", EXCHANGEABLE, NOT_EXCHANGEABLE])
    main(["status", "--out", str(out), "--all"])
    assert "not_applicable  " + NOT_EXCHANGEABLE in capsys.readouterr().out
    main(["list", "--posteriors", EXCHANGEABLE, NOT_EXCHANGEABLE, "--only-unfinished", "--out", str(out)])
    assert capsys.readouterr().out == ""
    main(["submit", "--analysis", f"{analyses_file}:write_free_length", "--out", str(out), "--posteriors", EXCHANGEABLE,
          "--dry-run", "--setup", "module load python"])
    assert "not submitted" in capsys.readouterr().out
    assert "module load python" in next(out.glob("_slurm/*/job.sbatch")).read_text()
