"""Command line: python -m posteriordb_jax.batch {list,local,submit,run-one,status} ...

Run with --help for each command's options."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .entry import Entry, select_entries
from .runner import FINISHED, load_analysis, run_entry, run_local, read_statuses
from .slurm import SlurmOptions, submit


def add_selection(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("which posteriors (default: every one in the catalog)")
    group.add_argument("--posteriors", nargs="+", metavar="NAME", help="posteriordb posterior names")
    group.add_argument("--exchangeable", action="store_true", help="only models with exchangeable data")
    group.add_argument("--with-reference-draws", action="store_true", help="only posteriors with reference draws")
    group.add_argument("--only-unfinished", action="store_true",
                       help="only posteriors whose status under --out is missing, failed or running")


def select_entries_from_args(args) -> list:
    chosen = select_entries(args.posteriors, exchangeable=True if args.exchangeable else None,
                     with_reference_draws=True if args.with_reference_draws else None)
    if getattr(args, "only_unfinished", False):
        records = read_statuses(args.out)
        chosen = [e for e in chosen if e.name not in records or records[e.name]["state"] not in FINISHED]
    return chosen


def add_slurm_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Slurm resources for each posterior's task")
    defaults = SlurmOptions()
    group.add_argument("--job-name", default=defaults.job_name)
    group.add_argument("--time", default=defaults.time)
    group.add_argument("--mem", default=defaults.mem)
    group.add_argument("--cpus-per-task", type=int, default=defaults.cpus_per_task)
    group.add_argument("--partition")
    group.add_argument("--account")
    group.add_argument("--gpus", type=int)
    group.add_argument("--max-concurrent", type=int, help="run at most this many tasks at once")
    group.add_argument("--setup", action="append", default=[], metavar="LINE",
                       help="shell line to run before Python, e.g. 'source ~/venv/bin/activate' (repeatable)")
    group.add_argument("--python", default=sys.executable, help="Python interpreter for the tasks (default: this one)")
    group.add_argument("--sbatch", action="append", default=[], metavar="OPTION",
                       help="extra sbatch option, e.g. '--qos=long' (repeatable)")
    group.add_argument("--dry-run", action="store_true", help="write the job script without submitting it")


def build_slurm_options(args) -> SlurmOptions:
    return SlurmOptions(job_name=args.job_name, time=args.time, mem=args.mem, cpus_per_task=args.cpus_per_task,
                        partition=args.partition, account=args.account, gpus=args.gpus,
                        max_concurrent=args.max_concurrent, setup=args.setup, python=args.python,
                        extra_sbatch=args.sbatch)


def handle_list(args) -> None:
    for entry in select_entries_from_args(args):
        print(entry.name)


def handle_local(args) -> None:
    states = run_local(analysis=load_analysis(args.analysis), entries=select_entries_from_args(args), out_root=args.out,
                       overwrite=args.overwrite)
    print(dict(Counter(states.values())))


def handle_submit(args) -> None:
    chosen = select_entries_from_args(args)
    script, job_id = submit(analysis=args.analysis, entries=chosen, out_root=args.out, options=build_slurm_options(args),
                            overwrite=args.overwrite, dry_run=args.dry_run)
    if job_id is None:
        print(f"wrote {script} for {len(chosen)} posteriors (not submitted)")
    else:
        print(f"submitted job {job_id}: {len(chosen)} posteriors, script {script}")


def handle_run_one(args) -> None:
    if args.name is not None:
        name = args.name
    else:
        name = Path(args.names_file).read_text().split()[args.index]
    state = run_entry(analysis=load_analysis(args.analysis), entry=Entry(name), out_root=args.out,
                      overwrite=args.overwrite)
    print(f"{state} {name}")
    if state == "failed":
        sys.exit(1)


def handle_status(args) -> None:
    records = read_statuses(args.out)
    print(dict(Counter(record["state"] for record in records.values())))
    for name, record in records.items():
        if record["state"] in ("failed", "running") or args.all:
            print(f"{record['state']:15s} {name}")
            if args.verbose and record.get("message"):
                print("    " + record["message"].strip().replace("\n", "\n    "))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m posteriordb_jax.batch", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="print the selected posterior names")
    listing.add_argument("--out", type=Path, help="results directory (for --only-unfinished)")
    add_selection(listing)
    listing.set_defaults(run=handle_list)

    for name, run, help_text in (("local", handle_local, "run the analysis on each posterior, in this process"),
                                 ("submit", handle_submit, "submit a Slurm array job, one task per posterior")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--analysis", required=True, help="'module:function' or 'file.py:function'")
        command.add_argument("--out", type=Path, required=True, help="results directory")
        command.add_argument("--overwrite", action="store_true", help="rerun posteriors that already finished")
        add_selection(command)
        if name == "submit":
            add_slurm_options(command)
        command.set_defaults(run=run)

    one = commands.add_parser("run-one", help="run the analysis on one posterior (what each Slurm task runs)")
    one.add_argument("--analysis", required=True)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--overwrite", action="store_true")
    which = one.add_mutually_exclusive_group(required=True)
    which.add_argument("--name", help="posterior name")
    which.add_argument("--names-file", type=Path, help="file of posterior names, one per line (with --index)")
    one.add_argument("--index", type=int, help="0-based line of --names-file")
    one.set_defaults(run=handle_run_one)

    status = commands.add_parser("status", help="summarize the results directory")
    status.add_argument("--out", type=Path, required=True)
    status.add_argument("--all", action="store_true", help="list every posterior, not only failed and running ones")
    status.add_argument("--verbose", action="store_true", help="show each listed posterior's message or traceback")
    status.set_defaults(run=handle_status)

    args = parser.parse_args(argv)
    if getattr(args, "only_unfinished", False) and args.out is None:
        parser.error("--only-unfinished needs --out")
    if args.command == "run-one" and args.names_file is not None and args.index is None:
        parser.error("--names-file needs --index")
    args.run(args)


if __name__ == "__main__":
    main()
