"""Subcommand dispatch for the baps command line interface."""

from __future__ import annotations

import argparse
import sys

from . import compare as compare_cmd
from . import demo as demo_cmd
from . import dynamic as dynamic_cmd
from . import heuristics as heuristics_cmd
from . import verify as verify_cmd

COMMANDS = {
    "demo": (demo_cmd, "solve one instance end to end and write a Gantt chart"),
    "verify": (verify_cmd, "check the formulations against independent optima"),
    "compare": (compare_cmd, "compare the exact model with the baselines"),
    "heuristics": (heuristics_cmd, "run the metaheuristics, optionally across seeds"),
    "dynamic": (dynamic_cmd, "disruptions and rolling-horizon experiments"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bapbench",
        description="Dynamic berth allocation: exact models, baselines and a rolling horizon.")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, rest = parser.parse_known_args(argv)

    module = COMMANDS[args.command][0]
    return module.main(rest)


if __name__ == "__main__":
    sys.exit(main())
