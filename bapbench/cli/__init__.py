"""Subcommand dispatch for the bapbench command line interface."""

from __future__ import annotations

import sys

from . import classical as classical_cmd
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
    "classical": (classical_cmd, "classical scheduling against enumeration"),
    "dynamic": (dynamic_cmd, "disruptions and rolling-horizon experiments"),
}


def _usage() -> str:
    lines = ["usage: bapbench <command> [options]", "", "commands:"]
    width = max(len(name) for name in COMMANDS)
    for name in sorted(COMMANDS):
        lines.append(f"  {name:<{width}}  {COMMANDS[name][1]}")
    lines += ["", "Run 'bapbench <command> --help' for the options of a command.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Dispatch on the first token by hand so that everything after the command
    # reaches that command's own parser. Letting a top-level parser handle it
    # would swallow --help and the subcommand would never see it.
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"bapbench: unknown command {command!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    return COMMANDS[command][0].main(rest)


if __name__ == "__main__":
    sys.exit(main())
