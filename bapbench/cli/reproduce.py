"""The reproduced published methods beside the package's own baselines."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from ..instances import make_toy_instance
from ..reproduction.comparison import compare_reproductions


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _print_rows(rows: list[dict]) -> None:
    print(f"  {'method':<22} {'objective':>10} {'gap %':>8} {'feasible':>9} "
          f"{'time s':>9}")
    for r in rows:
        print(f"  {r['method']:<22} {str(r['objective']):>10} "
              f"{str(r['gap_pct']):>8} {str(r['feasible']):>9} "
              f"{str(r['wall_time_s']):>9}")
        if not r["feasible"] and r["violations"]:
            print(f"  {'':<22} {r['violations']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="bapbench reproduce",
        description="Run the reproduced published methods beside the package's own "
                    "baselines on one instance.")
    p.add_argument("--instances", type=int, default=6)
    p.add_argument("--berths", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--objective", choices=("sum_C", "C_max"), default="sum_C")
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--procedures", nargs="+",
                   choices=("SIMPLE", "INDIVIDUAL", "INTERACT"),
                   default=["SIMPLE", "INDIVIDUAL", "INTERACT"],
                   help="which Imai repair procedures to run")
    p.add_argument("--max-iter", type=int, default=200,
                   help="subgradient iterations per Imai procedure")
    p.add_argument("--no-package-methods", action="store_true",
                   help="omit the package's own baselines and metaheuristics")
    p.add_argument("--no-iris", action="store_true",
                   help="omit the Iris ALNS reproduction")
    p.add_argument("--out", type=Path, default=Path("results/reproduce"))
    args = p.parse_args(argv)

    # The Imai reproductions solve the DBAP, which has no tidal windows and no
    # crane limit, so the instance is a plain one: on a constrained instance its
    # schedules would be infeasible here for reasons that say nothing about it.
    inst = make_toy_instance(f"reproduce_{args.instances}x{args.berths}",
                             args.instances, args.berths, seed=args.seed)
    print(inst.summary())
    print()

    rows = compare_reproductions(
        inst, args.objective, time_limit=args.time_limit, seed=args.seed,
        procedures=tuple(args.procedures),
        with_package_methods=not args.no_package_methods,
        with_iris=not args.no_iris,
        imai_max_iter=args.max_iter)

    _print_rows(rows)
    if rows:
        kind = rows[0]["reference_kind"]
        reference = rows[0]["reference"]
        print(f"\nreference for the gaps: {reference} ({kind})")

    path = args.out / "reproduce.csv"
    _write_csv(path, rows)
    print(f"wrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
