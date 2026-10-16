"""One instance, solved exactly and by every baseline, with a Gantt chart."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..heuristics.constructive import run_all_baselines
from ..instances import make_hard_instance, make_toy_instance
from ..metrics import draw_gantt, gap_pct
from ..models.bap import BAPTimeIndexed
from ..verification.exhaustive import verify_instance


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="baps demo",
                                description="Solve one instance with the exact model "
                                            "and the baselines.")
    p.add_argument("--instances", type=int, default=6)
    p.add_argument("--berths", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--slack", type=float, default=1.6,
                   help="horizon over the theoretical minimum; lower is tighter")
    p.add_argument("--objective", choices=("sum_C", "C_max"), default="sum_C")
    p.add_argument("--time-limit", type=float, default=120.0)
    p.add_argument("--out", type=Path, default=Path("results/demo"))
    p.add_argument("--verify", action="store_true",
                   help="also compute the exhaustive optimum (small instances only)")
    args = p.parse_args(argv)

    inst = make_hard_instance("demo", args.instances, args.berths, seed=args.seed,
                              slack=args.slack)
    problems = inst.validate()
    if problems:
        print("instance is infeasible by construction:")
        for problem in problems:
            print("  -", problem)
        return 1

    print(inst.summary())
    print()

    model = BAPTimeIndexed(inst, objective=args.objective,
                           time_limit=args.time_limit, mip_gap=0.0)
    exact = model.solve()
    if not exact.feasible:
        print("the exact model found no feasible schedule:")
        for v in exact.violations:
            print("  -", v)
        return 1
    optimum = exact.objective
    print(f"exact model           objective {optimum:>8.0f}   "
          f"{exact.wall_time:>7.3f}s")

    if args.verify:
        agrees, result = verify_instance(inst, optimum, args.objective,
                                         max_cases=2_000_000, time_budget=300.0)
        if result.objective is None:
            print(f"exhaustive enumeration skipped: {result.note}")
        else:
            print(f"exhaustive enumeration objective {result.objective:>7.0f}   "
                  f"{result.n_cases} cases   "
                  f"{'agrees' if agrees else 'DISAGREES'}")
            if not agrees:
                print("the two optima differ; one of the two models is wrong")
                return 1

    print()
    print(f"  {'method':<12} {'objective':>10} {'gap %':>8} {'feasible':>9} {'time s':>9}")
    schedules = run_all_baselines(inst, args.objective, seed=args.seed)
    for name, sch in schedules.items():
        gap = gap_pct(sch.objective, optimum) if sch.feasible else None
        print(f"  {name:<12} {str(sch.objective):>10} "
              f"{'' if gap is None else f'{gap:.2f}':>8} "
              f"{str(sch.feasible):>9} {sch.wall_time:>9.4f}")

    chosen = None
    for name, sch in schedules.items():
        if sch.feasible and (chosen is None or sch.objective < chosen[1].objective):
            chosen = (name, sch)
    if chosen:
        path = draw_gantt(inst, chosen[1].assign, args.out / "gantt.png",
                          title=f"{inst.name} - best baseline ({chosen[0]})")
        print(f"\ngantt chart written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
