"""The BACAP model of Iris, Pacino & Ropke (2017) as a command.

`bapbench reproduce` runs the three method reproductions on the package's own
`Instance`. This command runs the paper's own model instead: continuous berthing
position, a per-vessel crane range, the Appendix A crane pattern and the
per-period plan. It generates an instance from the published distributions, runs
the ALNS, and writes the assignments, the QC plan and a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from ..reproduction import bacap


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def _assignment_rows(inst: bacap.BACAPInstance,
                     result: bacap.BACAPResult) -> list[dict]:
    rows = []
    for vid, a in sorted(result.solution.assign.items(), key=lambda kv: kv[1].start):
        v = inst.vessels[vid]
        rows.append({
            "vessel": v.name, "class": v.kind, "length_sections": v.length,
            "berth": a.berth, "start": a.start, "finish": a.finish,
            "q_hat": a.config.q_hat, "f1": a.config.f1, "f2": a.config.f2,
            "crane_hours": a.config.crane_hours,
            "cost_thousand_usd": round(result.solution.cost_of(vid), 4),
        })
    return rows


def _plan_rows(result: bacap.BACAPResult) -> list[dict]:
    rows = []
    for vid in sorted(result.solution.cranes):
        for t, q in sorted(result.solution.cranes[vid].items()):
            rows.append({"vessel": result.solution.inst.vessels[vid].name,
                         "period": t, "cranes": q})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="bapbench bacap",
        description="Run the paper's own BACAP model on a reconstructed instance.")
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261014,
                   help="instance seed; it decides the vessel draws")
    p.add_argument("--alns-seed", type=int, default=1, help="search seed")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--quay", type=int, default=bacap.QUAY_SECTIONS,
                   help="quay length in 10 m sections")
    p.add_argument("--cranes", type=int, default=bacap.N_CRANES)
    p.add_argument("--horizon", type=int, default=bacap.HORIZON)
    p.add_argument("--max-candidates", type=int, default=120,
                   help="cost-ordered assignments kept per vessel")
    p.add_argument("--out", type=Path, default=Path("results/bacap"))
    args = p.parse_args(argv)

    try:
        inst = bacap.make_instance(f"bacap_{args.instances}", args.instances,
                                   seed=args.seed, quay=args.quay,
                                   cranes=args.cranes, horizon=args.horizon)
    except ValueError as exc:
        print(f"bapbench bacap: cannot build the instance: {exc}", file=sys.stderr)
        return 2

    mix = Counter(v.kind for v in inst.vessels)
    print(f"instance {inst.name}: {inst.n} vessels, quay {inst.quay * 10} m in "
          f"{inst.quay} sections, {inst.cranes} cranes, horizon {inst.horizon} h")
    print(f"  {', '.join(f'{count} {kind}' for kind, count in sorted(mix.items()))}"
          f"  |  {inst.meta['instance_kind']}")
    print()

    config = bacap.BACAPConfig(max_iterations=args.iterations,
                               max_candidates=args.max_candidates)
    try:
        result = bacap.alns(inst, config, seed=args.alns_seed)
    except RuntimeError as exc:
        print(f"bapbench bacap: the instance admits no plan: {exc}", file=sys.stderr)
        return 2

    problems = bacap.check_solution(inst, result.solution)
    print(f"objective {result.objective:.3f} (thousand USD)   "
          f"feasible {result.feasible}   unassigned {len(result.unassigned)}")
    print(f"{result.iterations} iterations in {result.wall_time:.2f}s")
    for name, weight in sorted(result.operators["destroy"].items()):
        print(f"  destroy {name:<10} weight {weight:.3f}")
    for name, weight in sorted(result.operators["insertion"].items()):
        print(f"  insert  {name:<10} weight {weight:.3f}")
    if problems:
        print("\nthe independent check rejects the plan:")
        for problem in problems:
            print(f"  - {problem}")
    if result.unassigned:
        names = [inst.vessels[vid].name for vid in result.unassigned]
        print(f"\nunassigned vessels: {names}")
    print()

    rows = _assignment_rows(inst, result)
    print(f"  {'vessel':<8} {'class':<8} {'pos':>4} {'start':>6} {'end':>4} "
          f"{'q_hat':>6} {'f1':>3} {'f2':>3} {'cr-h':>5} {'cost':>7}")
    for row in rows:
        print(f"  {row['vessel']:<8} {row['class']:<8} {row['berth']:>4} "
              f"{row['start']:>6} {row['finish']:>4} {row['q_hat']:>6} "
              f"{row['f1']:>3} {row['f2']:>3} {row['crane_hours']:>5} "
              f"{row['cost_thousand_usd']:>7.3f}")

    assignments = args.out / "assignments.csv"
    _write_csv(assignments, list(rows[0].keys()) if rows else [], rows)
    plan = args.out / "qc_plan.csv"
    plan_rows = _plan_rows(result)
    _write_csv(plan, ["vessel", "period", "cranes"], plan_rows)
    summary = args.out / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({
        "instance": inst.meta,
        "instance_seed": args.seed,
        "alns_seed": args.alns_seed,
        "iterations": result.iterations,
        "objective_thousand_usd": result.objective,
        "feasible": result.feasible,
        "unassigned": list(result.unassigned),
        "violations": problems,
        "wall_time_s": round(result.wall_time, 4),
        "operators": result.operators,
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {assignments} ({len(rows)} rows)")
    print(f"wrote {plan} ({len(plan_rows)} rows)")
    print(f"wrote {summary}")
    return 0 if result.feasible and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
