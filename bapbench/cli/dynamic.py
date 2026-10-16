"""Disruption experiments and rolling-horizon comparisons."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

from ..dynamics import Event, apply_events, evaluate_schedule_on, make_events, rolling_horizon
from ..instances import make_hard_instance
from ..metrics import gap_pct
from ..models.bap import BAPTimeIndexed

DEFAULT_SEEDS = 8


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _base(args) -> tuple:
    planned = make_hard_instance("base", args.instances, args.berths,
                                 seed=args.seed, slack=args.slack)
    return planned, BAPTimeIndexed(planned, time_limit=args.time_limit).solve()


# --------------------------------------------------------------------------
# single disruption types, evaluated offline with full information
# --------------------------------------------------------------------------
def cmd_events(args) -> int:
    planned, base = _base(args)
    if not base.feasible:
        print("the undisrupted instance is infeasible:", base.violations[:2])
        return 1
    print(planned.summary())
    print(f"undisrupted optimum {base.objective}\n")

    cases = [
        ("arrival_delay", lambda: [Event("arrival_delay", 0, v, 5, "arrival +5")
                                   for v in (0, 2, 4)]),
        ("handle_overrun", lambda: [Event("handle_overrun", 0, v, 3, "handling +3")
                                    for v in (1, 3)]),
        ("tidal_shift", lambda: [Event("tidal_shift", 0, 1, 4, "low tide +4")]),
        ("crane_loss", lambda: [Event("crane_loss", 10, None, 10, "one crane lost")]),
    ]
    rows = []
    print(f"  {'disruption':<18} {'objective':>10} {'vs base %':>10} {'feasible':>9}")
    for name, maker in cases:
        events = maker()
        disrupted = apply_events(planned, events)
        sch = BAPTimeIndexed(disrupted, time_limit=args.time_limit).solve()
        delta = gap_pct(sch.objective, base.objective) if sch.feasible else None
        print(f"  {name:<18} {str(sch.objective):>10} {str(delta):>10} "
              f"{str(sch.feasible):>9}")
        rows.append({"case": name, "n_events": len(events), "objective": sch.objective,
                     "delta_pct": delta, "feasible": sch.feasible,
                     "wall_time_s": round(sch.wall_time, 4),
                     "note": "; ".join(e.note for e in events),
                     "horizon": disrupted.horizon})
    _write_csv(args.out / "events.csv", rows)
    print(f"\nwrote {args.out / 'events.csv'}")
    return 0


# --------------------------------------------------------------------------
# rolling horizon against the perfect-information optimum
# --------------------------------------------------------------------------
def run_rolling(planned, n_events: int, seed: int, dt: int, time_limit: float) -> dict:
    rng = random.Random(seed)
    events = make_events(planned, rng, n_arrival=max(1, n_events // 2),
                         n_overrun=max(1, n_events // 3),
                         tidal_shift=(n_events >= 4), crane_loss=(n_events >= 4))
    events = events[:n_events]
    realised = apply_events(planned, events)

    perfect = BAPTimeIndexed(realised, time_limit=time_limit).solve()
    rr = rolling_horizon(planned, realised, events, dt=dt, time_limit=time_limit)
    ok, rolling_obj, violations = evaluate_schedule_on(realised, rr.schedule)

    regret = None
    if ok and rolling_obj is not None and perfect.feasible and perfect.objective:
        regret = round((rolling_obj - perfect.objective) / perfect.objective * 100.0, 3)
    return {
        "n_events": n_events, "seed": seed, "dt": dt,
        "perfect_objective": perfect.objective if perfect.feasible else None,
        "rolling_objective": rolling_obj if ok else None,
        "regret_pct": regret, "feasible": ok,
        "reoptimizations": rr.reoptimizations,
        "violation": violations[0] if violations else "",
        "event_kinds": ";".join(e.kind for e in events),
    }


def cmd_rolling(args) -> int:
    planned, base = _base(args)
    print(planned.summary())
    print(f"undisrupted optimum {base.objective}\n")

    rows = [run_rolling(planned, n, args.seed, args.dt, args.time_limit)
            for n in args.events]
    print(f"  {'events':>7} {'perfect':>9} {'rolling':>9} {'regret %':>9} "
          f"{'feasible':>9} {'replans':>8}")
    for r in rows:
        print(f"  {r['n_events']:>7} {str(r['perfect_objective']):>9} "
              f"{str(r['rolling_objective']):>9} {str(r['regret_pct']):>9} "
              f"{str(r['feasible']):>9} {r['reoptimizations']:>8}")
    _write_csv(args.out / "rolling.csv", rows)
    print(f"\nwrote {args.out / 'rolling.csv'}")
    return 0


def cmd_multiseed(args) -> int:
    planned, _ = _base(args)
    print(planned.summary())
    print(f"{args.seeds} seeds per setting, dt={args.dt}\n")

    rows = []
    print(f"  {'events':>7} {'feasible':>10} {'regret mean':>12} {'sd':>7} "
          f"{'median':>8} {'max':>8}")
    for n in args.events:
        sub = [run_rolling(planned, n, args.seed + i, args.dt, args.time_limit)
               for i in range(args.seeds)]
        for r in sub:
            r["scan"] = "multiseed"
        rows.extend(sub)
        regrets = [r["regret_pct"] for r in sub if r["regret_pct"] is not None]
        n_ok = sum(1 for r in sub if r["feasible"])
        if regrets:
            print(f"  {n:>7} {n_ok:>5}/{len(sub):<4} "
                  f"{statistics.mean(regrets):>12.3f} "
                  f"{statistics.pstdev(regrets) if len(regrets) > 1 else 0:>7.3f} "
                  f"{statistics.median(regrets):>8.3f} {max(regrets):>8.3f}")
        else:
            print(f"  {n:>7} {n_ok:>5}/{len(sub):<4} {'no feasible run':>12}")

    _write_csv(args.out / "multiseed.csv", rows)
    print(f"\nwrote {args.out / 'multiseed.csv'}")

    all_regrets = [r["regret_pct"] for r in rows if r["regret_pct"] is not None]
    n_infeasible = sum(1 for r in rows if not r["feasible"])
    if not all_regrets:
        print(f"warning: no run produced a regret value "
              f"({n_infeasible}/{len(rows)} runs had no feasible rolling schedule); "
              f"nothing can be concluded about commitment")
        return 1
    if all(r == 0 for r in all_regrets):
        print("warning: every regret is zero, which usually means commitments "
              "are not being enforced")
        return 1
    if n_infeasible:
        print(f"warning: {n_infeasible}/{len(rows)} runs produced no feasible "
              f"rolling schedule and are excluded from the regret statistics")
    return 0


def cmd_dt(args) -> int:
    planned, _ = _base(args)
    print(f"undisrupted optimum {BAPTimeIndexed(planned, time_limit=args.time_limit).solve().objective}")
    print(f"{args.seeds} seeds per window\n")
    rows = []
    print(f"  {'dt':>4} {'feasible':>10} {'regret mean':>12} {'sd':>7} {'replans':>9}")
    for dt in args.windows:
        sub = []
        for i in range(args.seeds):
            r = run_rolling(planned, args.dt_events, args.seed + i, dt, args.time_limit)
            r["scan"] = "dt"
            sub.append(r)
        rows.extend(sub)
        regrets = [r["regret_pct"] for r in sub if r["regret_pct"] is not None]
        n_ok = sum(1 for r in sub if r["feasible"])
        print(f"  {dt:>4} {n_ok:>5}/{len(sub):<4} "
              f"{statistics.mean(regrets) if regrets else float('nan'):>12.3f} "
              f"{statistics.pstdev(regrets) if len(regrets) > 1 else 0:>7.3f} "
              f"{statistics.mean([r['reoptimizations'] for r in sub]):>9.2f}")
    _write_csv(args.out / "dt.csv", rows)
    print(f"\nwrote {args.out / 'dt.csv'}")
    return 0


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bapbench dynamic",
                                description="Disruptions and rolling-horizon scheduling.")
    p.add_argument("mode", choices=("events", "rolling", "multiseed", "dt"),
                   nargs="?", default="events")
    p.add_argument("--out", type=Path, default=Path("results/dynamic"))
    p.add_argument("--instances", type=int, default=10)
    p.add_argument("--berths", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--slack", type=float, default=1.8)
    p.add_argument("--time-limit", type=float, default=30.0)
    p.add_argument("--dt", type=int, default=8, help="replanning interval in periods")
    p.add_argument("--dt-events", type=int, default=4, help="event count for the dt scan")
    p.add_argument("--events", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--windows", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    args = p.parse_args(argv)

    dispatch = {"events": cmd_events, "rolling": cmd_rolling,
                "multiseed": cmd_multiseed, "dt": cmd_dt}
    return dispatch[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
