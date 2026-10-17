"""Independent checks that the formulations and the models agree."""

from __future__ import annotations

import argparse
import sys

from ..instances import make_toy_instance
from ..metrics import objective_value
from ..models.bap import BAPDisjunctive, BAPTimeIndexed
from ..verification.benchmarks import (
    case_flow_bottleneck,
    case_parallel_balanced,
    case_single_machine,
    case_two_implementations_agree,
)
from ..verification.exhaustive import verify_instance
from ..verification.jobshop import job_shop_optimum


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))
    return ok


def formulation_agreement() -> bool:
    """The two formulations must return the same optimum on the same instance."""
    results = []
    for name, spec in (("toy_5x3", dict(n_vessels=5, n_berths=3, seed=20261014)),
                       ("toy_4x2", dict(n_vessels=4, n_berths=2, seed=7)),
                       ("tight_6x3", dict(n_vessels=6, n_berths=3, seed=11,
                                          tight=True))):
        inst = make_toy_instance(name, **spec)
        ti = BAPTimeIndexed(inst).solve()
        dj = BAPDisjunctive(inst).solve()
        same = (ti.feasible and dj.feasible
                and abs(ti.objective - dj.objective) < 1e-6)
        results.append(check(
            f"time-indexed and disjunctive agree on {name}",
            same, f"ti={ti.objective} dj={dj.objective}"))
    return all(results)


def exhaustive_agreement() -> bool:
    results = []
    for name, spec in (("exh_5x3", dict(n_vessels=5, n_berths=3, seed=20261014)),
                       ("exh_4x2", dict(n_vessels=4, n_berths=2, seed=3))):
        inst = make_toy_instance(name, **spec)
        sch = BAPTimeIndexed(inst).solve()
        agrees, res = verify_instance(inst, sch.objective, max_cases=2_000_000,
                                      time_budget=180.0)
        if res.objective is None:
            results.append(check(f"enumeration feasible for {name}", False, res.note))
            continue
        if not res.feasible:
            # The search hit its budget, so its incumbent is an upper bound and
            # comparing it would report a model disagreement the run has not shown.
            results.append(check(f"enumeration complete for {name}", False, res.note))
            continue
        results.append(check(
            f"exact model matches enumeration on {name}",
            agrees, f"model={sch.objective} enumerated={res.objective} "
                    f"cases={res.n_cases}"))
    return all(results)


def objective_definitions() -> bool:
    comp = {0: 5, 1: 9, 2: 12}
    ok = (objective_value(comp, "sum_C") == 26.0
          and objective_value(comp, "C_max") == 12.0)
    return check("objective conventions", ok,
                 f"sum_C={objective_value(comp, 'sum_C')} "
                 f"C_max={objective_value(comp, 'C_max')}")


def bound_sanity() -> bool:
    """No objective may fall below the trivial bound sum_i (r_i + p_i)."""
    inst = make_toy_instance("bound_6x3", 6, 3, seed=20261014)
    sch = BAPTimeIndexed(inst).solve()
    lower = sum(v.arrive + v.handle for v in inst.vessels)
    return check("objective respects the arrival bound",
                 sch.objective >= lower - 1e-6,
                 f"objective={sch.objective} bound={lower}")


def benchmark_families() -> bool:
    results = []
    for fn in (case_single_machine, case_parallel_balanced,
               case_flow_bottleneck, case_two_implementations_agree):
        r = fn()
        expected = r.get("expected", r.get("obj_per_operation", ""))
        results.append(check(r["name"], bool(r.get("match_expected")),
                             f"expected={expected}"))
    return all(results)


def job_shop_agreement() -> bool:
    """The job-shop MILP must match a solver-free exact search.

    The search branches over active schedules (Giffler-Thompson), which contain an
    optimal one, so its value is the optimum; a run that hits its budget is
    reported as an upper bound and is not compared here.
    """
    from ..models.classical import JobShopMILP, make_simple_problem, to_instance_view

    results = []
    for n, m in ((4, 3), (5, 3)):
        prob = make_simple_problem(f"js_{n}x{m}", n, m, seed=51 + n,
                                   horizon=n * 12 + 40)
        routes = {j: list(range(m)) for j in range(n)}
        milp = JobShopMILP(prob, routes=routes,
                           instance_view=to_instance_view(prob)).solve()
        proc = [[job.proc] * m for job in prob.jobs]
        search = job_shop_optimum([list(range(m)) for _ in range(n)], proc,
                                  time_budget=60.0)
        ok = (milp.feasible and search.feasible and search.objective is not None
              and abs(milp.objective - search.objective) < 1e-6)
        results.append(check(f"job shop MILP matches the exact search on {n}x{m}",
                             ok, f"milp={milp.objective} search={search.objective}"
                                 + (f" ({search.note})" if search.note else "")))
    return all(results)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bapbench verify",
                                description="Independent verification suite.")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["formulations", "exhaustive", "objectives",
                            "bounds", "benchmarks", "jobshop"])
    args = p.parse_args(argv)

    groups = [
        ("formulations", "two formulations agree", formulation_agreement),
        ("exhaustive", "exact model matches enumeration", exhaustive_agreement),
        ("objectives", "objective conventions", objective_definitions),
        ("bounds", "objective respects the arrival bound", bound_sanity),
        ("benchmarks", "known-optimum instance families", benchmark_families),
        ("jobshop", "job shop against a solver-free exact search",
         job_shop_agreement),
    ]

    failures = 0
    for key, title, fn in groups:
        if key in args.skip:
            continue
        print(f"{title}:")
        if not fn():
            failures += 1
        print()

    print("all checks passed" if not failures else f"{failures} group(s) failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
