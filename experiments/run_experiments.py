# -*- coding: utf-8 -*-
"""Experiment driver for the four lab reports.

Every number in `reports/` comes from a run of this script: it writes the raw
records to `data/` as CSV/JSON and the figures to `figures/`. Run it from the
repository root:

    .venv\\Scripts\\python.exe experiments\\run_experiments.py

The script is deterministic: every random draw is seeded, and every instance is
built by name and seed, so re-running it reproduces the same tables.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bapbench.bounds import lp_lower_bound, trivial_lower_bound
from bapbench.cli.dynamic import run_rolling
from bapbench.dynamics import apply_events
from bapbench.heuristics.constructive import BASELINES, run_all_baselines
from bapbench.heuristics.metaheuristics import HEURISTICS
from bapbench.heuristics.solver_configs import run_config_comparison
from bapbench.instances import make_hard_instance, make_toy_instance
from bapbench.metrics import gap_pct
from bapbench.models.bap import BAPDisjunctive, BAPTimeIndexed
from bapbench.models.classical import (
    FlowShopMILP,
    JobShopMILP,
    ParallelMachineMILP,
    brute_force_parallel,
    brute_force_single_machine,
    make_simple_problem,
    single_machine_spt,
    single_machine_wspt,
    to_instance_view,
)
from bapbench.verification.exhaustive import exhaustive_optimum
from bapbench.verification.jobshop import job_shop_optimum

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data"
FIGURES = HERE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write_csv(name: str, rows: list[dict]) -> Path:
    if not rows:
        return RESULTS / name
    path = RESULTS / name
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def write_json(name: str, payload) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def solve_exact(inst, objective: str = "sum_C", limit: float = 60.0):
    return BAPTimeIndexed(inst, objective=objective, time_limit=limit,
                          mip_gap=0.0).solve()


# ---------------------------------------------------------------------------
# experiment 1: MILP, LP and the basic scheduling model
# ---------------------------------------------------------------------------

def e1_model_scale() -> list[dict]:
    rows = []
    for n in (4, 6, 8, 10):
        inst = make_toy_instance(f"e1_{n}x3", n, 3, seed=100 + n, tight=True)
        problems = inst.validate()
        sch = solve_exact(inst, "sum_C", 60.0)
        rows.append({
            "n": n, "m": inst.m, "horizon": inst.horizon,
            "feasible": sch.feasible, "objective": sch.objective,
            "proved_optimal": bool(sch.extra.get("proved_optimal")),
            "status": sch.extra.get("status"),
            "vars": sch.extra.get("vars"), "cons": sch.extra.get("cons"),
            "wall_s": round(sch.wall_time, 4),
            "validate_problems": len(problems),
        })
        print(f"  e1 scale n={n}: obj={sch.objective} proved="
              f"{sch.extra.get('proved_optimal')} {sch.wall_time:.2f}s")
    return rows


def e1_formulations_and_enumeration() -> tuple[list[dict], list[dict]]:
    formulations, enumerations = [], []
    for n in (4, 5, 6):
        inst = make_toy_instance(f"e1_form_{n}x3", n, 3, seed=7, tight=True)
        ti = solve_exact(inst, "sum_C", 60.0)
        dj = BAPDisjunctive(inst, time_limit=60.0).solve()
        formulations.append({
            "n": n, "ti_objective": ti.objective, "dj_objective": dj.objective,
            "agree": bool(ti.feasible and dj.feasible and ti.objective == dj.objective),
            "ti_wall_s": round(ti.wall_time, 4),
            "dj_wall_s": round(dj.wall_time, 4),
            "ti_status": ti.extra.get("status"),
            "dj_status": dj.extra.get("status"),
        })
        print(f"  e1 formulations n={n}: ti={ti.objective} dj={dj.objective}")
        if n <= 5:
            res = exhaustive_optimum(inst, "sum_C", max_cases=2_000_000,
                                     time_budget=120.0)
            enumerations.append({
                "n": n, "milp": ti.objective, "enumerated": res.objective,
                "agree": (res.feasible and res.objective is not None
                          and ti.objective == res.objective),
                "cases": res.n_cases, "wall_s": round(res.wall_time, 3),
                "complete": res.feasible, "note": res.note,
            })
            print(f"  e1 enumeration n={n}: {res.objective} cases={res.n_cases}")
    return formulations, enumerations


def e1_bounds() -> list[dict]:
    cases = [
        ("toy_6x3", make_toy_instance("e1_bounds_6x3", 6, 3, seed=7, tight=True)),
        ("toy_8x3", make_toy_instance("e1_bounds_8x3", 8, 3, seed=7, tight=True)),
        ("hard_10x3", make_hard_instance("e1_bounds_10x3", 10, 3,
                                         seed=20261014, slack=1.8)),
    ]
    rows = []
    for label, inst in cases:
        sch = solve_exact(inst, "sum_C", 60.0)
        triv = trivial_lower_bound(inst, "sum_C")
        lp = lp_lower_bound(inst, "sum_C", time_limit=60.0)
        opt = sch.objective if sch.extra.get("proved_optimal") else None
        rows.append({
            "instance": label, "n": inst.n, "optimal": opt,
            "trivial": triv.value,
            "trivial_gap_pct": (round((opt - triv.value) / opt * 100, 2)
                                if opt and triv.value is not None else None),
            "lp": lp.value,
            "lp_gap_pct": (round((opt - lp.value) / opt * 100, 2)
                           if opt and lp.value is not None else None),
            "lp_wall_s": round(lp.wall_time, 3),
        })
        print(f"  e1 bounds {label}: opt={opt} lp={lp.value} triv={triv.value}")
    return rows


def e1_objectives() -> list[dict]:
    inst = make_toy_instance("e1_obj_7x3", 7, 3, seed=11, tight=True)
    rows = []
    for objective in ("sum_C", "C_max"):
        sch = solve_exact(inst, objective, 60.0)
        primary = sch.objective
        other = "C_max" if objective == "sum_C" else "sum_C"
        # Evaluate the same schedule under the other objective; the two optima are
        # optima of different functions and need not coincide.
        other_value = sch.compute_objective(inst, other)
        rows.append({
            "objective": objective, "value": primary,
            "other_objective": other, "other_value": other_value,
            "proved_optimal": bool(sch.extra.get("proved_optimal")),
            "wall_s": round(sch.wall_time, 4),
        })
        print(f"  e1 objective {objective}: {primary} (same plan {other}={other_value})")
    return rows


def e1_constraints() -> list[dict]:
    configs = [
        ("no_constraints", dict(tidal=False, crane_limit=None)),
        ("tidal_only", dict(tidal=True, crane_limit=None)),
        ("cranes_only", dict(tidal=False, crane_limit=2)),
        ("tidal_and_cranes", dict(tidal=True, crane_limit=2)),
    ]
    rows = []
    for label, cfg in configs:
        inst = make_hard_instance(f"e1_ablation_{label}", 8, 3, horizon=76,
                                  seed=20261014, **cfg)
        sch = solve_exact(inst, "sum_C", 30.0)
        rows.append({
            "config": label, "objective": sch.objective,
            "feasible": sch.feasible,
            "proved_optimal": bool(sch.extra.get("proved_optimal")),
            "wall_s": round(sch.wall_time, 4),
        })
        print(f"  e1 ablation {label}: {sch.objective}")
    return rows


def figure_model_size(rows: list[dict]) -> None:
    ns = [r["n"] for r in rows if r["vars"]]
    vars_ = [r["vars"] for r in rows if r["vars"]]
    cons = [r["cons"] for r in rows if r["cons"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, vars_, marker="o", label="variables")
    ax.plot(ns, cons, marker="s", label="constraints")
    ax.set_yscale("log")
    ax.set_xlabel("vessels n")
    ax.set_ylabel("count (log scale)")
    ax.set_title("Time-indexed MILP size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_model_size.png", dpi=150)
    plt.close(fig)


def figure_bounds(rows: list[dict]) -> None:
    labels = [r["instance"] for r in rows]
    opt = [r["optimal"] or 0 for r in rows]
    lp = [r["lp"] or 0 for r in rows]
    triv = [r["trivial"] or 0 for r in rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([i - 0.25 for i in x], opt, width=0.25, label="optimum")
    ax.bar(list(x), lp, width=0.25, label="LP bound")
    ax.bar([i + 0.25 for i in x], triv, width=0.25, label="trivial bound")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("sum C (periods)")
    ax.set_title("Optimum and lower bounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_bounds.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# experiment 2: classical scheduling
# ---------------------------------------------------------------------------

def e2_single_machine() -> tuple[list[dict], list[dict]]:
    from bapbench.cli.classical import brute_force_single_machine_weighted

    plain, release = [], []
    for n in (4, 5, 6, 7):
        P = make_simple_problem(f"e2_sm_{n}", n, 1, seed=17 + n, horizon=200)
        _, spt_val = single_machine_spt(P)
        _, wspt_val = single_machine_wspt(P)
        bf, _ = brute_force_single_machine(P, "sum_C")
        wbf, _ = brute_force_single_machine_weighted(P)
        plain.append({
            "n": n, "spt": spt_val, "brute": bf,
            "spt_optimal": (bf is not None and spt_val == bf),
            "wspt_wC": wspt_val, "brute_wC": wbf,
            "wspt_optimal": (wbf is not None and wspt_val == wbf),
        })
        print(f"  e2 single n={n}: spt={spt_val} brute={bf}")
    for n in (4, 5, 6):
        P = make_simple_problem(f"e2_smr_{n}", n, 1, seed=23 + n, release=True,
                                release_range=(0, 12), horizon=200)
        _, spt_val = single_machine_spt(P)
        _, wspt_val = single_machine_wspt(P)
        bf, _ = brute_force_single_machine(P, "sum_C")
        wbf, _ = brute_force_single_machine_weighted(P)
        release.append({
            "n": n, "spt": spt_val, "brute": bf,
            "spt_gap_pct": gap_pct(spt_val, bf),
            "spt_optimal": (bf is not None and spt_val == bf),
            "wspt_wC": wspt_val, "brute_wC": wbf,
            "wspt_gap_pct": gap_pct(wspt_val, wbf),
            "wspt_optimal": (wbf is not None and wspt_val == wbf),
        })
        print(f"  e2 release n={n}: spt={spt_val} brute={bf}")
    return plain, release


def e2_parallel() -> list[dict]:
    rows = []
    variants = [("plain", dict()), ("release", dict(release=True)),
                ("unavailable", dict(unavailable=True)),
                ("release+unavailable", dict(release=True, unavailable=True))]
    for label, kw in variants:
        P = make_simple_problem(f"e2_pm_{label}", 5, 3, seed=31, horizon=5 * 12 + 40, **kw)
        sch = ParallelMachineMILP(P, instance_view=to_instance_view(P)).solve()
        bf = brute_force_parallel(P, "sum_C", max_cases=400_000)
        rows.append({
            "variant": label, "milp": sch.objective, "brute": bf,
            "agree": (bf is not None and sch.objective == bf),
            "wall_s": round(sch.wall_time, 4),
        })
        print(f"  e2 parallel {label}: milp={sch.objective} brute={bf}")
    return rows


def _pfsp_makespan(proc: list[list[int]], order) -> int:
    done = [0] * len(proc[0])
    for j in order:
        t = 0
        for k, p in enumerate(proc[j]):
            t = max(t, done[k]) + p
            done[k] = t
    return max(done)


def e2_flow_shop() -> list[dict]:
    rows = []
    for n, m in ((4, 3), (5, 3)):
        P = make_simple_problem(f"e2_fs_{n}x{m}", n, m, seed=41 + n,
                                horizon=n * 12 + 40)
        sch = FlowShopMILP(P, instance_view=to_instance_view(P)).solve()
        proc = [[j.proc for _ in range(P.m)] for j in P.jobs]
        best = min(_pfsp_makespan(proc, order)
                   for order in itertools.permutations(range(P.n)))
        rows.append({
            "n": P.n, "m": P.m, "milp": sch.objective, "enumerated": best,
            "agree": sch.objective == best,
            "permutations": math.factorial(P.n),
            "wall_s": round(sch.wall_time, 4),
        })
        print(f"  e2 flow {n}x{m}: milp={sch.objective} enum={best}")
    return rows


def e2_job_shop() -> list[dict]:
    rows = []
    for n, m in ((4, 3), (5, 3), (4, 4)):
        P = make_simple_problem(f"e2_js_{n}x{m}", n, m, seed=51 + n,
                                horizon=n * 12 + 40)
        routes = {j: list(range(m)) for j in range(n)}
        sch = JobShopMILP(P, routes=routes, instance_view=to_instance_view(P)).solve()
        proc = [[job.proc] * m for job in P.jobs]
        search = job_shop_optimum([list(range(m)) for _ in range(n)], proc,
                                  time_budget=60.0)
        rows.append({
            "case": f"{n}x{m}", "milp": sch.objective,
            "branch_bound": search.objective,
            "agree": (search.feasible and sch.objective == search.objective),
            "nodes": search.nodes, "bb_wall_s": round(search.wall_time, 4),
        })
        print(f"  e2 jobshop {n}x{m}: milp={sch.objective} bb={search.objective}")

    # the instance the old non-delay rule could not solve
    routes = [[2, 1, 3, 0], [1, 3], [1, 2, 3], [1, 2, 3, 0]]
    proc = [[4, 5, 4, 7], [1, 1], [3, 1, 1], [2, 1, 2, 6]]
    search = job_shop_optimum(routes, proc, time_budget=60.0)
    rows.append({
        "case": "counterexample", "milp": None,
        "branch_bound": search.objective,
        "agree": search.objective == 20.0,
        "nodes": search.nodes, "bb_wall_s": round(search.wall_time, 4),
    })
    print(f"  e2 jobshop counterexample: bb={search.objective}")
    return rows


def figure_release_gap(rows: list[dict]) -> None:
    ns = [r["n"] for r in rows]
    gaps = [r["spt_gap_pct"] or 0 for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, gaps, marker="o")
    ax.set_xlabel("jobs n")
    ax.set_ylabel("SPT gap to optimum (%)")
    ax.set_title("1|r_j|sum C_j: SPT is not optimal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_release_gap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# experiment 3: heuristics, metaheuristics and solver configuration
# ---------------------------------------------------------------------------

def e3_rules() -> list[dict]:
    rows = []
    cases = [("e3_8x3_s2.0", 8, 2.0), ("e3_8x3_s1.8", 8, 1.8),
             ("e3_12x3_s2.0", 12, 2.0)]
    for name, n, slack in cases:
        inst = make_hard_instance(name, n, 3, seed=20261014, slack=slack)
        exact = solve_exact(inst, "sum_C", 60.0)
        optimum = exact.objective if exact.extra.get("proved_optimal") else None
        for method, sch in run_all_baselines(inst, "sum_C").items():
            rows.append({
                "instance": name, "n": n, "slack": slack, "method": method,
                "objective": sch.objective if sch.feasible else None,
                "gap_pct": (gap_pct(sch.objective, optimum)
                            if sch.feasible and optimum else None),
                "feasible": sch.feasible,
                "exact": optimum,
                "proved": bool(exact.extra.get("proved_optimal")),
            })
        print(f"  e3 rules {name}: exact={optimum}")
    return rows


def e3_multiseed() -> list[dict]:
    # 12 vessels at slack 1.7 is tight enough that the metaheuristics disagree
    # across seeds and with each other; on the looser instances every method hits
    # the optimum and the spread is zero by construction.
    inst = make_hard_instance("e3_ms_12x3", 12, 3, seed=20261014, slack=1.7)
    exact = solve_exact(inst, "sum_C", 60.0)
    optimum = exact.objective if exact.extra.get("proved_optimal") else None
    methods = ["FCFS", "LPT", "SPT", "GREEDY", "SA", "GA", "LS", "ILS", "ALNS"]
    seeds = 10
    rows = []
    for method in methods:
        vals, walls = [], []
        for seed in range(seeds):
            if method in BASELINES:
                fn = BASELINES[method]
                sch = (fn(inst, "sum_C", seed=seed) if method in ("SA", "GA")
                       else fn(inst, "sum_C"))
            else:
                sch = HEURISTICS[method](inst, "sum_C", seed)
            if sch.feasible and sch.objective is not None:
                vals.append(sch.objective)
                walls.append(sch.wall_time)
        rows.append({
            "method": method, "n_seeds": seeds, "n_valid": len(vals),
            "mean": statistics.mean(vals) if vals else None,
            "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "best": min(vals) if vals else None,
            "worst": max(vals) if vals else None,
            "gap_pct": (gap_pct(statistics.mean(vals), optimum)
                        if vals and optimum else None),
            "hit_optimal": sum(1 for v in vals if optimum and v == optimum),
            "mean_wall_s": round(statistics.mean(walls), 4) if walls else None,
            "exact": optimum,
        })
        print(f"  e3 multiseed {method}: mean={rows[-1]['mean']} sd={rows[-1]['sd']}")
    return rows


def e3_configs() -> list[dict]:
    # 20 vessels is large enough that presolve's cost is measurable; the 2 s
    # online budget is still not binding, which the table shows by every run
    # proving optimality.
    inst = make_hard_instance("e3_cfg_20x3", 20, 3, seed=20261014, slack=2.0)
    exact = solve_exact(inst, "sum_C", 60.0)
    optimum = exact.objective if exact.extra.get("proved_optimal") else None
    rows = run_config_comparison(inst, "sum_C", reference=optimum)
    for row in rows:
        row["exact"] = optimum
    print(f"  e3 configs: {[r['config'] for r in rows]}")
    return rows


def e3_robustness() -> list[dict]:
    inst = make_hard_instance("e3_tight_12x3", 12, 3, seed=20261014, slack=1.6)
    rows = []
    for method, sch in run_all_baselines(inst, "sum_C").items():
        rows.append({
            "method": method, "feasible": sch.feasible,
            "objective": sch.objective if sch.feasible else None,
            "violations": "; ".join(sch.violations[:1]),
        })
        print(f"  e3 robustness {method}: feasible={sch.feasible}")
    return rows


def figure_method_gaps(rows: list[dict]) -> None:
    instances = sorted({r["instance"] for r in rows})
    methods = ["FCFS", "LPT", "SPT", "GREEDY", "SA", "GA"]
    width = 0.8 / len(instances)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, name in enumerate(instances):
        vals = []
        for method in methods:
            hit = next(r for r in rows if r["instance"] == name and r["method"] == method)
            vals.append(hit["gap_pct"] or 0)
        ax.bar([i + k * width for i in range(len(methods))], vals, width=width,
               label=name)
    ax.set_xticks([i + width * (len(instances) - 1) / 2 for i in range(len(methods))])
    ax.set_xticklabels(methods)
    ax.set_ylabel("gap to optimum (%)")
    ax.set_title("Constructive rules against the exact optimum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_method_gaps.png", dpi=150)
    plt.close(fig)


def figure_multiseed(rows: list[dict]) -> None:
    methods = [r["method"] for r in rows]
    means = [r["mean"] or 0 for r in rows]
    sds = [r["sd"] or 0 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(methods, means, yerr=sds, fmt="o", capsize=4)
    ax.set_ylabel("mean objective (periods)")
    ax.set_title("Ten seeds per method, mean and standard deviation")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_multiseed.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# experiment 4: dynamics, rolling horizon and disruptions
# ---------------------------------------------------------------------------

def e4_events() -> list[dict]:
    from bapbench.dynamics import Event

    planned = make_hard_instance("e4_base", 8, 3, seed=20261014, slack=2.0)
    base = solve_exact(planned, "sum_C", 30.0)
    cases = [
        ("arrival_delay", [Event("arrival_delay", 0, v, 5, "arrival +5")
                           for v in (0, 2, 4)]),
        ("handle_overrun", [Event("handle_overrun", 0, v, 3, "handling +3")
                            for v in (1, 3)]),
        ("tidal_shift", [Event("tidal_shift", 0, 1, 4, "low tide +4")]),
        ("crane_loss", [Event("crane_loss", 10, None, 10, "one crane lost")]),
    ]
    rows = []
    for name, events in cases:
        disrupted = apply_events(planned, events)
        sch = solve_exact(disrupted, "sum_C", 30.0)
        rows.append({
            "case": name, "n_events": len(events),
            "objective": sch.objective,
            "delta_pct": gap_pct(sch.objective, base.objective),
            "feasible": sch.feasible,
            "horizon": disrupted.horizon,
        })
        print(f"  e4 event {name}: {sch.objective} vs {base.objective}")
    rows.insert(0, {"case": "undisrupted", "n_events": 0,
                    "objective": base.objective, "delta_pct": 0.0,
                    "feasible": base.feasible, "horizon": planned.horizon})
    return rows


def e4_rolling() -> tuple[list[dict], list[dict]]:
    planned = make_hard_instance("e4_roll_base", 8, 3, seed=20261014, slack=2.0)
    by_events, by_dt = [], []
    for n_events in (2, 4, 6):
        for seed in range(5):
            r = run_rolling(planned, n_events, seed, 8, 5.0)
            by_events.append({"n_events": n_events, "seed": seed, **r})
        got = [r["regret_pct"] for r in by_events if r["n_events"] == n_events
               and r["regret_pct"] is not None]
        print(f"  e4 rolling events={n_events}: regrets={got}")
    for dt in (4, 8, 16):
        for seed in range(5):
            r = run_rolling(planned, 6, seed, dt, 5.0)
            by_dt.append({"dt": dt, "seed": seed, **r})
        got = [r["regret_pct"] for r in by_dt if r["dt"] == dt
               and r["regret_pct"] is not None]
        print(f"  e4 rolling dt={dt}: regrets={got}")
    return by_events, by_dt


def figure_events(rows: list[dict]) -> None:
    names = [r["case"] for r in rows]
    values = [r["objective"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values)
    ax.set_ylabel("objective sum C")
    ax.set_title("Disruption impact on the optimum")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_events.png", dpi=150)
    plt.close(fig)


def figure_rolling(by_events: list[dict], by_dt: list[dict]) -> None:
    """Regret per seed; the black bar is the mean. Many runs recover the optimum,
    so a mean with a symmetric error bar would sit below zero -- plotting the
    seeds shows that the zeros are real rather than an artefact of averaging."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    panels = [
        (axes[0], by_events, "n_events",
         sorted({r["n_events"] for r in by_events}),
         "Regret by event count (5 seeds, dt=8)",
         "number of disruption events"),
        (axes[1], by_dt, "dt", sorted({r["dt"] for r in by_dt}),
         "Regret by replanning interval (5 seeds, 6 events)",
         "replanning interval dt (periods)"),
    ]
    for ax, rows, key, xs, title, xlabel in panels:
        for x in xs:
            vals = [r["regret_pct"] for r in rows
                    if r[key] == x and r["regret_pct"] is not None]
            if not vals:
                continue
            offsets = [(-0.08 + 0.16 * i / max(1, len(vals) - 1))
                       for i in range(len(vals))]
            ax.scatter([x + o for o in offsets], vals, s=16, alpha=0.65)
            ax.plot([x - 0.12, x + 0.12], [sum(vals) / len(vals)] * 2,
                    color="black", linewidth=2)
        ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("regret (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_rolling.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()

    print("[E1] MILP, LP and the basic model")
    scale = e1_model_scale()
    formulations, enumerations = e1_formulations_and_enumeration()
    bounds = e1_bounds()
    objectives = e1_objectives()
    constraints = e1_constraints()
    write_csv("e1_scale.csv", scale)
    write_csv("e1_formulations.csv", formulations)
    write_csv("e1_enumeration.csv", enumerations)
    write_csv("e1_bounds.csv", bounds)
    write_csv("e1_objectives.csv", objectives)
    write_csv("e1_constraints.csv", constraints)
    figure_model_size(scale)
    figure_bounds(bounds)

    print("[E2] classical scheduling")
    plain, release = e2_single_machine()
    parallel = e2_parallel()
    flow = e2_flow_shop()
    jobshop = e2_job_shop()
    write_csv("e2_single.csv", plain)
    write_csv("e2_release.csv", release)
    write_csv("e2_parallel.csv", parallel)
    write_csv("e2_flow.csv", flow)
    write_csv("e2_jobshop.csv", jobshop)
    figure_release_gap(release)

    print("[E3] heuristics, metaheuristics and solver configuration")
    rules = e3_rules()
    multiseed = e3_multiseed()
    configs = e3_configs()
    robust = e3_robustness()
    write_csv("e3_rules.csv", rules)
    write_csv("e3_multiseed.csv", multiseed)
    write_csv("e3_configs.csv", configs)
    write_csv("e3_robustness.csv", robust)
    figure_method_gaps(rules)
    figure_multiseed(multiseed)

    print("[E4] dynamics, rolling horizon and disruptions")
    events = e4_events()
    by_events, by_dt = e4_rolling()
    write_csv("e4_events.csv", events)
    write_csv("e4_rolling_events.csv", by_events)
    write_csv("e4_rolling_dt.csv", by_dt)
    figure_events(events)
    figure_rolling(by_events, by_dt)

    write_json("run_summary.json", {
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "note": "every table in reports/ comes from this run",
    })
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
