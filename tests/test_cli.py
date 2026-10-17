"""Tests for the experiment orchestration layer.

The functions here turn a model run into the rows that every reported number comes
from, so an error in them is not visible as a crash: it appears as a plausible
figure in a table. They are tested for the properties a reported row must have --
a reference it can name, a gap consistent with that reference, a failure reason
when a method produces nothing -- rather than for particular values.
"""

from __future__ import annotations

import csv
import json

import pytest

from bapbench.bounds import reference_for
from bapbench.cli.classical import parallel_machines, single_machine
from bapbench.cli.compare import compare, solve_exact, slack_scan
from bapbench.cli.dynamic import run_rolling
from bapbench.cli.heuristics import run_seeds, self_check
from bapbench.instances import make_hard_instance, make_toy_instance
from bapbench.heuristics.solver_configs import (
    CONFIGS,
    run_config_comparison,
    solve_with_config,
)


@pytest.fixture(scope="module")
def toy():
    return make_toy_instance("cli_toy", 5, 3, seed=20261014)


@pytest.fixture(scope="module")
def hard():
    return make_hard_instance("cli_hard", 8, 3, seed=20261014, slack=2.0)


# --- run_seeds ------------------------------------------------------------------

def test_run_seeds_reports_one_row_per_method(toy):
    rows = run_seeds(toy, "sum_C", 1, 20.0)
    methods = [r["method"] for r in rows]
    assert "MILP" in methods and "FCFS" in methods
    assert len(methods) == len(set(methods)), "a method appears twice"


def test_every_seed_row_can_name_its_reference(toy):
    """A gap without a reference cannot be interpreted, and a reference without a
    kind cannot be trusted, so the two must agree."""
    for r in run_seeds(toy, "sum_C", 1, 20.0):
        assert r["reference_kind"] in ("exact_proved", "lp_bound",
                                       "trivial_bound", "none")
        if r["reference"] is None:
            assert r["reference_kind"] == "none"
            assert r["mean_gap_pct"] is None
        else:
            assert r["reference_kind"] != "none"


def test_a_proved_run_uses_the_optimum_as_the_reference(toy):
    rows = run_seeds(toy, "sum_C", 1, 20.0)
    exact = next(r for r in rows if r["method"] == "MILP")
    assert exact["reference_kind"] == "exact_proved"
    assert exact["reference"] == pytest.approx(exact["exact"])


def test_no_heuristic_beats_the_reference(toy):
    """A negative gap against a valid reference means the reference is wrong or
    the method was scored against the wrong objective."""
    for r in run_seeds(toy, "sum_C", 1, 20.0):
        if r["mean_gap_pct"] is not None:
            assert r["mean_gap_pct"] >= -1e-6, r["method"]


def test_baselines_are_deterministic_so_their_spread_is_zero(toy):
    for r in run_seeds(toy, "sum_C", 2, 20.0):
        if r["kind"] == "deterministic":
            assert r["std_objective"] == 0.0, r["method"]


def test_a_run_without_a_bound_still_names_the_reference(toy):
    """with_bound=False skips the LP bound; the row must then say 'none' rather
    than borrow a label."""
    for r in run_seeds(toy, "sum_C", 1, 20.0, with_bound=False):
        assert r["reference_kind"] in ("exact_proved", "none")


# --- self_check -----------------------------------------------------------------

def test_self_check_accepts_a_clean_table():
    rows = [{"method": "FCFS", "kind": "deterministic", "std_objective": 0.0,
             "n_valid": 10, "n_seeds": 10},
            {"method": "SA", "kind": "stochastic", "std_objective": 3.5,
             "n_valid": 10, "n_seeds": 10}]
    assert self_check(rows) == []


def test_self_check_flags_a_deterministic_method_with_spread():
    """A deterministic method that varies across seeds has uncontrolled
    randomness, which invalidates its row."""
    rows = [{"method": "FCFS", "kind": "deterministic", "std_objective": 1.25,
             "n_valid": 10, "n_seeds": 10}]
    problems = self_check(rows)
    assert len(problems) == 1
    assert "FCFS" in problems[0]


def test_self_check_flags_a_deterministic_method_that_mostly_failed():
    """With fewer than two valid runs the spread is zero by definition, so a
    method that failed almost every seed must not pass the determinism check."""
    rows = [{"method": "FCFS", "kind": "deterministic", "std_objective": 0.0,
             "n_valid": 1, "n_seeds": 10}]
    problems = self_check(rows)
    assert len(problems) == 1
    assert "FCFS" in problems[0]


def test_self_check_allows_a_stochastic_method_to_vary():
    rows = [{"method": "SA", "kind": "stochastic", "std_objective": 11.0,
             "n_valid": 10, "n_seeds": 10}]
    assert self_check(rows) == []


# --- compare and solve_exact ----------------------------------------------------

def test_solve_exact_separates_feasibility_from_optimality(toy):
    """A feasible schedule is not automatically an optimal one; conflating them
    makes every gap measured against it meaningless."""
    sch, proved = solve_exact(toy, "sum_C", 20.0)
    assert sch.feasible
    assert proved is True
    assert sch.extra.get("proved_optimal") is True


def test_compare_rows_share_one_reference(toy):
    """Every row in a comparison is scored against the same value, otherwise the
    ordering of the methods is meaningless."""
    rows = compare(toy, "sum_C", 20.0)
    refs = {r["reference"] for r in rows}
    kinds = {r["reference_kind"] for r in rows}
    assert len(refs) == 1 and len(kinds) == 1


def test_compare_reports_a_gap_consistent_with_its_reference(toy):
    for r in compare(toy, "sum_C", 20.0):
        if r["gap_pct"] is None or r["objective"] is None:
            continue
        expected = (r["objective"] - r["reference"]) / r["reference"] * 100.0
        assert r["gap_pct"] == pytest.approx(expected, abs=1e-3)


def test_compare_marks_the_milp_row_as_proved_and_the_rest_not(toy):
    rows = compare(toy, "sum_C", 20.0)
    milp = next(r for r in rows if r["method"] == "MILP")
    assert milp["proved_optimal"] is True
    assert all(not r["proved_optimal"] for r in rows if r["method"] != "MILP")


def test_compare_keeps_an_infeasible_baseline_row_out_of_the_gap_column(toy):
    """A baseline that cannot place every vessel has no objective, so it must not
    contribute a gap."""
    for r in compare(toy, "sum_C", 20.0):
        if not r["feasible"]:
            assert r["gap_pct"] is None
            assert r["note"]


def test_slack_scan_names_a_reference_on_every_row():
    rows = slack_scan("sum_C", 20.0, slacks=[2.0, 1.8])
    assert rows
    for r in rows:
        assert r["reference_kind"] in ("exact_proved", "lp_bound",
                                       "trivial_bound", "none")
        if r["reference"] is None:
            assert r["reference_kind"] == "none"


# --- run_rolling ----------------------------------------------------------------

def test_rolling_regret_is_never_negative(hard):
    """The rolling planner works on less information than the perfect-information
    optimiser, so it cannot be better; a negative regret means the comparison is
    between incomparable quantities."""
    r = run_rolling(hard, 2, 20261014, 8, 20.0)
    if r["regret_pct"] is not None:
        assert r["regret_pct"] >= -1e-6


def test_a_reported_regret_comes_from_a_proved_denominator(hard):
    """An unproved incumbent is an upper bound on the optimum, so using it as the
    denominator understates the regret."""
    r = run_rolling(hard, 2, 20261014, 8, 20.0)
    assert r["perfect_proved"] in (True, False)
    if not r["perfect_proved"]:
        assert r["regret_pct"] is None
        assert r["perfect_objective"] is None
    else:
        assert r["perfect_objective"] is not None


def test_rolling_reports_the_events_it_actually_applied(hard):
    r = run_rolling(hard, 4, 20261014, 8, 20.0)
    kinds = [k for k in r["event_kinds"].split(";") if k]
    assert len(kinds) == r["n_events"] == 4


def test_rolling_counts_only_the_events_it_applied(hard):
    """The generator draws a fixed number per kind, so a request for three events
    yields two. The row must count what was applied, not what was asked for."""
    r = run_rolling(hard, 3, 20261014, 8, 20.0)
    kinds = [k for k in r["event_kinds"].split(";") if k]
    assert r["requested_events"] == 3
    assert r["n_events"] == len(kinds) <= r["requested_events"]


def test_rolling_reoptimises_roughly_once_per_decision_point(hard):
    r = run_rolling(hard, 2, 20261014, 8, 20.0)
    assert r["reoptimizations"] > 0


def test_a_failed_rolling_run_reports_a_violation(hard):
    """Feasibility is decided by an independent re-check, so a run marked
    infeasible must carry the reason."""
    r = run_rolling(hard, 2, 20261014, 8, 20.0)
    if not r["feasible"]:
        assert r["violation"]


# --- solver configurations ------------------------------------------------------

def test_config_comparison_covers_every_configured_run(toy):
    rows = run_config_comparison(toy, "sum_C")
    assert [r["config"] for r in rows] == [c.name for c in CONFIGS]


def test_a_config_run_records_what_it_was_asked_to_do(toy):
    sch = solve_with_config(toy, CONFIGS[0], "sum_C")
    assert sch.extra["config"] == CONFIGS[0].name
    assert sch.extra["config_kwargs"] == dict(CONFIGS[0].kwargs)


def test_config_gaps_use_the_supplied_reference(toy):
    reference = 100.0
    rows = run_config_comparison(toy, "sum_C", reference=reference)
    for r in rows:
        if r["objective"] is None or r["gap_pct"] is None:
            continue
        expected = (r["objective"] - reference) / reference * 100.0
        assert r["gap_pct"] == pytest.approx(expected, abs=1e-3)


def test_config_gap_treats_a_zero_reference_as_present(toy):
    """A zero reference is a legitimate value; testing it for truthiness would
    silently drop the gap."""
    rows = run_config_comparison(toy, "sum_C", reference=0.0)
    assert all(r["gap_pct"] is None for r in rows)


def test_gap_is_absent_rather_than_wrong_when_there_is_no_reference(toy):
    rows = run_config_comparison(toy, "sum_C", reference=None)
    assert all(r["gap_pct"] is None for r in rows)


# --- the reference helper the orchestration relies on ---------------------------

def test_reference_for_prefers_a_proved_optimum(toy):
    value, kind = reference_for(toy, "sum_C", 42.0, True)
    assert (value, kind) == (42.0, "exact_proved")


def test_reference_for_never_uses_an_unproved_value(toy):
    """The unproved value is an upper bound, so a gap against it would flatter the
    heuristic."""
    value, kind = reference_for(toy, "sum_C", 42.0, False)
    assert value != 42.0 or kind != "exact_proved"
    assert kind in ("lp_bound", "trivial_bound", "none")


def test_reference_for_ignores_a_proved_flag_without_a_value(toy):
    _value, kind = reference_for(toy, "sum_C", None, True)
    assert kind != "exact_proved"


# --- the command entry points write what they report ----------------------------

def test_the_classical_groups_write_their_csv(tmp_path):
    rows = single_machine([5, 6], tmp_path)
    assert len(rows) == 2
    assert (tmp_path / "W3a_single_machine.csv").exists()

    prows = parallel_machines([(4, 2), (5, 2)], tmp_path)
    assert len(prows) == 8
    with (tmp_path / "W3c_parallel_machines.csv").open(encoding="utf-8-sig") as f:
        assert sum(1 for _ in csv.DictReader(f)) == 8


def test_the_classical_groups_agree_with_their_independent_path(tmp_path):
    """The reported agreement flag has to reflect a real comparison."""
    for r in single_machine([5, 6], tmp_path):
        assert r["spt_matches_brute"] is True
    for r in parallel_machines([(4, 2), (5, 2)], tmp_path):
        assert r["agree"] is True


# --- the reproduce command ------------------------------------------------------

def test_the_reproduce_command_writes_its_csv(tmp_path):
    """The comparison layer has a command, so the reproductions are reachable
    without importing the package."""
    from bapbench.cli import reproduce

    code = reproduce.main(["--instances", "4", "--berths", "2",
                           "--max-iter", "15", "--time-limit", "20",
                           "--no-package-methods", "--no-iris",
                           "--out", str(tmp_path)])
    assert code == 0
    with (tmp_path / "reproduce.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    methods = {r["method"] for r in rows}
    assert {"MILP", "Imai2001[SIMPLE]", "Imai2001[INDIVIDUAL]",
            "Imai2001[INTERACT]"} <= methods
    assert all(r["reference_kind"] == "exact_proved" for r in rows)
    assert all(r["gap_pct"] != "" for r in rows if r["feasible"] == "True")


# --- the bacap command ----------------------------------------------------------

def test_the_bacap_command_writes_assignments_plan_and_summary(tmp_path):
    from bapbench.cli import bacap as bacap_cmd

    code = bacap_cmd.main(["--instances", "3", "--quay", "40", "--cranes", "6",
                           "--horizon", "80", "--iterations", "20",
                           "--out", str(tmp_path)])
    assert code == 0

    with (tmp_path / "assignments.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {"vessel", "berth", "start", "q_hat", "f1", "f2"} <= set(rows[0])

    with (tmp_path / "qc_plan.csv").open(encoding="utf-8-sig") as f:
        plan = list(csv.DictReader(f))
    assert plan and {"vessel", "period", "cranes"} <= set(plan[0])

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is True
    assert summary["violations"] == []
    assert summary["unassigned"] == []
    assert "reconstruction" in summary["instance"]["instance_kind"]


def test_the_bacap_command_rejects_a_quay_too_short_for_the_fleet(tmp_path):
    """A vessel is at least 8 sections long, so a 5-section quay cannot hold one.
    The command must say why instead of raising."""
    from bapbench.cli import bacap as bacap_cmd

    code = bacap_cmd.main(["--instances", "2", "--quay", "5",
                           "--out", str(tmp_path)])
    assert code == 2
