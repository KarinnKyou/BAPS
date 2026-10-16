"""M4 regression: a limit with no solution must not yield an assignment."""

from __future__ import annotations

from bapbench.instances import make_hard_instance
from bapbench.models.bap import BAPDisjunctive, BAPTimeIndexed


def test_disjunctive_reports_no_schedule_when_the_limit_gives_no_solution():
    """HiGHS leaves column values behind when it stops. Reading them produced an
    assignment that was infeasible everywhere while carrying a finite objective."""
    inst = make_hard_instance("m4", 10, 3, seed=20261014, slack=1.6)
    sch = BAPDisjunctive(inst, objective="sum_C", time_limit=0.3).solve()
    ai = BAPTimeIndexed(inst, objective="sum_C", time_limit=0.3).solve()
    # whichever solver path this machine takes, the contract is the same: either a
    # feasible schedule, or no objective at all
    for name, s in (("disjunctive", sch), ("time-indexed", ai)):
        if s.feasible:
            assert s.assign and s.objective is not None, name
        else:
            assert s.objective is None, f"{name} reported {s.objective} as infeasible"
            assert not s.assign, f"{name} returned an assignment it calls infeasible"


def test_no_solution_flag_is_recorded():
    inst = make_hard_instance("m4b", 10, 3, seed=20261014, slack=1.6)
    sch = BAPDisjunctive(inst, objective="sum_C", time_limit=0.3).solve()
    if not sch.feasible:
        assert sch.extra.get("has_solution") is False
        assert "no solution" in "; ".join(sch.violations) or sch.objective is None
