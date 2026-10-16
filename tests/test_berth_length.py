"""The models, the checkers and the enumerator must agree on quay length.

A vessel occupies its whole berth, so it can only berth where its length fits.
That condition is enforced by `Instance.validate`, `Schedule.check` and
`metrics.sanity_check_schedule`, which means the exact models and the solver-free
enumerator have to enforce it as well. When they do not, a model reports an
optimum the package's own check rejects, and that value becomes the reference
every subsequent gap is measured against.
"""

from __future__ import annotations

import pytest

from bapbench.instances import Berth, Instance, Vessel
from bapbench.metrics import sanity_check_schedule
from bapbench.models.bap import BAPDisjunctive, BAPTimeIndexed
from bapbench.verification.exhaustive import exhaustive_optimum


def _heterogeneous() -> Instance:
    """Two berths of different lengths and a vessel that fits only the longer."""
    return Instance(
        name="hetero", horizon=15,
        vessels=[Vessel(vid=0, name="V1", length=150, arrive=2, handle=3),
                 Vessel(vid=1, name="V2", length=200, arrive=1, handle=1),
                 Vessel(vid=2, name="V3", length=200, arrive=0, handle=3)],
        berths=[Berth(bid=0, name="B1", length=600),
                Berth(bid=1, name="B2", length=150)])


@pytest.mark.parametrize("model", [BAPTimeIndexed, BAPDisjunctive])
def test_a_reported_optimum_passes_the_packages_own_checks(model):
    """The model may not claim optimality for a schedule `check` rejects."""
    inst = _heterogeneous()
    sch = model(inst, objective="sum_C", time_limit=30.0).solve()
    assert sch.extra.get("proved_optimal"), "test needs a proved optimum"
    assert sch.feasible, sch.violations
    assert sch.check(inst).feasible
    assert sanity_check_schedule(inst, sch) == []


@pytest.mark.parametrize("objective", ["sum_C", "C_max"])
def test_the_model_matches_the_enumerator_when_berths_differ(objective):
    """Two independent paths must agree once both enforce the length condition."""
    inst = _heterogeneous()
    sch = BAPTimeIndexed(inst, objective=objective, time_limit=30.0).solve()
    exh = exhaustive_optimum(inst, objective, max_cases=4_000_000,
                             time_budget=60.0)
    assert exh.feasible, exh.note
    assert sch.objective == pytest.approx(exh.objective)


def test_a_vessel_longer_than_every_berth_leaves_no_schedule():
    """With no berth long enough the instance is infeasible, not silently solved."""
    inst = Instance(name="too_long", horizon=20,
                    vessels=[Vessel(vid=0, name="V1", length=400, arrive=0,
                                    handle=4)],
                    berths=[Berth(bid=0, name="B1", length=300)])
    sch = BAPTimeIndexed(inst, objective="sum_C", time_limit=20.0).solve()
    assert not sch.feasible


def test_the_length_condition_does_not_change_equal_berth_instances():
    """Every shipped generator uses equal berths, where the condition never bites."""
    from bapbench.instances import make_toy_instance
    inst = make_toy_instance("equal", 6, 3, seed=7)
    sch = BAPTimeIndexed(inst, objective="sum_C", time_limit=30.0).solve()
    assert sch.feasible and sch.extra.get("proved_optimal")
    exh = exhaustive_optimum(inst, "sum_C", max_cases=8_000_000,
                             time_budget=120.0)
    assert exh.feasible
    assert sch.objective == pytest.approx(exh.objective)
