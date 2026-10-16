"""Regression test for the presolve defect in the disjunctive formulation.

With HiGHS 1.15.1's default presolve the disjunctive model could report a
suboptimal objective while claiming status Optimal and a schedule its own check()
accepted. The model now disables presolve; this test pins the instance where the
disagreement appeared, against an exhaustive optimum so it cannot be satisfied by
two models sharing the same mistake.
"""

from __future__ import annotations

import pytest

from bapbench.instances import Berth, Instance, Vessel
from bapbench.models.bap import BAPDisjunctive, BAPTimeIndexed
from bapbench.verification.exhaustive import exhaustive_optimum


@pytest.fixture(scope="module")
def crane_limited():
    """4 vessels, 3 berths, a terminal-wide crane limit of 1 and phased tides.

    The crane limit is what makes presolve matter: it couples every vessel, and
    that is the family where the disagreement was reproduced.
    """
    return Instance(
        name="dj_presolve",
        vessels=[Vessel(vid=0, name="V1", length=100, arrive=4, handle=2),
                 Vessel(vid=1, name="V2", length=100, arrive=1, handle=8),
                 Vessel(vid=2, name="V3", length=100, arrive=2, handle=3),
                 Vessel(vid=3, name="V4", length=100, arrive=2, handle=4)],
        berths=[Berth(bid=k, name=f"B{k+1}", length=200) for k in range(3)],
        horizon=20,
        tidal_windows={"0": [(0, 3), (11, 14)], "1": [(4, 7), (15, 18)],
                       "2": [(8, 11), (19, 20)]},
        quay_crane_limit=1,
    )


def test_disjunctive_agrees_with_the_enumerated_optimum(crane_limited):
    enumerated = exhaustive_optimum(crane_limited, "sum_C", time_budget=60.0)
    assert enumerated.objective is not None, enumerated.note

    dj = BAPDisjunctive(crane_limited, objective="sum_C", time_limit=60.0).solve()
    assert dj.feasible, dj.violations
    assert dj.objective == pytest.approx(enumerated.objective)


def test_both_formulations_agree_on_that_instance(crane_limited):
    ti = BAPTimeIndexed(crane_limited, objective="sum_C", time_limit=60.0).solve()
    dj = BAPDisjunctive(crane_limited, objective="sum_C", time_limit=60.0).solve()
    assert ti.feasible and dj.feasible
    assert ti.objective == pytest.approx(dj.objective)


def test_presolve_is_disabled_by_default_on_the_disjunctive_model(crane_limited):
    """The guard that keeps the defect from returning silently."""
    model = BAPDisjunctive(crane_limited, objective="sum_C")
    assert model.solver_kwargs == {}
    assert model.solve().feasible
    # presolve is forced off inside solve(); a caller cannot re-enable it by
    # leaving solver_kwargs empty
    assert "presolve" not in model.solver_kwargs
