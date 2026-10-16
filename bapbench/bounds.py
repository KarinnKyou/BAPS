"""Lower bounds for the berth allocation problem.

References for the gap columns. Where the exact model proves an optimum, that
value is the reference. Where it does not, a lower bound still makes the result
reportable: "within X% of a valid bound" is a claim that holds, whereas a gap
against an unproved value does not.

Two bounds are provided:

* ``trivial`` - every vessel finishes no earlier than its arrival plus its
  handling time, so the sum over vessels is a bound on ``sum C_i``. Free, and
  valid for every objective as an arrival-side bound.
* ``lp`` - the optimum of the LP relaxation of the time-indexed formulation.
  Dominates the trivial bound and is exact on many instances, at the cost of one
  LP solve.
"""

from __future__ import annotations

from dataclasses import dataclass

from .instances import Instance


@dataclass
class LowerBound:
    """A valid lower bound on the optimal objective value."""

    value: float | None
    kind: str                  # "trivial" | "lp" | "none"
    detail: str = ""
    wall_time: float = 0.0

    def __float__(self) -> float:
        if self.value is None:
            raise ValueError(f"no bound available ({self.detail})")
        return float(self.value)


def trivial_lower_bound(inst: Instance, objective: str = "sum_C") -> LowerBound:
    """Arrival-side bound, in the units of the requested objective.

    Every vessel is handled after it arrives, so its completion time is at least
    ``arrival_i + handling_i``. Summing those minima bounds ``sum C_i``; taking
    their maximum bounds ``C_max``. Using the sum for ``C_max`` is not a bound at
    all -- it can exceed the optimum by a wide margin and turn every gap
    negative.
    """
    if inst.n == 0:
        return LowerBound(0.0, "trivial", "empty instance")
    earliest = [v.arrive + v.handle for v in inst.vessels]
    if objective == "sum_C":
        return LowerBound(float(sum(earliest)), "trivial",
                          "sum_i (arrival_i + handling_i)")
    if objective == "C_max":
        return LowerBound(float(max(earliest)), "trivial",
                          "max_i (arrival_i + handling_i)")
    raise ValueError(f"unknown objective: {objective}")


def lp_lower_bound(inst: Instance, objective: str = "sum_C",
                   time_limit: float | None = None) -> LowerBound:
    """Optimum of the LP relaxation of the time-indexed formulation.

    Relaxing integrality enlarges the feasible set, so the LP optimum can only be
    below the integer optimum.
    """
    import time

    from .models.bap import BAPTimeIndexed

    t0 = time.perf_counter()
    model = BAPTimeIndexed(inst, objective=objective, relax=True,
                           time_limit=time_limit)
    try:
        sch = model.solve()
    except RuntimeError as exc:
        return LowerBound(None, "none", f"LP not built: {exc}",
                          time.perf_counter() - t0)
    wall = time.perf_counter() - t0
    if not sch.extra.get("proved_optimal") or sch.objective is None:
        return LowerBound(None, "none",
                          f"LP not solved to optimality ({sch.extra.get('status')})",
                          wall)
    return LowerBound(float(sch.objective), "lp",
                      "optimum of the LP relaxation", wall)


def best_lower_bound(inst: Instance, objective: str = "sum_C",
                     time_limit: float | None = None,
                     with_lp: bool = True) -> LowerBound:
    """The strongest bound that could be computed, preferring the LP."""
    trivial = trivial_lower_bound(inst, objective)
    if not with_lp:
        return trivial
    lp = lp_lower_bound(inst, objective, time_limit)
    if lp.value is None:
        return trivial
    # Both are bounds on the same objective, so the larger is the stronger.
    if trivial.value is None or lp.value >= trivial.value:
        return lp
    return trivial


def reference_for(inst: Instance, objective: str = "sum_C",
                  exact_value: float | None = None,
                  exact_proved: bool = False,
                  time_limit: float | None = 30.0) -> tuple[float | None, str]:
    """Pick the reference value for a gap and name what it is.

    Returns ``(value, kind)`` where kind is one of ``exact_proved``,
    ``lp_bound`` or ``trivial_bound``. An unproved exact value is never used: a
    time-limited objective is an upper bound on the optimum, so a gap against it
    would overstate the heuristic's quality.
    """
    if exact_proved and exact_value is not None:
        return float(exact_value), "exact_proved"
    bound = best_lower_bound(inst, objective, time_limit=time_limit)
    if bound.value is None:
        return None, "none"
    return float(bound.value), f"{bound.kind}_bound"
