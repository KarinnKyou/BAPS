# -*- coding: utf-8 -*-
"""Adaptive Large Neighborhood Search of Iris, Pacino & Ropke (2017).

The module re-implements the BACAP ALNS of

    C. Iris, D. Pacino, S. Ropke, "Improved formulations and an Adaptive Large
    Neighborhood Search heuristic for the integrated berth allocation and quay
    crane assignment problem", Transportation Research Part E 105 (2017) 123-147

on the instance and objective layer of this package, so its schedules are
verified by the same independent checks as every other method here.

What is reproduced (from the prose of Sections 5 and 6.1.1)
-----------------------------------------------------------
* The four destroy operators: Shaw removal with the three-term relatedness
  measure ``a|b_i - b_j| + b|s_i - s_j| + c|e_i - e_j|``, cost and
  time-relatedness removal, cost and berth-relatedness removal, and random
  removal with its time-relatedness test.
* The two insertion operators: basic greedy insertion, which takes the first
  feasible candidate in increasing cost order under a determinism factor ``p``,
  and the smarter insertion, which reevaluates the candidate instead of skipping
  it and decides between all feasible candidates. Both are offered in the
  deterministic (``p = 1``) and the stochastic (``p`` drawn from [0.25, 1.0])
  variant, which is the pair of versions the paper describes for each.
* The adaptive weight mechanism: per-operator weights and scores, the three
  score tiers ``rho1/rho2/rho3`` for "new global best" / "improves the current
  objective" / "accepted but not improving", the reaction factor ``gamma``, the
  reset of all scores every ``d`` iterations, and roulette-wheel selection.
* The acceptance criterion: simulated annealing with the temperature cooled by a
  geometric factor derived from ``T_start = u%`` and ``T_end = n%`` of the
  initial objective over ``e`` iterations, reheated to ``T_start`` every ``e``
  iterations, and termination after ``c_max`` iterations.
* The greedy construction heuristic of Section 5.1: a random vessel order, each
  vessel's candidate assignments tried in increasing cost order, first feasible
  one placed.
* The tuned parameter vector of Section 6.1.1,
  ``(a, b, c, phi_r, u, n, e, rho1, rho2, rho3, gamma, d) =
  (0.01, 2.0, 0.01, 4, 0.05, 0.0005, 30000, 2, 5, 10, 0.8, 1000)``, the
  alternative ``phi_r = 6`` of Table 6, and ``c_max = 270000``.

What could not be reproduced
----------------------------
* The four algorithm listings (Algorithms 1-4) are images in the published PDF
  and were not extracted with the text. The operators and the search loop here
  follow the surrounding prose, which specifies them step by step; no
  pseudocode step was invented and then reported as the paper's.
* The QC assignment layer. The paper's solution carries a detailed per-period QC
  plan, and its two insertion operators differ exactly in how that plan is built
  (Appendix A's ``q_hat, f1, f2`` and the "as much as possible" plan of Section
  5.3.1). This package models quay cranes as a limit on simultaneously handled
  vessels, with no per-vessel minimum/maximum crane counts and no per-period
  plan, so there is no QC plan to construct or to repair. The insertion
  operators are therefore separated by their candidate decision rule, which is
  representable and checkable here. Both enforce the package's crane limit, so
  every schedule returned is feasible under it.
* The paper's objective. ``min sum_i (c1_i dETA_i + c2_i dEFT_i + c3_i u_i +
  c4 sum_t sum_q q r_itq)`` is minimised here as the package's objective for the
  same vessel set (``sum_C`` or ``C_max``), because the paper's cost
  coefficients, desired berthing positions, ETAs, EFTs and LFTs belong to the
  Meisel & Bierwirth (2009) instances. `vessel_slot_cost` states the
  correspondence used for the destroy operators' cost ordering.
* Any comparison with the published tables. The paper's primary benchmark is the
  30-instance set of Meisel & Bierwirth (2009), which is not publicly available,
  and its objective and constraints differ from this package's as described
  above. `make_meisel_bierwirth_instance` generates instances with the published
  characteristics (vessel counts 20/30/40, 1000 m quay, 10 m discretisation, 10
  quay cranes, 168 h horizon, interference ``alpha = 0.9``, berthing-deviation
  ``beta = 0.01``), but a direct comparison against the paper's table values is
  not possible.

Solution representation
-----------------------
The package's existing metaheuristics search permutations of vessels and decode
each permutation with a shared placement routine, so their neighbourhood is
"reorder the whole fleet" and cannot express "remove these eight vessels and
leave every other vessel exactly where it is". This module therefore represents a
solution as the assignment itself, ``{vid: (berth, start)}``, and maintains the
derived berth occupancy and crane load incrementally, so a destroy operator can
remove an arbitrary subset and the insertion operators put back only those
vessels while the rest of the plan is untouched. Every solution returned is
re-checked with :func:`bapbench.metrics.sanity_check_schedule` and re-scored from
the assignment alone.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence

from ..instances import Berth, Instance, Schedule, Vessel
from ..metrics import objective_value, sanity_check_schedule

# Parameter vector of Section 6.1.1, in the order the paper lists it:
# (a, b, c, phi_r, u, n, e, rho1, rho2, rho3, gamma, d).
PAPER_PARAMETERS = (0.01, 2.0, 0.01, 4, 0.05, 0.0005, 30000, 2, 5, 10, 0.8, 1000)

# Section 6.1.1 reports that phi_r = 6 can replace 4 at almost no cost in
# solution quality and a large saving in runtime (Table 6).
PAPER_PHI_RUNTIME_ALTERNATIVE = 6

# Section 6.1: the heuristic terminates after this many iterations per run.
PAPER_MAX_ITERATIONS = 270000

# Section 5.2 and Algorithm 2: the destroy determinism parameter is drawn from
# this range, and is pushed to 1.0 when the destroy list still needs entries.
DESTROY_P_MIN, DESTROY_P_MAX = 0.6, 1.0
# Section 5.3: the stochastic insertion draws its determinism factor here.
INSERTION_P_MIN, INSERTION_P_MAX = 0.25, 1.0


# ---------------------------------------------------------------------------
# Instance generation with the published characteristics
# ---------------------------------------------------------------------------

# Meisel & Bierwirth (2009) use feeder, medium and jumbo vessels that differ in
# technical properties. The benchmark itself is not public, so these are the
# three size classes with a length range in 10 m sections, a handling-time range
# in hours, and the range of cranes a vessel of that class works with. The QC
# demand of each vessel is then derived, as in the paper, from its handling time,
# its crane count and the interference exponent.
@dataclass(frozen=True)
class _VesselClass:
    length_range: tuple[int, int]
    handling_range: tuple[int, int]
    crane_range: tuple[int, int]


_VESSEL_CLASSES = (
    _VesselClass((8, 14), (6, 16), (2, 5)),      # feeder
    _VesselClass((18, 24), (10, 28), (3, 7)),    # medium
    _VesselClass((26, 34), (20, 48), (4, 9)),    # jumbo
)


def make_meisel_bierwirth_instance(
    name: str = "iris2017_20x20",
    n_vessels: int = 20,
    quay_length: int = 1000,
    section: int = 10,
    n_cranes: int = 10,
    horizon: int = 168,
    n_berths: int | None = None,
    alpha: float = 0.9,
    beta: float = 0.01,
    seed: int = 20261014,
    *,
    quay_crane_limit: int | None = None,
) -> Instance:
    """Generate an instance with the characteristics of Section 6.1.

    Vessel counts are 20, 30 or 40, the quay is 1000 m discretised into 10 m
    sections, 10 quay cranes are available, and the planning horizon is 168
    working hours. Vessels are drawn from the three published size classes, with
    the QC-hour demand that gives the handling time implied by the crane count
    and the interference exponent, and arrive inside the part of the horizon from
    which every vessel can still be served.

    The quay is represented as ``n_berths`` discrete berths whose lengths sum to
    ``quay_length``. The paper's berthing *position* has no analogue in this
    package, where a berth is occupied whole, so the desired position of each
    vessel is recorded in ``Instance.meta`` rather than used as a placement
    variable. With whole-berth occupancy two vessels can never share a
    berth-period, so the interference coefficient ``alpha`` and the
    berthing-deviation coefficient ``beta`` cannot change a placement here; both
    are stored in the metadata as the statement of the paper's setting.

    ``quay_crane_limit`` defaults to ``n_cranes``, the paper's crane count, and
    is the binding resource: with a 1000 m quay cut into 20 berths, berth
    capacity alone almost never constrains a 20-40 vessel fleet.
    """
    if n_vessels < 1:
        raise ValueError("n_vessels must be positive")
    if section <= 0 or quay_length % section != 0:
        raise ValueError("quay_length must be a multiple of section")
    if horizon < 2:
        raise ValueError("horizon must leave room for at least one handling")
    sections = quay_length // section
    if n_berths is None:
        n_berths = max(2, min(20, n_vessels))
    if n_berths < 1:
        raise ValueError("n_berths must be positive")

    rng = random.Random(seed)
    vessels: list[Vessel] = []
    desired: list[int] = []
    qc_demands: list[int] = []
    for i in range(n_vessels):
        spec = _VESSEL_CLASSES[i % len(_VESSEL_CLASSES)]
        length = rng.randint(*spec.length_range)
        handle = rng.randint(*spec.handling_range)
        hi = min(spec.crane_range[1], n_cranes)
        cranes = rng.randint(min(spec.crane_range[0], hi), hi)
        # As in the paper's constraint (2): q effective QC hours per period, so
        # the demand is the handling time scaled by the interference exponent.
        qc_demands.append(int(round(alpha * cranes * handle)))
        vessels.append(Vessel(vid=i, name=f"V{i + 1:02d}", length=length,
                              arrive=0, handle=handle, priority=i % 3 + 1))
        desired.append(rng.randint(0, max(0, sections - length)))

    # Arrivals are spread over the window from which a vessel can still finish,
    # leaving the longest handling of the fleet at least a period of slack.
    longest = max(v.handle for v in vessels)
    latest_arrive = max(0, horizon - longest - 1)
    for v in vessels:
        v.arrive = rng.randint(0, latest_arrive)

    berths = _split_quay(n_berths, quay_length, [v.length for v in vessels])
    inst = Instance(
        name=name, vessels=vessels, berths=berths, horizon=horizon, dt=1.0,
        quay_crane_limit=n_cranes if quay_crane_limit is None else quay_crane_limit,
        meta={
            "generator": "make_meisel_bierwirth_instance",
            "seed": seed,
            "source": "Iris, Pacino & Ropke (2017), Section 6.1",
            "benchmark": "Meisel & Bierwirth (2009), not publicly available",
            "quay_length_m": quay_length,
            "section_m": section,
            "sections": sections,
            "n_cranes": n_cranes,
            "horizon_hours": horizon,
            "interference_alpha": alpha,
            "berthing_deviation_beta": beta,
            "desired_positions": desired,
            "qc_hour_demand": qc_demands,
        },
    )
    problems = inst.validate()
    if problems:
        raise ValueError(f"generated instance is infeasible by construction: {problems}")
    return inst


def _split_quay(n_berths: int, quay_length: int,
                vessel_lengths: Sequence[int]) -> list[Berth]:
    """Cut the quay into berths that each hold the longest vessel.

    Equal cuts are used where possible; where an equal cut would not hold the
    longest vessel, every berth gets that minimum and the remainder is spread
    evenly. Whole-berth occupancy means a vessel must fit its berth entirely, so
    this keeps generated instances feasible without making any berth longer than
    it needs to be.
    """
    longest = max(vessel_lengths) if vessel_lengths else 1
    width = max(quay_length // n_berths, longest)
    if width * n_berths <= quay_length:
        widths = [width] * n_berths
        widths[0] += quay_length - width * n_berths
    else:
        widths = [longest] * n_berths
        spare = quay_length - longest * n_berths
        if spare < 0:
            raise ValueError(
                f"{n_berths} berths cannot hold a {longest} m vessel within "
                f"{quay_length} m of quay")
        for k in range(spare):
            widths[k % n_berths] += 1
    return [Berth(bid=k, name=f"B{k + 1:02d}", length=widths[k])
            for k in range(n_berths)]


# ---------------------------------------------------------------------------
# Solution, candidate assignments and feasibility primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Slot:
    """A candidate assignment: one vessel at one berth and start period."""
    berth: int
    start: int
    cost: float = 0.0


def vessel_slot_cost(inst: Instance, vid: int, start: int, objective: str) -> float:
    """The paper's per-assignment cost component under this package's objectives.

    The paper's insertion operators try each vessel's candidate assignments in
    increasing cost ``c_j``, the time-dependent part of its objective (1). Under
    ``sum_C`` the part of the package's objective that a single assignment
    contributes on its own is the vessel's completion time ``s_i + p_i``; under
    ``C_max`` it is its finishing time, the same expression.
    """
    if objective not in ("sum_C", "C_max"):
        raise ValueError(f"unknown objective: {objective}")
    return float(start + inst.vessels[vid].handle)


def generate_candidates(inst: Instance, objective: str = "sum_C",
                        max_per_vessel: int | None = 400) -> list[list[Slot]]:
    """Every feasible (berth, start) assignment per vessel, in increasing cost.

    Feasible here means release time, horizon and tidal windows - exactly the
    conditions under which the start domain of ``BAPTimeIndexed`` is non-empty.
    Berth occupancy and the crane limit couple vessels, so they are enforced
    during insertion instead. ``max_per_vessel`` keeps a cost-ordered prefix,
    which stands in for the paper's column reduction: the paper likewise discards
    assignments a priori on cost grounds, and a cost-ordered insertion only ever
    reaches the cheapest few.
    """
    out: list[list[Slot]] = []
    for v in inst.vessels:
        slots: list[Slot] = []
        for k in range(inst.m):
            if v.length > inst.berths[k].length:
                # A vessel occupies the whole berth, so it can only berth where its
                # length fits, the same condition the models and the checkers apply.
                continue
            for st in range(v.arrive, inst.horizon - v.handle + 1):
                if all(not inst.blocked(k, t) for t in range(st, st + v.handle)):
                    slots.append(Slot(k, st,
                                      vessel_slot_cost(inst, v.vid, st, objective)))
        slots.sort(key=lambda s: (s.cost, s.berth, s.start))
        if max_per_vessel is not None:
            slots = slots[:max_per_vessel]
        if not slots:
            raise RuntimeError(
                f"vessel {v.name} has no feasible start on any berth: the "
                f"instance admits no schedule")
        out.append(slots)
    return out


@dataclass
class ALNSSolution:
    """A berth-and-time assignment with its derived resource usage.

    ``assign`` is the assignment itself and is the search space; ``occupancy``
    (whether a berth is busy in a period) and ``crane_load`` (vessels handled per
    period) are derived state, maintained incrementally so a single vessel can be
    removed or inserted without rebuilding them. A vessel absent from ``assign``
    is unassigned, exactly as in the paper's partial solution after destruction.
    """
    inst: Instance
    assign: dict[int, tuple[int, int]] = field(default_factory=dict)
    occupancy: list[list[bool]] | None = None
    crane_load: list[int] | None = None

    def __post_init__(self) -> None:
        # None marks derived state that was never supplied. Testing the lists for
        # emptiness instead would reallocate them whenever they happened to hold
        # only zeros, silently discarding the occupancy a caller had built.
        if self.occupancy is None:
            self.occupancy = [[False] * self.inst.horizon for _ in range(self.inst.m)]
        if self.crane_load is None:
            self.crane_load = [0] * self.inst.horizon

    @classmethod
    def empty(cls, inst: Instance) -> "ALNSSolution":
        return cls(inst=inst)

    @classmethod
    def from_assign(cls, inst: Instance,
                    assign: dict[int, tuple[int, int]]) -> "ALNSSolution":
        """Build a solution from an assignment, deriving its resource usage.

        The occupancy and crane load are replayed from the assignment rather than
        taken on trust: a solution whose derived state disagrees with its own
        assignment would report a schedule that overlaps on a berth.
        """
        solution = cls.empty(inst)
        for vid, (k, st) in assign.items():
            insert_slot(solution, vid, Slot(k, st))
        return solution

    def copy(self) -> "ALNSSolution":
        return ALNSSolution(inst=self.inst,
                            assign=dict(self.assign),
                            occupancy=[row[:] for row in self.occupancy],
                            crane_load=self.crane_load[:])

    @property
    def assigned(self) -> list[int]:
        return sorted(self.assign)

    @property
    def unassigned(self) -> list[int]:
        return [v.vid for v in self.inst.vessels if v.vid not in self.assign]

    @property
    def complete(self) -> bool:
        return len(self.assign) == self.inst.n

    def objective(self, objective: str = "sum_C") -> float:
        """Objective from the assignment alone; ``inf`` while incomplete."""
        if not self.complete:
            return math.inf
        comp = {vid: st + self.inst.vessels[vid].handle
                for vid, (_k, st) in self.assign.items()}
        return objective_value(comp, objective)

    def completion(self, vid: int) -> float:
        """Completion period of one vessel, used as its cost in the operators."""
        got = self.assign.get(vid)
        if got is None:
            return math.inf
        return float(got[1] + self.inst.vessels[vid].handle)

    def to_schedule(self, inst: Instance, objective: str = "sum_C",
                    solver: str = "ALNS-IRIS2017", wall_time: float = 0.0,
                    extra: dict | None = None) -> Schedule:
        """Verify and score the assignment through the package's own paths."""
        sch = Schedule(assign=dict(self.assign), solver=solver, wall_time=wall_time,
                       extra=dict(extra or {}))
        sch.check(inst)
        if sch.feasible:
            sch.compute_objective(inst, objective)
        return sch


def _spans(solution: ALNSSolution, vid: int) -> range:
    got = solution.assign[vid]
    return range(got[1], got[1] + solution.inst.vessels[vid].handle)


def _overlaps(solution: ALNSSolution, vid: int, k: int, st: int) -> bool:
    """Berth conflict: a berth handles one vessel at a time."""
    for t in range(st, st + solution.inst.vessels[vid].handle):
        if solution.occupancy[k][t]:
            return True
    return False


def _crane_ok(solution: ALNSSolution, vid: int, st: int) -> bool:
    """Whether the usable crane capacity still admits one more vessel."""
    inst = solution.inst
    p = inst.vessels[vid].handle
    for t in range(st, min(st + p, inst.horizon)):
        cap = inst.crane_capacity(t)
        if cap is not None and solution.crane_load[t] + 1 > cap:
            return False
    return True


def feasible_slot(solution: ALNSSolution, vid: int, slot: Slot) -> bool:
    """Whether assignment ``slot`` may be added to the partial solution."""
    return (not _overlaps(solution, vid, slot.berth, slot.start)
            and _crane_ok(solution, vid, slot.start))


def insert_slot(solution: ALNSSolution, vid: int, slot: Slot) -> None:
    """Add a feasible assignment of ``vid`` to the solution."""
    solution.assign[vid] = (slot.berth, slot.start)
    for t in _spans(solution, vid):
        solution.occupancy[slot.berth][t] = True
        solution.crane_load[t] += 1


def remove_vessels(solution: ALNSSolution, vids: Iterable[int]) -> None:
    """Remove the given vessels and leave the rest of the plan untouched."""
    for vid in vids:
        if vid not in solution.assign:
            continue
        k, st = solution.assign[vid]
        for t in range(st, st + solution.inst.vessels[vid].handle):
            solution.occupancy[k][t] = False
            solution.crane_load[t] -= 1
        del solution.assign[vid]


# ---------------------------------------------------------------------------
# Destroy operators (Section 5.2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Destroyed:
    """The assignments a destroy operator selected, and which operator it was."""
    removed: tuple[int, ...]
    operator: str


class _DestroyContext:
    """State shared by the destroy operators: the plan, the rng, the budget."""

    def __init__(self, inst: Instance, solution: ALNSSolution,
                 rng: random.Random, config: "ALNSConfig") -> None:
        self.inst = inst
        self.solution = solution
        self.rng = rng
        self.config = config

    def remove_count(self) -> int:
        """How many assignments to destroy.

        Section 5.2: a random integer between ``phi_u / 2`` and ``phi_u``, where
        ``phi_u`` is the instance size divided by ``phi_r``, subject to
        ``2 <= phi <= phi_u``. On an instance too small for ``phi_r`` to leave
        room for two assignments the lower bound of the paper's own range wins.
        """
        size = len(self.solution.assign)
        upper = max(2, int(size / self.config.phi_r))
        lower = max(2, upper // 2)
        return min(self.rng.randint(lower, upper), size)

    def relation_probability(self) -> float:
        return self.rng.uniform(DESTROY_P_MIN, DESTROY_P_MAX)

    def time_related(self, i: int, j: int) -> bool:
        """Time-related: the two handling intervals overlap on different berths.

        The paper's wording is "different berth positions at the same time"; with
        whole-berth occupancy that is an overlap of the two intervals.
        """
        a, b = self.solution.assign[i], self.solution.assign[j]
        if a[0] == b[0] and a[1] != b[1]:
            return False
        s1, e1 = a[1], a[1] + self.inst.vessels[i].handle
        s2, e2 = b[1], b[1] + self.inst.vessels[j].handle
        return s1 < e2 and s2 < e1

    def berth_related(self, i: int, j: int) -> bool:
        """Berth-related: the same berthing area at different times."""
        a, b = self.solution.assign[i], self.solution.assign[j]
        return a[0] == b[0] and a[1] != b[1]

    def relatedness(self, i: int, j: int) -> float:
        """The paper's measure (48): ``a|b_i-b_j| + b|s_i-s_j| + c|e_i-e_j|``.

        Berthing positions of the paper's quay plan have no counterpart in a
        discrete-berth model, so the berth index stands in for the position and
        ``a`` scales a difference of berth indices rather than of metres.
        """
        a, b = self.solution.assign[i], self.solution.assign[j]
        e1 = a[1] + self.inst.vessels[i].handle
        e2 = b[1] + self.inst.vessels[j].handle
        return (self.config.shaw_a * abs(a[0] - b[0])
                + self.config.shaw_b * abs(a[1] - b[1])
                + self.config.shaw_c * abs(e1 - e2))


def destroy_shaw(ctx: _DestroyContext) -> Destroyed:
    """Shaw removal: prefer assignments related to a randomly chosen one.

    Relatedness is a distance: the paper's (48) states that a *lower* value of
    M(i, j) means a *higher* relatedness, so the remaining assignments are walked
    from the smallest M upwards, each tested against the determinism parameter.
    Sorting the other way round would select the least related assignments, which
    would defeat the operator the paper describes.
    """
    order = ctx.solution.assigned
    if not order:
        return Destroyed((), "shaw")
    phi = ctx.remove_count()
    chosen = [ctx.rng.choice(order)]
    p = ctx.relation_probability()
    while len(chosen) < phi:
        remaining = [i for i in order if i not in set(chosen)]
        if not remaining:
            break
        remaining.sort(key=lambda j: (ctx.relatedness(chosen[0], j), j))
        cand = remaining.pop(0)
        # Algorithm 2 sets p to 1.0 once only as many assignments are left as the
        # destroy list still needs, which guarantees exactly phi are collected.
        if len(remaining) < phi - len(chosen):
            p = DESTROY_P_MAX
        if ctx.rng.random() <= p:
            chosen.append(cand)
        else:
            p = min(DESTROY_P_MAX, p + (DESTROY_P_MAX - p) / 2.0)
    return Destroyed(tuple(chosen), "shaw")


def _cost_relatedness_destroy(ctx: _DestroyContext,
                              related: Callable[[int, int], bool],
                              name: str) -> Destroyed:
    """Decreasing cost order; the costliest vessel and its related ones."""
    order = ctx.solution.assigned
    if not order:
        return Destroyed((), name)
    phi = ctx.remove_count()
    order.sort(key=lambda i: (-ctx.solution.completion(i), i))
    chosen: list[int] = []
    while len(chosen) < phi:
        seed = next((i for i in order if i not in chosen), None)
        if seed is None:
            break
        chosen.append(seed)
        for j in order:
            if len(chosen) >= phi:
                break
            if j not in chosen and related(seed, j):
                chosen.append(j)
    return Destroyed(tuple(chosen[:phi]), name)


def destroy_cost_time(ctx: _DestroyContext) -> Destroyed:
    """Cost and time-relatedness removal (Section 5.2.2)."""
    return _cost_relatedness_destroy(ctx, ctx.time_related, "cost_time")


def destroy_cost_berth(ctx: _DestroyContext) -> Destroyed:
    """Cost and berth-relatedness removal (Section 5.2.3)."""
    return _cost_relatedness_destroy(ctx, ctx.berth_related, "cost_berth")


def destroy_random(ctx: _DestroyContext) -> Destroyed:
    """Random removal with the time-relatedness test of Section 5.2.4."""
    order = ctx.solution.assigned
    if not order:
        return Destroyed((), "random")
    phi = ctx.remove_count()
    chosen = [ctx.rng.choice(order)]
    p = ctx.relation_probability()
    while len(chosen) < phi:
        remaining = [i for i in order if i not in chosen]
        if not remaining:
            break
        cand = ctx.rng.choice(remaining)
        if any(ctx.time_related(cand, j) for j in chosen):
            chosen.append(cand)
        elif len(remaining) <= phi - len(chosen):
            chosen.append(cand)
        elif ctx.rng.random() <= p:
            chosen.append(cand)
        else:
            p = min(DESTROY_P_MAX, p + (DESTROY_P_MAX - p) / 2.0)
    return Destroyed(tuple(chosen), "random")


DESTROY_OPERATORS: dict[str, Callable[[_DestroyContext], Destroyed]] = {
    "shaw": destroy_shaw,
    "cost_time": destroy_cost_time,
    "cost_berth": destroy_cost_berth,
    "random": destroy_random,
}


# ---------------------------------------------------------------------------
# Insertion operators (Section 5.3)
# ---------------------------------------------------------------------------

def _objective_delta(solution: ALNSSolution, vid: int, slot: Slot,
                     objective: str) -> float:
    """Increase in the objective if ``slot`` is fixed for ``vid``.

    Inserting a vessel never changes the completion time of a vessel already
    placed, so under ``sum_C`` the increase is the new vessel's completion time
    and under ``C_max`` it is the new maximum.
    """
    finish = float(slot.start + solution.inst.vessels[vid].handle)
    if objective == "sum_C" or not solution.assign:
        return finish
    current = max(solution.completion(j) for j in solution.assign)
    return max(0.0, finish - current)


def repair_greedy(solution: ALNSSolution, candidates: list[list[Slot]],
                  order: Sequence[int], rng: random.Random,
                  p: float) -> list[int]:
    """Basic greedy insertion (Section 5.3.1).

    For each vessel of the insertion list the assignments are tried one by one in
    increasing cost order; an assignment is considered with probability ``p`` and
    the first feasible one is fixed. A vessel that cannot be inserted is left
    unassigned and reported, for the caller to penalise, rather than being
    silently dropped.

    A final pass with ``p = 1`` places any vessel the probabilistic pass skipped,
    so the stochastic variant differs from the deterministic one in which
    candidate it settles on rather than in how many vessels it strands.
    """
    stuck: list[int] = []
    for vid in order:
        placed = False
        for slot in candidates[vid]:
            if p < 1.0 and rng.random() > p:
                continue
            if feasible_slot(solution, vid, slot):
                insert_slot(solution, vid, slot)
                placed = True
                break
        if not placed:
            stuck.append(vid)
    for vid in stuck:
        for slot in candidates[vid]:
            if feasible_slot(solution, vid, slot):
                insert_slot(solution, vid, slot)
                break
    return [vid for vid in stuck if vid not in solution.assign]


def repair_smarter(solution: ALNSSolution, candidates: list[list[Slot]],
                   order: Sequence[int], objective: str) -> list[int]:
    """Smarter insertion (Section 5.3.2).

    The paper's improvement over the basic operator is that it does not abandon
    an assignment whose stored QC plan cannot be fitted: it reevaluates the
    assignment, produces an alternative plan, and stops as soon as the remaining
    assignments are no worse than the one it produced. With no per-period QC plan
    in this package, the representable form of that rule is to reevaluate every
    feasible candidate instead of accepting the first, fixing the one with the
    smallest resulting objective. Both operators therefore enforce the package's
    crane limit, but the smarter one chooses rather than settles.
    """
    stuck: list[int] = []
    for vid in order:
        best: Slot | None = None
        best_delta = math.inf
        for slot in candidates[vid]:
            if not feasible_slot(solution, vid, slot):
                continue
            delta = _objective_delta(solution, vid, slot, objective)
            if delta < best_delta:
                best, best_delta = slot, delta
        if best is None:
            stuck.append(vid)
        else:
            insert_slot(solution, vid, best)
    return stuck


# ---------------------------------------------------------------------------
# Adaptive weights (Section 5, "Adaptive Weight Adjustment")
# ---------------------------------------------------------------------------

@dataclass
class _OperatorState:
    """Weights, scores and selection counts of one family of operators."""
    weights: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    used: dict[str, int] = field(default_factory=dict)
    total_used: dict[str, int] = field(default_factory=dict)
    rounds: int = 0

    @classmethod
    def uniform(cls, names: Iterable[str]) -> "_OperatorState":
        names = list(names)
        return cls(weights={n: 1.0 for n in names},
                   scores={n: 0.0 for n in names},
                   used={n: 0 for n in names},
                   total_used={n: 0 for n in names})

    def roulette(self, rng: random.Random) -> str:
        """Roulette-wheel selection over the weights, as the paper describes.

        Each weight is used as a selection probability up to the total weight, so
        a cumulative distribution is formed and inverted for a uniform draw.
        """
        total = sum(self.weights.values())
        if total <= 0.0:
            return rng.choice(sorted(self.weights))
        draw = rng.random() * total
        acc = 0.0
        for name in sorted(self.weights):
            acc += self.weights[name]
            if draw < acc:
                return name
        return sorted(self.weights)[-1]

    def update(self, gamma: float) -> None:
        """Apply the weight update and reset the scores.

        ``w_i`` is kept when the operator was not used during the last ``d``
        iterations, and becomes ``(1 - gamma) w_i + gamma p_i / W_i`` otherwise;
        every score is then set to zero again.
        """
        for name in self.scores:
            count = self.used[name]
            if count:
                self.weights[name] = ((1.0 - gamma) * self.weights[name]
                                      + gamma * self.scores[name] / count)
            self.scores[name] = 0.0
            self.used[name] = 0
            self.total_used[name] = self.total_used.get(name, 0) + count
        self.rounds += 1


# ---------------------------------------------------------------------------
# Configuration and result
# ---------------------------------------------------------------------------

@dataclass
class ALNSConfig:
    """ALNS parameters, defaulting to the tuned vector of Section 6.1.1."""
    max_iterations: int = PAPER_MAX_ITERATIONS
    phi_r: int = 4                     # assignments destroyed = size / phi_r
    shaw_a: float = 0.01               # weight of berth relatedness
    shaw_b: float = 2.0                # weight of start-time relatedness
    shaw_c: float = 0.01               # weight of end-time relatedness
    start_pct: float = 0.05            # T_start as a fraction of the initial objective
    end_pct: float = 0.0005            # T_end as a fraction of the initial objective
    reheat_period: int = 30000         # iterations per cooling cycle, then reheat
    rho1: float = 2.0                  # score: new global best
    rho2: float = 5.0                  # score: improves the current solution
    rho3: float = 10.0                 # score: accepted but not improving
    gamma: float = 0.8                 # reaction factor
    update_interval: int = 1000        # d: iterations between weight updates
    max_candidates: int | None = 400   # candidate assignments kept per vessel
    seed_baselines: bool = True        # also seed from the package's rules

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.phi_r < 1:
            raise ValueError("phi_r must be at least 1")
        if self.update_interval < 1:
            raise ValueError("update_interval must be at least 1")
        if self.reheat_period < 1:
            raise ValueError("reheat_period must be at least 1")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if self.start_pct < 0.0 or self.end_pct < 0.0:
            raise ValueError("temperature fractions must be non-negative")
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ValueError("max_candidates must be positive or None")


@dataclass
class ALNSResult:
    """A completed run: the schedule plus the state of the search mechanisms."""
    schedule: Schedule
    objective: float
    iterations: int
    temperature_start: float
    temperature_end: float
    cooling_factor: float
    destroy_weights: dict[str, float]
    insertion_weights: dict[str, float]
    destroy_scores: dict[str, float]
    insertion_scores: dict[str, float]
    destroy_used: dict[str, int]
    insertion_used: dict[str, int]
    best_objective_trajectory: list[float]
    seed: int
    initial_objective: float

    @property
    def assign(self) -> dict[int, tuple[int, int]]:
        return self.schedule.assign

    @property
    def operator_usage(self) -> dict[str, int]:
        return {**{f"destroy:{k}": v for k, v in self.destroy_used.items()},
                **{f"insert:{k}": v for k, v in self.insertion_used.items()}}


INSERTION_OPERATORS: tuple[str, ...] = (
    "greedy", "greedy_stochastic", "smarter", "smarter_stochastic",
)

#: The six operator families of Sections 5.2 and 5.3.
ALNS_OPERATORS: dict[str, tuple[str, ...]] = {
    "destroy": tuple(DESTROY_OPERATORS),
    "insertion": INSERTION_OPERATORS,
}


# ---------------------------------------------------------------------------
# The ALNS
# ---------------------------------------------------------------------------

def _initial_solution(inst: Instance, candidates: list[list[Slot]],
                      objective: str, rng: random.Random,
                      config: ALNSConfig) -> ALNSSolution:
    """The greedy construction heuristic of Section 5.1.

    A random vessel order; for each vessel the assignments are sorted by cost and
    the first feasible one is placed. The paper obtains its initial solution from
    a constructive heuristic or a mathematical model, so the package's FCFS/LPT/
    SPT rules are also tried and the best of the four is returned; starting from
    a better feasible solution only shortens the search.
    """
    order = list(range(inst.n))
    rng.shuffle(order)
    greedy = ALNSSolution.empty(inst)
    repair_greedy(greedy, candidates, order, rng, 1.0)
    if not config.seed_baselines:
        return greedy
    better = _best_baseline_seed(inst, objective, greedy)
    return better if better is not None else greedy


def _best_baseline_seed(inst: Instance, objective: str,
                        fallback: ALNSSolution) -> ALNSSolution | None:
    """The cheapest feasible baseline schedule, or None if none is better."""
    from ..heuristics.constructive import fcfs, lpt, spt

    best: ALNSSolution | None = None
    best_val = fallback.objective(objective)
    for rule in (fcfs, lpt, spt):
        try:
            sch = rule(inst, objective)
        except RuntimeError:
            continue
        if sch.feasible and sch.objective is not None and sch.objective < best_val:
            best = ALNSSolution.from_assign(inst, dict(sch.assign))
            best_val = float(sch.objective)
    return best


def _accept(candidate: float, current: float, temperature: float,
            rng: random.Random) -> bool:
    """Simulated-annealing acceptance: better always, worse with ``e^{-d/T}``."""
    if candidate <= current:
        return True
    if not math.isfinite(candidate) or temperature <= 0.0:
        return False
    return rng.random() < math.exp(-(candidate - current) / temperature)


def _penalty(inst: Instance, base: float, unassigned: int) -> float:
    """Cost charged for leaving vessels unassigned (Section 5.3).

    The paper adds a high penalty per unassigned vessel so such solutions are
    unattractive without being infeasible. It is scaled to the instance's own
    objective magnitude, so the smallest penalty charged for a single unassigned
    vessel already exceeds the search's whole temperature range and an incomplete
    solution can never be preferred to a complete one.
    """
    if unassigned == 0:
        return 0.0
    return max(base, 1.0) * (inst.n + 1) * (unassigned + 1)


def alns(inst: Instance, objective: str = "sum_C", *, seed: int = 20261014,
         config: ALNSConfig | None = None,
         max_iterations: int | None = None) -> ALNSResult:
    """Run the BACAP ALNS of Iris, Pacino & Ropke (2017) on ``inst``.

    ``max_iterations`` overrides ``config.max_iterations`` so a short search can
    be run without rebuilding the rest of the parameter vector. The result
    carries the operator weights, selection counts and the best-objective
    trajectory alongside the verified schedule.
    """
    config = config or ALNSConfig()
    if max_iterations is not None:
        config = replace(config, max_iterations=max_iterations)
    if objective not in ("sum_C", "C_max"):
        raise ValueError(f"unknown objective: {objective}")

    wall_start = time.perf_counter()
    rng = random.Random(seed)
    candidates = generate_candidates(inst, objective, config.max_candidates)
    current = _initial_solution(inst, candidates, objective, rng, config)

    current_val = current.objective(objective)
    initial_val = current_val
    if current.complete:
        best, best_val = current.copy(), current_val
    else:
        best, best_val = ALNSSolution.empty(inst), math.inf

    # The temperature is derived from the initial solution's objective, as the
    # paper describes, and healed to a usable value when the search starts from
    # an incomplete construction with no finite objective.
    reference = initial_val if math.isfinite(initial_val) else _trivial_sum(inst)
    start_temp = config.start_pct * abs(reference) or 1.0
    end_temp = config.end_pct * abs(reference)
    cooling = (end_temp / start_temp) ** (1.0 / config.reheat_period)
    temperature = start_temp

    if not math.isfinite(current_val):
        current_val = _trivial_sum(inst) * (inst.n + 1)

    destroy_state = _OperatorState.uniform(DESTROY_OPERATORS)
    insertion_state = _OperatorState.uniform(INSERTION_OPERATORS)

    def scored(solution: ALNSSolution) -> float:
        base = best_val if math.isfinite(best_val) else current_val
        return solution.objective(objective) + _penalty(
            inst, base, len(solution.unassigned))

    trajectory: list[float] = []
    for iteration in range(config.max_iterations):
        destroy_name = destroy_state.roulette(rng)
        insertion_name = insertion_state.roulette(rng)
        destroy_state.used[destroy_name] += 1
        insertion_state.used[insertion_name] += 1

        destroyed = DESTROY_OPERATORS[destroy_name](
            _DestroyContext(inst, current, rng, config))
        candidate = current.copy()
        remove_vessels(candidate, destroyed.removed)

        order = list(destroyed.removed)
        rng.shuffle(order)
        if insertion_name.startswith("smarter"):
            repair_smarter(candidate, candidates, order, objective)
        else:
            p = (rng.uniform(INSERTION_P_MIN, INSERTION_P_MAX)
                 if insertion_name.endswith("stochastic") else 1.0)
            repair_greedy(candidate, candidates, order, rng, p)

        candidate_val = candidate.objective(objective)
        scored_val = scored(candidate)

        gain = 0.0
        accepted = False
        if candidate_val < best_val:
            best, best_val = candidate.copy(), candidate_val
            gain, accepted = config.rho1, True
        elif candidate_val < current_val:
            gain, accepted = config.rho2, True
        elif _accept(scored_val, current_val, temperature, rng):
            gain, accepted = config.rho3, True
        if accepted:
            current, current_val = candidate, scored_val
        destroy_state.scores[destroy_name] += gain
        insertion_state.scores[insertion_name] += gain

        temperature *= cooling
        if (iteration + 1) % config.reheat_period == 0:
            temperature = start_temp
        if (iteration + 1) % config.update_interval == 0:
            destroy_state.update(config.gamma)
            insertion_state.update(config.gamma)
        # The trajectory reports the best complete objective seen so far; the
        # leading infinities, while no complete solution existed yet, are dropped.
        if math.isfinite(best_val):
            trajectory.append(best_val)

    destroy_state.update(config.gamma)
    insertion_state.update(config.gamma)

    if not math.isfinite(best_val):
        raise RuntimeError(
            f"ALNS found no complete schedule for {inst.name} in "
            f"{config.max_iterations} iterations")

    wall = time.perf_counter() - wall_start
    sanity = sanity_check_schedule(inst, best.assign)
    extra = {
        "iterations": config.max_iterations,
        "seed": seed,
        "initial_objective": initial_val,
        "params": {
            "phi_r": config.phi_r,
            "shaw": (config.shaw_a, config.shaw_b, config.shaw_c),
            "start_pct": config.start_pct, "end_pct": config.end_pct,
            "reheat_period": config.reheat_period,
            "rho1": config.rho1, "rho2": config.rho2, "rho3": config.rho3,
            "gamma": config.gamma, "update_interval": config.update_interval,
        },
        "destroy_weights": dict(destroy_state.weights),
        "insertion_weights": dict(insertion_state.weights),
        "destroy_used": dict(destroy_state.total_used),
        "insertion_used": dict(insertion_state.total_used),
        "weight_updates": destroy_state.rounds,
        "unassigned": best.unassigned,
        "sanity_issues": [issue.kind for issue in sanity],
    }
    schedule = best.to_schedule(inst, objective, "ALNS-IRIS2017", wall, extra)
    return ALNSResult(
        schedule=schedule,
        objective=best_val,
        iterations=config.max_iterations,
        temperature_start=start_temp,
        temperature_end=end_temp,
        cooling_factor=cooling,
        destroy_weights=dict(destroy_state.weights),
        insertion_weights=dict(insertion_state.weights),
        destroy_scores=dict(destroy_state.scores),
        insertion_scores=dict(insertion_state.scores),
        destroy_used=dict(destroy_state.total_used),
        insertion_used=dict(insertion_state.total_used),
        best_objective_trajectory=trajectory,
        seed=seed,
        initial_objective=initial_val,
    )


def _trivial_sum(inst: Instance) -> float:
    """Sum of arrival plus handling: a finite reference for the temperature."""
    return float(sum(v.arrive + v.handle for v in inst.vessels))


def run_alns(inst: Instance, objective: str = "sum_C", *, seed: int = 20261014,
             config: ALNSConfig | None = None,
             max_iterations: int | None = None) -> Schedule:
    """Convenience entry point returning only the verified schedule."""
    return alns(inst, objective, seed=seed, config=config,
                max_iterations=max_iterations).schedule
