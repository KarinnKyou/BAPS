# -*- coding: utf-8 -*-
"""The BACAP model of Iris, Pacino & Ropke (2017), with its quay-crane layer.

The ALNS in `iris2017.py` runs on this package's discrete-berth model, where a
vessel occupies a whole berth and cranes are only a limit on how many vessels are
handled at once. The paper's problem is broader: a vessel is berthed at a position
along the quay, and a per-period quay-crane plan is part of the solution, so the
handling time depends on how many cranes the vessel gets. This module implements
that model: `BACAPInstance` (continuous position, per-vessel QC-hour demand, crane
range, arrival and time-window costs), the economic crane pattern of Appendix A,
the per-period plan built from the remaining cranes, the paper's objective, and an
ALNS over assignments that carries the plan.

What is reproduced, and from where
----------------------------------
* The objective (1): ``sum_i (c1_i dETA_i + c2_i dEFT_i + c3_i u_i)
  + c4 sum_t sum_q q r_itq`` with ``dETA = max(ETA - s, 0)``,
  ``dEFT = max(e - EFT, 0)`` and ``u = 1`` when ``e > LFT``.
* The crane-workload constraint ``sum_t sum_q q^alpha r_itq >= (1 + beta db) m_i``
  and the terminal constraint ``sum_i sum_q q r_itq <= Q`` per period.
* Appendix A's economic crane pattern: the base crane count
  ``q_hat = floor((D_eff / process)^(1/alpha))`` and the split of the handling
  period into ``f1`` periods at ``q_hat`` cranes and ``f2`` at ``q_hat + 1``,
  found by the printed loop over ``p``.
* Algorithms 1-4: the ALNS loop with simulated-annealing acceptance and adaptive
  operator weights, the four destroy operators, and the two insertion operators.
* The instance generator of Section 6.1: the vessel classes and their length,
  QC-hour demand, crane range and cost parameters, 60/30/10 class mix, 1000 m
  quay in 10 m sections, 10 cranes, a 168 h horizon, ``alpha = 0.9`` and
  ``beta = 0.01``.

What is not reproduced
----------------------
* The 30 original instances of Meisel & Bierwirth (2009). They are not publicly
  available, so `make_instance` builds instances from the published distributions
  and records that they are literature reconstructions, not the original data.
  Nothing here should be read as reproducing the paper's table values.
* The exact random seeds, rounding of ETA/EFT/LFT and the filtering that produced
  the published instances. `make_instance` states the choices it makes.

Readings of the source text that are not fully determined by it
---------------------------------------------------------------
1. **The per-period plan.** Appendix A fixes the *pattern* -- how many periods get
   ``q_hat`` and how many ``q_hat + 1`` cranes -- but not which periods receive the
   extra crane. The plan here gives the extra crane to the ``f2`` periods with the
   most remaining terminal capacity. That rule is complete: it fails exactly when
   no choice of the ``f2`` periods would fit, because the periods with the most
   room are the best candidates for the extra one.
2. **The assignment's handling time.** ``process`` is an input to Appendix A, so an
   assignment carries both a position and a duration. For a fixed base crane count
   the *smallest* feasible duration dominates every larger one: it finishes no
   later, and it needs no more crane-hours. Candidate generation therefore keeps
   one duration per (position, base crane count).
3. **``q_hat`` outside ``[r_min, r_max]``.** The paper determines ``q_hat`` within
   the vessel's crane range; it is not computed and then clamped. An assignment
   whose computed ``q_hat`` falls outside the range is rejected, not repaired.
4. **Insertion operators.** Both are represented by how they choose among the
   candidates: the basic operator takes the first feasible one under a determinism
   probability, the smarter one evaluates every feasible candidate and fixes the
   cheapest. That is the part of their difference this model can express; the
   paper's smarter operator additionally rebuilds a QC plan the basic one
   abandons, and with the plan built on insertion both end up feasible.
5. **Arrivals.** ``make_instance`` draws an ETA from the part of the horizon from
   which the vessel can still be served. The paper's rule is a uniform draw over
   the horizon; taking it literally can produce a vessel that cannot finish, which
   is a property of the instance rather than of the method, so the generator keeps
   the instance feasible.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

# Objective (1): c1..c3 are per vessel and in thousand USD, c4 is thousand USD per
# crane-hour, so every term of the objective is in the same unit.
C4 = 0.1

# Defaults of Section 6.1.
QUAY_SECTIONS = 100        # 1000 m in 10 m sections
N_CRANES = 10
HORIZON = 168
ALPHA = 0.9
BETA = 0.01


@dataclass(frozen=True)
class VesselClass:
    """One row of the published vessel table: length, demand, cranes, costs."""
    name: str
    length: tuple[int, int]        # 10 m sections
    demand: tuple[int, int]        # m_i, QC-hours
    cranes: tuple[int, int]        # r_i^min, r_i^max
    c1: float                      # cost per hour berthed before ETA
    c2: float                      # cost per hour finishing after EFT
    c3: float                      # one-off cost for finishing after LFT


# The 60/30/10 mix and the parameter ranges of the published instances.
VESSEL_CLASSES: dict[str, VesselClass] = {
    "feeder": VesselClass("feeder", (8, 21), (5, 15), (1, 2), 1.0, 1.0, 3.0),
    "medium": VesselClass("medium", (21, 30), (15, 50), (2, 4), 2.0, 2.0, 6.0),
    "jumbo": VesselClass("jumbo", (30, 40), (50, 60), (4, 6), 3.0, 3.0, 9.0),
}
CLASS_MIX = (("feeder", 0.6), ("medium", 0.3), ("jumbo", 0.1))


@dataclass(frozen=True)
class BACAPVessel:
    """A vessel in the paper's notation.

    Times are hours from the start of the horizon, positions are 10 m sections,
    and ``demand`` is the total crane-hour demand ``m_i`` (in QC-hours before the
    ``alpha`` interference is applied, as the paper defines it).
    """
    vid: int
    name: str
    kind: str
    length: int
    eta: int
    est: int              # earliest start, 0.9 * ETA
    eft: int              # expected finish time
    lft: int              # latest finish time
    demand: int           # m_i
    r_min: int
    r_max: int
    b0: int               # desired berthing position
    c1: float
    c2: float
    c3: float

    def d_min(self, alpha: float = ALPHA) -> int:
        """Shortest handling time at the desired position.

        With every period using ``r_max`` cranes, ``m / r_max^alpha`` periods are
        needed; the paper's ``alpha`` is the default, and a caller that changed the
        interference exponent should pass it in rather than inherit this one.
        """
        return math.ceil(self.demand / self.r_max ** alpha)


@dataclass
class BACAPInstance:
    """A BACAP instance: vessels, quay, cranes, horizon and the paper's constants."""
    name: str
    vessels: list[BACAPVessel]
    quay: int = QUAY_SECTIONS
    cranes: int = N_CRANES
    horizon: int = HORIZON
    alpha: float = ALPHA
    beta: float = BETA
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for position, v in enumerate(self.vessels):
            if v.vid != position:
                raise ValueError(f"vessel at position {position} has vid {v.vid}; "
                                 f"vid must equal the index")
            if v.length > self.quay:
                raise ValueError(f"{v.name} is longer than the quay")
            if not 0 <= v.b0 <= self.quay - v.length:
                raise ValueError(f"{v.name} desired position {v.b0} leaves the quay")
            if v.r_min < 1 or v.r_max < v.r_min:
                raise ValueError(f"{v.name} has an empty crane range")
            if not v.est <= v.eta <= v.eft <= v.lft:
                raise ValueError(f"{v.name} time window is not ordered")
            if v.demand <= 0:
                raise ValueError(f"{v.name} has no crane-hour demand")
        if self.alpha <= 0 or self.beta < 0:
            raise ValueError("alpha must be positive and beta non-negative")

    @property
    def n(self) -> int:
        return len(self.vessels)


# ---------------------------------------------------------------------------
# Appendix A: the economic crane pattern of one assignment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QCConfig:
    """The two-level crane pattern of one assignment.

    ``f1`` periods are worked with ``q_hat`` cranes and ``f2`` with ``q_hat + 1``;
    the two add up to the assignment's handling time. ``crane_hours`` is what the
    objective charges, i.e. the actual number of cranes, not their effective
    capacity.
    """
    q_hat: int
    f1: int
    f2: int

    @property
    def process(self) -> int:
        return self.f1 + self.f2

    @property
    def crane_hours(self) -> int:
        return self.f1 * self.q_hat + self.f2 * (self.q_hat + 1)


def effective_demand(inst: BACAPInstance, vessel: BACAPVessel, berth: int) -> float:
    """``(1 + beta |b - b^0|) m_i``: the crane-hours the position makes necessary."""
    return (1.0 + inst.beta * abs(berth - vessel.b0)) * vessel.demand


def qc_pattern(inst: BACAPInstance, vessel: BACAPVessel, berth: int,
               process: int) -> QCConfig | None:
    """Appendix A's ``q_hat, f1, f2`` for one assignment, or None if invalid.

    ``q_hat`` is computed from the demand and the handling time and must fall in
    the vessel's crane range: an assignment whose base crane count is outside it is
    not serviced at a clamped count, it is rejected. ``f2`` is the smallest number
    of periods that have to use one crane more, which is what the printed loop
    leaves behind (it walks ``p`` down and keeps the last feasible value).
    """
    if process < 1:
        return None
    demand = effective_demand(inst, vessel, berth)
    q_hat = math.floor((demand / process) ** (1.0 / inst.alpha))
    if q_hat < vessel.r_min or q_hat > vessel.r_max:
        return None

    f1 = f2 = None
    for p in range(process, -1, -1):
        capacity = (p * (q_hat + 1) ** inst.alpha
                    + (process - p) * q_hat ** inst.alpha)
        if capacity >= demand - 1e-9:
            f1, f2 = process - p, p
    if f1 is None:
        return None
    if f2 and q_hat + 1 > vessel.r_max:
        return None
    return QCConfig(q_hat=q_hat, f1=f1, f2=f2)


def min_process(inst: BACAPInstance, vessel: BACAPVessel, berth: int,
                q_hat: int) -> int | None:
    """Shortest handling time whose base crane count is ``q_hat``.

    ``q_hat = floor((D/p)^(1/alpha))`` holds exactly on
    ``D/(q_hat+1)^alpha < p <= D/q_hat^alpha``, so the smallest integer in that
    interval is ``floor(D/(q_hat+1)^alpha) + 1``. With ``q_hat = r_max`` the extra
    crane is unavailable, so every period must use ``r_max`` and the duration is
    ``ceil(D/r_max^alpha)``. Floating point can put the estimate one off, so the
    candidate is verified against `qc_pattern` before it is returned.
    """
    demand = effective_demand(inst, vessel, berth)
    if q_hat >= vessel.r_max:
        estimate = math.ceil(demand / q_hat ** inst.alpha)
    else:
        estimate = math.floor(demand / (q_hat + 1) ** inst.alpha) + 1
    estimate = max(estimate, 1)
    for process in (estimate, estimate + 1, estimate - 1):
        if process < 1:
            continue
        config = qc_pattern(inst, vessel, berth, process)
        if config is not None and config.q_hat == q_hat:
            return process
    return None


@dataclass(frozen=True)
class Assignment:
    """One vessel's place in the plan: position, start and crane pattern."""
    berth: int
    start: int
    config: QCConfig

    @property
    def process(self) -> int:
        return self.config.process

    @property
    def finish(self) -> int:
        return self.start + self.process


# ---------------------------------------------------------------------------
# The solution: assignments plus the per-period crane plan
# ---------------------------------------------------------------------------

class BACAPSolution:
    """A partial or complete plan and the resources it uses.

    ``assign`` holds one `Assignment` per serviced vessel; ``cranes`` holds the
    per-period crane count of each assignment, which is the plan Appendix A's
    pattern is realised into. ``crane_use`` and ``busy`` are derived from those two
    and are maintained incrementally so a destroy operator can remove an arbitrary
    subset without rebuilding the whole plan.
    """

    def __init__(self, inst: BACAPInstance) -> None:
        self.inst = inst
        self.assign: dict[int, Assignment] = {}
        self.cranes: dict[int, dict[int, int]] = {}
        self.crane_use: list[int] = [0] * inst.horizon
        self.busy: list[list[tuple[int, int]]] = [[] for _ in range(inst.horizon)]

    # -- state -------------------------------------------------------------
    def copy(self) -> "BACAPSolution":
        other = BACAPSolution(self.inst)
        other.assign = dict(self.assign)
        other.cranes = {vid: dict(plan) for vid, plan in self.cranes.items()}
        other.crane_use = list(self.crane_use)
        other.busy = [list(intervals) for intervals in self.busy]
        return other

    @property
    def assigned(self) -> list[int]:
        return sorted(self.assign)

    def cost_of(self, vid: int) -> float:
        """The vessel's own contribution to objective (1)."""
        a, v = self.assign[vid], self.inst.vessels[vid]
        return (v.c1 * max(v.eta - a.start, 0)
                + v.c2 * max(a.finish - v.eft, 0)
                + (v.c3 if a.finish > v.lft else 0.0)
                + C4 * a.config.crane_hours)

    def objective(self) -> float:
        """Objective (1) over the serviced vessels.

        An incomplete plan is scored on what it contains; the ALNS adds the
        penalty for the missing vessels itself, which is where the paper puts it.
        """
        return sum(self.cost_of(vid) for vid in self.assign)

    # -- feasibility -------------------------------------------------------
    def position_free(self, vid: int, a: Assignment) -> bool:
        """Whether the vessel's stretch of quay is free for its whole stay."""
        start, end = a.berth, a.berth + self.inst.vessels[vid].length
        for t in range(a.start, a.finish):
            for (lo, hi) in self.busy[t]:
                if start < hi and lo < end:
                    return False
        return True

    def plan_for(self, a: Assignment) -> dict[int, int] | None:
        """Realise the pattern into per-period cranes, or None if it does not fit.

        The extra crane goes to the ``f2`` periods with the most remaining
        capacity. Choosing them that way is complete: if any set of ``f2`` periods
        can take the extra crane, so can the ``f2`` with the most room.
        """
        periods = range(a.start, a.finish)
        room = {t: self.inst.cranes - self.crane_use[t] for t in periods}
        boosted = set(sorted(periods, key=lambda t: (-room[t], t))[:a.config.f2])
        plan: dict[int, int] = {}
        for t in periods:
            need = a.config.q_hat + (1 if t in boosted else 0)
            if need > room[t]:
                return None
            plan[t] = need
        return plan

    def insert(self, vid: int, a: Assignment) -> bool:
        """Fix an assignment and its plan, or report that it does not fit."""
        if vid in self.assign:
            raise ValueError(f"vessel {vid} is already assigned")
        plan = self.plan_for(a)
        if plan is None or not self.position_free(vid, a):
            return False
        self.assign[vid] = a
        self.cranes[vid] = plan
        length = self.inst.vessels[vid].length
        for t, q in plan.items():
            self.crane_use[t] += q
            self.busy[t].append((a.berth, a.berth + length))
        return True

    def remove(self, vids) -> None:
        """Remove vessels and rebuild the derived state from the survivors.

        The remaining vessels keep their plans: the paper's destroy step removes
        assignments and leaves the rest of the plan untouched, and rebuilding them
        from the survivors is the same plan.
        """
        removed = [vid for vid in vids if vid in self.assign]
        for vid in removed:
            del self.assign[vid]
            del self.cranes[vid]
        self.crane_use = [0] * self.inst.horizon
        self.busy = [[] for _ in range(self.inst.horizon)]
        for vid, a in self.assign.items():
            length = self.inst.vessels[vid].length
            for t, q in self.cranes[vid].items():
                self.crane_use[t] += q
                self.busy[t].append((a.berth, a.berth + length))


def check_solution(inst: BACAPInstance, solution: BACAPSolution) -> list[str]:
    """Independent feasibility check of a plan against the paper's constraints.

    Recomputes every total from ``assign`` and ``cranes`` instead of trusting the
    derived state, so a solution that mutated one without the other is reported
    rather than passing. Returns one string per violation.
    """
    problems: list[str] = []

    missing = [v.name for v in inst.vessels if v.vid not in solution.assign]
    if missing:
        problems.append(f"{len(missing)} vessels unassigned: {missing[:5]}")
    unknown = [vid for vid in solution.assign if not 0 <= vid < inst.n]
    if unknown:
        return problems + [f"unknown vessels {unknown[:5]}"]

    length = {v.vid: v.length for v in inst.vessels}
    for vid, a in solution.assign.items():
        v = inst.vessels[vid]
        if a.berth < 0 or a.berth + v.length > inst.quay:
            problems.append(f"{v.name} at position {a.berth} leaves the quay")
        if a.start < v.est:
            problems.append(f"{v.name} starts at {a.start}, before EST {v.est}")
        if a.finish > inst.horizon:
            problems.append(f"{v.name} finishes at {a.finish}, beyond the horizon")
        plan = solution.cranes.get(vid, {})
        if set(plan) != set(range(a.start, a.finish)):
            problems.append(f"{v.name} has no crane plan for every period it occupies")
            continue
        for t, q in plan.items():
            if q < v.r_min or q > v.r_max:
                problems.append(f"{v.name} has {q} cranes at {t}, outside "
                                f"[{v.r_min}, {v.r_max}]")
                break
        capacity = sum(q ** inst.alpha for q in plan.values())
        need = effective_demand(inst, v, a.berth)
        if capacity < need - 1e-9:
            problems.append(f"{v.name} gets {capacity:.3f} effective crane-hours, "
                            f"needs {need:.3f}")

    busy = [0] * inst.horizon
    for vid, plan in solution.cranes.items():
        for t, q in plan.items():
            if 0 <= t < inst.horizon:
                busy[t] += q
    for t, used in enumerate(busy):
        if used > inst.cranes:
            problems.append(f"{used} cranes at {t}, terminal has {inst.cranes}")
            break

    for t in range(inst.horizon):
        spans = sorted((solution.assign[vid].berth,
                        solution.assign[vid].berth + length[vid])
                       for vid in solution.assign
                       if solution.assign[vid].start <= t < solution.assign[vid].finish)
        for (lo1, hi1), (lo2, _hi2) in zip(spans, spans[1:]):
            if lo2 < hi1:
                problems.append(f"two vessels overlap on the quay at {t}: "
                                f"[{lo1}, {hi1}) and [{lo2}, ...)")
                break
        else:
            continue
        break

    return problems


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------

def make_instance(name: str = "bacap_20", n_vessels: int = 20,
                  seed: int = 20261014, quay: int = QUAY_SECTIONS,
                  cranes: int = N_CRANES, horizon: int = HORIZON,
                  classes: dict[str, VesselClass] | None = None) -> BACAPInstance:
    """Reconstruct an instance from the published distributions.

    The 30 original instances are not public, so this generates vessels from the
    published class table and mix (60/30/10), and records in `meta` that it is a
    literature reconstruction. ETA is drawn from the part of the horizon from which
    the vessel can still be served, EST is 0.9 ETA, EFT is ETA plus the shortest
    handling time, LFT is ETA plus 1.5 times it.
    """
    table = classes or VESSEL_CLASSES
    rng = random.Random(seed)
    kinds = list(table)
    # A caller-supplied class table need not use the published names, in which
    # case every class is drawn with equal weight; the mix only applies to the
    # three published classes it describes.
    mix = dict(CLASS_MIX)
    weights = [mix.get(k, 1.0) for k in kinds]

    vessels: list[BACAPVessel] = []
    for i in range(n_vessels):
        kind = rng.choices(kinds, weights=weights)[0]
        spec = table[kind]
        length = rng.randint(*spec.length)
        demand = rng.randint(*spec.demand)
        r_min, r_max = spec.cranes
        d_min = math.ceil(demand / r_max ** ALPHA)
        latest_eta = max(0, horizon - d_min)
        eta = rng.randint(0, latest_eta)
        vessels.append(BACAPVessel(
            vid=i, name=f"V{i + 1:02d}", kind=kind, length=length,
            eta=eta, est=math.floor(0.9 * eta), eft=eta + d_min,
            lft=eta + math.ceil(1.5 * d_min), demand=demand,
            r_min=r_min, r_max=r_max,
            b0=rng.randint(0, max(0, quay - length)),
            c1=spec.c1, c2=spec.c2, c3=spec.c3))

    return BACAPInstance(
        name=name, vessels=vessels, quay=quay, cranes=cranes, horizon=horizon,
        meta={
            "generator": "make_instance",
            "source": "Iris, Pacino & Ropke (2017), Section 6.1 distributions",
            "instance_kind": "literature reconstruction, not a Meisel & Bierwirth "
                             "(2009) original",
            "seed": seed,
            "class_mix": dict(CLASS_MIX),
            "quay_sections": quay, "section_m": 10,
            "cranes": cranes, "horizon_hours": horizon,
            "alpha": ALPHA, "beta": BETA, "c4": C4,
        })


# ---------------------------------------------------------------------------
# Candidate assignments
# ---------------------------------------------------------------------------

def assignment_cost(inst: BACAPInstance, vid: int, a: Assignment) -> float:
    """The vessel's own objective contribution (1), without the ALNS penalty."""
    v = inst.vessels[vid]
    return (v.c1 * max(v.eta - a.start, 0)
            + v.c2 * max(a.finish - v.eft, 0)
            + (v.c3 if a.finish > v.lft else 0.0)
            + C4 * a.config.crane_hours)


def generate_candidates(inst: BACAPInstance, max_per_vessel: int | None = 120
                        ) -> list[list[Assignment]]:
    """Candidate assignments per vessel, cheapest first.

    An assignment is a (position, base crane count, start) triple, with the
    handling time that base crane count implies. Only the shortest duration per
    (position, base crane count) is kept, which dominates the longer ones, and the
    cost-ordered prefix stands in for the paper's candidate reduction.
    """
    out: list[list[Assignment]] = []
    for v in inst.vessels:
        rows: list[tuple[float, int, int, QCConfig]] = []
        for berth in range(0, inst.quay - v.length + 1):
            for q_hat in range(v.r_min, v.r_max + 1):
                process = min_process(inst, v, berth, q_hat)
                if process is None:
                    continue
                config = qc_pattern(inst, v, berth, process)
                if config is None or config.q_hat != q_hat:
                    continue
                latest = inst.horizon - process
                if latest < v.est:
                    continue
                for start in range(v.est, latest + 1):
                    a = Assignment(berth=berth, start=start, config=config)
                    rows.append((assignment_cost(inst, v.vid, a),
                                 berth, start, config))
        rows.sort(key=lambda row: (row[0], row[1], row[2]))
        if max_per_vessel is not None:
            rows = rows[:max_per_vessel]
        slots = [Assignment(berth=berth, start=start, config=config)
                 for (_cost, berth, start, config) in rows]
        if not slots:
            raise RuntimeError(
                f"{v.name} has no feasible assignment: the instance admits no plan")
        out.append(slots)
    return out


# ---------------------------------------------------------------------------
# ALNS (Algorithms 1-4)
# ---------------------------------------------------------------------------

DESTROY_P_MIN, DESTROY_P_MAX = 0.6, 1.0
INSERTION_P_MIN, INSERTION_P_MAX = 0.25, 1.0


@dataclass
class BACAPConfig:
    """Parameters, defaulting to the paper's tuned vector of Section 6.1.1."""
    max_iterations: int = 2000
    phi_r: int = 4
    shaw_a: float = 0.01
    shaw_b: float = 2.0
    shaw_c: float = 0.01
    start_pct: float = 0.05
    end_pct: float = 0.0005
    reheat_period: int = 30000
    rho1: float = 2.0
    rho2: float = 5.0
    rho3: float = 10.0
    gamma: float = 0.8
    update_interval: int = 1000
    max_candidates: int | None = 120


@dataclass
class BACAPResult:
    solution: BACAPSolution
    objective: float
    iterations: int
    feasible: bool
    unassigned: tuple[int, ...]
    wall_time: float
    operators: dict


class _Operators:
    """Weights, scores and counts of one operator family (Section 5)."""

    def __init__(self, names) -> None:
        self.names = list(names)
        self.weights = {n: 1.0 for n in self.names}
        self.scores = {n: 0.0 for n in self.names}
        self.used = {n: 0 for n in self.names}

    def pick(self, rng: random.Random) -> str:
        total = sum(self.weights.values())
        if total <= 0.0:
            return rng.choice(sorted(self.names))
        draw = rng.random() * total
        acc = 0.0
        for name in sorted(self.names):
            acc += self.weights[name]
            if draw < acc:
                return name
        return sorted(self.names)[-1]

    def update(self, gamma: float) -> None:
        for name in self.names:
            count = self.used[name]
            if count:
                self.weights[name] = ((1.0 - gamma) * self.weights[name]
                                      + gamma * self.scores[name] / count)
            self.scores[name] = 0.0
            self.used[name] = 0


def _remove_count(solution: BACAPSolution, config: BACAPConfig,
                  rng: random.Random) -> int:
    size = len(solution.assign)
    if size <= 1:
        return size
    upper = max(2, int(size / config.phi_r))
    lower = max(2, upper // 2)
    return min(rng.randint(lower, upper), size)


def _relatedness(solution: BACAPSolution, i: int, j: int,
                 config: BACAPConfig) -> float:
    """The paper's (48): a|b_i - b_j| + b|s_i - s_j| + c|e_i - e_j|."""
    a, b = solution.assign[i], solution.assign[j]
    return (config.shaw_a * abs(a.berth - b.berth)
            + config.shaw_b * abs(a.start - b.start)
            + config.shaw_c * abs(a.finish - b.finish))


def _time_related(solution: BACAPSolution, i: int, j: int) -> bool:
    """``i`` and ``j`` are worked at the same time (different positions)."""
    a, b = solution.assign[i], solution.assign[j]
    return a.start < b.finish and b.start < a.finish


def _berth_related(solution: BACAPSolution, i: int, j: int) -> bool:
    """``i`` and ``j`` share quay space at different times."""
    a, b = solution.assign[i], solution.assign[j]
    if a.start == b.start:
        return False
    length_i = solution.inst.vessels[i].length
    length_j = solution.inst.vessels[j].length
    return a.berth < b.berth + length_j and b.berth < a.berth + length_i


def destroy_shaw(solution: BACAPSolution, config: BACAPConfig,
                 rng: random.Random) -> list[int]:
    """Shaw removal: prefer assignments related to a randomly chosen one."""
    order = solution.assigned
    if not order:
        return []
    phi = _remove_count(solution, config, rng)
    chosen = [rng.choice(order)]
    p = rng.uniform(DESTROY_P_MIN, DESTROY_P_MAX)
    while len(chosen) < phi:
        remaining = [i for i in order if i not in set(chosen)]
        if not remaining:
            break
        remaining.sort(key=lambda j: (_relatedness(solution, chosen[0], j, config), j))
        cand = remaining.pop(0)
        if len(remaining) < phi - len(chosen):
            p = DESTROY_P_MAX
        if rng.random() <= p:
            chosen.append(cand)
        else:
            p = min(DESTROY_P_MAX, p + (DESTROY_P_MAX - p) / 2.0)
    return chosen


def _cost_related(solution: BACAPSolution, config: BACAPConfig, rng: random.Random,
                  related) -> list[int]:
    order = solution.assigned
    if not order:
        return []
    phi = _remove_count(solution, config, rng)
    order.sort(key=lambda i: (-solution.cost_of(i), i))
    chosen: list[int] = []
    while len(chosen) < phi:
        seed = next((i for i in order if i not in chosen), None)
        if seed is None:
            break
        chosen.append(seed)
        for j in order:
            if len(chosen) >= phi:
                break
            if j not in chosen and related(solution, seed, j):
                chosen.append(j)
    return chosen[:phi]


def destroy_cost_time(solution: BACAPSolution, config: BACAPConfig,
                      rng: random.Random) -> list[int]:
    return _cost_related(solution, config, rng, _time_related)


def destroy_cost_berth(solution: BACAPSolution, config: BACAPConfig,
                       rng: random.Random) -> list[int]:
    return _cost_related(solution, config, rng, _berth_related)


def destroy_random(solution: BACAPSolution, config: BACAPConfig,
                   rng: random.Random) -> list[int]:
    order = solution.assigned
    if not order:
        return []
    phi = _remove_count(solution, config, rng)
    chosen = [rng.choice(order)]
    p = rng.uniform(DESTROY_P_MIN, DESTROY_P_MAX)
    while len(chosen) < phi:
        remaining = [i for i in order if i not in chosen]
        if not remaining:
            break
        cand = rng.choice(remaining)
        if any(_time_related(solution, cand, j) for j in chosen):
            chosen.append(cand)
        elif len(remaining) <= phi - len(chosen):
            chosen.append(cand)
        elif rng.random() <= p:
            chosen.append(cand)
        else:
            p = min(DESTROY_P_MAX, p + (DESTROY_P_MAX - p) / 2.0)
    return chosen


DESTROY_OPERATORS = {
    "shaw": destroy_shaw,
    "cost_time": destroy_cost_time,
    "cost_berth": destroy_cost_berth,
    "random": destroy_random,
}


def repair_greedy(solution: BACAPSolution, candidates: list[list[Assignment]],
                  order, rng: random.Random, p: float) -> list[int]:
    """Basic greedy insertion (Algorithm 3): first feasible candidate under ``p``."""
    stuck: list[int] = []
    for vid in order:
        placed = False
        for a in candidates[vid]:
            if p < 1.0 and rng.random() > p:
                continue
            if solution.insert(vid, a):
                placed = True
                break
        if not placed:
            stuck.append(vid)
    # A final deterministic pass places whatever the probabilistic one skipped,
    # so the stochastic variant differs in which candidate it settles on rather
    # than in how many vessels it strands.
    for vid in stuck:
        for a in candidates[vid]:
            if solution.insert(vid, a):
                break
    return [vid for vid in stuck if vid not in solution.assign]


def repair_smarter(solution: BACAPSolution, candidates: list[list[Assignment]],
                   order) -> list[int]:
    """Smarter insertion: evaluate every feasible candidate, fix the cheapest.

    The paper's smarter operator re-examines a candidate whose stored plan cannot
    be realised instead of dropping it. Here the plan is built on insertion, so
    every candidate that fits is already realised; the representable difference is
    choosing the best one instead of the first.
    """
    stuck: list[int] = []
    for vid in order:
        best: Assignment | None = None
        best_cost = math.inf
        for a in candidates[vid]:
            if not solution.position_free(vid, a):
                continue
            if solution.plan_for(a) is None:
                continue
            cost = assignment_cost(solution.inst, vid, a)
            if cost < best_cost:
                best, best_cost = a, cost
        if best is None:
            stuck.append(vid)
        else:
            solution.insert(vid, best)
    return stuck


INSERTION_OPERATORS = {"greedy": repair_greedy, "smarter": repair_smarter}


def _initial_solution(inst: BACAPInstance, candidates: list[list[Assignment]],
                      rng: random.Random) -> BACAPSolution:
    """Greedy construction in a random order (Section 5.1)."""
    solution = BACAPSolution(inst)
    order = list(range(inst.n))
    rng.shuffle(order)
    stuck = repair_greedy(solution, candidates, order, rng, 1.0)
    if stuck:
        # A second pass in a different order can place what the first stranded;
        # the order of insertion decides which quay space is taken first.
        for _ in range(4):
            if not stuck:
                break
            retry = BACAPSolution(inst)
            order = list(range(inst.n))
            rng.shuffle(order)
            stuck = repair_greedy(retry, candidates, order, rng, 1.0)
            if not stuck:
                solution = retry
    return solution


def _penalty(inst: BACAPInstance, base: float, unassigned: int) -> float:
    return max(1.0, base) * (inst.n + 1) * (unassigned + 1)


def alns(inst: BACAPInstance, config: BACAPConfig | None = None,
         seed: int = 20261014) -> BACAPResult:
    """Algorithm 1 over BACAP assignments, with the QC plan built on insertion."""
    config = config or BACAPConfig()
    rng = random.Random(seed)
    t0 = time.perf_counter()

    candidates = generate_candidates(inst, config.max_candidates)
    current = _initial_solution(inst, candidates, rng)
    current_obj = current.objective() + _penalty(inst, current.objective(),
                                                 inst.n - len(current.assign))
    # The incumbent is the best *complete* plan. An incomplete plan can be
    # cheaper -- it services fewer vessels -- so it must not be allowed to win by
    # default: that is exactly how a run reports a small objective with half the
    # fleet missing.
    best: BACAPSolution | None = None
    best_obj = math.inf
    if len(current.assign) == inst.n:
        best, best_obj = current.copy(), current.objective()

    destroy_state = _Operators(DESTROY_OPERATORS)
    insert_state = _Operators(INSERTION_OPERATORS)

    temperature = config.start_pct * max(1.0, current.objective())
    cooling = ((config.end_pct / config.start_pct) ** (1.0 / max(1, config.reheat_period))
               if config.start_pct > 0 else 1.0)

    for it in range(config.max_iterations):
        d_name = destroy_state.pick(rng)
        i_name = insert_state.pick(rng)
        destroy_state.used[d_name] += 1
        insert_state.used[i_name] += 1

        cand = current.copy()
        removed = DESTROY_OPERATORS[d_name](cand, config, rng)
        cand.remove(removed)
        order = list(removed)
        rng.shuffle(order)
        if i_name == "greedy":
            p = rng.uniform(INSERTION_P_MIN, INSERTION_P_MAX)
            stuck = repair_greedy(cand, candidates, order, rng, p)
        else:
            stuck = repair_smarter(cand, candidates, order)

        real = cand.objective()
        scored = real + _penalty(inst, real, len(stuck))
        gain = 0.0
        accepted = False
        if not stuck and real < best_obj - 1e-9:
            best, best_obj = cand.copy(), real
            gain = config.rho1
            accepted = True
        elif scored < current_obj - 1e-9:
            gain = config.rho2
            accepted = True
        elif rng.random() < math.exp(-(scored - current_obj) / max(temperature, 1e-9)):
            gain = config.rho3
            accepted = True
        if accepted:
            current, current_obj = cand, scored
        destroy_state.scores[d_name] += gain
        insert_state.scores[i_name] += gain

        temperature *= cooling
        if config.reheat_period and (it + 1) % config.reheat_period == 0:
            temperature = config.start_pct * max(1.0, current.objective())
        if config.update_interval and (it + 1) % config.update_interval == 0:
            destroy_state.update(config.gamma)
            insert_state.update(config.gamma)

    if best is None:
        # No complete plan was ever found: report the best partial one, and say so
        # through `feasible`/`unassigned` rather than by a small objective.
        best = current.copy()
        best_obj = current.objective()
    unassigned = tuple(sorted(set(range(inst.n)) - set(best.assign)))
    return BACAPResult(
        solution=best, objective=best_obj, iterations=config.max_iterations,
        feasible=not unassigned, unassigned=unassigned,
        wall_time=time.perf_counter() - t0,
        operators={
            "destroy": dict(destroy_state.weights),
            "insertion": dict(insert_state.weights),
        })


def run_alns(inst: BACAPInstance, seed: int = 20261014,
             max_iterations: int = 2000) -> BACAPResult:
    """Convenience entry point with a bounded iteration count."""
    return alns(inst, BACAPConfig(max_iterations=max_iterations), seed=seed)
