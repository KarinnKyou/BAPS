# -*- coding: utf-8 -*-
"""Instance model: vessels, berths, tidal windows, crane resources.

A single Instance type is shared by the exact models, the heuristics and the
dynamic extensions, so every method is compared on identical data.

Time is measured in integer periods of `dt` hours (0.5 by default), which keeps
the time-indexed formulations and the exhaustive verifier simple. Any result
derived from a discrete model should state `dt`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

DT_DEFAULT = 0.5          # hours per period
HORIZON_DEFAULT = 48      # planning horizon in periods (24 h)
# Horizon used to draw the fleet. Arrivals are drawn from a range that scales
# with the horizon, so a sweep over `slack` must reuse one fixed fleet.
BASE_HORIZON = 96


@dataclass
class Vessel:
    """A vessel. Field names follow port practice, so the model maps back to it."""
    vid: int
    name: str
    length: int            # length overall (m), i.e. quay length occupied
    arrive: int            # arrival period (earliest possible start)
    handle: int            # periods of berth occupancy
    priority: int = 1      # weight for multi-objective extensions

    @property
    def finish_min(self) -> int:
        """Earliest possible completion (start work on arrival)."""
        return self.arrive + self.handle


@dataclass
class Berth:
    """A discrete berth: a fixed stretch of the quay."""
    bid: int
    name: str
    length: int            # usable quay length (m)


@dataclass
class Instance:
    """A complete scheduling instance."""
    name: str
    vessels: list[Vessel]
    berths: list[Berth]
    horizon: int = HORIZON_DEFAULT
    dt: float = DT_DEFAULT
    events: list[dict] = field(default_factory=list)   # dynamic disruptions
    meta: dict = field(default_factory=dict)

    # tidal_windows[bid] = [(start, end), ...]: periods in which the berth
    # cannot be worked (low tide, maintenance, closure).
    tidal_windows: dict = field(default_factory=dict)
    # Max number of vessels handled simultaneously terminal-wide; None = no limit.
    quay_crane_limit: int | None = None
    # [(start, end, reduction), ...]: within [start, end) the usable crane count
    # drops by `reduction`. A time window rather than a permanent cut, because
    # with two cranes a permanent cut leaves the instance infeasible.
    crane_failures: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.vessels)

    @property
    def m(self) -> int:
        return len(self.berths)

    def __post_init__(self) -> None:
        """Enforce that a vessel's `vid` is its index in `vessels`.

        Across the package a vessel is reached as `vessels[vid]` -- in both exact
        models, the heuristics, the enumerator, the metrics and the dynamics. That
        only works while `vid == index`, and an instance built any other way would
        fail deep inside a solve with a KeyError or, worse, be scored against the
        wrong vessel. Berth ids are compared against `berths[bid]` the same way.
        """
        for position, v in enumerate(self.vessels):
            if v.vid != position:
                raise ValueError(
                    f"vessel at position {position} has vid {v.vid}; the package "
                    f"requires vid == index, so renumber the vessels 0..n-1")
        for position, b in enumerate(self.berths):
            if b.bid != position:
                raise ValueError(
                    f"berth at position {position} has bid {b.bid}; the package "
                    f"requires bid == index, so renumber the berths 0..m-1")

    def blocked(self, bid: int, t: int) -> bool:
        """Whether berth `bid` cannot be worked in period `t`."""
        for (a, b) in self.tidal_windows.get(str(bid), self.tidal_windows.get(bid, [])):
            if a <= t < b:
                return True
        return False

    def blocked_periods(self, bid: int) -> int:
        return sum(1 for t in range(self.horizon) if self.blocked(bid, t))

    def crane_capacity(self, t: int) -> int | None:
        """Usable cranes in period `t`, after failure windows; None = unlimited."""
        if self.quay_crane_limit is None:
            return None
        cap = self.quay_crane_limit
        for (a, b, red) in self.crane_failures:
            if a <= t < b:
                cap = min(cap, max(0, self.quay_crane_limit - red))
        return cap

    def min_crane_capacity(self) -> int | None:
        """Smallest usable crane count over the horizon."""
        caps = [self.crane_capacity(t) for t in range(self.horizon)]
        caps = [c for c in caps if c is not None]
        return min(caps) if caps else None

    def max_handle(self) -> int:
        return max(v.handle for v in self.vessels)

    def total_length(self) -> int:
        return sum(v.length for v in self.vessels)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "horizon": self.horizon,
            "dt": self.dt,
            "vessels": [asdict(v) for v in self.vessels],
            "berths": [asdict(b) for b in self.berths],
            "events": self.events,
            "meta": self.meta,
            "tidal_windows": self.tidal_windows,
            "quay_crane_limit": self.quay_crane_limit,
            "crane_failures": self.crane_failures,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        return cls(
            name=d["name"],
            vessels=[Vessel(**v) for v in d["vessels"]],
            berths=[Berth(**b) for b in d["berths"]],
            horizon=d.get("horizon", HORIZON_DEFAULT),
            dt=d.get("dt", DT_DEFAULT),
            events=d.get("events", []),
            meta=d.get("meta", {}),
            tidal_windows=d.get("tidal_windows", {}),
            quay_crane_limit=d.get("quay_crane_limit", None),
            crane_failures=d.get("crane_failures", []),
        )

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dumps(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Instance":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> list[str]:
        """Report reasons the instance cannot admit any feasible schedule."""
        problems: list[str] = []
        if not self.vessels:
            problems.append("no vessels")
        if not self.berths:
            problems.append("no berths")
        max_berth = max((b.length for b in self.berths), default=0)
        for v in self.vessels:
            if v.length > max_berth:
                problems.append(
                    f"vessel {v.name} length {v.length} exceeds largest berth "
                    f"{max_berth}, infeasible"
                )
            if v.arrive + v.handle > self.horizon:
                problems.append(
                    f"vessel {v.name} arrival {v.arrive} + handling {v.handle} "
                    f"exceeds horizon {self.horizon}, would be truncated"
                )
        # Length demand cannot exceed total berth length times the horizon.
        cap = sum(b.length for b in self.berths) * self.horizon
        demand = sum(v.length * v.handle for v in self.vessels)
        if demand > cap:
            problems.append(
                f"total demand {demand} (length x handling) > capacity {cap}, "
                f"infeasible"
            )

        # Every vessel needs at least one start period clear of tidal blocks.
        for v in self.vessels:
            ok = False
            for k in range(self.m):
                for st in range(v.arrive, self.horizon - v.handle + 1):
                    if all(not self.blocked(k, t) for t in range(st, st + v.handle)):
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                problems.append(
                    f"vessel {v.name} has no unblocked handling window on any berth, "
                    f"infeasible"
                )

        if self.quay_crane_limit is not None and self.quay_crane_limit < 1:
            problems.append(f"crane limit {self.quay_crane_limit} < 1, infeasible")

        return problems

    def summary(self) -> str:
        extra = []
        if self.tidal_windows:
            nb = sum(self.blocked_periods(k) for k in range(self.m))
            extra.append(f"tidal blocks {nb} berth-periods")
        if self.quay_crane_limit is not None:
            extra.append(f"crane limit {self.quay_crane_limit}")
        tail = ("  |  " + ", ".join(extra)) if extra else ""
        return (
            f"instance {self.name}: {self.n} vessels x {self.m} berths | "
            f"horizon {self.horizon}x{self.dt}h = {self.horizon*self.dt:.1f}h | "
            f"{len(self.events)} disruption events{tail}"
        )


def make_toy_instance(
    name: str = "toy_5x3",
    n_vessels: int = 5,
    n_berths: int = 3,
    berth_length: int | None = None,
    horizon: int = 48,
    seed: int = 20261014,
    *,
    tight: bool = False,
) -> Instance:
    """Build a toy instance.

    With tight=True the berth length is derived from the longest vessel so that
    quay capacity actually binds: ceil(1.5 * max vessel length) lets any single
    vessel berth but keeps two large ones from sharing one berth. Simply halving
    a longer berth instead would make the instance infeasible.

    With berth_length=None the default is 2.5 * the longest vessel, which leaves
    quay capacity slack.
    """
    import math
    import random

    rng = random.Random(seed)

    vessels = []
    for i in range(n_vessels):
        length = rng.choice([120, 150, 180, 200, 240])
        arrive = rng.randint(0, max(1, horizon // 4))
        handle = rng.randint(4, 12)        # periods, i.e. 2-6 h at dt=0.5
        vessels.append(
            Vessel(vid=i, name=f"V{i+1:02d}", length=length,
                   arrive=arrive, handle=handle, priority=rng.randint(1, 3))
        )

    max_len = max(v.length for v in vessels)
    if berth_length is None:
        factor = 1.5 if tight else 2.5
        berth_length = int(math.ceil(factor * max_len))

    berths = [
        Berth(bid=k, name=f"B{k+1}", length=berth_length) for k in range(n_berths)
    ]

    return Instance(
        name=name, vessels=vessels, berths=berths, horizon=horizon,
        meta={"generator": "make_toy_instance", "seed": seed,
              "tight": tight, "berth_length": berth_length},
    )


def make_hard_instance(
    name: str = "hard_10x3",
    n_vessels: int = 10,
    n_berths: int = 3,
    horizon: int | None = None,
    seed: int = 20261014,
    *,
    tidal: bool = True,
    tidal_period: int = 24,
    tidal_block: int = 8,
    crane_limit: int | None = 2,
    slack: float = 1.0,
) -> Instance:
    """Build an instance with business constraints and a controllable horizon.

    The business constraints are not what makes this instance hard: with a loose
    horizon the earliest-available-berth greedy rule is already optimal whatever
    the constraints are. Difficulty comes from tight slack, defined as

        slack = horizon / lower_bound_makespan

    where the lower bound is `max_j handling_j + ceil(sum_j handling_j / m)`:
    the last vessel to finish needs one full handling plus, on average, the
    berth-time of all the others. The horizon is derived from `slack` when not
    given, so `slack = 1.0` means the horizon equals that bound and the instance
    is infeasible by construction; larger slack is looser.

    Note that feasibility is not determined by `slack` alone. Tidal windows
    remove about `tidal_block / tidal_period` of every berth's time and the crane
    limit caps concurrency, so an instance can be infeasible at a slack well
    above 1.0 and feasible again only once the horizon accommodates the blocked
    periods. Check the exact model rather than assuming a threshold.

    The constraints enabled by default are tidal windows, blocking `tidal_block`
    periods out of every `tidal_period` on each berth with a per-berth phase,
    and a terminal-wide limit of `crane_limit` vessels handled at once.
    """
    # The vessel set must not depend on the horizon being swept. `arrive` is
    # drawn from a range that scales with the horizon, so generating the instance
    # afresh per horizon would compare different fleets and make feasibility
    # non-monotone in `slack`. Build the fleet once and retarget the horizon.
    base = make_toy_instance(name, n_vessels, n_berths, horizon=BASE_HORIZON,
                             seed=seed, tight=True)
    p_sum = sum(v.handle for v in base.vessels)
    p_max = max(v.handle for v in base.vessels)
    lower_bound = p_max + math.ceil(p_sum / max(1, n_berths))
    if horizon is None:
        horizon = max(int(math.ceil(lower_bound * slack)), p_max + 1)

    inst = make_toy_instance(name, n_vessels, n_berths, horizon=horizon,
                             seed=seed, tight=True)
    inst.vessels = base.vessels
    if tidal:
        windows = {}
        for k in range(n_berths):
            blocks = []
            phase = k * (tidal_period // max(1, n_berths))
            for start in range(phase, horizon, tidal_period):
                end = min(start + tidal_block, horizon)
                if start < end:
                    blocks.append((start, end))
            windows[k] = blocks
        inst.tidal_windows = {str(k): v for k, v in windows.items()}
    inst.quay_crane_limit = crane_limit
    inst.meta.update({"generator": "make_hard_instance", "seed": seed,
                      "tidal": tidal, "tidal_period": tidal_period,
                      "tidal_block": tidal_block, "crane_limit": crane_limit,
                      "slack": slack, "horizon": horizon,
                      "lower_bound_makespan": lower_bound,
                      "horizon_over_lower_bound": round(horizon / lower_bound, 3)})
    return inst


@dataclass
class Schedule:
    """A feasible or infeasible solution: assign[v] = (berth_id, start_period)."""
    assign: dict[int, tuple[int, int]]
    objective: float | None = None
    feasible: bool = True
    violations: list[str] = field(default_factory=list)
    solver: str = ""
    wall_time: float = 0.0
    extra: dict = field(default_factory=dict)

    def completion(self, inst: Instance) -> dict[int, int]:
        """Completion time of each vessel.

        The explicit coverage check keeps an incomplete assignment reported as a
        construction failure instead of a bare KeyError.
        """
        missing = [v.vid for v in inst.vessels if v.vid not in self.assign]
        if missing:
            known = {v.vid: v.name for v in inst.vessels}
            names = [known[m] for m in missing]
            raise ValueError(
                f"incomplete schedule: {len(missing)} vessels unassigned {names}. "
                f"This normally means the constructive heuristic failed to place "
                f"them, not that the instance is infeasible.")
        return {v.vid: self.assign[v.vid][1] + v.handle for v in inst.vessels}

    def check(self, inst: Instance) -> "Schedule":
        """Verify feasibility independently of the model's constraint code."""
        viol: list[str] = []
        cmap = {v.vid: v for v in inst.vessels}
        bmap = {b.bid: b for b in inst.berths}

        # A schedule that does not mention every vessel is not feasible. Without
        # this, an empty assignment passes every remaining check vacuously.
        unassigned = [v.name for v in inst.vessels if v.vid not in self.assign]
        if unassigned:
            viol.append(f"{len(unassigned)} vessels unassigned: {unassigned[:5]}")

        unknown = [vid for vid in self.assign if vid not in cmap]
        if unknown:
            # The deeper checks index vessels by vid, so stop here rather than
            # raising: an unknown vessel is itself the finding.
            viol.append(f"assignment names unknown vessels {sorted(unknown)[:5]}")
            self.violations, self.feasible = viol, False
            return self

        for vid, (bid, st) in self.assign.items():
            v, b = cmap[vid], bmap.get(bid)
            if b is None:
                viol.append(f"vessel V{vid} assigned to unknown berth {bid}")
                continue
            if st < v.arrive:
                viol.append(f"vessel {v.name} starts at {st} before arrival {v.arrive}")
            if st + v.handle > inst.horizon:
                viol.append(
                    f"vessel {v.name} completes at {st+v.handle} "
                    f"beyond horizon {inst.horizon}"
                )

        # Quay length available on each berth, period by period.
        for b in inst.berths:
            occ = [0] * inst.horizon
            for vid, (bid, st) in self.assign.items():
                if bid != b.bid or vid not in cmap:
                    continue          # unknown vessels are reported above
                v = cmap[vid]
                for t in range(st, min(st + v.handle, inst.horizon)):
                    occ[t] += v.length
            over = [t for t, o in enumerate(occ) if o > b.length]
            if over:
                worst = max(occ)
                viol.append(
                    f"berth {b.name} exceeds capacity in {len(over)} periods "
                    f"(peak {worst} > capacity {b.length}), first at t={over[0]}"
                )

        # Whole-berth occupancy: one vessel at a time per berth. Checking the
        # summed vessel lengths instead would let two short vessels share a berth,
        # which the exact model does not permit.
        for b in inst.berths:
            spans = sorted((st, st + cmap[vid].handle, vid)
                           for vid, (bid, st) in self.assign.items() if bid == b.bid)
            for (s1, e1, v1), (s2, _e2, v2) in zip(spans, spans[1:]):
                if s2 < e1:
                    viol.append(
                        f"berth {b.name}: {cmap[v1].name} and {cmap[v2].name} "
                        f"occupy the berth simultaneously at t={s2}")

        # Tidal windows: blocked berth periods cannot be worked.
        for vid, (bid, st) in self.assign.items():
            v = cmap[vid]
            hit = [t for t in range(st, min(st + v.handle, inst.horizon))
                   if inst.blocked(bid, t)]
            if hit:
                viol.append(
                    f"vessel {v.name} on berth {bmap[bid].name} crosses a tidal "
                    f"block ({len(hit)} periods, first at t={hit[0]})"
                )

        # Crane limit, including failure windows.
        if inst.quay_crane_limit is not None:
            busy = [0] * inst.horizon
            for vid, (bid, st) in self.assign.items():
                v = cmap[vid]
                for t in range(st, min(st + v.handle, inst.horizon)):
                    busy[t] += 1
            over = []
            for t, c in enumerate(busy):
                cap = inst.crane_capacity(t)
                if cap is not None and c > cap:
                    over.append(t)
            if over:
                viol.append(
                    f"crane limit exceeded in {len(over)} periods "
                    f"(peak {max(busy)}, minimum usable {inst.min_crane_capacity()}, "
                    f"first at t={over[0]})"
                )

        self.violations = viol
        self.feasible = not viol
        return self

    def compute_objective(self, inst: Instance, objective: str = "sum_C") -> float:
        """Compute the objective.

        sum_C: minimise total completion time sum_i C_i (classic BAP objective).
        C_max: minimise the makespan.
        """
        comp = self.completion(inst)
        if objective == "sum_C":
            val: float = float(sum(comp.values()))
        elif objective == "C_max":
            val = float(max(comp.values()))
        else:
            raise ValueError(f"unknown objective: {objective}")
        self.objective = val
        return val

    def metrics(self, inst: Instance) -> dict:
        cmap = {v.vid: v for v in inst.vessels}
        comp = self.completion(inst)
        wait = {vid: self.assign[vid][1] - cmap[vid].arrive for vid in self.assign}
        total_berth_cap = sum(b.length for b in inst.berths) * inst.horizon
        used = sum(cmap[vid].length * cmap[vid].handle for vid in self.assign)
        return {
            "n_vessels": inst.n,
            "sum_completion": sum(comp.values()),
            "C_max": max(comp.values()),
            "sum_waiting": sum(wait.values()),
            "mean_waiting": sum(wait.values()) / max(1, len(wait)),
            "berth_utilization": used / total_berth_cap if total_berth_cap else 0.0,
            "feasible": self.feasible,
            "violations": len(self.violations),
            "solver": self.solver,
            "wall_time_s": round(self.wall_time, 4),
            "objective": self.objective,
        }


def save_instances(out_dir: str | Path, insts: list[Instance]) -> list[Path]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for inst in insts:
        p = d / f"{inst.name}.json"
        inst.save(p)
        paths.append(p)
    return paths
