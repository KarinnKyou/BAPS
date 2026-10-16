"""Disruption events and the rolling-horizon scheduler.

Events are arrival delay, handling overrun, tidal shift and crane loss, with
amplitudes bounded so the instance stays feasible. The scheduler reveals
events only up to the current period and plans the rest from predictions.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

from .instances import Instance, Schedule


# Disruption events

@dataclass
class Event:
    """A single disruption event."""
    kind: str                 # arrival_delay, handle_overrun, tidal_shift, crane_loss
    period: int
    target: int | None = None  # vessel vid, or berth bid; None means global
    delta: int = 0            # delay, overrun, blocked length or cranes lost
    note: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "period": self.period, "target": self.target,
                "delta": self.delta, "note": self.note}


def make_events(inst: Instance, rng: random.Random, *,
                n_arrival: int = 3, n_overrun: int = 2,
                tidal_shift: bool = True, crane_loss: bool = True,
                max_delay: int = 8, max_overrun: int = 4) -> list[Event]:
    """Generate disruption events; bounded amplitudes keep the instance feasible."""
    events: list[Event] = []
    vids = list(range(inst.n))

    for vid in rng.sample(vids, min(n_arrival, len(vids))):
        events.append(Event("arrival_delay", period=0, target=vid,
                            delta=rng.randint(1, max_delay),
                            note="arrival delay (weather or port congestion)"))

    for vid in rng.sample(vids, min(n_overrun, len(vids))):
        events.append(Event("handle_overrun", period=rng.randint(0, inst.horizon // 3),
                            target=vid, delta=rng.randint(1, max_overrun),
                            note="handling overrun (equipment failure or cargo)"))

    if tidal_shift and inst.tidal_windows:
        bid = rng.randrange(inst.m)
        events.append(Event("tidal_shift", period=rng.randint(0, inst.horizon // 2),
                            target=bid, delta=rng.choice([-3, 3]),
                            note="low-tide window shifted"))

    if crane_loss and inst.quay_crane_limit is not None:
        events.append(Event("crane_loss", period=rng.randint(0, inst.horizon // 2),
                            target=None, delta=rng.choice([6, 8, 10]),
                            note="crane failure (one crane unavailable)"))

    events.sort(key=lambda e: e.period)
    return events


def apply_events(inst: Instance, events: list[Event],
                 upto: int | None = None) -> Instance:
    """Apply events to a copy of the instance, never mutating the input.

    ``upto`` keeps only events already revealed, modelling partial information.
    """
    new = copy.deepcopy(inst)
    new.events = [e.to_dict() for e in events]

    for e in events:
        if upto is not None and e.period > upto:
            continue
        if e.kind == "arrival_delay":
            v = new.vessels[e.target]
            v.arrive = v.arrive + e.delta
        elif e.kind == "handle_overrun":
            v = new.vessels[e.target]
            v.handle = v.handle + e.delta
        elif e.kind == "tidal_shift":
            key = str(e.target)
            blocks = new.tidal_windows.get(key, [])
            shifted = []
            for (a, b) in blocks:
                a2, b2 = a + e.delta, b + e.delta
                if b2 <= 0 or a2 >= new.horizon:
                    continue
                shifted.append((max(0, a2), min(new.horizon, b2)))
            new.tidal_windows[key] = shifted
        elif e.kind == "crane_loss":
            # A crane loss is a temporary window, not a permanent limit cut,
            # which would make the instance infeasible. `delta` is its duration.
            if new.quay_crane_limit is not None:
                dur = max(1, e.delta if e.delta else 8)
                new.crane_failures = list(new.crane_failures) + [
                    (e.period, min(e.period + dur, new.horizon), 1)
                ]
        else:
            raise ValueError(f"unknown event kind: {e.kind}")

    # Grow the horizon so truncation alone cannot make the instance infeasible.
    need = max((v.arrive + v.handle for v in new.vessels), default=0)
    for (a, b, _r) in new.crane_failures:
        need = max(need, b)
    if need > new.horizon:
        new.horizon = need + 2
    return new


def observed_arrivals(inst_true: Instance, planned: Instance,
                      now: int) -> dict[int, int]:
    """Realised arrivals for vessels already due, planned values for the rest.

    A vessel counts as observed once its scheduled arrival has passed. Testing the
    realised arrival against `now` instead would hide a delay: a vessel due at 6
    but actually arriving at 15 is still reported as arriving at 6 until t=15.
    """
    out = {}
    for v_true, v_plan in zip(inst_true.vessels, planned.vessels):
        if v_plan.arrive <= now:
            out[v_true.vid] = v_true.arrive
        else:
            out[v_plan.vid] = v_plan.arrive
    return out


# Rolling-horizon solver

@dataclass
class RollingResult:
    schedule: Schedule | None
    events: list[Event]
    reoptimizations: int = 0
    total_wall: float = 0.0
    log: list[dict] = field(default_factory=list)
    # Lock set for the final rebuild: {vid: (berth, start_lower_bound, fixed)}
    final_locks: dict = field(default_factory=dict)
    # Whether the final rebuild produced a schedule valid on the realised instance
    feasible: bool = False


def _ensure_horizon(view: Instance, locked: dict) -> Instance:
    """Widen the planning horizon so a commitment cannot strand a vessel.

    ``apply_events`` grows the horizon from the *predicted* arrivals, but the
    rolling loop then overwrites them with the observed ones, and a commitment's
    start lower bound can sit beyond the horizon as well. Either can leave a
    vessel with no feasible start period, which the model reports by raising
    rather than by returning an infeasible schedule.
    """
    need = max((v.arrive + v.handle for v in view.vessels), default=0)
    for (a, b, _r) in view.crane_failures:
        need = max(need, b)
    for vid, spec in locked.items():
        if len(spec) > 1 and isinstance(spec[1], int):
            need = max(need, spec[1] + view.vessels[vid].handle)
    if need > view.horizon:
        view.horizon = need + 2
    return view


def rolling_horizon(planned: Instance, true_inst: Instance, events: list[Event],
                    *, dt: int = 8, time_limit: float = 5.0) -> RollingResult:
    """Rolling-horizon scheduling with tiered commitment.

    A berth can be committed because a berthed vessel cannot move, but a start
    period can only be postponed, never advanced. Level one locks the berth and
    allows a start at or after a lower bound; level two locks the exact start
    for vessels that have actually begun handling. Commitment waits until the
    arrival has been observed, so no decision rests on information that may
    later prove wrong.
    """
    t0 = time.perf_counter()
    horizon_end = true_inst.horizon
    result = RollingResult(schedule=None, events=events)
    prev: Schedule | None = None
    locked: dict[int, tuple[int, int, bool]] = {}

    for now in range(0, horizon_end, dt):
        obs = observed_arrivals(true_inst, planned, now)
        view = apply_events(planned, events, upto=now)
        for v in view.vessels:
            v.arrive = obs.get(v.vid, v.arrive)

        # Berth first; the exact start is pinned only once handling has begun.
        # Pinning uses the realised arrival, not the planning view's prediction:
        # a vessel the view still expects to have arrived may in fact be late, so
        # its start has not happened and must not be frozen.
        if prev is not None and prev.assign:
            for vid, (bid, st) in prev.assign.items():
                real_arr = true_inst.vessels[vid].arrive
                if real_arr > now:
                    continue
                lb = max(st, real_arr)
                if vid not in locked and st <= now:
                    locked[vid] = (bid, lb, False)
                if vid in locked and not locked[vid][2] and st < now and real_arr < now:
                    locked[vid] = (bid, lb, True)

        # Commitments and observed delays can both push a vessel past the
        # horizon the view was built with.
        view = _ensure_horizon(view, locked)

        try:
            from .models.bap import BAPTimeIndexed
            sch = BAPTimeIndexed(view, time_limit=time_limit,
                                 lock=dict(locked)).solve()
            if not sch.feasible:
                # A later event can overturn a commitment; free the start and retry.
                sch = BAPTimeIndexed(view, time_limit=time_limit,
                                     lock=dict(locked), lock_start_free=True).solve()
                result.log.append({"now": now, "fallback": "lock_start_free",
                                   "objective": sch.objective,
                                   "feasible": sch.feasible, "n_locked": len(locked)})
        except Exception as e:                                   # noqa: BLE001
            result.log.append({"now": now, "error": f"{type(e).__name__}: {e}"})
            continue

        result.reoptimizations += 1
        prev = sch
        result.log.append({
            "now": now, "objective": sch.objective, "feasible": sch.feasible,
            "n_locked": len(locked),
            "n_locked_fixed": sum(1 for v in locked.values() if v[2]),
            "horizon": view.horizon,
        })
        if len(locked) >= true_inst.n and all(v[2] for v in locked.values()):
            break

    # Rebuild on the realised instance and the commitments: the last planning
    # view keeps predicted arrivals and can violate start >= arrival.
    final_locks = {vid: (bid, lb, False) for vid, (bid, lb, _f) in locked.items()}
    result.final_locks = final_locks
    try:
        from .models.bap import BAPTimeIndexed
        realised = _ensure_horizon(true_inst, final_locks)
        budget = max(time_limit, 30.0)

        final = BAPTimeIndexed(realised, time_limit=budget,
                               lock=final_locks).solve()
        kept = dict(final_locks)
        if not final.feasible:
            final = BAPTimeIndexed(realised, time_limit=budget,
                                   lock=final_locks, lock_start_free=True).solve()

        # A commitment to a berth cannot outlast that berth's availability. On a
        # tightly constrained instance the accumulated locks can leave no legal
        # assignment at all, and because locks are never released the planner
        # would then never recover. Release them one at a time, newest first --
        # the oldest commitments are the ones most defensible -- and report how
        # many survived.
        if not final.feasible:
            order = sorted(final_locks, key=lambda vid: final_locks[vid][1])
            for drop in range(1, len(order) + 1):
                trial = {vid: spec for vid, spec in final_locks.items()
                         if vid not in set(order[:drop])}
                attempt = BAPTimeIndexed(realised, time_limit=budget,
                                         lock=trial).solve()
                if not attempt.feasible:
                    attempt = BAPTimeIndexed(realised, time_limit=budget,
                                             lock=trial, lock_start_free=True).solve()
                if attempt.feasible:
                    final, kept, dropped = attempt, trial, drop
                    result.log.append({"now": "final_rebuild",
                                       "released_locks": drop})
                    break

        result.schedule = final if final.feasible else prev
        result.feasible = bool(final.feasible)
        result.final_locks = kept
        result.log.append({"now": "final_rebuild",
                           "objective": result.schedule.objective
                           if result.schedule else None,
                           "feasible": result.schedule.feasible
                           if result.schedule else False,
                           "n_locked": len(kept),
                           "n_committed_before": len(final_locks)})
    except Exception as e:                                       # noqa: BLE001
        result.log.append({"now": "final_rebuild",
                           "error": f"{type(e).__name__}: {e}"})
        result.schedule = prev
        result.feasible = False

    result.total_wall = time.perf_counter() - t0
    return result


def evaluate_schedule_on(inst: Instance, sch: Schedule) -> tuple[bool, float | None, list[str]]:
    """Cost a schedule on the realised instance, returning
    ``(feasible, objective, violations)``. Decisions are made on predictions
    but settled on reality; scoring a plan on its own predictions is circular.
    """
    if sch is None or not sch.assign:
        return False, None, ["no schedule"]
    from .instances import Schedule as S
    probe = S(assign=dict(sch.assign), solver=sch.solver)
    probe.check(inst)
    if not probe.feasible:
        return False, None, probe.violations
    probe.compute_objective(inst, "sum_C")
    return True, probe.objective, []
