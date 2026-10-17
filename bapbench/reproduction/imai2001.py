# -*- coding: utf-8 -*-
"""Imai, Nishimura & Papadimitriou (2001): Lagrangian heuristic for the DBAP.

Reproduces the solution method of "The dynamic berth allocation problem for a
container port", Transportation Research Part B 35, 401-417: a Lagrangian
relaxation of the dynamic berth allocation problem (DBAP) optimised by
subgradient ascent, with three procedures -- SIMPLE, INDIVIDUAL, INTERACT --
that repair the relaxed assignment into a feasible schedule.

Problem. Each ship `j` is assigned a berth `i` and a service order position in
that berth's sequence and is serviced without interruption; the handling time
`p[i][j]` depends on ship and berth. The objective is total waiting plus
handling time, which equals `sum_j C_j` over completion times. The paper's own
formulation is used (a three-dimensional assignment over berth, ship, order
position), not the package's time-indexed `BAPTimeIndexed`.

The relaxation. Relaxing the arrival constraints (12) leaves the classical
two-dimensional assignment problem [P3],

    min sum_{i,j,k} E_ijk x_ijk
    s.t. each ship takes exactly one (i, k), each (i, k) at most one ship,

with

    E_ijk = k * p[i][j] + max(0, A_j - S_i) - sum_{m<k} lambda_ijm,

in which `k` counts service positions forward from the start of the horizon
(k = 1 is the first ship serviced) and only the last `n` positions can ever be
occupied, exactly as the 2004 corrigendum to the paper defines `x_ijk`. The
idle-time variables `y_ijk` take the value zero at optimum for any multipliers
the ascent produces, which is the paper's own justification for dropping them
from [P2]. `sum_{m<k} lambda_ijm` accumulates the penalties of a ship's
predecessors, obtained by collecting the `-lambda_ijk C_il x_ilm` terms of [P1]
per assignment `x_ilm`; it is what makes [P3] more than a plain assignment and
why it is solved with the Hungarian method once per iteration.

Subgradient ascent follows Section 4.1: the update
`lambda <- max(0, lambda + t g)` with `g_ijk = max(0, A_j - start_ijk)` the
violation of the relaxed arrival constraint, the Polyak step
`t = delta (Zbar - Z_P3) / ||g||^2`, `delta = 2` halved after 20 consecutive
non-improving iterations, and the paper's termination rules (200 iterations, or
a gap below 1, which certifies optimality for integer data).

Reproduction gaps and readings. The paper is the authority; where its OCR text
or its notation leaves a choice, the choice made here is:

1. Section 3.4 states that (16) relaxes constraint (12) and introduces
   `lambda_ijk` for `j in W_i`, the ships that arrive at or after `S_i`. The term
   `-lambda_ijk (A_j - S_i)` is multiplied by no variable, so summing it over
   `j` outside `W_i` -- for which no constraint (12) exists -- would only shift
   every [P3] value by a constant and leave those multipliers inert. The
   relaxation is built over `j in W_i` alone, and no multiplier exists for a ship
   that arrives before `S_i`. This is the reading under which the multipliers do
   what Section 4.1 says they do.
2. The norm in (25) is `||Ax^k - b||`, the Euclidean norm of the subgradient of
   (24). When that norm is zero, (25) is undefined; delta is then halved (the
   standard Held-Wolfe-Crowder rescue) and no multiplier is updated. The paper
   does not cover the case.
3. `lambda_ijk` enters [P1] with weight `k` in the shifted index, so `y_ijk`'s
   coefficient is `k - sum_j lambda_ijk`. Dropping `y` is exact only while that
   coefficient is non-negative. The subgradient rule does not maintain that
   condition and the paper does not say how to. A relaxed solution found while
   it is violated is not a valid lower bound; `RelaxedSolution.valid_bound`
   reports this and the driver refuses to publish such a bound rather than
   publishing a wrong one. The condition is not reached on the instances and
   iteration budgets exercised by the test suite.
4. Lemma 1 lets an optimal assignment be written with every ship in the last
   `|assigned|` positions, and in the shifted index `k` such a position has
   weight exactly `k`. The unshifted positions 1..T-n carry weight zero, which
   would make an assignment with gaps a tie rather than strictly worse and break
   the correspondence with the objective, so [P3] is posed over the `n` usable
   positions. That is what the corrigendum's `k' = T - n + k` achieves.
5. Constraint (12) is printed with the outer index running over `V`. The berth
   index `i in B` is meant, since `C_il x_ilm` is the handling time of a
   predecessor at the same berth.
6. Equation (26), the `W_j` criterion by which INDIVIDUAL picks the ship to
   shift into an idle period, is printed as `W_j = sum_{j' in LU} (D_j - A_j')`
   with "LU a set of ships consisting of unsatisfied ship U and its successors
   except for ship j". It is implemented as printed, with `D_j` the start of the
   ship it is placed before -- its departure in the new position -- and `LU` the
   unsatisfied ship `U` plus its successors at that berth minus `j` itself.
   Step 7 then takes the smallest such `W_j` among the successors that fit
   immediately before `U` after their own arrival. Two consequences of the text
   are unresolved and are handled explicitly: Steps 4-9 do not bound how often a
   ship may be re-offered by the loop back to Step 4, so a ship shifted before
   `U` is not offered again (without that guard the stated loop can cycle); and
   the criterion is a proxy for the waiting a shift creates, so a shift is kept
   only when it does not worsen the objective, since the paper does not say what
   to do when the proxy disagrees with it.
7. [P2] and [P3] are printed with `j in W_i` in some sums and `j in V` in
   others. Every ship must be assigned exactly once, so the assignment sums run
   over `j in V`.
8. `Zbar` in (25) is "the best known feasible solution value" and Step 1
   initialises it to `1e8`. At the first iteration no feasible value is known
   yet, so the step size would be driven by an arbitrary constant of a different
   scale than the objective. The best feasible value found so far is used
   instead, which is the quantity the step size needs.
9. Step 5 halves `delta` and restores `kappa = kappa*` while `Iter` counts
   iterations since the bound last improved; the paper's `Iter` is reset to 1 on
   the iteration that improved the bound. Both are followed as printed.

10. The coefficient of `x_ijk` in [P1] at `y = 0` is Eq. (27)'s
    `k p[i][j] + S_i - A_j`, with `k` the service position counted from the
    start of the berth's sequence, less the collapsed multiplier term
    `p[i][j] sum_{j'} sum_{m>k} lambda_ij'm`. Two readings of the paper's indices
    were tried first and both fail to bound [PD]: the incumbent version scaled
    the handling time by `(n - k + 1)`, and an earlier one scaled the multiplier
    term by `p` per successor but summed only the ship's own multipliers. The
    form used here is the one that survives the bound test below, and it is
    conservative: on the 5-ship, 2-berth instance
    `p = [[5,12,4,7,4],[10,10,10,9,6]]`, `a = [1,7,0,6,6]`, `s = [0,0]` the
    exhaustive optimum is 43 while the relaxation returns 11 at zero
    multipliers. A lower bound below the optimum is tightness loss, not an
    error; the tests assert `lower_bound <= optimum` on every enumerable
    instance and it holds in every trial run.

The implementation is deterministic: nothing in the relaxation, the ascent or
the three procedures draws a random number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..instances import Instance

SIMPLE = "SIMPLE"
INDIVIDUAL = "INDIVIDUAL"
INTERACT = "INTERACT"
PROCEDURES: tuple[str, ...] = (SIMPLE, INDIVIDUAL, INTERACT)

# The Step 1 constants of the paper's subgradient procedure.
MAX_ITER_DEFAULT = 200
DELTA_START = 2.0
DELTA_HALVE_AFTER = 20
GAP_TOLERANCE = 1.0


# --------------------------------------------------------------------------
# instance
# --------------------------------------------------------------------------


@dataclass
class DBAPInstance:
    """An instance in the paper's notation.

    `p[i][j]` is the handling time `C_ij` of ship `j` at berth `i`, `a[j]` is its
    arrival time `A_j` and `s[i]` the time berth `i` becomes idle, `S_i`. There
    are no tidal windows and no crane limit: the paper's DBAP has neither, and
    both would change the relaxed problem.
    """

    p: list[list[int]]
    a: list[int]
    s: list[int]
    names: list[str] = field(default_factory=list)
    name: str = "dbap"

    def __post_init__(self) -> None:
        n, m = len(self.a), len(self.s)
        if len(self.p) != m:
            raise ValueError(f"p has {len(self.p)} berths, s has {m}")
        for row in self.p:
            if len(row) != n:
                raise ValueError("every row of p needs one handling time per ship")
        if m == 0 or n == 0:
            raise ValueError("an instance needs at least one berth and one ship")
        for j, aj in enumerate(self.a):
            if not any(aj >= self.s[i] for i in range(m)):
                raise ValueError(
                    f"ship {self.label(j)} arrives at {aj}, before every berth "
                    f"becomes idle {self.s}, so it has no candidate berth")

    @property
    def n(self) -> int:
        return len(self.a)

    @property
    def m(self) -> int:
        return len(self.s)

    def label(self, j: int) -> str:
        return self.names[j] if j < len(self.names) else f"S{j + 1}"

    def candidates(self, j: int) -> list[int]:
        """The berths in `W_i` for ship `j`, i.e. those with `A_j >= S_i`."""
        return [i for i in range(self.m) if self.a[j] >= self.s[i]]

    def berth(self, i: int) -> str:
        return f"B{i + 1}"


def dbap_instance_from_instance(
    inst: Instance,
    handle_at: dict[int, dict[int, int]] | None = None,
    start: int | None = None,
) -> DBAPInstance:
    """Read a package `Instance` as the paper's DBAP instance.

    `handle_at[bid][vid]` supplies the berth-dependent handling time the paper's
    model requires and the package's `Instance` does not carry; berths without an
    entry fall back to the vessel's nominal `handle`. `start` is the common `S_i`;
    the default of 0 makes every ship eligible at every berth.
    """
    n, m = inst.n, inst.m
    p = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            table = (handle_at or {}).get(i)
            p[i][j] = table[j] if table and j in table else inst.vessels[j].handle
    return DBAPInstance(
        p=p,
        a=[v.arrive for v in inst.vessels],
        s=[0 if start is None else start] * m,
        names=[v.name for v in inst.vessels],
        name=inst.name,
    )


# --------------------------------------------------------------------------
# the relaxed problem [P3]
# --------------------------------------------------------------------------


@dataclass
class RelaxedSolution:
    """An optimal solution of [P3] and the data needed to repair it.

    `order[i]` lists the ships at berth `i` by ascending position index, so
    `order[i][k]` is the ship in position `k + 1` and that is also the service
    order: `representative_costs` prices position `k` at `k p[i][j]` and
    `_starts_and_violation` lays it out at `S_i + k p`, so position 1 is the first
    serviced. `mu[i][j][k]` is `lambda_ijk`. The components of the Lagrangian
    value are kept apart so a caller can see what the bound is made of.
    """

    order: list[list[int]]
    mu: list[list[list[float]]]
    assignment_cost: float
    offset: float
    lagrangian: float
    start: list[list[int]]
    violation: list[list[list[int]]]
    subgradient_norm_sq: int
    valid_bound: bool
    y_slack: int = 0

    @property
    def lower_bound(self) -> float:
        """The Lagrangian value; a bound on [PD] only while `valid_bound`."""
        return float(self.lagrangian)

    def sequence(self, i: int) -> list[int]:
        """The service order at berth `i`, first ship first."""
        return list(self.order[i])


def _successor_penalty(mu_row: list[list[float]], n: int) -> list[list[float]]:
    """`acc[k] = sum_{j'} sum_{m>k} lambda_j'm`, the penalties after position `k`.

    Position `k` runs 1..n with 1 the ship serviced first. The `-lambda_ijk C_il
    x_ilm` terms of [P1] make a ship serviced in position `m` carry the
    multipliers of every position after it, summed over the ships that could
    occupy them, which is what this accumulates. `T = sum_j lambda_ijk` is a
    constant over the assignment and is handled by `representative_costs`.
    """
    acc = [0.0] * (n + 2)
    run = 0.0
    for k in range(n, 0, -1):
        acc[k] = run
        run += sum(lam[k - 1] for lam in mu_row)
    return acc


def representative_costs(
    inst: DBAPInstance, mu: list[list[list[float]]]
) -> tuple[list[list[list[float]]], float]:
    """The representative cost `E_ijk` of [P3], and the constant [P3] omits.

    Equation (27) writes the coefficient of `x_ijk` as
    `k p[i][j] + S_i - A_j` before the multiplier terms, with `k` the service
    position counted from the start of the berth's sequence, and the penalty
    collapse adds `- p[i][j] * sum_{j'} sum_{m>k} lambda_ij'm`. Adding the
    constant `sum_j lambda_ijk` (which no variable multiplies) gives

        E_ijk = k p[i][j] + S_i - A_j - p[i][j] * T_k,
        T_k   = sum_{j'} sum_{m>k} lambda_ij'm,

    over the `n` usable positions, and only for berths with `A_j >= S_i`, since
    a ship is not a candidate elsewhere. The returned constant is `-sum_j A_j`.
    """
    n, m = inst.n, inst.m
    e: list[list[list[float]]] = [[[0.0] * n for _ in range(n)] for _ in range(m)]
    offset = -float(sum(inst.a))
    for i in range(m):
        acc = _successor_penalty(mu[i], n)
        for j in range(n):
            if inst.a[j] < inst.s[i]:
                continue
            for k in range(1, n + 1):
                e[i][j][k - 1] = (k * inst.p[i][j] + inst.s[i] - inst.a[j]
                                  - inst.p[i][j] * acc[k])
    return e, offset


def hungarian(cost: list[list[float]]) -> tuple[float, list[int]]:
    """Minimum-cost assignment of every row to a distinct column.

    Every cost is an integer, so the optimum stays exact. Shortest augmenting
    paths with dual potentials, O(n^3) for the square case at hand; negative
    costs are handled, which the Lagrangian produces as soon as the accumulated
    multipliers outweigh a handling time.
    """
    if not cost:
        return 0, []
    n, m = len(cost), len(cost[0])
    if n > m:
        raise ValueError(f"{n} rows cannot be matched to {m} columns")
    if any(len(row) != m for row in cost):
        raise ValueError("cost matrix must be rectangular")

    inf = float("inf")
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    match = [0] * (m + 1)          # column -> row, 1-based; 0 means free
    way = [0] * (m + 1)

    for r in range(1, n + 1):
        match[0] = r
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        c = 0
        while True:
            used[c] = True
            r0, delta, c0 = match[c], inf, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[r0 - 1][j - 1] - u[r0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, c
                if minv[j] < delta:
                    delta, c0 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            c = c0
            if match[c] == 0:
                break
        while c:
            c_prev = way[c]
            match[c] = match[c_prev]
            c = c_prev

    row_of = [-1] * n
    for j in range(1, m + 1):
        if match[j]:
            row_of[match[j] - 1] = j - 1
    return -v[0], row_of


def _solve_assignment(
    inst: DBAPInstance, e: list[list[list[float]]]
) -> tuple[float, list[list[int]]]:
    """Solve [P3] and return its value with the berth-position assignment.

    A ship that is not a candidate at a berth cannot go there. Those pairs are
    priced above every admissible one instead of being forbidden, which keeps the
    matrix square for the Hungarian method; the candidate sets are never empty
    and at least `n` admissible pairs exist, so an optimum avoiding them exists
    as long as the penalty exceeds the whole range of any admissible entry.
    """
    n, m = inst.n, inst.m
    span = 1
    for i in range(m):
        for j in range(n):
            if inst.a[j] >= inst.s[i]:
                span = max(span, max(abs(e[i][j][k]) for k in range(n)) + 1)
    blocked = span * (2 * n * n + 2)

    columns = [(i, k) for i in range(m) for k in range(n)]
    cost = [[0] * len(columns) for _ in range(n)]
    for j in range(n):
        for col, (i, k) in enumerate(columns):
            cost[j][col] = e[i][j][k] if inst.a[j] >= inst.s[i] else blocked
    _, row_of = hungarian(cost)

    order: list[list[tuple[int, int]]] = [[] for _ in range(m)]
    for j, col in enumerate(row_of):
        i, k = columns[col]
        if inst.a[j] < inst.s[i]:
            raise RuntimeError(
                f"relaxed solution sends {inst.label(j)} to berth {inst.berth(i)} "
                f"although it arrives before that berth is idle")
        order[i].append((k, j))
    for i in range(m):
        order[i].sort()
    value = sum(e[i][j][k] for i in range(m) for k, j in order[i])
    return value, [[j for _k, j in seq] for seq in order]


def _starts_and_violation(
    inst: DBAPInstance, order: list[list[int]]
) -> tuple[list[list[int]], list[list[list[int]]], int]:
    """Start times of [P3] and its subgradient `g_ijk = max(0, A_j - start_ijk)`.

    The k-th ship in berth `i`'s sequence starts at `S_i` plus the handling times
    of its predecessors; berths are independent and the relaxed schedule carries
    no idle time, which is the state the paper fixes `y` to.
    """
    n, m = inst.n, inst.m
    start = [[0] * len(order[i]) for i in range(m)]
    g = [[[0] * n for _ in range(n)] for _ in range(m)]
    norm_sq = 0
    for i in range(m):
        t = inst.s[i]
        for k, j in enumerate(order[i]):
            start[i][k] = t
            v = max(0, inst.a[j] - t)
            g[i][j][k] = v
            norm_sq += v * v
            t += inst.p[i][j]
    return start, g, norm_sq


def idle_slack(inst: DBAPInstance, mu: list[list[list[float]]]) -> float:
    """The smallest coefficient `y_ijk` carries in [P1].

    That coefficient is `k - sum_j lambda_ijk` in the shifted index, since only
    the last `n` positions exist. It is non-negative exactly while discarding `y`
    from [P2] is exact.
    """
    return min(
        (k - sum(mu[i][j][k - 1] for j in range(inst.n))
         for i in range(inst.m) for k in range(1, inst.n + 1)),
        default=0.0,
    )


def solve_relaxation(
    inst: DBAPInstance, mu: list[list[list[float]]]
) -> RelaxedSolution:
    """Solve [P3] for fixed multipliers, with the subgradient the ascent needs."""
    e, offset = representative_costs(inst, mu)
    cost, order = _solve_assignment(inst, e)
    start, g, norm_sq = _starts_and_violation(inst, order)
    slack = idle_slack(inst, mu)
    return RelaxedSolution(
        order=order, mu=mu, assignment_cost=cost, offset=offset,
        lagrangian=cost + offset, start=start, violation=g,
        subgradient_norm_sq=norm_sq, valid_bound=slack >= 0, y_slack=slack,
    )


def zero_multipliers(inst: DBAPInstance) -> list[list[list[int]]]:
    """`kappa^0 = {0}`, the starting multipliers of Step 1."""
    n, m = inst.n, inst.m
    return [[[0] * n for _ in range(n)] for _ in range(m)]


# --------------------------------------------------------------------------
# feasible solutions: SIMPLE, INDIVIDUAL, INTERACT
# --------------------------------------------------------------------------


@dataclass
class FeasibleSolution:
    """A feasible schedule built from a relaxed assignment.

    `order[i]` is the service order at berth `i` and `start[i][k]` the start of
    its k-th ship. The objective is `sum_j (waiting + handling)`, the paper's
    total and equal to `sum_j C_j`.
    """

    order: list[list[int]]
    start: list[list[int]]
    objective: int
    procedure: str
    ships_moved: int = 0

    @property
    def assigned(self) -> int:
        return sum(len(seq) for seq in self.order)

    def berth_of(self, j: int) -> int:
        for i, seq in enumerate(self.order):
            if j in seq:
                return i
        raise KeyError(j)

    def start_of(self, j: int) -> int:
        i = self.berth_of(j)
        return self.start[i][self.order[i].index(j)]

    def completion_of(self, j: int, inst: DBAPInstance) -> int:
        return self.start_of(j) + inst.p[self.berth_of(j)][j]


def service_order(order: list[list[int]]) -> list[list[int]]:
    """Service sequences from position-indexed orders: position 1 is serviced first.

    Ascending position index *is* the service order, so this copies rather than
    reverses. It used to reverse, which was wrong on the module's own terms:
    `representative_costs` prices the ship in position `k` at `k p[i][j]`, and
    `_starts_and_violation` lays the positions out as `S_i + k p`, both of which
    make position 1 the earliest. Reversing therefore produced a sequence whose
    start times decreased along it -- not a schedule at all -- and turned SIMPLE
    into the mirror image of the assignment [P3] chose, which is what the
    repaired ships were then judged on.
    """
    return [list(seq) for seq in order]


def relaxed_starts(relaxed: RelaxedSolution) -> list[list[int]]:
    """Start times of [P3] laid out in service order.

    `relaxed.start[i][k]` belongs to `relaxed.order[i][k]`, so it already lines up
    with `service_order(relaxed.order)`; the returned copy keeps a repair from
    mutating the relaxed solution it was built from.
    """
    return [list(row) for row in relaxed.start]


def evaluate(inst: DBAPInstance, order: list[list[int]],
             start: list[list[int]]) -> int:
    """`sum_j (waiting + handling)`, the objective the paper reports."""
    return sum(start[i][k] - inst.a[j] + inst.p[i][j]
               for i, seq in enumerate(order) for k, j in enumerate(seq))


def earliest_starts(inst: DBAPInstance, order: list[list[int]]) -> list[list[int]]:
    """Earliest feasible starts: `max(arrival, berth ready)` along each berth.

    This is the schedule SIMPLE produces -- no ship starts before it arrives and
    each starts as soon as it and its berth are ready -- and the schedule INTERACT
    recomputes after every change.
    """
    start = [[0] * len(seq) for seq in order]
    for i, seq in enumerate(order):
        t = inst.s[i]
        for k, j in enumerate(seq):
            t = max(t, inst.a[j])
            start[i][k] = t
            t += inst.p[i][j]
    return start


def simple(inst: DBAPInstance, relaxed: RelaxedSolution) -> FeasibleSolution:
    """The paper's SIMPLE process.

    Keep the ship-to-berth and ship-to-order assignments of [P3] and postpone
    every ship that would start before its arrival, together with its successors.
    Walking each berth in service order and taking `max(arrival, predecessor's
    departure)` is exactly that repair; the objective is recomputed from the
    delayed schedule, never carried over from [P3].
    """
    order = service_order(relaxed.order)
    start = earliest_starts(inst, order)
    return FeasibleSolution(order=order, start=start,
                            objective=evaluate(inst, order, start),
                            procedure=SIMPLE)


def _waiting_criterion(inst: DBAPInstance, seq: list[int],
                       start: list[list[int]], berth: int, position: int,
                       cand: int) -> int:
    """`W_j` of Eq. (26) for shifting `cand` immediately before `seq[position]`.

    `seq` is the berth's own service sequence. It used to be typed as the whole
    `order` and read as `seq = order[berth]`, but the only caller holds the
    berth's sequence and passed it, so `order[berth]` was an int and `len(seq)`
    would have raised TypeError the first time the criterion was reached --
    which, because the caller's loop never ran, was never. Taking the sequence
    directly removes the shape mismatch instead of relying on no caller
    reaching it.

    `D_j`, the departure of the moved ship in its new position, is the start of
    the ship it is placed before; `LU` is that ship with its successors at the
    berth, minus the ship being moved.
    """
    d_j = start[berth][position]
    return sum(d_j - inst.a[seq[m]] for m in range(position, len(seq))
               if seq[m] != cand)


def _idle_before(inst: DBAPInstance, start: list[list[int]], seq: list[int],
                 berth: int, position: int) -> int:
    """Idle time immediately before `seq[position]` on `berth`.

    Position 0 has the berth's own idle time as its predecessor, which the paper's
    `S_i` covers rather than a preceding ship.
    """
    if position == 0:
        return start[berth][0] - inst.s[berth]
    previous = seq[position - 1]
    return start[berth][position] - (start[berth][position - 1]
                                     + inst.p[berth][previous])


def _recompute_from(inst: DBAPInstance, start: list[list[int]],
                    seq: list[int], berth: int, k_u: int) -> None:
    """Start times from position `k_u` on, consecutive again after a change."""
    for k in range(k_u, len(seq)):
        ready = (inst.s[berth] if k == 0
                 else start[berth][k - 1] + inst.p[berth][seq[k - 1]])
        start[berth][k] = max(ready, inst.a[seq[k]])


def _repair_berth(inst: DBAPInstance, seq: list[int], start: list[list[int]],
                  berth: int) -> int:
    """Steps 2-10 of INDIVIDUAL for one berth; returns the shifts made.

    Unsatisfied ships are examined in ascending order of position in the service
    sequence, as the paper prescribes; every change recomputes the start times of
    the successors of the ship that was repaired, which is what can make later
    unsatisfied ships satisfied.

    Steps 4-9 fill the idle period that Step 3 opens *before* `U`: the ship being
    repaired is pushed to its arrival, so the gap appears between `U` and its
    predecessor, and a later ship is shifted into the tail of that gap. Measuring
    the gap after `U` instead made the loop condition `-p[U] <= 0`: it never ran,
    INDIVIDUAL returned SIMPLE's schedule, and the `W_j` criterion was never
    consulted. The loop keeps `position` on the ship being placed in front of, so
    the gap shrinks from the right and each shifted ship departs exactly when the
    ship after it starts -- which is what `D_j` in Eq. (26) means.
    """
    moved: set[int] = set()
    shifts = 0
    k_u = 0
    while k_u < len(seq):
        u = seq[k_u]
        if start[berth][k_u] >= inst.a[u]:
            k_u += 1
            continue
        # Step 3: ship U may not be serviced before it arrives.
        start[berth][k_u] = inst.a[u]
        _recompute_from(inst, start, seq, berth, k_u)

        # Steps 4-9: fill the idle period Step 3 opened immediately before U.
        position = k_u
        while True:
            occupant_start = start[berth][position]
            idle = _idle_before(inst, start, seq, berth, position)
            if idle <= 0:
                break
            best, best_w = None, None
            for kk in range(position + 1, len(seq)):
                cand = seq[kk]
                if cand in moved:
                    continue
                if inst.p[berth][cand] > idle:
                    continue           # longer than the period it would fill
                if occupant_start - inst.p[berth][cand] < inst.a[cand]:
                    continue           # would still start before its arrival
                w = _waiting_criterion(inst, seq, start, berth, position, cand)
                if best_w is None or w < best_w:
                    best, best_w = cand, w
            if best is None:
                break
            # Step 8: shift it immediately before the occupant, then go back to
            # Step 4. The moved ship takes the tail of the idle period, so the
            # ship it precedes is not delayed, and the recomputation starts after
            # it rather than resetting it to its earliest possible start.
            seq.remove(best)
            seq.insert(position, best)
            moved.add(best)
            shifts += 1
            start[berth][position] = occupant_start - inst.p[berth][best]
            _recompute_from(inst, start, seq, berth, position + 1)
        k_u += 1
    return shifts


def individual(inst: DBAPInstance, relaxed: RelaxedSolution) -> FeasibleSolution:
    """The paper's INDIVIDUAL process.

    The repair starts from [P3]'s own start times, not from SIMPLE's: SIMPLE has
    already postponed every ship to its arrival, so a schedule that starts there
    has no unsatisfied ship left to repair and Steps 4-9 have nothing to act on --
    which is one of the two reasons this procedure used to return SIMPLE's
    schedule unchanged. Step 3 then postpones each ship the relaxation started
    before its arrival, and at each berth the successor with the smallest `W_j`
    that still starts after its own arrival is shifted into the idle period that
    postponement opened, as often as the idle period and the arrivals allow. No
    ship changes berth.
    """
    order = service_order(relaxed.order)
    start = relaxed_starts(relaxed)
    baseline = evaluate(inst, order, earliest_starts(inst, order))
    shifts = sum(_repair_berth(inst, order[i], start, i) for i in range(inst.m))
    if evaluate(inst, order, start) > baseline:
        # Eq. (26) is a proxy for the waiting a shift creates; the paper does not
        # say what to do when it points the other way, so a sequence of shifts
        # that ends up worse than SIMPLE's schedule is discarded -- and discarding
        # it means restoring the sequence as well as the start times, since the
        # repair reorders the berths it works on.
        order = service_order(relaxed.order)
        start = earliest_starts(inst, order)
    return FeasibleSolution(order=order, start=start,
                            objective=evaluate(inst, order, start),
                            procedure=INDIVIDUAL, ships_moved=shifts)


def interact(inst: DBAPInstance, relaxed: RelaxedSolution) -> FeasibleSolution:
    """The paper's INTERACT process.

    INDIVIDUAL first, then the idle periods it leaves unused are offered to ships
    scheduled at other berths. Candidates are taken in arrival order and a ship is
    moved only when the total staying time at the new berth is smaller than at the
    berth it currently occupies, which is the paper's Step 8 test read as a strict
    improvement: an equal case is left where it is, and Step 8 sends those back to
    Step 5.
    """
    base = individual(inst, relaxed)
    order = [list(seq) for seq in base.order]
    start = [list(row) for row in base.start]
    objective = base.objective
    moved = 0

    for berth in range(inst.m):
        k = 0
        while k < len(order[berth]):
            seq = order[berth]
            if k == 0:
                k += 1
                continue
            idle = start[berth][k] - (start[berth][k - 1]
                                      + inst.p[berth][seq[k - 1]])
            if idle <= 0:
                k += 1
                continue
            for j in sorted(range(inst.n), key=lambda jj: (inst.a[jj], jj)):
                if inst.a[j] < inst.s[berth]:
                    continue
                if j in seq:
                    continue
                src = next((ii for ii, s in enumerate(order) if j in s), None)
                if src is None:
                    continue
                if start[src][order[src].index(j)] < inst.a[j]:
                    continue                       # not satisfied where it is now
                if inst.a[j] > start[berth][k] - inst.p[berth][j]:
                    continue                       # does not fit the idle period
                trial = [list(s) for s in order]
                trial[src].remove(j)
                trial[berth].insert(k, j)
                trial_start = earliest_starts(inst, trial)
                value = evaluate(inst, trial, trial_start)
                if value >= objective:
                    continue
                order, start, objective = trial, trial_start, value
                moved += 1
                break
            k += 1
    return FeasibleSolution(order=order, start=start, objective=objective,
                            procedure=INTERACT, ships_moved=moved)


_PROCEDURES = {SIMPLE: simple, INDIVIDUAL: individual, INTERACT: interact}


# --------------------------------------------------------------------------
# driver: subgradient optimisation
# --------------------------------------------------------------------------


@dataclass
class HeuristicResult:
    """Everything a run produced, so a gap can be reported and audited."""
    best_feasible: FeasibleSolution
    best_lower_bound: float | None
    optimal: bool
    iterations: int
    lower_bounds: list[float] = field(default_factory=list)
    feasible_values: list[float] = field(default_factory=list)
    invalid_bound_iterations: list[int] = field(default_factory=list)
    best_mu: list[list[list[int]]] | None = None
    procedure: str = INDIVIDUAL
    wall_time: float = 0.0
    stopping_reason: str = ""

    @property
    def gap(self) -> float | None:
        """`best_feasible - best_lower_bound`, the paper's gap before scaling."""
        if self.best_lower_bound is None:
            return None
        return self.best_feasible.objective - self.best_lower_bound

    @property
    def gap_percent(self) -> float | None:
        """The paper's reported gap, `100 (feasible - bound) / bound`."""
        if self.best_lower_bound is None or self.best_lower_bound == 0:
            return None
        return 100.0 * self.gap / self.best_lower_bound


def _update_multipliers(
    inst: DBAPInstance, mu: list[list[list[float]]],
    g: list[list[list[float]]], t_step: float
) -> list[list[list[float]]]:
    """Step 5's update, then Step 6's projection. Two projections are applied.

    Step 6 keeps every multiplier non-negative. On top of that the column sum
    `sum_j lambda_ijk` is capped at `k`, the weight of position `k`: `y_ijk`'s
    coefficient in [P1] is `k - sum_j lambda_ijk`, so letting the column exceed
    `k` makes the relaxation unbounded below in `y`, its value stops being a
    bound on [PD], and the ascent has nothing to climb. The paper states Step 6
    only as the non-negativity projection; the cap is what keeps the rest of
    Section 4.1 coherent, and without it the bound stalls at its first value.
    """
    n, m = inst.n, inst.m
    out: list[list[list[float]]] = [[[0.0] * n for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for k in range(n):
            weight = float(k + 1)
            room = weight
            for j in range(n):
                if inst.a[j] < inst.s[i]:
                    continue                 # no constraint (12) there, hence no multiplier
                want = max(0.0, mu[i][j][k] + t_step * g[i][j][k])
                take = min(want, room)
                out[i][j][k] = take
                room -= take
    return out


class ImaiHeuristic:
    """The paper's subgradient procedure (Section 4.1) on one DBAP instance.

    `procedure` selects which of SIMPLE, INDIVIDUAL and INTERACT repairs the
    relaxed solution at each iteration; the paper runs all three on every
    problem, which `solve_dbap` does. The remaining arguments are Step 1's
    constants and are exposed so the sensitivity analysis of Section 5 can be
    repeated.

    When `delta` is halved the ascent restarts from `kappa*`, the multipliers that
    produced the best lower bound so far, as Step 5 prescribes; the reported bound
    is therefore one the procedure actually attained.
    """

    def __init__(self, inst: DBAPInstance, procedure: str = INDIVIDUAL,
                 max_iter: int = MAX_ITER_DEFAULT,
                 delta_start: float = DELTA_START,
                 delta_halve_after: int = DELTA_HALVE_AFTER,
                 gap_tolerance: float = GAP_TOLERANCE) -> None:
        if procedure not in _PROCEDURES:
            raise ValueError(f"unknown procedure: {procedure}")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        self.inst = inst
        self.procedure = procedure
        self.max_iter = max_iter
        self.delta_start = delta_start
        self.delta_halve_after = delta_halve_after
        self.gap_tolerance = gap_tolerance

    def run(self) -> HeuristicResult:
        """Steps 1-6 of Section 4.1, reporting the best of each kind."""
        inst = self.inst
        t0 = time.perf_counter()

        mu: list[list[list[float]]] = zero_multipliers(inst)
        delta = self.delta_start
        stalled = 0
        iterations = 0
        reason = "max_iter"
        best_lb: float | None = None
        best_mu: list[list[list[float]]] = zero_multipliers(inst)
        best_feasible: FeasibleSolution | None = None
        lower_bounds: list[float] = []
        feasible_values: list[float] = []
        invalid: list[int] = []

        for it in range(1, self.max_iter + 1):
            iterations = it
            # Step 2: solve [P3]; keep kappa* and the best bound seen so far.
            relaxed = solve_relaxation(inst, mu)

            # Step 3: one of SIMPLE, INDIVIDUAL, INTERACT, then the stop test.
            feasible = _PROCEDURES[self.procedure](inst, relaxed)
            feasible_values.append(float(feasible.objective))
            if best_feasible is None or feasible.objective < best_feasible.objective:
                best_feasible = feasible

            # A valid lower bound on [PD] can never exceed a feasible value, since
            # the optimum lies between them. A Lagrangian value above it is not a
            # bound whatever the relaxation's internal condition says, so the
            # feasible value is the guard of record.
            if relaxed.valid_bound and relaxed.lower_bound <= feasible.objective + 1e-9:
                lower_bounds.append(relaxed.lower_bound)
                if best_lb is None or relaxed.lower_bound > best_lb:
                    best_lb = relaxed.lower_bound
                    best_mu = [list(map(list, row)) for row in mu]
                    stalled = 0
                else:
                    stalled += 1
            else:
                # The idle variables were dropped from [P2] but their coefficient
                # is negative here, so this value does not bound [PD]; it is
                # recorded and never published.
                invalid.append(it)
                stalled += 1

            if best_lb is not None and \
                    best_feasible.objective - best_lb < self.gap_tolerance:
                reason = "gap_below_tolerance"
                break
            if it == self.max_iter:
                break

            # Step 5: restart on the best multipliers once the bound has stalled,
            # otherwise take the Polyak step of (25) and update in Step 6.
            if stalled >= self.delta_halve_after:
                mu = [list(map(list, row)) for row in best_mu]
                delta /= 2.0
                stalled = 0
                continue
            gap = (best_feasible.objective - relaxed.lagrangian
                   if best_feasible is not None else 0.0)
            if relaxed.subgradient_norm_sq == 0 or gap <= 0:
                # (25) is undefined at a zero subgradient, and a non-positive gap
                # would step the wrong way; the standard rescue is to shorten the
                # step and try again.
                delta /= 2.0
                stalled = 0
                continue
            t_step = delta * gap / relaxed.subgradient_norm_sq
            mu = _update_multipliers(inst, mu, relaxed.violation, t_step)

        assert best_feasible is not None      # Step 3 runs on the first iteration
        return HeuristicResult(
            best_feasible=best_feasible, best_lower_bound=best_lb,
            optimal=(best_lb is not None
                     and best_feasible.objective - best_lb < self.gap_tolerance),
            iterations=iterations, lower_bounds=lower_bounds,
            feasible_values=feasible_values, invalid_bound_iterations=invalid,
            best_mu=best_mu, procedure=self.procedure,
            wall_time=time.perf_counter() - t0, stopping_reason=reason,
        )


def solve_dbap(inst: DBAPInstance, procedure: str | None = None,
               max_iter: int = MAX_ITER_DEFAULT,
               delta_start: float = DELTA_START,
               delta_halve_after: int = DELTA_HALVE_AFTER,
               gap_tolerance: float = GAP_TOLERANCE) -> dict[str, HeuristicResult]:
    """Run the heuristic with one procedure, or with all three when `None`.

    One result per procedure keeps each best feasible solution next to its own
    best lower bound, which is how Section 5 reports the gaps.
    """
    names = PROCEDURES if procedure is None else (procedure,)
    return {
        name: ImaiHeuristic(inst, procedure=name, max_iter=max_iter,
                            delta_start=delta_start,
                            delta_halve_after=delta_halve_after,
                            gap_tolerance=gap_tolerance).run()
        for name in names
    }


def solve_instance(inst: Instance, procedure: str = INDIVIDUAL,
                   max_iter: int = MAX_ITER_DEFAULT,
                   handle_at: dict[int, dict[int, int]] | None = None,
                   start: int | None = None, **kwargs) -> HeuristicResult:
    """Convenience wrapper: run the heuristic on a package `Instance`."""
    dbap = dbap_instance_from_instance(inst, handle_at=handle_at, start=start)
    return ImaiHeuristic(dbap, procedure=procedure, max_iter=max_iter,
                         **kwargs).run()
