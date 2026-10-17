# -*- coding: utf-8 -*-
"""Reproduction of Yalaoui & Chu (2006) for ``P_m | r_j | sum C_j``.

Reference
---------
F. Yalaoui, C. Chu, "New exact method to solve the Pm/rj/PCj schedule problem",
International Journal of Production Economics 100 (2006) 168-179.

Scope
-----
N jobs with an integer release date ``r_j`` and an integer processing time ``p_j``
are scheduled without preemption on M identical machines that are all available
at time 0; the criterion is the sum of completion times.  The module reproduces

* Observation 1 and dominance Properties 2-7 as individually callable predicates
  (``observation1_discard``, ``property2_dominated`` ... ``property7_dominated``);
* the polynomial lower bounds ``lb_srpt`` (job-splitting relaxation, Property 8),
  ``lb_spt`` (Proposition 1, release dates relaxed to 0) and their combination
  ``lb_srpt_spt``, which is what the paper's Tables 2 and 3 compare;
* the two priority heuristics HPRTF and HAL that provide the initial upper bound,
  and the branch and bound of Section 4 (``solve_parallel_bnb``);
* ``generate_instance``, the Section 5 instance generator.

Every schedule in this module is a semi-active schedule, i.e. a permutation of a
subset of the jobs: appending job i to a partial schedule K places i on the
machine that becomes available earliest, m(K), starting it at ``R_i(phi_m(K))``.
``YalaouiSchedule`` is that partial schedule; ``K + i`` is ``K.append(i)`` and
``O(K, i)`` is the completion of K + i by an optimal schedule of the rest.

Readings of the source text that are not fully determined by it
--------------------------------------------------------------
The paper's plain-text extraction shows ``p`` for ``<=``, ``X`` for ``>=``,
``o`` for ``<``, ``2`` for ``-`` and loses the ``+`` of ``K + {i}``.  The
following points go beyond that transliteration and are recorded instead of
being silently resolved:

1. **Property 7.**  The statement reads ``F(K') <= F(K - {i})``, but the proof
   compares K' with the partial schedule "K + {i}" and reroutes its tail, so the
   second schedule is read as K + i and the predicate requires
   ``F(K') <= F(K + i)``.  ``d = max_k (phi'_k - phi_k)`` is taken over the k-th
   smallest machine finishing times of K + i and of K', as the proof specifies.
2. **Property 5.**  The right-hand side is read literally as
   ``min(R_j(c) - R_j(phi_m), min_{k in J(K)} p_k)`` with J(K) the jobs already
   scheduled.  ``min_{k in J(K)} p_k`` is undefined for the empty schedule, so
   the predicate does not fire at the root; the source gives no explanation of
   that term beyond a reference to Chu (1995).
3. **Property 6, which is not applied.**  Printed as
   ``E_i(phi_m) <= E_j(phi_m)`` and ``(p_i - p_j) <= [E_i(phi_m) - E_j(phi_m)] *
   [U_m(K) - 1]`` it is not a dominance rule: it prunes O(K, i) although every
   optimal schedule goes through K + i.  The counterexample is small and is
   asserted in ``tests/test_reproduction_yalaoui.py`` -- six jobs on two machines
   (``generate_instance(6, 2, 0.5, 1)``), where the two optimal schedules are
   mirror images, each node of each path is pruned by the printed condition, and
   the branch and bound returns 536 instead of 534.  The paper's own proof gives
   the reason it cannot be sound as printed: it uses ``R_i(phi_m) >=
   R_j(phi_m)``, which the two stated hypotheses do not imply.  The printed form,
   its transposition, the printed form with that release condition added, with
   the cross-machine ``U(K)`` in place of ``U_m(K)``, with ``U_m(K) >= 1``
   required, and with the release condition alone were each tried against the
   enumeration and each loses the optimum on some instance, so no form of the
   rule is applied: ``property6_dominated`` always returns False and the search
   decides without it.  It stays in the module, and in ``PROPERTY_NAMES``, so the
   property that could not be reproduced is visible rather than silently missing.
4. **D_i(K).**  The notation list defines D_i(K) as the completion time of the
   job immediately preceding i on its machine, and 0 if i is first.  Properties
   2, 4 and 5 use D_j exactly in that sense, so ``E_j(D_j) = C_j(K)``.
5. **Property 3.**  ``phi_m`` is read as the finishing time of the earliest
   available machine m(K), and "machine g != m" as a machine other than m(K).
6. **Corollary 1.**  The proofs of Properties 2, 4 and 6 bound the number of
   delayed jobs by ``U_m(K)`` (resp. ``U(K)``) via Corollary 1 of Yalaoui and
   Chu (2002), which is not part of this paper.  The coefficients enter the
   predicates exactly as printed.
7. **Table 1 (the extended-SRPT example).**  The printed completion times
   (3, 5, 7, 4, 12, 16, 10, total 57) are not feasible for the printed data:
   jobs 1, 2 and 4 must be finished by time 5, which needs 9 units of machine
   time while the machines (availability 0, 2, 5) offer only 8 in [0, 5].  The
   continuous rule of Property 8 yields 57 1/6 for that data (and 66 for the
   non-split problem), so the example appears to assume whole-period processing.
   The rule implemented here allows splitting at arbitrary times, which only
   tightens the bound.
8. **HPRTF.**  With ``t`` the earliest machine availability, every job released
   at t has ``PRTF(i, t) = 2 max(t, r_i) + p_i = 2t + p_i``, so evaluating PRTF
   only on the released jobs would rank them exactly as HAL does.  The PRTF
   priority is therefore read as Chu (1992)'s rule over all unscheduled jobs,
   which may leave the machine idle waiting for an unreleased job; HAL keeps the
   paper's wording, SPT among the jobs already released.
9. **U_m(K) and N <= M.**  Observation 1 presumes N > M; for N <= M the branch
   and bound enumerates without the properties (one job per machine is optimal).

The properties and bounds are verified against the independent enumeration in
``bapbench.models.classical`` in ``tests/test_reproduction_yalaoui.py``.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from itertools import count as _counter
from typing import Iterable, Sequence

from ..models.classical import Job, Machine, SimpleProblem

EPS = 1e-9

PROPERTY_NAMES: tuple[str, ...] = (
    "observation1",
    "property2",
    "property3",
    "property4",
    "property5",
    "property6",
    "property7",
)


# --------------------------------------------------------------------------- #
# Partial schedules
# --------------------------------------------------------------------------- #

def earliest_start(t: float, release: float) -> float:
    """R_i(t) = max(t, r_i): earliest starting time of job i at time t."""
    return t if t > release else float(release)


def earliest_completion(t: float, release: float, proc: float) -> float:
    """E_i(t) = max(t, r_i) + p_i: earliest completion time of job i at time t."""
    return earliest_start(t, release) + proc


@dataclass(frozen=True)
class YalaouiSchedule:
    """A (partial) semi-active schedule for P_m | r_j | sum C_j.

    ``sequences[m]`` is the order in which the jobs are processed on machine m.
    Machines without jobs finish at 0.  A job starts as soon as both its
    predecessor on the machine and its release date allow it, so a schedule is
    fully described by the per-machine sequences and the instance data.
    """

    releases: tuple[int, ...]
    procs: tuple[int, ...]
    sequences: tuple[tuple[int, ...], ...]

    _completion: dict[int, int] = field(init=False, repr=False, compare=False)
    _predecessor: dict[int, int] = field(init=False, repr=False, compare=False)
    _machine_finish: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _machine_of: dict[int, int] = field(init=False, repr=False, compare=False)
    _total: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        completion: dict[int, int] = {}
        predecessor: dict[int, int] = {}
        machine_of: dict[int, int] = {}
        finish: list[int] = []
        for machine, seq in enumerate(self.sequences):
            t = 0
            for job in seq:
                predecessor[job] = t
                machine_of[job] = machine
                t = int(earliest_completion(t, self.releases[job], self.procs[job]))
                completion[job] = t
            finish.append(t)
        object.__setattr__(self, "_completion", completion)
        object.__setattr__(self, "_predecessor", predecessor)
        object.__setattr__(self, "_machine_finish", tuple(finish))
        object.__setattr__(self, "_machine_of", machine_of)
        object.__setattr__(self, "_total", sum(completion.values()))

    @property
    def machines(self) -> int:
        return len(self.sequences)

    @property
    def jobs(self) -> frozenset[int]:
        """J(K): the jobs scheduled in K."""
        return frozenset(self._completion)

    @property
    def total(self) -> int:
        """F(K): sum of the completion times of the jobs scheduled in K."""
        return self._total

    @property
    def machine_finish(self) -> tuple[int, ...]:
        """phi_m(K): the finishing time of every machine; 0 if it is empty."""
        return self._machine_finish

    def completion(self, job: int) -> int:
        """C_i(K)."""
        return self._completion[job]

    def predecessor_completion(self, job: int) -> int:
        """D_i(K): completion time of the job preceding i on its machine, else 0."""
        return self._predecessor[job]

    def machine_of(self, job: int) -> int:
        return self._machine_of[job]

    def unscheduled(self) -> tuple[int, ...]:
        return tuple(j for j in range(len(self.releases)) if j not in self._completion)

    def earliest_machine(self) -> int:
        """m(K): the smallest indexed machine that becomes available earliest."""
        return min(range(self.machines), key=lambda k: (self._machine_finish[k], k))

    def sorted_finish(self) -> tuple[int, ...]:
        return tuple(sorted(self._machine_finish))

    def second_earliest_finish(self) -> int:
        """c in Property 5: the second smallest machine finishing time in K."""
        if self.machines < 2:
            raise ValueError("the second machine finishing time needs at least two machines")
        return sorted(self._machine_finish)[1]

    def append(self, job: int) -> "YalaouiSchedule":
        """K + {i}: schedule job i after K on the earliest available machine."""
        machine = self.earliest_machine()
        sequences = list(self.sequences)
        sequences[machine] = sequences[machine] + (job,)
        return YalaouiSchedule(self.releases, self.procs, tuple(sequences))


def schedule_from_problem(problem: SimpleProblem,
                          sequences: Sequence[Sequence[int]]) -> YalaouiSchedule:
    """Build a partial schedule for ``problem`` from per-machine job sequences."""
    return YalaouiSchedule(tuple(j.release for j in problem.jobs),
                           tuple(j.proc for j in problem.jobs),
                           tuple(tuple(seq) for seq in sequences))


def empty_schedule(problem: SimpleProblem) -> YalaouiSchedule:
    """K = the empty partial schedule of ``problem``."""
    return YalaouiSchedule(tuple(j.release for j in problem.jobs),
                           tuple(j.proc for j in problem.jobs),
                           tuple(() for _ in range(problem.m)))


# --------------------------------------------------------------------------- #
# Observation 1
# --------------------------------------------------------------------------- #

def max_additional_jobs(sched: YalaouiSchedule, machine: int) -> int:
    """U_m(K) = min(N - M + 1 - |J_m(K)|, N - |J(K)|).

    The largest number of jobs that a non-dominated completion of K can still
    add to machine m: no machine may hold more than N - M + 1 jobs (Observation
    1), and only N - |J(K)| jobs are left.
    """
    n = len(sched.releases)
    return min(n - sched.machines + 1 - len(sched.sequences[machine]),
               n - len(sched.jobs))


def max_additional_jobs_over_machines(sched: YalaouiSchedule) -> int:
    """U(K) = max_{1 <= m <= M} U_m(K)."""
    return max(max_additional_jobs(sched, k) for k in range(sched.machines))


def observation1_discard(sched: YalaouiSchedule) -> bool:
    """Observation 1: K can be discarded when a machine holds strictly more than
    N - M + 1 jobs.

    Some other machine is then empty and the last job of the overloaded machine
    can be moved to it, where it starts no later; the moved schedule is at least
    as good and is not an extension of K, so K carries no optimal schedule.
    Defined for N > M, which is the paper's setting; with N <= M every job can be
    given its own machine and the rule would be wrong.
    """
    n = len(sched.releases)
    if n <= sched.machines:
        return False
    limit = n - sched.machines + 1
    return any(len(seq) > limit for seq in sched.sequences)


# --------------------------------------------------------------------------- #
# Dominance Properties 2-7
# --------------------------------------------------------------------------- #

def property2_dominated(sched: YalaouiSchedule, job: int) -> bool:
    """Property 2 (group 1): O(K, i) is dominated when some job j scheduled on the
    earliest available machine m(K) satisfies

        E_i(D_j) <= E_j(D_j)   and   E_i(D_j) - E_j(D_j) <= (p_i - p_j) * U_m(K).

    Job i put in j's slot finishes no later than j does, so the jobs behind j on
    machine m are not delayed, while j moves to i's slot; the jobs that follow
    lose at most (p_j - p_i) each and there are at most U_m(K) of them.
    """
    machine = sched.earliest_machine()
    capacity = max_additional_jobs(sched, machine)
    proc_i, release_i = sched.procs[job], sched.releases[job]
    for j in sched.sequences[machine]:
        e_i = earliest_completion(sched.predecessor_completion(j), release_i, proc_i)
        e_j = sched.completion(j)          # E_j(D_j) = C_j(K)
        if e_i <= e_j and e_i - e_j <= (proc_i - sched.procs[j]) * capacity:
            return True
    return False


def property3_dominated(sched: YalaouiSchedule, job: int) -> bool:
    """Property 3 (group 1): O(K, i) is dominated when a job j on a machine
    g != m(K) satisfies

        D_j <= phi_m(K)   and   R_i(phi_m(K)) < r_j,

    i.e. machine g is idle from the end of j's predecessor until r_j, and that
    idle stretch begins no later than the instant at which machine m(K) would
    start job i.  Exchanging the two machine tails then leaves every completion
    time no larger, so O(K, i) is not an active schedule.
    """
    machine = sched.earliest_machine()
    phi = sched.machine_finish[machine]
    start_i = earliest_start(phi, sched.releases[job])
    for other, seq in enumerate(sched.sequences):
        if other == machine:
            continue
        for j in seq:
            if sched.predecessor_completion(j) <= phi and start_i < sched.releases[j]:
                return True
    return False


def property4_dominated(sched: YalaouiSchedule, job: int) -> bool:
    """Property 4 (group 2): O(K, i) is dominated when another unscheduled job j
    satisfies

        E_i(phi_m) >= E_j(phi_m)
        and E_i(phi_m) - E_j(phi_m) >= (p_i - p_j) * (U(K) - 1),

    i.e. taking j next completes no later than taking i, by a margin that covers
    the delay the exchange of i and j can impose on the jobs scheduled after
    them (at most U(K) - 1 of them).
    """
    machine = sched.earliest_machine()
    phi = sched.machine_finish[machine]
    proc_i = sched.procs[job]
    e_i = earliest_completion(phi, sched.releases[job], proc_i)
    slack = max_additional_jobs_over_machines(sched) - 1
    for j in sched.unscheduled():
        if j == job:
            continue
        e_j = earliest_completion(phi, sched.releases[j], sched.procs[j])
        if e_i >= e_j and e_i - e_j >= (proc_i - sched.procs[j]) * slack:
            return True
    return False


def property5_dominated(sched: YalaouiSchedule, job: int) -> bool:
    """Property 5 (group 2): O(K, i) is dominated when another unscheduled job j
    satisfies

        [E_j(phi_m) - R_i(phi_m)] * [U_m(K) - 1]
            <= min( R_j(c) - R_j(phi_m), min_{k in J(K)} p_k ),

    where c is the second smallest machine finishing time in K.  The left side
    bounds the delay that inserting j before i causes to the jobs scheduled after
    i (at most U_m(K) - 1 of them), the right side the gain of moving j off its
    own machine.
    """
    if sched.machines < 2 or not sched.jobs:
        return False
    machine = sched.earliest_machine()
    phi = sched.machine_finish[machine]
    slack = max_additional_jobs(sched, machine) - 1
    start_i = earliest_start(phi, sched.releases[job])
    scheduled_min_proc = min(sched.procs[k] for k in sched.jobs)
    for j in sched.unscheduled():
        if j == job:
            continue
        e_j = earliest_completion(phi, sched.releases[j], sched.procs[j])
        release = sched.releases[j]
        gain = earliest_start(sched.second_earliest_finish(), release) \
            - earliest_start(phi, release)
        if (e_j - start_i) * slack <= min(gain, scheduled_min_proc):
            return True
    return False


def property6_dominated(sched: YalaouiSchedule, job: int) -> bool:
    """Property 6 (group 2): not applied.

    The paper's exchange argument for this group is meant to let a shorter job take
    the slot of a longer one.  Applied to the branch and bound, every reading of it
    loses the optimum on instances of six jobs: on `generate_instance(6, 2,
    lam=0.5, seed=1)` the enumeration finds 534 while the search returns 536, and
    the predicate fires only a handful of times in the whole search, so those
    firings are the cause.  The paper's own proof gives the reason it cannot be
    sound as printed: it uses `R_i(phi_m) >= R_j(phi_m)`, which the two stated
    hypotheses do not imply.

    Tried and rejected as repairs, each checked against the enumeration: the
    printed form, its transposition, the printed form with that additional release
    condition, with the cross-machine `U(K)` in place of `U_m(K)`, with `U_m(K) >= 1`
    required, and with the release condition alone.  None is exact; only not
    applying the predicate is.

    The function is therefore always false and is never consulted for a dominance
    decision.  It stays in the module, and in `PROPERTY_NAMES`, so the property that
    could not be reproduced is visible rather than silently missing.
    """
    return False


def property7_dominated(sched: YalaouiSchedule, job: int,
                        alternatives: Iterable[YalaouiSchedule]) -> bool:
    """Property 7 (group 3, dynamic programming): O(K, i) is dominated when a
    partial schedule K' with J(K') = J(K) + {i} satisfies

        F(K') <= F(K + i)   and   [N - |J(K)| - 1] * d <= F(K + i) - F(K'),

    where ``d = max_k (phi'_k - phi_k)`` compares the k-th smallest machine
    finishing times of K + i and K'.  K' covers the same jobs more cheaply, and
    the at most N - |J(K)| - 1 jobs still to come lose at most d each when the
    tail of O(K, i) is re-attached to K' in order of the machine finishing times.
    """
    target = sched.append(job)
    remaining = len(sched.releases) - len(sched.jobs) - 1
    target_finish = target.sorted_finish()
    for alternative in alternatives:
        if alternative.jobs != target.jobs:
            continue
        if alternative.sequences == target.sequences:
            continue
        if alternative.total > target.total:
            continue
        d = max(b - a for a, b in zip(target_finish, alternative.sorted_finish()))
        if remaining * d <= target.total - alternative.total:
            return True
    return False


# --------------------------------------------------------------------------- #
# Property 8 and Proposition 1: the two relaxed problems
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SplitSchedule:
    """Completion times of the job-splitting relaxation (Property 8)."""

    completion: tuple[float, ...]
    total: float


def property8_extended_srpt(releases: Sequence[float], procs: Sequence[float],
                           availability: Sequence[float]) -> SplitSchedule:
    """Property 8: with job splitting the extended SRPT rule is optimal.

    A job may be interrupted and processed simultaneously by several machines.
    At any time the available job with the shortest remaining processing time is
    processed on all machines that are available, and the processing is
    interrupted when a job becomes available whose processing time is strictly
    shorter than the remaining processing time of the job in process; a machine
    joins the job in process as soon as it becomes available.  The rule is
    optimal for the relaxation and runs in O(N log N) with a priority queue (the
    implementation below rescans the released jobs and is O(N^2) in the worst
    case, which is not on the critical path of the branch and bound).
    """
    n = len(releases)
    finish = [0.0] * n
    remaining = [float(p) for p in procs]
    machine_times = sorted(float(a) for a in availability)
    arrivals = sorted(range(n), key=lambda j: (releases[j], j))
    n_machines = len(machine_times)

    pool = 0                 # machines already available
    next_machine = 0
    next_arrival = 0
    released: list[int] = []
    t = 0.0
    while True:
        while next_machine < n_machines and machine_times[next_machine] <= t + EPS:
            pool += 1
            next_machine += 1
        while next_arrival < n and releases[arrivals[next_arrival]] <= t + EPS:
            released.append(arrivals[next_arrival])
            next_arrival += 1
        pending = [j for j in released if remaining[j] > EPS]
        events: list[float] = []
        if next_arrival < n:
            events.append(float(releases[arrivals[next_arrival]]))
        if next_machine < n_machines:
            events.append(machine_times[next_machine])
        if not pending:
            if not events:
                break
            t = max(t, min(events))
            continue
        if pool == 0:
            t = max(t, machine_times[next_machine])
            continue
        current = min(pending, key=lambda j: (remaining[j], j))
        next_t = min([t + remaining[current] / pool, *events])
        remaining[current] -= (next_t - t) * pool
        t = next_t
        if remaining[current] <= EPS:
            remaining[current] = 0.0
            finish[current] = t
    return SplitSchedule(tuple(finish), float(sum(finish)))


@dataclass(frozen=True)
class GeneralizedSPTSchedule:
    """Completion times of the equal-release-dates relaxation (Proposition 1)."""

    completion: tuple[float, ...]
    total: float


def proposition1_generalized_spt(procs: Sequence[float],
                                 availability: Sequence[float],
                                 common_release: float = 0.0) -> GeneralizedSPTSchedule:
    """Proposition 1: with a common release date for all jobs, the generalized SPT
    rule minimizes the sum of completion times on identical parallel machines
    even when the machines have unequal availability dates.

    The jobs are taken in non-decreasing processing time (ties by index) and each
    is assigned to the machine that becomes available earliest (ties by machine
    index), starting as soon as both that machine and the common release date
    allow.
    """
    free = [float(a) for a in availability]
    completion = [0.0] * len(procs)
    for job in sorted(range(len(procs)), key=lambda j: (procs[j], j)):
        machine = min(range(len(free)), key=lambda k: (free[k], k))
        start = max(free[machine], float(common_release))
        completion[job] = start + procs[job]
        free[machine] = completion[job]
    return GeneralizedSPTSchedule(tuple(completion), float(sum(completion)))


# --------------------------------------------------------------------------- #
# Lower bounds
# --------------------------------------------------------------------------- #

def lb_srpt(sched: YalaouiSchedule) -> float:
    """LB_SRPT: F(K) plus the optimal value of the job-splitting relaxation of the
    remaining jobs (Property 8), with the machines available at phi_m(K).

    Splitting is a relaxation of ``P_m | r_j | sum C_j``, so the value is a lower
    bound on the best completion of K.
    """
    remaining = sched.unscheduled()
    if not remaining:
        return float(sched.total)
    split = property8_extended_srpt(
        tuple(sched.releases[j] for j in remaining),
        tuple(sched.procs[j] for j in remaining),
        sched.machine_finish,
    )
    return sched.total + split.total


def lb_spt(sched: YalaouiSchedule) -> float:
    """LB_SPT: F(K) plus the optimal value of the relaxation in which all release
    dates are set to 0 (Proposition 1), with the machines available at phi_m(K).

    Release dates are relaxed because 0 <= r_j: every schedule of the node is
    feasible for the relaxed problem, so its optimum cannot exceed the node's.
    """
    remaining = sched.unscheduled()
    if not remaining:
        return float(sched.total)
    spt = proposition1_generalized_spt(
        tuple(sched.procs[j] for j in remaining), sched.machine_finish, 0.0)
    return sched.total + spt.total


def lb_srpt_spt(sched: YalaouiSchedule) -> float:
    """LB_SRPT-SPT: the best of the two bounds, i.e. their maximum.

    Both are valid lower bounds, so the larger one is valid and weaker nodes are
    eliminated sooner; this is the combination of Tables 2 and 3.
    """
    return max(lb_srpt(sched), lb_spt(sched))


# --------------------------------------------------------------------------- #
# Heuristics for the initial upper bound
# --------------------------------------------------------------------------- #

def _greedy_schedule(problem: SimpleProblem, choose) -> YalaouiSchedule:
    sched = empty_schedule(problem)
    while sched.unscheduled():
        sched = sched.append(choose(sched))
    return sched


def heuristic_hal(problem: SimpleProblem) -> YalaouiSchedule:
    """HAL: at each step the earliest available machine takes the job with the
    smallest processing time among the jobs already released; if none of them is
    released, the machine waits for the earliest release date.
    """
    def choose(sched: YalaouiSchedule) -> int:
        now = sched.machine_finish[sched.earliest_machine()]
        waiting = sched.unscheduled()
        ready = [j for j in waiting if sched.releases[j] <= now]
        if not ready:
            now = min(sched.releases[j] for j in waiting)
            ready = [j for j in waiting if sched.releases[j] <= now]
        return min(ready, key=lambda j: (sched.procs[j], j))

    return _greedy_schedule(problem, choose)


def heuristic_hprtf(problem: SimpleProblem) -> YalaouiSchedule:
    """HPRTF: at each step the earliest available machine takes the job with the
    smallest PRTF value ``PRTF(i, t) = 2 max(t, r_i) + p_i``, evaluated at the
    earliest machine availability t over all unscheduled jobs, so the machine may
    stay idle for a job that is not released yet.
    """
    def choose(sched: YalaouiSchedule) -> int:
        now = sched.machine_finish[sched.earliest_machine()]
        return min(sched.unscheduled(),
                   key=lambda j: (2 * max(now, sched.releases[j]) + sched.procs[j], j))

    return _greedy_schedule(problem, choose)


def initial_upper_bound(problem: SimpleProblem) -> YalaouiSchedule:
    """The paper's first upper bound: the better of HPRTF and HAL."""
    candidates = [heuristic_hal(problem), heuristic_hprtf(problem)]
    return min(candidates, key=lambda sched: (sched.total, sched.sequences))


# --------------------------------------------------------------------------- #
# Instance generator of Section 5
# --------------------------------------------------------------------------- #

def generate_instance(n: int, m: int, lam: float, seed: int) -> SimpleProblem:
    """Generate an instance as in Section 5 of the paper.

    Processing times are uniform integers on [1, 100] and release dates are
    uniform integers on [0, lam * n * 50.5 / m], the mean processing time of
    U[1, 100] being 50.5; lam ranges over {0.2, 0.4, ..., 3.0} there.
    """
    rng = random.Random(seed)
    procs = [rng.randint(1, 100) for _ in range(n)]
    latest_release = int(lam * n * 50.5 / m)
    jobs = [Job(jid=j, name=f"J{j + 1}", release=rng.randint(0, latest_release),
                proc=procs[j]) for j in range(n)]
    machines = [Machine(mid=k, name=f"M{k + 1}") for k in range(m)]
    horizon = max((j.release for j in jobs), default=0) + sum(procs) + 1
    return SimpleProblem(name=f"yc2006_n{n}_m{m}_l{lam}", jobs=jobs,
                         machines=machines, horizon=horizon)


# --------------------------------------------------------------------------- #
# Branch and bound
# --------------------------------------------------------------------------- #

@dataclass
class BnBResult:
    """Outcome of the branch and bound of Section 4."""

    objective: int | None
    sequences: tuple[tuple[int, ...], ...] | None
    nodes: int
    pruned: dict[str, int]
    initial_upper_bound: int | None = None
    exhaustive: bool = True

    def schedule(self, problem: SimpleProblem) -> YalaouiSchedule | None:
        if self.sequences is None:
            return None
        return schedule_from_problem(problem, self.sequences)


def solve_parallel_bnb(problem: SimpleProblem, *,
                       properties: Iterable[str] | None = None,
                       use_bounds: bool = True,
                       strategy: str = "best_first",
                       node_limit: int | None = None) -> BnBResult:
    """Solve ``P_m | r_j | sum C_j`` exactly by the paper's branch and bound.

    Each node is a partial schedule; branching assigns one more unscheduled job
    to the machine that becomes available earliest, the initial upper bound is the
    better of HAL and HPRTF, and a node is discarded when Observation 1 or one of
    Properties 2-7 applies or when its lower bound reaches the best known complete
    schedule.  ``best_first`` explores the node with the smallest lower bound
    first, breaking ties towards the node with more jobs scheduled, as in the
    paper; ``depth_first`` is available to compare the effect of the strategy.

    ``properties`` selects which of :data:`PROPERTY_NAMES` are applied (None
    applies all of them); ``use_bounds`` turns the lower bound pruning off, which
    together with an empty ``properties`` enumerates every schedule.
    """
    if strategy not in ("best_first", "depth_first"):
        raise ValueError(f"unknown strategy: {strategy!r}")
    if any(machine.unavailable for machine in problem.machines):
        raise ValueError(
            "P_m | r_j | sum C_j has no machine unavailability; the properties and "
            "bounds of this method assume that every machine is available at time 0")
    enabled = set(PROPERTY_NAMES) if properties is None else set(properties)
    unknown = enabled - set(PROPERTY_NAMES)
    if unknown:
        raise ValueError(f"unknown properties: {sorted(unknown)}")
    if problem.n <= problem.m:
        # One job per machine is optimal and U_m(K) <= 0 breaks the properties.
        enabled = set()

    releases = tuple(job.release for job in problem.jobs)
    n = len(releases)

    best = initial_upper_bound(problem)
    initial = best.total
    pruned = {name: 0 for name in (*PROPERTY_NAMES, "bound")}
    archive: dict[frozenset[int], YalaouiSchedule] = {}
    tie = _counter()
    frontier: list = []

    def push(sched: YalaouiSchedule, bound: float) -> None:
        if strategy == "best_first":
            heapq.heappush(frontier,
                           (bound, -len(sched.jobs), next(tie), sched))
        else:
            frontier.append(sched)

    def pop() -> YalaouiSchedule:
        if strategy == "best_first":
            return heapq.heappop(frontier)[3]
        return frontier.pop()

    root = empty_schedule(problem)
    push(root, lb_srpt_spt(root) if use_bounds else 0.0)
    nodes = 0
    exhaustive = True
    while frontier:
        if node_limit is not None and nodes >= node_limit:
            exhaustive = False
            break
        sched = pop()
        nodes += 1
        for job in sched.unscheduled():
            child = sched.append(job)
            if "observation1" in enabled and observation1_discard(child):
                pruned["observation1"] += 1
                continue
            child_key = child.jobs
            fired = False
            for name, predicate in (("property2", property2_dominated),
                                    ("property3", property3_dominated),
                                    ("property4", property4_dominated),
                                    ("property5", property5_dominated),
                                    ("property6", property6_dominated)):
                if name in enabled and predicate(sched, job):
                    pruned[name] += 1
                    fired = True
            if "property7" in enabled:
                known = archive.get(child_key)
                if known is not None and \
                        property7_dominated(sched, job, (known,)):
                    pruned["property7"] += 1
                    fired = True
            if fired:
                continue
            known = archive.get(child_key)
            if known is None or child.total < known.total:
                archive[child_key] = child
            if len(child.jobs) == n:
                if child.total < best.total:
                    best = child
                continue
            if use_bounds:
                bound = lb_srpt_spt(child)
                if bound >= best.total - EPS:
                    pruned["bound"] += 1
                    continue
                push(child, bound)
            else:
                push(child, 0.0)

    return BnBResult(objective=best.total if best is not None else None,
                     sequences=best.sequences if best is not None else None,
                     nodes=nodes, pruned=pruned, initial_upper_bound=initial,
                     exhaustive=exhaustive)


__all__ = [
    "BnBResult",
    "GeneralizedSPTSchedule",
    "PROPERTY_NAMES",
    "SplitSchedule",
    "YalaouiSchedule",
    "earliest_completion",
    "earliest_start",
    "empty_schedule",
    "generate_instance",
    "heuristic_hal",
    "heuristic_hprtf",
    "initial_upper_bound",
    "lb_spt",
    "lb_srpt",
    "lb_srpt_spt",
    "max_additional_jobs",
    "max_additional_jobs_over_machines",
    "observation1_discard",
    "property2_dominated",
    "property3_dominated",
    "property4_dominated",
    "property5_dominated",
    "property6_dominated",
    "property7_dominated",
    "property8_extended_srpt",
    "proposition1_generalized_spt",
    "schedule_from_problem",
    "solve_parallel_bnb",
]
