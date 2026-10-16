# BAPS

Berth allocation benchmark: exact MILP models for the dynamic berth allocation
problem, a set of baseline and metaheuristic solvers, an exhaustive verifier, and
a rolling-horizon scheduler for disruptions.

The point of this repository is not to propose a new algorithm. It is to give the
berth allocation problem an implementation where **the optima are trustworthy**, so
that later method comparisons mean something.

## Why this exists

Most BERth allocation papers compare heuristics against each other, on instances
they generate themselves, with no external reference point. That is a weak test:
if the baseline is wrong or the exact model is subtly incorrect, every subsequent
comparison inherits the error and nothing in the results reveals it.

So this code is built around independent verification. For small instances the
optimum is computed three ways that share no code:

1. A time-indexed MILP assigning each vessel a berth and a start period.
2. A disjunctive MILP using pairwise ordering variables instead of a time grid.
3. Exhaustive enumeration with no solver involved at all.

All three must agree. In addition, several instance families are constructed so
that the optimum equals a provable lower bound, which gives the formulations an
external reference rather than a self-consistency check.

## Install

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

PuLP is used as the modelling layer. Only the HiGHS backend is required; no
commercial solver is needed.

## Quick start

```
bapbench demo
```

Solves one instance exactly, runs every baseline against it, prints the gaps, and
writes a Gantt chart to `results/demo/gantt.png`. Add `--verify` to also compute
the exhaustive optimum and assert the two agree.

```
bapbench verify
```

Runs the full verification suite: the two formulations agree, the exact model
matches enumeration, objective conventions are consistent, objectives never fall
below the trivial bound, and the known-optimum instance families reproduce.

```
bapbench compare          # exact versus baselines, constraint ablation, slack and size scans
bapbench heuristics       # metaheuristics across seeds; solver configurations
bapbench dynamic          # disruption events and rolling-horizon regret
```

Every command writes its output under `results/`, and `--out` redirects it
elsewhere. `bapbench <command> --help` lists the flags.

## What is in the box

**Instances** (`bapbench/instances.py`) -- one `Instance` type shared by every
method, so nothing is compared across different data. Time is in integer periods
of `dt` hours. Alongside vessels and berths it carries two business constraints:
berth closures (`tidal_windows`) and a quay-wide limit on simultaneously handled
vessels, with temporary crane outages modelled as time windows.

**Exact models** (`bapbench/models/bap.py`) -- the two MILP formulations above,
for `sum_i C_i` and `C_max`. `bapbench/models/classical.py` holds the classical
scheduling models they are related to: single machine, parallel machine,
permutation flow shop, job shop.

**Solvers** (`bapbench/heuristics/`) -- FCFS, LPT, SPT, greedy, simulated
annealing and a genetic algorithm, plus local search, iterated local search and
ALNS. Every method searches over a permutation of vessels and shares one
placement routine, so differences come from the priority rule or the search
rather than from the decoding.

**Verification** (`bapbench/verification/`) -- the solver-free enumerator and the
known-optimum instance families.

**Dynamics** (`bapbench/dynamics.py`) -- four disruption kinds and a rolling
horizon that commits to decisions in tiers: a berth can be committed, since a
berthed vessel cannot move, but a start period can only be postponed. Commitment
happens only after the vessel's arrival has been observed. The final schedule is
rebuilt against the realised instance, because decisions are made on predictions
but must be costed on reality.

## Calibration notes

Some things were measured rather than assumed, and they are easy to get wrong.

**Difficulty comes from the horizon, not from the constraints.** Adding tidal
closures and a crane limit barely changes the gap between the exact optimum and
the best heuristic -- with a loose horizon, vessels rarely contend for the same
berth and simple greedy rules are already optimal. The parameter that matters is
`slack = horizon / theoretical minimum horizon`. `make_hard_instance` exposes it.
Tightening `slack` also eventually breaks the exact model itself: at `slack`
around 1.3 optimality can no longer be proved within a 120 s limit, which is the
regime this repository is meant to populate.

**A big-M that is too small fails silently.** It does not produce infeasibility;
it returns a plausible wrong answer, often with objective zero because the
ordering constraints stop binding. `safe_big_m` recomputes the bound per instance
rather than hard-coding a large constant.

**The naive flow-shop lower bound is wrong.** Machine load alone under-estimates
the makespan; the minimum tail time after the machine must be added. Similarly,
`max_j r_j + sum_j p_j` is a valid makespan bound only for a single machine; with
`m` machines the safe upper bound is `max_j r_j + m * sum_j p_j`.

**Stochastic methods need repeated runs.** On 12-vessel instances the spread of a
metaheuristic across seeds can exceed the difference between two methods. A
single run can rank a method wrongly. `bapbench heuristics multiseed` reports the
spread and asserts that deterministic methods have zero spread, which catches
uncontrolled randomness leaking into the implementation.

**Rolling-horizon results are meaningless without decision commitment.** If every
decision point re-optimises the whole horizon from scratch, the last decision
point has already observed every disruption and the schedule collapses onto the
perfect-information optimum, so regret is identically zero. Commitments are
therefore enforced, and `bapbench dynamic multiseed` fails loudly if all regrets come
out zero.

**Checking quay length is not the same as checking berth occupancy.** The exact
model gives a vessel its whole berth while handling, so no two vessels may hold
the same berth in the same period. An earlier version of the feasibility check
only compared the summed vessel lengths against the berth length, which accepts
two short vessels side by side and silently disagrees with the model. The
enumerator had the same flaw, and because its schedule builder also ignored berth
closures and the crane limit, it was solving a strictly easier problem and
reporting optima below the true one. Both now enforce every constraint the model
does; that is the only way agreement between them means anything.

**A search that fills in start times greedily is not a search.** Choosing each
vessel's earliest feasible start, in one order, misses feasible schedules even for
a fixed berth assignment, and it reported "no feasible schedule" on instances the
model solves. The enumerator now backtracks over each vessel's full candidate
window. Two further defects surfaced while cross-checking it: occupancy was
tracked in one array shared across berths, so a vessel on one berth blocked
another, and the pruning bound used a sum where `C_max` needs a suffix maximum,
which made the pruning unsound.

**Partial information has to be modelled on the schedule, not the outcome.** The
rolling horizon decided a delayed vessel was observable by testing its *realised*
arrival against the current period. A vessel due at 6 but actually arriving at 15
therefore still reported 6 until t=15, so the branch that falls back to the plan
was unreachable for precisely the vessels it existed for. Which vessels are
observable now follows from their scheduled arrival.

**Checking the disjunctive formulation only on unrestricted instances hides its
gaps.** `BAPDisjunctive` encodes berth non-overlap and quay capacity, but not
berth closures or the crane limit, so on a constrained instance it returns a
schedule that fails the independent feasibility check. It reports that honestly as
`feasible=False`; it is not silently wrong. Use `BAPTimeIndexed` for instances
with closures or a crane limit.

## Limitations

- The exact models are practical only for small instances. The time-indexed
  formulation is `O(n*m*T)` in binaries and becomes slow well before real port
  scale.
- A vessel occupies its whole berth while handling. This makes the capacity
  constraint equivalent to one vessel per berth-period and keeps the formulation
  compact, but it discards the ability to berth two short vessels on one berth
  simultaneously. Relaxing it requires interval or position-based constraints.
- Time is discretised at `dt` hours, so solutions carry a quantisation error of
  that order. `dt` is reported in instance metadata and should accompany any
  result.
- Disruption amplitudes in `make_events` are bounded so instances stay feasible.
  Behaviour at genuinely infeasible disruption levels is not modelled.
- The instance families with provable optima are small by construction. They
  validate the models and the exhaustive verifier; they do not benchmark
  heuristics at scale.
- No standard published benchmark set is bundled. Attempts to transcribe FT06
  from memory produced inconsistent optima, which is why the provable-bound
  families exist instead. Supplying authoritative benchmark files would be a
  genuine improvement.

## Layout

```
bapbench/
  instances.py              instance model, generators, schedule checking
  metrics.py                objective conventions, gap, Gantt charts
  dynamics.py               disruption events, tiered-commitment rolling horizon
  models/
    bap.py                  time-indexed and disjunctive MILP formulations
    classical.py            single machine, parallel machine, flow shop, job shop
  heuristics/
    constructive.py         FCFS, LPT, SPT, greedy, SA, GA
    metaheuristics.py       local search, ILS, ALNS
    solver_configs.py       backend configuration comparisons
  verification/
    exhaustive.py           solver-free enumeration
    benchmarks.py           instance families with provable optima
  cli/                      the `bapbench` command
tests/                      pytest suite
results/                    experiment output (not tracked)
```

## Tests

```
pip install -e ".[dev]"
pytest
```

25 tests covering the two formulations, their agreement with the solver-free
enumerator, the known-optimum families, schedule checking, and the disruption and
rolling-horizon behaviour.

## License

MIT. See `LICENSE`.
