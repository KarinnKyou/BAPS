# BAPS

Berth allocation benchmark: exact MILP models for the dynamic berth allocation
problem, baseline and metaheuristic solvers, an exhaustive verifier, and a
rolling-horizon scheduler for disruptions.

The purpose of this repository is not to propose a new algorithm. It is to give
the berth allocation problem an implementation whose optima can be trusted, so
that later method comparisons mean something.

## Why this exists

Most berth allocation papers compare heuristics against each other on instances
they generate themselves, with no external reference point. That is a weak test:
if a baseline is wrong or the exact model is subtly incorrect, every subsequent
comparison inherits the error and nothing in the results reveals it.

This code is therefore built around independent verification. On small instances
the optimum is computed three ways that share no code:

1. A time-indexed MILP assigning each vessel a berth and a start period.
2. A disjunctive MILP using pairwise ordering variables instead of a time grid.
3. Exhaustive enumeration with no solver involved.

All three must agree. Several instance families are additionally constructed so
that the optimum equals a provable lower bound, which gives the formulations an
external reference rather than only a self-consistency check.

## Install

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install setuptools
pip install -e .
```

PuLP is the modelling layer and `highspy` supplies the HiGHS backend it calls;
both are dependencies, so no commercial solver is needed. matplotlib is required
only to draw the Gantt chart and is a dependency for that reason.

The package also runs directly from the project root without installation, using
`python -m bapbench.cli`.

## Usage

```
bapbench demo
```

Solves one instance exactly, runs every baseline against it, prints the gaps, and
writes a Gantt chart to `results/demo/gantt.png`. Add `--verify` to compute the
exhaustive optimum as well and assert that the two agree.

```
bapbench verify
```

Runs the verification suite: the two formulations agree, the exact model matches
enumeration, objective conventions are consistent, objectives never fall below
the trivial bound, and the known-optimum instance families reproduce.

```
bapbench compare      # exact versus baselines, constraint ablation, slack and size scans
bapbench heuristics   # metaheuristics across seeds; solver configurations
bapbench dynamic      # disruption events and rolling-horizon regret
bapbench classical    # classical scheduling models against independent enumeration
bapbench reproduce    # reproduced published methods beside the package's own
bapbench bacap        # the paper's own BACAP model, with QC plans, on a reconstruction
```

Every experiment command (`demo`, `compare`, `heuristics`, `classical`,
`dynamic`, `reproduce`, `bacap`) writes under `results/`, and `--out` redirects it.
`verify` is the exception: it is a pass/fail suite and writes nothing. `bapbench
<command> --help` lists the flags.

## What is in the box

**Instances** (`bapbench/instances.py`) — one `Instance` type shared by every
method, so nothing is compared across different data. Time is in integer periods
of `dt` hours. Alongside vessels and berths it carries two business constraints:
berth closures (`tidal_windows`) and a quay-wide limit on simultaneously handled
vessels, with temporary crane outages modelled as time windows.

**Exact models** (`bapbench/models/bap.py`) — the two MILP formulations above,
for `sum_i C_i` and `C_max`. `bapbench/models/classical.py` holds the classical
scheduling models they relate to: single machine, parallel machine, permutation
flow shop, job shop.

**Bounds** (`bapbench/bounds.py`) — a trivial arrival-side bound and the LP
relaxation of the time-indexed formulation. Where no optimum is proved, a gap is
measured against one of these and the row says which, rather than against an
unproved incumbent.

**Reproductions** (`bapbench/reproduction/`) — three published solution methods
re-implemented on the same `Instance`: the Imai, Nishimura & Papadimitriou (2001)
Lagrangian heuristic for the DBAP, the Iris, Pacino & Ropke (2017) ALNS for the
integrated berth-and-crane problem, and the Yalaoui & Chu (2006) branch and bound
for `P_m | r_j | sum C_j`. `comparison.py` runs them beside the package's own
baselines, scoring every row from the assignment alone and re-checking it twice.
`bapbench reproduce` drives that comparison from the command line.

`reproduction/bacap.py` is the exception that carries its own model: the paper's
problem needs a continuous berthing position, a per-vessel crane range and a
per-period quay-crane plan, none of which `Instance` expresses. It implements
objective (1), Appendix A's `q_hat, f1, f2` pattern, the per-period plan and the
ALNS on that model, and generates instances from the published distributions. The
30 original Meisel & Bierwirth (2009) instances are not public, so a generated
instance is labelled a literature reconstruction and its values are not the
paper's table values. `bapbench bacap` runs it from the command line and writes
the assignments, the QC plan and a summary.

**Solvers** (`bapbench/heuristics/`) — FCFS, LPT, SPT, greedy, simulated
annealing and a genetic algorithm, plus local search, iterated local search and
ALNS. Every method searches over a permutation of vessels and shares one
placement routine, so differences come from the priority rule or the search
rather than from the decoding.

**Verification** (`bapbench/verification/`) — the solver-free enumerator, the
known-optimum instance families, and a solver-free exact search used as a
second reference for the job shop.

**Dynamics** (`bapbench/dynamics.py`) — four disruption kinds and a rolling
horizon that commits to decisions in tiers: a berth can be committed, since a
berthed vessel cannot move, but a start period can only be postponed. Commitment
happens only after the vessel's arrival has been observed. The final schedule is
rebuilt against the realised instance, because decisions are made on predictions
but must be costed on reality.

## Modelling notes

Points that determine whether results are interpretable.

**Difficulty comes from the horizon, not from the constraints.** Adding berth
closures and a crane limit barely changes the gap between the exact optimum and
the best heuristic. With a loose horizon, vessels rarely contend for the same
berth and simple greedy rules are already optimal. The parameter that matters is
`slack = horizon / theoretical minimum horizon`, exposed by `make_hard_instance`.
Tightening it also eventually defeats the exact model itself: near `slack` 1.3
optimality can no longer be proved within a 120 s limit, which is the regime this
repository is meant to populate. Note that `make_hard_instance` defaults to
`slack = 1.0`, which is infeasible once closures and a crane limit both apply —
pass a larger value for constrained instances.

**A big-M that is too small fails silently.** It does not produce infeasibility;
it returns a plausible wrong answer, often with objective zero because the
ordering constraints stop binding. `safe_big_m` recomputes the bound per instance
rather than hard-coding a large constant.

**Bounds must include the tail.** Machine load alone under-estimates a flow-shop
makespan; the minimum remaining work after the machine has to be added. Likewise
`max_j r_j + sum_j p_j` bounds the makespan only for a single machine; with `m`
machines the safe upper bound is `max_j r_j + m * sum_j p_j`.

**A vessel occupies its whole berth while handling.** This makes the capacity
constraint equivalent to one vessel per berth-period and keeps the formulations
compact. It also means the summed vessel lengths are not the relevant test:
checking them would accept two short vessels side by side, which the models do
not permit. `bapbench/metrics.py` provides `berth_overlaps` for the correct check.

**Stochastic methods need repeated runs.** On 12-vessel instances the spread of a
metaheuristic across seeds can exceed the difference between two methods, so a
single run can rank a method wrongly. `bapbench heuristics multiseed` reports the
spread and asserts that deterministic methods have zero spread.

**Rolling-horizon results depend on decision commitment.** If every decision
point re-optimises the whole horizon from scratch, the last decision point has
already observed every disruption and the schedule collapses onto the
perfect-information optimum, so regret is identically zero. Commitments are
enforced here, and `bapbench dynamic multiseed` reports if all regrets come out
zero. Which vessels are observable at a given time follows from their scheduled
arrival, not their realised one.

## Limitations

- The exact models are practical only for small instances. The time-indexed
  formulation is `O(n*m*T)` in binaries and becomes slow well before real port
  scale.
- A vessel occupies its whole berth, so two short vessels cannot share one berth.
  Relaxing this requires interval or position-based constraints.
- Time is discretised at `dt` hours, so solutions carry a quantisation error of
  that order. `dt` is reported in instance metadata and should accompany any
  result.
- Disruption amplitudes in `make_events` are bounded so instances stay feasible.
  Behaviour at genuinely infeasible disruption levels is not modelled.
- The instance families with provable optima are small by construction. They
  validate the models and the enumerator; they do not benchmark heuristics at
  scale.
- No standard published benchmark set is bundled. The provable-bound families are
  used instead of transcribed tables, which are easy to get wrong. Supplying
  authoritative benchmark files would be a genuine improvement.
- The BACAP instances of `reproduction/bacap.py` are reconstructions from the
  published distributions, not the 30 original Meisel & Bierwirth (2009)
  instances, which are not public. No value computed on them is a reproduction of
  the paper's tables. The per-period QC plan and the smallest handling time per
  base crane count are readings of the paper, documented in that module.

## Layout

```
bapbench/
  instances.py              instance model, generators, schedule checking
  metrics.py                objective conventions, gap, Gantt charts
  bounds.py                 trivial and LP lower bounds used as gap references
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
    jobshop.py              solver-free exact branch and bound (active schedules)
  reproduction/
    imai2001.py             Lagrangian heuristic for the DBAP
    iris2017.py             ALNS for the integrated berth-and-crane problem
    yalaoui2006.py          branch and bound for P_m | r_j | sum C_j
    bacap.py                the paper's own model: quay position, QC plans, objective (1)
    comparison.py           the reproductions beside the package's baselines
  cli/                      the `bapbench` command
tests/                      pytest suite
experiments/                four lab reports with their script, data and figures
results/                    experiment output (not tracked)
```

## Tests

```
pip install -e ".[dev]"
pytest
```

The suite covers the two formulations and their agreement on constrained
instances, the solver-free enumerator, the known-optimum families, schedule
checking, the classical scheduling models against enumeration, the disruption and
rolling-horizon behaviour, the experiment orchestration layer, and each
reproduction against evidence it does not produce itself.

## Experiments

`experiments/` holds four lab reports — MILP/LP and the basic model, classical
scheduling, heuristics and solvers, dynamics and rolling horizons — together with
the script that produces every number in them, the raw CSV records and the
figures:

```
python experiments/run_experiments.py
```

See `experiments/README.md` for the mapping between reports and data files.

## License

MIT. See `LICENSE`.
