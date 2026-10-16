"""Compare the enumerator's optimum against the model's feasibility rules."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\DSH_Workspace\01_Code\BAPS")))

from bapbench.instances import Schedule, make_hard_instance
from bapbench.metrics import sanity_check_schedule
from bapbench.models.bap import BAPTimeIndexed
from bapbench.verification.exhaustive import exhaustive_optimum

inst = make_hard_instance("demo", 5, 3, seed=20261014, slack=1.6)
print(inst.summary())
print("berth lengths:", [b.length for b in inst.berths])
print("vessel lengths:", [v.length for v in inst.vessels])
print("berth 0 blocked:", inst.tidal_windows.get("0"))
print()

res = exhaustive_optimum(inst, "sum_C", max_cases=2_000_000, time_budget=60.0)
print("enumerated optimum:", res.objective, "cases:", res.n_cases)
print("assignment:", res.assign)

sch = Schedule(assign=res.assign).check(inst)
print("Schedule.check feasible:", sch.feasible)
for v in sch.violations:
    print("   violation:", v)
issues = sanity_check_schedule(inst, res.assign)
print("sanity issues:", [(i.kind, i.detail) for i in issues])

m = BAPTimeIndexed(inst, time_limit=60.0)
exact = m.solve()
print("MILP objective:", exact.objective, "feasible:", exact.feasible)
print("MILP assignment:", exact.assign)

comp = {v.vid: res.assign[v.vid][1] + v.handle for v in inst.vessels}
print("enumerated completion:", comp, "sum:", sum(comp.values()))
print("enumerated starts:", {v: res.assign[v] for v in sorted(res.assign)})
