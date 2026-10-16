# -*- coding: utf-8 -*-
"""Compare the two LP files (pulp writeLP vs HiGHS in-memory writeModel)."""
import highspy

for f in ("_dj.lp", "_dj_mem.lp"):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(f"D:/DSH_Workspace/01_Code/BAPS/{f}")
    h.run()
    info = h.getInfo()
    lp = h.getLp()
    print(f"{f}: status={h.getModelStatus()} obj={info.objective_function_value} "
          f"rows={lp.num_row_} cols={lp.num_col_} "
          f"integrality={sum(1 for v in lp.integrality_ if v != 0)}")
