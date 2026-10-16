# -*- coding: utf-8 -*-
"""Compare the two HiGHS models in memory (matrices) and by file."""
import numpy as np
import highspy

def load(path):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(path)
    return h

A = load("D:/DSH_Workspace/01_Code/BAPS/_dj.lp")
B = load("D:/DSH_Workspace/01_Code/BAPS/_dj_mem.lp")
la, lb = A.getLp(), B.getLp()
print("costs equal:", np.allclose(la.col_cost_, lb.col_cost_))
print("col bounds equal:", np.allclose(la.col_lower_, lb.col_lower_),
      np.allclose(la.col_upper_, lb.col_upper_))
print("row bounds equal:", np.allclose(la.row_lower_, lb.row_lower_),
      np.allclose(la.row_upper_, lb.row_upper_))
print("integrality equal:", list(la.integrality_) == list(lb.integrality_))
aa, ab = la.a_matrix_, lb.a_matrix_
print("matrix shape:", aa.num_col_, aa.num_row_, ab.num_col_, ab.num_row_)
print("matrix values equal:",
      np.allclose(aa.value_, ab.value_) if len(aa.value_) == len(ab.value_) else
      f"different lengths {len(aa.value_)} vs {len(ab.value_)}")
if len(aa.value_) == len(ab.value_):
    d = np.abs(np.array(aa.value_) - np.array(ab.value_))
    print("max abs diff:", d.max(), "n diff:", int((d > 1e-12).sum()))
    print("starts equal:", list(aa.start_) == list(ab.start_))
    print("index equal:", list(aa.index_) == list(ab.index_))
