# -*- coding: utf-8 -*-
"""Canonical (order-independent) comparison of the two HiGHS models, colwise."""
from collections import Counter, defaultdict
import numpy as np
import highspy


def load(path):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(path)
    return h


def canon(h):
    lp = h.getLp()
    m = lp.a_matrix_
    rows = defaultdict(list)
    for j in range(lp.num_col_):
        for k in range(m.start_[j], m.start_[j + 1]):
            rows[m.index_[k]].append((j, round(m.value_[k], 9)))
    rowset = Counter((round(lp.row_lower_[i], 9), round(lp.row_upper_[i], 9),
                      tuple(sorted(rows.get(i, []))))
                     for i in range(lp.num_row_))
    cols = Counter((round(lp.col_cost_[j], 9), round(lp.col_lower_[j], 9),
                    round(lp.col_upper_[j], 9), int(lp.integrality_[j]))
                   for j in range(lp.num_col_))
    return rowset, cols


A = load("D:/DSH_Workspace/01_Code/BAPS/_dj.lp")
B = load("D:/DSH_Workspace/01_Code/BAPS/_dj_mem.lp")
ra, ca = canon(A)
rb, cb = canon(B)
print("column definitions identical:", ca == cb)
print("row multisets identical:", ra == rb)
print("rows only in writeLP file:", len(ra - rb),
      " rows only in in-memory file:", len(rb - ra))
for k in list((ra - rb))[:3]:
    print("  A-only:", k)
for k in list((rb - ra))[:3]:
    print("  B-only:", k)
