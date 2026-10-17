# 实验：四组课程实验与报告

本目录是四份实验报告及其可复现的数据、脚本和图表。所有数值都由
`run_experiments.py` 的一次运行产生（约 3–4 分钟），报告里不引用任何手工填写的数字。

## 目录

```
experiments/
  run_experiments.py          四组实验的驱动脚本
  data/                       原始记录（CSV）与运行摘要（JSON）
  figures/                    报告中的图（PNG）
  reports/
    01_milp_lp_basic_model.md     MILP、LP 与调度基本模型
    02_classical_scheduling.md    经典调度问题与建模
    03_heuristics_and_solvers.md  启发式、元启发式与求解器
    04_dynamic_rolling.md         动态调度、滚动时域与扰动
```

## 复现

在仓库根目录执行：

```powershell
.venv\Scripts\python.exe experiments\run_experiments.py
```

脚本是确定性的：实例由固定 seed 生成，随机搜索的种子固定为 0–9，重复运行得到同一组表。
它会覆盖 `data/` 和 `figures/`，因此提交前若重跑过，请一并更新报告中的数字
（正常情况下数字不变）。

## 四组实验对应关系

| 报告 | 回答的问题 | 主要数据文件 |
|---|---|---|
| 01 | 模型怎么写、最优值可不可信、下界有多紧、目标与约束如何影响最优 | `e1_scale.csv`、`e1_formulations.csv`、`e1_enumeration.csv`、`e1_bounds.csv`、`e1_objectives.csv`、`e1_constraints.csv` |
| 02 | BAP 与经典调度怎么对应、规则何时最优、MILP 建模对不对 | `e2_single.csv`、`e2_release.csv`、`e2_parallel.csv`、`e2_flow.csv`、`e2_jobshop.csv` |
| 03 | 规则与元启发式差多少、随机性多大、求解器配置影响什么 | `e3_rules.csv`、`e3_multiseed.csv`、`e3_configs.csv`、`e3_robustness.csv` |
| 04 | 扰动代价、部分信息代价（regret）、重规划间隔与承诺 | `e4_events.csv`、`e4_rolling_events.csv`、`e4_rolling_dt.csv` |

## 读数约定

- 目标值默认是 `sum_C = Σ C_i`，单位是 0.5 h 的周期（`dt = 0.5`）。
- `gap %` 一律相对**已证明最优**的值计算；没有证明最优时不报 gap，或注明参照是下界。
- 所有调度都经过与算法无关的检查器（`Schedule.check` 与 `sanity_check_schedule`）；
  报告中的"可行"指两者都通过。
- 报告里的分节与 `data/` 中的子实验一一对应，便于核对。
