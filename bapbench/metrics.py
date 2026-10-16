"""Objective conventions and plotting helpers shared by all experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OBJECTIVES = {
    "sum_C": "min sum_i C_i (total completion time)",
    "C_max": "min max_i C_i (makespan)",
}

DEFAULT_OBJECTIVE = "sum_C"


def objective_value(completion: dict[int, int],
                    objective: str = DEFAULT_OBJECTIVE) -> float:
    if not completion:
        raise ValueError("empty completion dict")
    if objective == "sum_C":
        return float(sum(completion.values()))
    if objective == "C_max":
        return float(max(completion.values()))
    raise ValueError(f"unknown objective {objective!r}, expected one of {list(OBJECTIVES)}")


def gap_pct(heuristic: float | None, optimal: float | None) -> float | None:
    """Relative gap in percent. Returns None when undefined rather than inf."""
    if heuristic is None or optimal is None or optimal == 0:
        return None
    return round((heuristic - optimal) / optimal * 100.0, 3)


def gantt_rows(inst, assign: dict[int, tuple[int, int]]) -> list[dict]:
    rows = []
    for vid, (bid, st) in assign.items():
        v = inst.vessels[vid]
        rows.append({
            "berth": inst.berths[bid].name,
            "bid": bid,
            "name": v.name,
            "start": st,
            "end": st + v.handle,
            "length": v.length,
        })
    rows.sort(key=lambda r: (r["bid"], r["start"]))
    return rows


# matplotlib ships DejaVu Sans, which has no CJK glyphs. Plotting Chinese labels
# without a CJK font yields boxes and only a stderr warning, so the figure looks
# fine while being unusable. Fall back to English labels instead.
_CJK_CANDIDATES = ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                   "Source Han Sans SC", "WenQuanYi Micro Hei", "Arial Unicode MS")


def setup_matplotlib_cjk() -> bool:
    import matplotlib
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in have:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def cjk_available() -> bool:
    try:
        from matplotlib import font_manager
    except ImportError:
        return False
    have = {f.name for f in font_manager.fontManager.ttflist}
    return any(n in have for n in _CJK_CANDIDATES[:4])


def draw_gantt(inst, assign: dict[int, tuple[int, int]], path: str | Path,
               title: str = "", dt_hours: float | None = None) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to draw a Gantt chart") from exc

    cjk = setup_matplotlib_cjk()
    dt = dt_hours if dt_hours is not None else inst.dt
    rows = gantt_rows(inst, assign)
    ypos = {b.name: i for i, b in enumerate(inst.berths)}

    fig, ax = plt.subplots(figsize=(10, max(2.0, 0.6 * len(ypos) + 1.2)))
    for r in rows:
        ax.barh(ypos[r["berth"]], (r["end"] - r["start"]) * dt,
                left=r["start"] * dt, height=0.5,
                color="#4C72B0", edgecolor="black", linewidth=0.6)
        ax.text((r["start"] + (r["end"] - r["start"]) / 2) * dt,
                ypos[r["berth"]], r["name"],
                ha="center", va="center", fontsize=8, color="white")

    for k in range(inst.m):
        t = 0
        while t < inst.horizon:
            if inst.blocked(k, t):
                t0 = t
                while t < inst.horizon and inst.blocked(k, t):
                    t += 1
                ax.axvspan(t0 * dt, t * dt, color="#DD8452", alpha=0.18)
            else:
                t += 1

    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()))
    if cjk:
        ax.set_xlabel(f"Time (hours, 1 period = {dt}h)")
        ax.set_title(title or f"\u6cca\u4f4d\u8c03\u5ea6 Gantt \u2014 {inst.name}")
    else:
        ax.set_xlabel(f"Time (hours, 1 period = {dt}h)")
        ax.set_title(title or f"Berth schedule Gantt - {inst.name}")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


@dataclass
class SanityIssue:
    kind: str
    detail: str


def sanity_check_schedule(inst, assign: dict[int, tuple[int, int]]) -> list[SanityIssue]:
    """Re-check a schedule without going through Schedule.check."""
    issues: list[SanityIssue] = []
    if set(assign) != set(range(inst.n)):
        issues.append(SanityIssue(
            "coverage",
            f"assignment covers {sorted(assign)}, expected {list(range(inst.n))}"))
    for vid, (bid, st) in assign.items():
        v = inst.vessels[vid]
        if not (0 <= bid < inst.m):
            issues.append(SanityIssue("berth_range", f"V{vid} berth {bid} out of range"))
            continue
        if st < v.arrive:
            issues.append(SanityIssue("release",
                                      f"{v.name} starts at {st} before arrival {v.arrive}"))
        if st + v.handle > inst.horizon:
            issues.append(SanityIssue("horizon",
                                      f"{v.name} finishes at {st + v.handle} beyond horizon"))
        if any(inst.blocked(bid, t) for t in range(st, st + v.handle)):
            issues.append(SanityIssue("tidal", f"{v.name} straddles a blocked period"))
    issues.extend(SanityIssue("berth_overlap", p)
                  for p in berth_overlaps(inst, assign))
    return issues


def berth_overlaps(inst, assign: dict[int, tuple[int, int]]) -> list[str]:
    """Berths where two vessels occupy the same period.

    A vessel occupies its whole berth while handling, so periods may not overlap.
    This is stronger than checking that summed vessel lengths fit within the berth
    length, which would let two short vessels share one berth.
    """
    by_berth: dict[int, list[tuple[int, int, int]]] = {}
    for vid, (bid, st) in assign.items():
        v = inst.vessels[vid]
        by_berth.setdefault(bid, []).append((st, st + v.handle, vid))
    problems = []
    for bid, spans in by_berth.items():
        spans.sort()
        for (s1, e1, v1), (s2, _e2, v2) in zip(spans, spans[1:]):
            if s2 < e1:
                problems.append(
                    f"berth {inst.berths[bid].name}: {inst.vessels[v1].name} and "
                    f"{inst.vessels[v2].name} overlap at t={s2}")
    return problems


def check_bound_sanity(obj_value: float, lower_bound: float,
                       label: str) -> SanityIssue | None:
    if obj_value < lower_bound - 1e-6:
        return SanityIssue(
            "below_bound",
            f"{label}: objective {obj_value} below lower bound {lower_bound}")
    return None
