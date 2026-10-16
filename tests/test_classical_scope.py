"""The shop models reject instances they cannot represent."""

from __future__ import annotations

import pytest

from bapbench.models.classical import (
    FlowShopMILP,
    JobShopMILP,
    ParallelMachineMILP,
    make_simple_problem,
    to_instance_view,
)


@pytest.fixture(scope="module")
def with_releases():
    return make_simple_problem("rel", 4, 3, seed=7, horizon=80,
                               release=True, release_range=(9, 10))


def test_flow_shop_rejects_release_dates(with_releases):
    with pytest.raises(ValueError, match="release dates"):
        FlowShopMILP(with_releases,
                     instance_view=to_instance_view(with_releases)).solve()


def test_job_shop_rejects_release_dates(with_releases):
    routes = {j: list(range(with_releases.m)) for j in range(with_releases.n)}
    with pytest.raises(ValueError, match="release dates"):
        JobShopMILP(with_releases, routes=routes,
                    instance_view=to_instance_view(with_releases)).solve()


def test_parallel_machine_still_supports_release_dates(with_releases):
    """The model that does carry r_j must keep working."""
    sch = ParallelMachineMILP(with_releases, objective="sum_C",
                              instance_view=to_instance_view(with_releases)).solve()
    assert sch.feasible, sch.violations
    for j, (_k, t) in sch.assign.items():
        assert t >= with_releases.jobs[j].release


def test_shop_models_accept_instances_without_releases():
    prob = make_simple_problem("norel", 4, 3, seed=7, horizon=80)
    fs = FlowShopMILP(prob, instance_view=to_instance_view(prob)).solve()
    assert fs.feasible, fs.violations
