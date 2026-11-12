"""Unit tests for the parts whose correctness every result depends on.

    cd endsem && python -m pytest -q tests
"""
import numpy as np

from engine.costs import CostModel
from engine.link import DEMAND, SPEC, Link, Transfer
from engine.timeline import SimConfig, simulate
from predictors.budget import AdaptiveBudget
from predictors.calibration import competition_rank
from predictors.cross_layer import CrossLayerPredictor
from traces.workload import make_uniform_workload


def test_link_fifo_and_preemption():
    bw, chunk = 1e9, 1e6
    # FIFO: the demand waits for the whole 10 MB speculative transfer
    link = Link(bw, chunk, "fifo")
    spec = Transfer("s", 10e6, SPEC, 0.0, 5)
    link.enqueue(spec)
    link.run_until(0.0025)
    dem = Transfer("d", 1e6, DEMAND, 0.0025, 1)
    link.enqueue(dem)
    assert abs(link.finish_time(dem) - 0.011) < 1e-9
    # prio: the demand preempts at the next 1 MB chunk boundary (3 ms), then the spec resumes
    link = Link(bw, chunk, "prio")
    spec = Transfer("s", 10e6, SPEC, 0.0, 5)
    link.enqueue(spec)
    link.run_until(0.0025)
    dem = Transfer("d", 1e6, DEMAND, 0.0025, 1)
    link.enqueue(dem)
    assert abs(link.finish_time(dem) - 0.004) < 1e-9
    assert abs(link.finish_time(spec) - 0.011) < 1e-9


def test_competition_rank_ties_share_best_position():
    r = competition_rank(np.array([[3.0, 0.0, 3.0, 1.0, 0.0]]))
    assert r.tolist() == [[0, 3, 0, 2, 3]]


def test_cross_layer_counter_halves_on_overflow():
    c = CrossLayerPredictor(L=2, N=4, D=1, bits=2)
    for _ in range(5):
        c.update(0, 1, np.array([[0, 1]]), np.array([[2, 3]]))
    assert c.C[1, 0].max() <= 3
    s = c.scores(0, 1, np.array([[0, 1]]))
    assert s[0, 2] > s[0, 0] and s[0, 3] > s[0, 1]


def test_budget_tracks_tau_quantile():
    b = AdaptiveBudget(L=1, D=0, N=8, tau=0.75, off_threshold=0.0)
    for delta in [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]:
        pos = np.array([0, 1 + delta])       # needed experts at ranks 0 and 1+delta, k_hat = 2
        b.update(0, 0, pos, 2)
    M, k_hat = b.size(0, 0, 2.0)
    assert k_hat == 2 and M == 2             # 75th percentile of delta* is 0
    b.tau = 0.9
    M, _ = b.size(0, 0, 2.0)
    assert M == 4                            # 90th percentile reaches the delta* = 2 cases (r* = 4)


def test_validation_gate_batch1_exact():
    c = CostModel("A100", "mixtral", "C2C-900", 1)
    wl = make_uniform_workload("mixtral", 8, 2, 32, 1, 8, seed=0)
    r = simulate(wl, wl.needed(), None, SimConfig(gpu="A100", link="C2C-900", cache_ratio=0.0))
    assert abs(r["token_latency_ms"] / c.q2("ondemand_step_ms") - 1) < 1e-3


def test_infinite_cache_hits_after_first_step():
    wl = make_uniform_workload("mixtral", 8, 2, 32, 1, 16, seed=0)
    r = simulate(wl, wl.needed(), None, SimConfig(cache_ratio=1.0, eviction="lru"))
    assert r["miss_rate"] <= 8 * 32 / (16 * 32 * 2) + 1e-9   # at most one miss per (layer, expert)
