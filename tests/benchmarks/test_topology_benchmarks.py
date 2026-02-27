"""Scalability benchmarks for topology analysis.

Validates that numpy-backed spectral analysis scales to real-world
graph sizes and catches performance regressions.

Run with:
    uv run pytest tests/benchmarks/ -v
"""

from __future__ import annotations

import math

import pytest

from agenticraft_foundation.topology.hypergraph import HypergraphNetwork
from agenticraft_foundation.topology.laplacian import NetworkGraph

# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------


def make_ring(n: int) -> NetworkGraph:
    """Ring topology with n nodes."""
    return NetworkGraph.create_ring(n)


def make_complete(n: int) -> NetworkGraph:
    """Complete (fully connected) topology with n nodes."""
    return NetworkGraph.create_complete(n)


def make_star(n: int) -> NetworkGraph:
    """Star topology with n nodes (1 center + n-1 leaves)."""
    return NetworkGraph.create_star(n)


def make_grid(n: int) -> NetworkGraph:
    """Grid topology with side length ~ sqrt(n)."""
    side = int(math.isqrt(n))
    return NetworkGraph.create_mesh(side, side)


def make_hypergraph(n: int) -> HypergraphNetwork:
    """Hypergraph with n nodes and n//3 hyperedges (3-5 nodes each)."""
    hg = HypergraphNetwork()
    for i in range(n):
        hg.add_node(f"n{i}")

    num_edges = max(1, n // 3)
    for e in range(num_edges):
        # Each hyperedge connects 3-5 consecutive nodes (wrapping)
        size = min(3 + e % 3, n)
        nodes = {f"n{(e * 3 + k) % n}" for k in range(size)}
        hg.add_hyperedge(f"he{e}", nodes)

    return hg


# ---------------------------------------------------------------------------
# Benchmarks — parameterized over graph size
# ---------------------------------------------------------------------------

SIZES = [50, 100, 500]


@pytest.mark.benchmark
@pytest.mark.parametrize("n", SIZES, ids=[f"n={s}" for s in SIZES])
def test_bench_ring_analyze(benchmark, n):
    """Benchmark NetworkGraph.analyze() on a ring topology."""
    graph = make_ring(n)
    result = benchmark(graph.analyze)
    assert result.is_connected
    assert result.algebraic_connectivity > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("n", SIZES, ids=[f"n={s}" for s in SIZES])
def test_bench_complete_analyze(benchmark, n):
    """Benchmark NetworkGraph.analyze() on a complete topology."""
    graph = make_complete(n)
    result = benchmark(graph.analyze)
    assert result.is_connected
    assert result.algebraic_connectivity > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("n", SIZES, ids=[f"n={s}" for s in SIZES])
def test_bench_star_analyze(benchmark, n):
    """Benchmark NetworkGraph.analyze() on a star topology."""
    graph = make_star(n)
    result = benchmark(graph.analyze)
    assert result.is_connected
    assert result.algebraic_connectivity > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("n", SIZES, ids=[f"n={s}" for s in SIZES])
def test_bench_grid_analyze(benchmark, n):
    """Benchmark NetworkGraph.analyze() on a grid topology."""
    graph = make_grid(n)
    result = benchmark(graph.analyze)
    assert result.is_connected
    assert result.algebraic_connectivity > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("n", SIZES, ids=[f"n={s}" for s in SIZES])
def test_bench_hypergraph_analyze(benchmark, n):
    """Benchmark HypergraphNetwork.analyze() on a hypergraph."""
    hg = make_hypergraph(n)
    result = benchmark(hg.analyze)
    assert result.num_nodes == n
