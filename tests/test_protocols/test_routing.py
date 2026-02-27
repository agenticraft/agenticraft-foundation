"""Tests for protocol-aware routing algorithms.

Covers:
- Protocol-Constrained BFS (minimum-hop routing)
- Resilient Multi-Protocol Routing (failover strategies)
"""

from __future__ import annotations

import pytest

from agenticraft_foundation.protocols.compatibility import (
    PROTOCOL_COMPATIBILITY,
    ProtocolCompatibilityMatrix,
)
from agenticraft_foundation.protocols.cost import PathCostCalculator
from agenticraft_foundation.protocols.graph import ProtocolGraph
from agenticraft_foundation.protocols.routing import (
    ProtocolConstrainedBFS,
    ResilientRouter,
)
from agenticraft_foundation.types import ProtocolName

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def compat_matrix() -> ProtocolCompatibilityMatrix:
    """Default protocol compatibility matrix."""
    return ProtocolCompatibilityMatrix(PROTOCOL_COMPATIBILITY)


@pytest.fixture
def diamond_graph() -> ProtocolGraph:
    """Diamond graph: A->B, A->C, B->D, C->D with mixed protocols."""
    graph = ProtocolGraph()
    graph.add_agent("A", ["routing"], {ProtocolName.MCP})
    graph.add_agent("B", ["processing"], {ProtocolName.MCP, ProtocolName.A2A})
    graph.add_agent("C", ["analysis"], {ProtocolName.A2A})
    graph.add_agent("D", ["output"], {ProtocolName.MCP, ProtocolName.A2A})
    graph.add_edge("A", "B", {ProtocolName.MCP})
    graph.add_edge("A", "C", {ProtocolName.A2A})
    graph.add_edge("B", "D", {ProtocolName.MCP, ProtocolName.A2A})
    graph.add_edge("C", "D", {ProtocolName.A2A})
    return graph


@pytest.fixture
def linear_graph() -> ProtocolGraph:
    """Linear graph: A->B->C with MCP only."""
    graph = ProtocolGraph()
    graph.add_agent("A", ["start"], {ProtocolName.MCP})
    graph.add_agent("B", ["middle"], {ProtocolName.MCP})
    graph.add_agent("C", ["end"], {ProtocolName.MCP})
    graph.add_edge("A", "B", {ProtocolName.MCP})
    graph.add_edge("B", "C", {ProtocolName.MCP})
    return graph


# ── Protocol-Constrained BFS ────────────────────────────────────────


class TestProtocolConstrainedBFS:
    """Minimum-hop routing with protocol constraints."""

    def _make_bfs(
        self,
        graph: ProtocolGraph,
        compat: ProtocolCompatibilityMatrix,
    ) -> ProtocolConstrainedBFS:
        calc = PathCostCalculator(graph, compat)
        return ProtocolConstrainedBFS(graph, compat, calc)

    def test_shortest_path_exists(
        self,
        diamond_graph,
        compat_matrix,
    ):
        bfs = self._make_bfs(diamond_graph, compat_matrix)
        route = bfs.find_shortest_path("A", "D", ProtocolName.MCP)
        assert route is not None
        assert route.path[0] == "A"
        assert route.path[-1] == "D"

    def test_shortest_path_hop_count(
        self,
        diamond_graph,
        compat_matrix,
    ):
        bfs = self._make_bfs(diamond_graph, compat_matrix)
        route = bfs.find_shortest_path("A", "D", ProtocolName.MCP)
        assert route is not None
        # A->B->D = 2 hops (shortest via MCP)
        assert route.num_hops == 2

    def test_shortest_path_a2a(
        self,
        diamond_graph,
        compat_matrix,
    ):
        bfs = self._make_bfs(diamond_graph, compat_matrix)
        route = bfs.find_shortest_path("A", "D", ProtocolName.A2A)
        assert route is not None
        # A->C->D = 2 hops via A2A
        assert route.num_hops == 2

    def test_no_path_available(
        self,
        linear_graph,
        compat_matrix,
    ):
        # Test with a disconnected target
        linear_graph.add_agent("X", ["isolated"], {ProtocolName.A2A})
        bfs = self._make_bfs(linear_graph, compat_matrix)
        route = bfs.find_shortest_path("A", "X", ProtocolName.MCP)
        assert route is None

    def test_same_source_target(
        self,
        diamond_graph,
        compat_matrix,
    ):
        bfs = self._make_bfs(diamond_graph, compat_matrix)
        route = bfs.find_shortest_path("A", "A", ProtocolName.MCP)
        assert route is not None
        assert route.path == ["A"]
        assert route.num_hops == 0

    def test_adjacent_nodes(
        self,
        diamond_graph,
        compat_matrix,
    ):
        bfs = self._make_bfs(diamond_graph, compat_matrix)
        route = bfs.find_shortest_path("A", "B", ProtocolName.MCP)
        assert route is not None
        assert route.path == ["A", "B"]
        assert route.num_hops == 1


# ── Resilient Router ────────────────────────────────────────────────


class TestResilientRouter:
    """Resilient multi-protocol routing with failover strategies."""

    def _make_router(
        self,
        graph: ProtocolGraph,
        compat: ProtocolCompatibilityMatrix,
    ) -> ResilientRouter:
        calc = PathCostCalculator(graph, compat)
        return ResilientRouter(graph, compat, calc)

    def test_route_with_no_failures(
        self,
        diamond_graph,
        compat_matrix,
    ):
        router = self._make_router(diamond_graph, compat_matrix)
        route = router.find_resilient_route(
            "A",
            "D",
            ProtocolName.MCP,
            failed_protocols=set(),
            failed_agents=set(),
        )
        assert route is not None
        assert route.path[0] == "A"
        assert route.path[-1] == "D"

    def test_route_with_protocol_failure(
        self,
        diamond_graph,
        compat_matrix,
    ):
        router = self._make_router(diamond_graph, compat_matrix)
        # Fail MCP — should fall back to A2A path
        route = router.find_resilient_route(
            "A",
            "D",
            ProtocolName.A2A,
            failed_protocols={ProtocolName.MCP},
            failed_agents=set(),
        )
        assert route is not None
        assert route.path[-1] == "D"

    def test_route_with_agent_failure(
        self,
        diamond_graph,
        compat_matrix,
    ):
        router = self._make_router(diamond_graph, compat_matrix)
        # Fail agent B — route must go via C
        route = router.find_resilient_route(
            "A",
            "D",
            ProtocolName.A2A,
            failed_protocols=set(),
            failed_agents={"B"},
        )
        assert route is not None
        assert "B" not in route.path

    def test_no_route_all_protocols_failed(
        self,
        linear_graph,
        compat_matrix,
    ):
        router = self._make_router(linear_graph, compat_matrix)
        route = router.find_resilient_route(
            "A",
            "C",
            ProtocolName.MCP,
            failed_protocols={ProtocolName.MCP},
            failed_agents=set(),
        )
        assert route is None
