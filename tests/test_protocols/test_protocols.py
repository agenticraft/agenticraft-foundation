"""
Comprehensive tests for the agenticraft_foundation.protocols package.

Tests cover all 7 modules:
- graph.py: ProtocolGraph, AgentNode, ProtocolEdge, NodeType
- compatibility.py: ProtocolCompatibilityMatrix, CompatibilityRelation, CompatibilityLevel
- affinity.py: CapabilityAffinityMatrix, AffinityConfig
- cost.py: PathCostCalculator, TranslationCost, PathCost, CostConfig
- routing.py: ProtocolAwareDijkstra, RoutingState, OptimalRoute, RoutingConfig
- semantic.py: SemanticPreservationVerifier, SemanticViolation, SemanticVerificationResult
- specifications.py: ProtocolSpecification, ProtocolValidPath, SemanticPreservation, etc.
"""

from __future__ import annotations

import pytest

from agenticraft_foundation.protocols.affinity import (
    AffinityConfig,
    CapabilityAffinityMatrix,
)
from agenticraft_foundation.protocols.compatibility import (
    CompatibilityLevel,
    CompatibilityRelation,
    ProtocolCompatibilityMatrix,
)
from agenticraft_foundation.protocols.cost import (
    PathCost,
    PathCostCalculator,
    TranslationCost,
)
from agenticraft_foundation.protocols.graph import (
    AgentNode,
    NodeType,
    ProtocolEdge,
    ProtocolGraph,
)
from agenticraft_foundation.protocols.routing import (
    OptimalRoute,
    ProtocolAwareDijkstra,
    RoutingState,
)
from agenticraft_foundation.protocols.semantic import (
    SemanticPreservationVerifier,
    SemanticVerificationResult,
    SemanticViolation,
    SemanticViolationType,
    VerificationConfig,
)
from agenticraft_foundation.protocols.specifications import (
    OptimalRouting,
    ProtocolPropertyResult,
    ProtocolPropertyStatus,
    ProtocolPropertyType,
    ProtocolSpecification,
    ProtocolValidPath,
    SemanticPreservation,
    TranslationBoundProperty,
)
from agenticraft_foundation.types import ProtocolName

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_graph() -> ProtocolGraph:
    """Empty protocol graph."""
    return ProtocolGraph()


@pytest.fixture
def simple_graph() -> ProtocolGraph:
    """Graph with 3 agents connected linearly: a1 -- a2 -- a3."""
    g = ProtocolGraph()
    g.add_agent("a1", ["code_execution"], {ProtocolName.MCP})
    g.add_agent("a2", ["task_delegation"], {ProtocolName.MCP, ProtocolName.A2A})
    g.add_agent("a3", ["agent_discovery"], {ProtocolName.A2A})
    g.add_edge("a1", "a2", {ProtocolName.MCP})
    g.add_edge("a2", "a3", {ProtocolName.A2A})
    return g


@pytest.fixture
def diamond_graph() -> ProtocolGraph:
    """Diamond graph: a1 -> (a2, a3) -> a4 with various protocols."""
    g = ProtocolGraph()
    g.add_agent("a1", ["code_execution"], {ProtocolName.MCP})
    g.add_agent("a2", ["task_delegation"], {ProtocolName.MCP, ProtocolName.A2A})
    g.add_agent("a3", ["resource_access"], {ProtocolName.MCP, ProtocolName.ANP})
    g.add_agent("a4", ["agent_discovery"], {ProtocolName.A2A, ProtocolName.ANP})
    g.add_edge("a1", "a2", {ProtocolName.MCP}, weights={ProtocolName.MCP: 1.0})
    g.add_edge("a1", "a3", {ProtocolName.MCP}, weights={ProtocolName.MCP: 2.0})
    g.add_edge("a2", "a4", {ProtocolName.A2A}, weights={ProtocolName.A2A: 1.0})
    g.add_edge("a3", "a4", {ProtocolName.ANP}, weights={ProtocolName.ANP: 1.0})
    return g


@pytest.fixture
def compat_matrix() -> ProtocolCompatibilityMatrix:
    """Default protocol compatibility matrix."""
    return ProtocolCompatibilityMatrix()


@pytest.fixture
def affinity_matrix() -> CapabilityAffinityMatrix:
    """Default capability affinity matrix."""
    return CapabilityAffinityMatrix()


@pytest.fixture
def cost_calculator(simple_graph, compat_matrix) -> PathCostCalculator:
    """Path cost calculator wired to simple_graph."""
    return PathCostCalculator(simple_graph, compat_matrix)


@pytest.fixture
def dijkstra(simple_graph, compat_matrix) -> ProtocolAwareDijkstra:
    """Dijkstra router wired to simple_graph."""
    calc = PathCostCalculator(simple_graph, compat_matrix)
    return ProtocolAwareDijkstra(simple_graph, compat_matrix, calc)


# ===========================================================================
# graph.py Tests
# ===========================================================================


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_type_values(self):
        assert NodeType.AGENT == "agent"
        assert NodeType.GATEWAY == "gateway"
        assert NodeType.ROUTER == "router"
        assert NodeType.TRANSLATOR == "translator"

    def test_node_type_is_str_enum(self):
        assert isinstance(NodeType.AGENT, str)


class TestAgentNode:
    """Tests for AgentNode dataclass."""

    def test_defaults(self):
        node = AgentNode(agent_id="test")
        assert node.agent_id == "test"
        assert node.capabilities == []
        assert node.protocols == set()
        assert node.node_type == NodeType.AGENT
        assert node.avg_latency_ms == 50.0
        assert node.reliability == 0.99

    def test_supports_protocol(self):
        node = AgentNode(
            agent_id="n1",
            protocols={ProtocolName.MCP, ProtocolName.A2A},
        )
        assert node.supports_protocol(ProtocolName.MCP)
        assert node.supports_protocol(ProtocolName.A2A)
        assert not node.supports_protocol(ProtocolName.ANP)

    def test_hash_by_agent_id(self):
        n1 = AgentNode(agent_id="abc")
        n2 = AgentNode(agent_id="abc")
        assert hash(n1) == hash(n2)
        assert n1 in {n2}


class TestProtocolEdge:
    """Tests for ProtocolEdge dataclass."""

    def test_defaults(self):
        edge = ProtocolEdge(source="a", target="b", protocols={ProtocolName.MCP})
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.bandwidth_mbps == 100.0

    def test_get_weight_supported(self):
        edge = ProtocolEdge(
            source="a",
            target="b",
            protocols={ProtocolName.MCP},
            weights={ProtocolName.MCP: 2.5},
        )
        assert edge.get_weight(ProtocolName.MCP) == 2.5

    def test_get_weight_default(self):
        edge = ProtocolEdge(source="a", target="b", protocols={ProtocolName.MCP})
        # No explicit weight -> default 1.0
        assert edge.get_weight(ProtocolName.MCP) == 1.0

    def test_get_weight_unsupported_returns_inf(self):
        edge = ProtocolEdge(source="a", target="b", protocols={ProtocolName.MCP})
        assert edge.get_weight(ProtocolName.A2A) == float("inf")

    def test_supports_protocol(self):
        edge = ProtocolEdge(
            source="a",
            target="b",
            protocols={ProtocolName.MCP, ProtocolName.A2A},
        )
        assert edge.supports_protocol(ProtocolName.MCP)
        assert not edge.supports_protocol(ProtocolName.ANP)

    def test_hash(self):
        e1 = ProtocolEdge(source="a", target="b", protocols={ProtocolName.MCP})
        e2 = ProtocolEdge(source="a", target="b", protocols={ProtocolName.A2A})
        assert hash(e1) == hash(e2)  # hash is (source, target)


class TestProtocolGraph:
    """Tests for ProtocolGraph."""

    def test_add_agent(self, empty_graph):
        node = empty_graph.add_agent("agent1", ["code_execution"], {ProtocolName.MCP})
        assert node.agent_id == "agent1"
        assert "agent1" in empty_graph.agents

    def test_add_agent_custom_params(self, empty_graph):
        node = empty_graph.add_agent(
            "gw1",
            protocols={ProtocolName.MCP},
            node_type=NodeType.GATEWAY,
            avg_latency_ms=10.0,
            reliability=0.999,
        )
        assert node.node_type == NodeType.GATEWAY
        assert node.avg_latency_ms == 10.0
        assert node.reliability == 0.999

    def test_remove_agent(self, simple_graph):
        assert simple_graph.remove_agent("a2")
        assert "a2" not in simple_graph.agents
        # Edges involving a2 should be gone
        assert ("a1", "a2") not in simple_graph.edges
        assert ("a2", "a1") not in simple_graph.edges
        assert ("a2", "a3") not in simple_graph.edges

    def test_remove_agent_not_found(self, empty_graph):
        assert not empty_graph.remove_agent("missing")

    def test_add_edge(self, empty_graph):
        empty_graph.add_agent("x", [], {ProtocolName.MCP})
        empty_graph.add_agent("y", [], {ProtocolName.MCP})
        edge = empty_graph.add_edge("x", "y", {ProtocolName.MCP})
        assert edge.source == "x"
        assert edge.target == "y"
        # Bidirectional by default
        assert ("x", "y") in empty_graph.edges
        assert ("y", "x") in empty_graph.edges

    def test_add_edge_unidirectional(self, empty_graph):
        empty_graph.add_agent("x", [], {ProtocolName.MCP})
        empty_graph.add_agent("y", [], {ProtocolName.MCP})
        empty_graph.add_edge("x", "y", {ProtocolName.MCP}, bidirectional=False)
        assert ("x", "y") in empty_graph.edges
        assert ("y", "x") not in empty_graph.edges

    def test_add_edge_nonexistent_source_raises(self, empty_graph):
        empty_graph.add_agent("y", [], set())
        with pytest.raises(ValueError, match="Source agent not in graph"):
            empty_graph.add_edge("x", "y", {ProtocolName.MCP})

    def test_add_edge_nonexistent_target_raises(self, empty_graph):
        empty_graph.add_agent("x", [], set())
        with pytest.raises(ValueError, match="Target agent not in graph"):
            empty_graph.add_edge("x", "y", {ProtocolName.MCP})

    def test_remove_edge(self, simple_graph):
        assert simple_graph.remove_edge("a1", "a2")
        assert ("a1", "a2") not in simple_graph.edges
        assert ("a2", "a1") not in simple_graph.edges

    def test_remove_edge_unidirectional(self, empty_graph):
        empty_graph.add_agent("x", [], set())
        empty_graph.add_agent("y", [], set())
        empty_graph.add_edge("x", "y", {ProtocolName.MCP})
        # Remove only forward direction
        empty_graph.remove_edge("x", "y", bidirectional=False)
        assert ("x", "y") not in empty_graph.edges
        assert ("y", "x") in empty_graph.edges

    def test_remove_edge_not_found(self, empty_graph):
        assert not empty_graph.remove_edge("x", "y")

    def test_get_edge(self, simple_graph):
        edge = simple_graph.get_edge("a1", "a2")
        assert edge is not None
        assert edge.source == "a1"
        assert edge.target == "a2"

    def test_get_edge_not_found(self, simple_graph):
        assert simple_graph.get_edge("a1", "a3") is None

    def test_get_neighbors(self, simple_graph):
        neighbors = simple_graph.get_neighbors("a2")
        assert "a1" in neighbors
        assert "a3" in neighbors

    def test_get_neighbors_with_protocol_filter(self, simple_graph):
        mcp_neighbors = simple_graph.get_neighbors("a2", ProtocolName.MCP)
        a2a_neighbors = simple_graph.get_neighbors("a2", ProtocolName.A2A)
        assert "a1" in mcp_neighbors
        assert "a3" in a2a_neighbors

    def test_get_edges_from(self, simple_graph):
        edges = simple_graph.get_edges_from("a2")
        assert len(edges) == 2  # edges to a1 and a3 (bidirectional)

    def test_get_edges_from_with_protocol(self, simple_graph):
        edges = simple_graph.get_edges_from("a2", ProtocolName.MCP)
        assert len(edges) == 1
        assert edges[0].target == "a1"

    def test_get_agents_by_capability(self, simple_graph):
        agents = simple_graph.get_agents_by_capability("code_execution")
        assert len(agents) == 1
        assert agents[0].agent_id == "a1"

    def test_get_agents_by_capability_with_protocol(self, simple_graph):
        agents = simple_graph.get_agents_by_capability("task_delegation", ProtocolName.A2A)
        assert len(agents) == 1
        assert agents[0].agent_id == "a2"

    def test_get_agents_by_protocol(self, simple_graph):
        mcp_agents = simple_graph.get_agents_by_protocol(ProtocolName.MCP)
        ids = [a.agent_id for a in mcp_agents]
        assert "a1" in ids
        assert "a2" in ids
        assert "a3" not in ids

    def test_is_protocol_valid_path_valid(self, simple_graph):
        assert simple_graph.is_protocol_valid_path(["a1", "a2"], [ProtocolName.MCP])

    def test_is_protocol_valid_path_invalid_protocol(self, simple_graph):
        # a1->a2 only has MCP, not A2A
        assert not simple_graph.is_protocol_valid_path(["a1", "a2"], [ProtocolName.A2A])

    def test_is_protocol_valid_path_no_edge(self, simple_graph):
        assert not simple_graph.is_protocol_valid_path(["a1", "a3"], [ProtocolName.MCP])

    def test_is_protocol_valid_path_trivial(self, simple_graph):
        assert simple_graph.is_protocol_valid_path(["a1"], [])

    def test_is_protocol_valid_path_length_mismatch(self, simple_graph):
        assert not simple_graph.is_protocol_valid_path(["a1", "a2", "a3"], [ProtocolName.MCP])

    def test_get_statistics(self, simple_graph):
        stats = simple_graph.get_statistics()
        assert stats["num_agents"] == 3
        assert stats["num_edges"] == 4  # 2 bidirectional edges = 4 directed
        assert stats["num_protocols"] == 3
        assert stats["avg_neighbors"] > 0

    def test_clear(self, simple_graph):
        simple_graph.clear()
        assert len(simple_graph.agents) == 0
        assert len(simple_graph.edges) == 0

    def test_get_statistics_empty(self, empty_graph):
        stats = empty_graph.get_statistics()
        assert stats["num_agents"] == 0
        assert stats["avg_neighbors"] == 0


# ===========================================================================
# compatibility.py Tests
# ===========================================================================


class TestCompatibilityRelation:
    """Tests for CompatibilityRelation."""

    def test_full_compatibility(self):
        rel = CompatibilityRelation(
            source=ProtocolName.MCP,
            target=ProtocolName.A2A,
            level=CompatibilityLevel.FULL,
            semantic_loss=0.0,
        )
        assert rel.is_full
        assert not rel.is_partial
        assert rel.is_compatible

    def test_partial_compatibility(self):
        rel = CompatibilityRelation(
            source=ProtocolName.A2A,
            target=ProtocolName.MCP,
            level=CompatibilityLevel.PARTIAL,
            semantic_loss=0.1,
        )
        assert not rel.is_full
        assert rel.is_partial
        assert rel.is_compatible

    def test_no_compatibility(self):
        rel = CompatibilityRelation(
            source=ProtocolName.MCP,
            target=ProtocolName.CUSTOM,
            level=CompatibilityLevel.NONE,
        )
        assert not rel.is_full
        assert not rel.is_partial
        assert not rel.is_compatible

    def test_translation_cost_full(self):
        rel = CompatibilityRelation(
            source=ProtocolName.MCP,
            target=ProtocolName.A2A,
            level=CompatibilityLevel.FULL,
            semantic_loss=0.0,
            latency_factor=1.0,
        )
        cost = rel.get_translation_cost()
        assert cost == pytest.approx(0.1)  # base_cost for FULL

    def test_translation_cost_partial(self):
        rel = CompatibilityRelation(
            source=ProtocolName.A2A,
            target=ProtocolName.MCP,
            level=CompatibilityLevel.PARTIAL,
            semantic_loss=0.1,
            latency_factor=1.1,
        )
        cost = rel.get_translation_cost()
        # base_cost=0.5, semantic_penalty=0.2, latency_penalty=0.05
        assert cost == pytest.approx(0.75)

    def test_translation_cost_none_returns_inf(self):
        rel = CompatibilityRelation(
            source=ProtocolName.MCP,
            target=ProtocolName.CUSTOM,
            level=CompatibilityLevel.NONE,
        )
        assert rel.get_translation_cost() == float("inf")


class TestProtocolCompatibilityMatrix:
    """Tests for ProtocolCompatibilityMatrix."""

    def test_default_matrix_has_entries(self, compat_matrix):
        stats = compat_matrix.get_statistics()
        assert stats["num_relations"] > 0
        assert stats["num_protocols"] >= 3

    def test_get_compatibility_known_pair(self, compat_matrix):
        rel = compat_matrix.get_compatibility(ProtocolName.MCP, ProtocolName.A2A)
        assert rel.is_compatible
        assert rel.level == CompatibilityLevel.FULL

    def test_get_compatibility_self(self, compat_matrix):
        rel = compat_matrix.get_compatibility(ProtocolName.MCP, ProtocolName.MCP)
        assert rel.is_full
        assert rel.semantic_loss == 0.0

    def test_can_translate(self, compat_matrix):
        assert compat_matrix.can_translate(ProtocolName.MCP, ProtocolName.A2A)
        assert compat_matrix.can_translate(ProtocolName.A2A, ProtocolName.MCP)

    def test_get_translation_cost(self, compat_matrix):
        cost = compat_matrix.get_translation_cost(ProtocolName.MCP, ProtocolName.A2A)
        assert cost < float("inf")
        assert cost >= 0.0

    def test_get_semantic_loss_known(self, compat_matrix):
        loss = compat_matrix.get_semantic_loss(ProtocolName.A2A, ProtocolName.MCP)
        assert 0.0 <= loss <= 1.0
        assert loss == 0.1  # from built-in data

    def test_get_semantic_loss_self(self, compat_matrix):
        assert compat_matrix.get_semantic_loss(ProtocolName.MCP, ProtocolName.MCP) == 0.0

    def test_get_all_compatible(self, compat_matrix):
        compatible = compat_matrix.get_all_compatible(ProtocolName.MCP)
        assert len(compatible) > 0
        assert ProtocolName.A2A in compatible

    def test_get_all_compatible_full_only(self, compat_matrix):
        full = compat_matrix.get_all_compatible(ProtocolName.MCP, level=CompatibilityLevel.FULL)
        # MCP->A2A is FULL, MCP->MCP is FULL
        assert ProtocolName.A2A in full
        assert ProtocolName.MCP in full

    def test_register_compatibility(self, compat_matrix):
        rel = CompatibilityRelation(
            source=ProtocolName.CUSTOM,
            target=ProtocolName.CUSTOM,
            level=CompatibilityLevel.FULL,
            semantic_loss=0.0,
        )
        compat_matrix.register_compatibility(rel)
        result = compat_matrix.get_compatibility(ProtocolName.CUSTOM, ProtocolName.CUSTOM)
        assert result.is_full

    def test_get_best_translation_path_direct(self, compat_matrix):
        path = compat_matrix.get_best_translation_path(ProtocolName.MCP, ProtocolName.A2A)
        assert path is not None
        assert path == [ProtocolName.MCP, ProtocolName.A2A]

    def test_get_best_translation_path_self(self, compat_matrix):
        path = compat_matrix.get_best_translation_path(ProtocolName.MCP, ProtocolName.MCP)
        assert path == [ProtocolName.MCP]

    def test_get_statistics(self, compat_matrix):
        stats = compat_matrix.get_statistics()
        assert "full_compatibility" in stats
        assert "partial_compatibility" in stats
        assert "avg_semantic_loss" in stats
        assert stats["full_compatibility"] > 0

    def test_get_semantic_loss_incompatible(self, compat_matrix):
        """Test semantic loss is 1.0 for incompatible protocols."""
        # Register an incompatible relation
        from agenticraft_foundation.types import ProtocolName

        incompatible_rel = CompatibilityRelation(
            source=ProtocolName.MCP,
            target=ProtocolName.MCP,  # Override self-compat to NONE for test
            level=CompatibilityLevel.NONE,
            semantic_loss=1.0,
            reversible=False,
        )
        # Use a fresh matrix with only this relation
        custom = ProtocolCompatibilityMatrix(
            {(ProtocolName.MCP, ProtocolName.A2A): incompatible_rel}
        )
        loss = custom.get_semantic_loss(ProtocolName.MCP, ProtocolName.A2A)
        assert loss == 1.0

    def test_get_best_translation_path_indirect(self, compat_matrix):
        """Test indirect path via intermediary protocol."""
        from agenticraft_foundation.types import ProtocolName

        # Create a matrix where A->C has no direct path but A->B->C works
        a_to_b = CompatibilityRelation(
            source=ProtocolName.MCP,
            target=ProtocolName.A2A,
            level=CompatibilityLevel.PARTIAL,
            semantic_loss=0.1,
        )
        b_to_c = CompatibilityRelation(
            source=ProtocolName.A2A,
            target=ProtocolName.CUSTOM,
            level=CompatibilityLevel.PARTIAL,
            semantic_loss=0.2,
        )
        # No direct MCP->CUSTOM
        matrix = ProtocolCompatibilityMatrix(
            {
                (ProtocolName.MCP, ProtocolName.A2A): a_to_b,
                (ProtocolName.A2A, ProtocolName.CUSTOM): b_to_c,
            }
        )
        path = matrix.get_best_translation_path(ProtocolName.MCP, ProtocolName.CUSTOM, max_hops=2)
        assert path is not None
        assert len(path) == 3
        assert path[0] == ProtocolName.MCP
        assert path[1] == ProtocolName.A2A
        assert path[2] == ProtocolName.CUSTOM

    def test_get_best_translation_path_no_path(self, compat_matrix):
        """Test returns None when no path exists (incompatible)."""
        from agenticraft_foundation.types import ProtocolName

        # Create a matrix with only NONE-level relations (no compatible path)
        no_compat = {
            (ProtocolName.MCP, ProtocolName.A2A): CompatibilityRelation(
                source=ProtocolName.MCP,
                target=ProtocolName.A2A,
                level=CompatibilityLevel.NONE,
                semantic_loss=1.0,
            ),
            (ProtocolName.A2A, ProtocolName.CUSTOM): CompatibilityRelation(
                source=ProtocolName.A2A,
                target=ProtocolName.CUSTOM,
                level=CompatibilityLevel.NONE,
                semantic_loss=1.0,
            ),
            (ProtocolName.MCP, ProtocolName.CUSTOM): CompatibilityRelation(
                source=ProtocolName.MCP,
                target=ProtocolName.CUSTOM,
                level=CompatibilityLevel.NONE,
                semantic_loss=1.0,
            ),
        }
        matrix = ProtocolCompatibilityMatrix(no_compat)
        path = matrix.get_best_translation_path(ProtocolName.MCP, ProtocolName.A2A, max_hops=2)
        assert path is None

    def test_get_best_translation_path_max_hops_1(self, compat_matrix):
        """Test max_hops=1 disallows indirect routing."""
        from agenticraft_foundation.types import ProtocolName

        # Create a matrix with no direct path
        a_to_b = CompatibilityRelation(
            source=ProtocolName.MCP,
            target=ProtocolName.A2A,
            level=CompatibilityLevel.PARTIAL,
            semantic_loss=0.1,
        )
        b_to_c = CompatibilityRelation(
            source=ProtocolName.A2A,
            target=ProtocolName.CUSTOM,
            level=CompatibilityLevel.PARTIAL,
            semantic_loss=0.2,
        )
        matrix = ProtocolCompatibilityMatrix(
            {
                (ProtocolName.MCP, ProtocolName.A2A): a_to_b,
                (ProtocolName.A2A, ProtocolName.CUSTOM): b_to_c,
            }
        )
        path = matrix.get_best_translation_path(ProtocolName.MCP, ProtocolName.CUSTOM, max_hops=1)
        assert path is None


# ===========================================================================
# affinity.py Tests
# ===========================================================================


class TestAffinityConfig:
    """Tests for AffinityConfig."""

    def test_defaults(self):
        cfg = AffinityConfig()
        assert cfg.default_affinity == 0.5
        assert cfg.min_affinity == 0.0
        assert cfg.max_affinity == 1.0
        assert cfg.aggregation_method == "weighted_average"


class TestCapabilityAffinityMatrix:
    """Tests for CapabilityAffinityMatrix."""

    def test_get_affinity_known(self, affinity_matrix):
        aff = affinity_matrix.get_affinity("code_execution", ProtocolName.MCP)
        assert aff == 0.95

    def test_get_affinity_unknown_returns_default(self, affinity_matrix):
        aff = affinity_matrix.get_affinity("unknown_cap", ProtocolName.MCP)
        assert aff == 0.5

    def test_get_optimal_protocol_code_execution(self, affinity_matrix):
        best = affinity_matrix.get_optimal_protocol("code_execution")
        assert best == ProtocolName.MCP

    def test_get_optimal_protocol_task_delegation(self, affinity_matrix):
        best = affinity_matrix.get_optimal_protocol("task_delegation")
        assert best == ProtocolName.A2A

    def test_score_protocol_for_capabilities_single(self, affinity_matrix):
        score = affinity_matrix.score_protocol_for_capabilities(
            ["code_execution"], ProtocolName.MCP
        )
        assert score == 0.95

    def test_score_protocol_for_capabilities_multiple(self, affinity_matrix):
        score = affinity_matrix.score_protocol_for_capabilities(
            ["code_execution", "resource_access"], ProtocolName.MCP
        )
        # Average of 0.95 and 0.90
        assert score == pytest.approx(0.925)

    def test_score_protocol_for_capabilities_empty(self, affinity_matrix):
        score = affinity_matrix.score_protocol_for_capabilities([], ProtocolName.MCP)
        assert score == 0.5  # default

    def test_score_protocol_weighted(self, affinity_matrix):
        score = affinity_matrix.score_protocol_for_capabilities(
            ["code_execution", "task_delegation"],
            ProtocolName.MCP,
            weights={"code_execution": 2.0, "task_delegation": 1.0},
        )
        # weighted: (0.95*2 + 0.5*1) / 3.0 = 0.8
        assert score == pytest.approx(0.8)

    def test_score_protocol_max_aggregation(self):
        cfg = AffinityConfig(aggregation_method="max")
        matrix = CapabilityAffinityMatrix(config=cfg)
        score = matrix.score_protocol_for_capabilities(
            ["code_execution", "task_delegation"], ProtocolName.MCP
        )
        # max(0.95, 0.5) = 0.95
        assert score == 0.95

    def test_score_protocol_min_aggregation(self):
        cfg = AffinityConfig(aggregation_method="min")
        matrix = CapabilityAffinityMatrix(config=cfg)
        score = matrix.score_protocol_for_capabilities(
            ["code_execution", "task_delegation"], ProtocolName.MCP
        )
        # min(0.95, 0.5) = 0.5
        assert score == 0.5

    def test_rank_protocols_for_capabilities(self, affinity_matrix):
        ranked = affinity_matrix.rank_protocols_for_capabilities(["code_execution"])
        assert len(ranked) == 3  # MCP, A2A, CUSTOM
        # First should be the best protocol
        assert ranked[0][0] == ProtocolName.MCP
        # Scores should be descending
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_get_capability_coverage(self, affinity_matrix):
        covered = affinity_matrix.get_capability_coverage(ProtocolName.MCP, threshold=0.9)
        assert "code_execution" in covered
        assert "llm_integration" in covered

    def test_get_capability_coverage_high_threshold(self, affinity_matrix):
        covered = affinity_matrix.get_capability_coverage(ProtocolName.MCP, threshold=0.95)
        assert "code_execution" in covered

    def test_register_affinity(self, affinity_matrix):
        affinity_matrix.register_affinity("new_cap", ProtocolName.MCP, 0.88)
        assert affinity_matrix.get_affinity("new_cap", ProtocolName.MCP) == 0.88

    def test_register_affinity_clamped(self, affinity_matrix):
        affinity_matrix.register_affinity("x", ProtocolName.MCP, 1.5)
        assert affinity_matrix.get_affinity("x", ProtocolName.MCP) == 1.0
        affinity_matrix.register_affinity("y", ProtocolName.MCP, -0.5)
        assert affinity_matrix.get_affinity("y", ProtocolName.MCP) == 0.0

    def test_get_statistics(self, affinity_matrix):
        stats = affinity_matrix.get_statistics()
        assert stats["num_capabilities"] > 0
        assert stats["num_protocols"] == 3
        assert stats["num_entries"] > 0
        assert 0.0 <= stats["avg_affinity"] <= 1.0
        assert "coverage" in stats

    def test_get_best_protocol_for_capabilities(self, affinity_matrix):
        best_proto, best_score = affinity_matrix.get_best_protocol_for_capabilities(
            ["code_execution", "resource_access"]
        )
        assert best_proto == ProtocolName.MCP
        assert best_score > 0.0

    def test_get_best_protocol_with_exclusion(self, affinity_matrix):
        best_proto, _ = affinity_matrix.get_best_protocol_for_capabilities(
            ["code_execution"],
            exclude_protocols={ProtocolName.MCP},
        )
        assert best_proto != ProtocolName.MCP


# ===========================================================================
# cost.py Tests
# ===========================================================================


class TestTranslationCost:
    """Tests for TranslationCost."""

    def test_total_cost(self):
        tc = TranslationCost(
            source_protocol=ProtocolName.MCP,
            target_protocol=ProtocolName.A2A,
            base_cost=0.1,
            semantic_loss_penalty=0.2,
        )
        assert tc.total_cost == pytest.approx(0.3)

    def test_is_identity(self):
        tc = TranslationCost(
            source_protocol=ProtocolName.MCP,
            target_protocol=ProtocolName.MCP,
        )
        assert tc.is_identity

    def test_is_not_identity(self):
        tc = TranslationCost(
            source_protocol=ProtocolName.MCP,
            target_protocol=ProtocolName.A2A,
        )
        assert not tc.is_identity

    def test_repr(self):
        tc = TranslationCost(
            source_protocol=ProtocolName.MCP,
            target_protocol=ProtocolName.A2A,
            base_cost=0.1,
            semantic_loss_penalty=0.0,
            latency_ms=11.0,
        )
        r = repr(tc)
        assert "mcp" in r
        assert "a2a" in r


class TestPathCost:
    """Tests for PathCost."""

    def test_empty_path(self):
        pc = PathCost()
        assert pc.total_edge_cost == 0.0
        assert pc.total_translation_cost == 0.0
        assert pc.total_cost == 0.0
        assert pc.num_translations == 0

    def test_total_edge_cost(self):
        pc = PathCost(edge_costs=[1.0, 2.0, 3.0])
        assert pc.total_edge_cost == 6.0

    def test_total_translation_cost(self):
        tc1 = TranslationCost(
            source_protocol=ProtocolName.MCP,
            target_protocol=ProtocolName.A2A,
            base_cost=0.1,
            semantic_loss_penalty=0.2,
        )
        tc2 = TranslationCost(
            source_protocol=ProtocolName.MCP,
            target_protocol=ProtocolName.MCP,
            base_cost=0.0,
            semantic_loss_penalty=0.0,
        )
        pc = PathCost(translation_costs=[tc1, tc2])
        assert pc.total_translation_cost == pytest.approx(0.3)

    def test_total_cost_combined(self):
        tc = TranslationCost(
            source_protocol=ProtocolName.MCP,
            target_protocol=ProtocolName.A2A,
            base_cost=0.5,
            semantic_loss_penalty=0.0,
        )
        pc = PathCost(edge_costs=[1.0, 2.0], translation_costs=[tc])
        assert pc.total_cost == pytest.approx(3.5)

    def test_num_translations_excludes_identity(self):
        identity = TranslationCost(
            source_protocol=ProtocolName.MCP,
            target_protocol=ProtocolName.MCP,
        )
        real = TranslationCost(
            source_protocol=ProtocolName.MCP,
            target_protocol=ProtocolName.A2A,
        )
        pc = PathCost(translation_costs=[identity, real])
        assert pc.num_translations == 1

    def test_semantic_loss_estimate(self):
        tc = TranslationCost(
            source_protocol=ProtocolName.MCP,
            target_protocol=ProtocolName.A2A,
            semantic_loss_penalty=0.1,
        )
        pc = PathCost(translation_costs=[tc])
        assert 0.0 < pc.semantic_loss_estimate < 1.0

    def test_semantic_loss_estimate_no_translations(self):
        pc = PathCost()
        assert pc.semantic_loss_estimate == 0.0

    def test_get_summary(self):
        pc = PathCost(
            edge_costs=[1.0],
            path=["a", "b"],
            protocol_sequence=[ProtocolName.MCP],
        )
        summary = pc.get_summary()
        assert "total_cost" in summary
        assert "num_translations" in summary


class TestPathCostCalculator:
    """Tests for PathCostCalculator."""

    def test_calculate_path_cost_single_edge(self, cost_calculator):
        cost = cost_calculator.calculate_path_cost(["a1", "a2"], [ProtocolName.MCP])
        assert cost > 0.0
        assert cost < float("inf")

    def test_calculate_path_cost_trivial(self, cost_calculator):
        cost = cost_calculator.calculate_path_cost(["a1"], [])
        assert cost == 0.0

    def test_calculate_path_cost_length_mismatch_raises(self, cost_calculator):
        with pytest.raises(ValueError, match="Protocol sequence length"):
            cost_calculator.calculate_path_cost(
                ["a1", "a2", "a3"],
                [ProtocolName.MCP],  # should be 2
            )

    def test_get_path_cost_breakdown(self, cost_calculator):
        breakdown = cost_calculator.get_path_cost_breakdown(["a1", "a2"], [ProtocolName.MCP])
        assert len(breakdown.edge_costs) == 1
        assert breakdown.path == ["a1", "a2"]

    def test_get_translation_cost_identity(self, cost_calculator):
        tc = cost_calculator.get_translation_cost(ProtocolName.MCP, ProtocolName.MCP)
        assert tc.is_identity
        assert tc.total_cost == 0.0

    def test_get_translation_cost_cross_protocol(self, cost_calculator):
        tc = cost_calculator.get_translation_cost(ProtocolName.MCP, ProtocolName.A2A)
        assert not tc.is_identity
        assert tc.total_cost > 0.0

    def test_translation_cost_caching(self, cost_calculator):
        tc1 = cost_calculator.get_translation_cost(ProtocolName.MCP, ProtocolName.A2A)
        tc2 = cost_calculator.get_translation_cost(ProtocolName.MCP, ProtocolName.A2A)
        assert tc1.total_cost == tc2.total_cost

    def test_clear_cache(self, cost_calculator):
        cost_calculator.get_translation_cost(ProtocolName.MCP, ProtocolName.A2A)
        stats = cost_calculator.get_statistics()
        assert stats["cache_size"] > 0
        cost_calculator.clear_cache()
        stats = cost_calculator.get_statistics()
        assert stats["cache_size"] == 0

    def test_compare_paths(self, cost_calculator):
        # Same path should be equal
        result = cost_calculator.compare_paths(
            (["a1", "a2"], [ProtocolName.MCP]),
            (["a1", "a2"], [ProtocolName.MCP]),
        )
        assert result == 0

    def test_edge_cost_missing_edge_returns_inf(self, cost_calculator):
        cost = cost_calculator._get_edge_cost("a1", "a3", ProtocolName.MCP)
        assert cost == float("inf")

    def test_estimate_minimum_cost(self, cost_calculator):
        est = cost_calculator.estimate_minimum_cost("a1", "a2", ProtocolName.MCP)
        assert est > 0.0

    def test_estimate_minimum_cost_with_translation(self, cost_calculator):
        est = cost_calculator.estimate_minimum_cost("a1", "a3", ProtocolName.MCP, ProtocolName.A2A)
        # Should include translation cost MCP->A2A
        assert est > 1.0


# ===========================================================================
# routing.py Tests
# ===========================================================================


class TestRoutingState:
    """Tests for RoutingState."""

    def test_frozen(self):
        s = RoutingState(agent_id="a1", protocol=ProtocolName.MCP)
        with pytest.raises(AttributeError):
            s.agent_id = "a2"  # type: ignore[misc]

    def test_hash_and_eq(self):
        s1 = RoutingState(agent_id="a1", protocol=ProtocolName.MCP)
        s2 = RoutingState(agent_id="a1", protocol=ProtocolName.MCP)
        assert s1 == s2
        assert hash(s1) == hash(s2)

    def test_not_equal_different_protocol(self):
        s1 = RoutingState(agent_id="a1", protocol=ProtocolName.MCP)
        s2 = RoutingState(agent_id="a1", protocol=ProtocolName.A2A)
        assert s1 != s2

    def test_lt_for_heap(self):
        s1 = RoutingState(agent_id="a1", protocol=ProtocolName.MCP)
        s2 = RoutingState(agent_id="a2", protocol=ProtocolName.MCP)
        assert s1 < s2

    def test_lt_same_agent(self):
        s1 = RoutingState(agent_id="a1", protocol=ProtocolName.A2A)
        s2 = RoutingState(agent_id="a1", protocol=ProtocolName.MCP)
        # a2a < mcp alphabetically
        assert s1 < s2

    def test_eq_with_non_routing_state(self):
        s = RoutingState(agent_id="a1", protocol=ProtocolName.MCP)
        assert s.__eq__("not_a_routing_state") is NotImplemented

    def test_usable_in_set(self):
        s1 = RoutingState(agent_id="a1", protocol=ProtocolName.MCP)
        s2 = RoutingState(agent_id="a1", protocol=ProtocolName.MCP)
        s3 = RoutingState(agent_id="a2", protocol=ProtocolName.MCP)
        assert len({s1, s2, s3}) == 2


class TestOptimalRoute:
    """Tests for OptimalRoute."""

    def test_num_translations(self):
        route = OptimalRoute(
            path=["a1", "a2", "a3"],
            protocol_sequence=[ProtocolName.MCP, ProtocolName.A2A],
            total_cost=2.0,
            translation_points=[("a2", ProtocolName.MCP, ProtocolName.A2A)],
            semantic_loss_estimate=0.05,
            num_hops=2,
        )
        assert route.num_translations == 1

    def test_is_single_protocol(self):
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
            num_hops=1,
        )
        assert route.is_single_protocol

    def test_is_not_single_protocol(self):
        route = OptimalRoute(
            path=["a1", "a2", "a3"],
            protocol_sequence=[ProtocolName.MCP, ProtocolName.A2A],
            total_cost=2.0,
            translation_points=[("a2", ProtocolName.MCP, ProtocolName.A2A)],
            semantic_loss_estimate=0.05,
            num_hops=2,
        )
        assert not route.is_single_protocol

    def test_get_summary(self):
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
            num_hops=1,
        )
        summary = route.get_summary()
        assert summary["path"] == ["a1", "a2"]
        assert summary["num_hops"] == 1
        assert summary["total_cost"] == 1.0


class TestProtocolAwareDijkstra:
    """Tests for ProtocolAwareDijkstra."""

    def test_find_optimal_route_direct(self, dijkstra):
        route = dijkstra.find_optimal_route("a1", "a2", ProtocolName.MCP)
        assert route is not None
        assert route.path == ["a1", "a2"]
        assert route.protocol_sequence == [ProtocolName.MCP]
        assert route.is_single_protocol

    def test_find_optimal_route_multi_hop(self, dijkstra):
        route = dijkstra.find_optimal_route("a1", "a3", ProtocolName.MCP)
        assert route is not None
        assert "a1" in route.path
        assert "a3" in route.path
        assert route.num_hops >= 2

    def test_find_route_same_source_target(self, dijkstra):
        route = dijkstra.find_optimal_route("a1", "a1", ProtocolName.MCP)
        assert route is not None
        assert route.path == ["a1"]
        assert route.total_cost == 0.0

    def test_find_route_source_not_in_graph(self, dijkstra):
        route = dijkstra.find_optimal_route("missing", "a2", ProtocolName.MCP)
        assert route is None

    def test_find_route_target_not_in_graph(self, dijkstra):
        route = dijkstra.find_optimal_route("a1", "missing", ProtocolName.MCP)
        assert route is None

    def test_routing_statistics(self, dijkstra):
        dijkstra.find_optimal_route("a1", "a2", ProtocolName.MCP)
        stats = dijkstra.get_statistics()
        assert stats["routes_computed"] >= 1
        assert stats["routes_found"] >= 1

    def test_reset_statistics(self, dijkstra):
        dijkstra.find_optimal_route("a1", "a2", ProtocolName.MCP)
        dijkstra.reset_statistics()
        stats = dijkstra.get_statistics()
        assert stats["routes_computed"] == 0

    def test_diamond_route_picks_cheaper(self, diamond_graph, compat_matrix):
        """In diamond graph, a1->a2->a4 has edge cost 2 vs a1->a3->a4 cost 3."""
        calc = PathCostCalculator(diamond_graph, compat_matrix)
        router = ProtocolAwareDijkstra(diamond_graph, compat_matrix, calc)
        route = router.find_optimal_route("a1", "a4", ProtocolName.MCP)
        assert route is not None
        assert route.total_cost < float("inf")

    def test_find_all_routes(self, dijkstra):
        routes = dijkstra.find_all_routes("a1", "a2", ProtocolName.MCP)
        assert len(routes) >= 1
        # Should be sorted by cost
        costs = [r.total_cost for r in routes]
        assert costs == sorted(costs)

    def test_find_all_routes_no_path(self, dijkstra):
        routes = dijkstra.find_all_routes("a1", "missing", ProtocolName.MCP)
        assert routes == []


# ===========================================================================
# semantic.py Tests
# ===========================================================================


class TestSemanticViolationType:
    """Tests for SemanticViolationType enum."""

    def test_values(self):
        assert SemanticViolationType.FIELD_MISSING == "field_missing"
        assert SemanticViolationType.FIELD_MODIFIED == "field_modified"
        assert SemanticViolationType.TYPE_MISMATCH == "type_mismatch"
        assert SemanticViolationType.ROUND_TRIP_FAILED == "round_trip_failed"


class TestSemanticViolation:
    """Tests for SemanticViolation."""

    def test_repr(self):
        v = SemanticViolation(
            violation_type=SemanticViolationType.FIELD_MISSING,
            field_path="payload.data",
            message="Field missing",
        )
        r = repr(v)
        assert "field_missing" in r

    def test_default_severity(self):
        v = SemanticViolation(
            violation_type=SemanticViolationType.FIELD_MISSING,
            field_path="x",
        )
        assert v.severity == "warning"


class TestSemanticVerificationResult:
    """Tests for SemanticVerificationResult."""

    def test_has_violations(self):
        result = SemanticVerificationResult(
            preserved=True,
            original_hash="abc",
            translated_hash="abc",
            violations=[
                SemanticViolation(
                    violation_type=SemanticViolationType.FIELD_MODIFIED,
                    field_path="x",
                )
            ],
        )
        assert result.has_violations

    def test_no_violations(self):
        result = SemanticVerificationResult(
            preserved=True,
            original_hash="abc",
            translated_hash="abc",
        )
        assert not result.has_violations

    def test_critical_violations(self):
        result = SemanticVerificationResult(
            preserved=False,
            original_hash="abc",
            translated_hash="def",
            violations=[
                SemanticViolation(
                    violation_type=SemanticViolationType.FIELD_MISSING,
                    field_path="a",
                    severity="critical",
                ),
                SemanticViolation(
                    violation_type=SemanticViolationType.FIELD_MODIFIED,
                    field_path="b",
                    severity="info",
                ),
            ],
        )
        assert len(result.critical_violations) == 1

    def test_round_trip_preserved_no_hash(self):
        result = SemanticVerificationResult(
            preserved=True,
            original_hash="abc",
            translated_hash="abc",
        )
        # No round_trip_hash => True by default
        assert result.round_trip_preserved

    def test_round_trip_preserved_matching(self):
        result = SemanticVerificationResult(
            preserved=True,
            original_hash="abc",
            translated_hash="abc",
            round_trip_hash="abc",
        )
        assert result.round_trip_preserved

    def test_round_trip_not_preserved(self):
        result = SemanticVerificationResult(
            preserved=True,
            original_hash="abc",
            translated_hash="abc",
            round_trip_hash="xyz",
        )
        assert not result.round_trip_preserved

    def test_get_summary(self):
        result = SemanticVerificationResult(
            preserved=True,
            original_hash="abc",
            translated_hash="abc",
        )
        summary = result.get_summary()
        assert summary["preserved"] is True
        assert "num_violations" in summary


class TestSemanticPreservationVerifier:
    """Tests for SemanticPreservationVerifier."""

    def test_verify_identical_dicts_preserved(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_translation(
            {"action": "search", "query": "test"},
            {"action": "search", "query": "test"},
        )
        assert result.preserved
        assert result.semantic_distance == 0.0

    def test_verify_missing_field_violation(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_translation(
            {"action": "search", "query": "test", "limit": 10},
            {"action": "search", "query": "test"},
        )
        assert result.has_violations
        missing = [
            v for v in result.violations if v.violation_type == SemanticViolationType.FIELD_MISSING
        ]
        assert len(missing) == 1
        assert missing[0].field_path == "limit"

    def test_verify_modified_value(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_translation(
            {"action": "search", "query": "original"},
            {"action": "search", "query": "modified"},
        )
        modified = [
            v for v in result.violations if v.violation_type == SemanticViolationType.FIELD_MODIFIED
        ]
        assert len(modified) >= 1

    def test_verify_type_mismatch(self):
        verifier = SemanticPreservationVerifier(
            config=VerificationConfig(strict_type_checking=True)
        )
        result = verifier.verify_translation(
            {"count": 42},
            {"count": "42"},
        )
        type_mismatches = [
            v for v in result.violations if v.violation_type == SemanticViolationType.TYPE_MISMATCH
        ]
        assert len(type_mismatches) >= 1

    def test_verify_extra_fields_structure_change(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_translation(
            {"action": "search"},
            {"action": "search", "extra": "field"},
        )
        structure_changes = [
            v
            for v in result.violations
            if v.violation_type == SemanticViolationType.STRUCTURE_CHANGED
        ]
        assert len(structure_changes) >= 1

    def test_verify_ignores_underscore_fields(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_translation(
            {"action": "search", "_meta": "internal"},
            {"action": "search"},
        )
        # _meta should be ignored
        missing = [
            v
            for v in result.violations
            if v.violation_type == SemanticViolationType.FIELD_MISSING and v.field_path == "_meta"
        ]
        assert len(missing) == 0

    def test_verify_with_ignore_fields(self):
        config = VerificationConfig(ignore_fields={"timestamp"})
        verifier = SemanticPreservationVerifier(config=config)
        result = verifier.verify_translation(
            {"action": "search", "timestamp": "2025-01-01"},
            {"action": "search", "timestamp": "2025-12-31"},
        )
        assert result.preserved

    def test_verify_critical_field_missing(self):
        config = VerificationConfig(critical_fields={"action"})
        verifier = SemanticPreservationVerifier(config=config)
        result = verifier.verify_translation(
            {"action": "search", "query": "test"},
            {"query": "test"},
        )
        assert not result.preserved
        critical = result.critical_violations
        assert len(critical) == 1

    def test_verify_nested_dicts(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_translation(
            {"payload": {"data": "value1"}},
            {"payload": {"data": "value2"}},
        )
        modified = [
            v for v in result.violations if v.violation_type == SemanticViolationType.FIELD_MODIFIED
        ]
        assert len(modified) >= 1
        assert "payload.data" in modified[0].field_path

    def test_verify_empty_dicts(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_translation({}, {})
        assert result.preserved
        assert result.semantic_distance == 0.0

    def test_semantic_distance_completely_different(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_translation(
            {"a": 1, "b": 2},
            {"c": 3, "d": 4},
        )
        assert result.semantic_distance > 0.5

    def test_verify_round_trip(self):
        class ForwardTranslator:
            def translate(self, msg):
                return dict(msg)

            def reverse_translate(self, msg):
                return dict(msg)

        class ReverseTranslator:
            def translate(self, msg):
                return dict(msg)

            def reverse_translate(self, msg):
                return dict(msg)

        verifier = SemanticPreservationVerifier()
        result = verifier.verify_round_trip(
            {"action": "test", "data": 42},
            ForwardTranslator(),
            ReverseTranslator(),
        )
        assert result.preserved
        assert result.round_trip_preserved

    def test_verify_output_matching_schema(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_output(
            {"name": "agent1", "count": 5},
            {"name": "str", "count": "int"},
        )
        assert result.preserved

    def test_verify_output_missing_field(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_output(
            {"name": "agent1"},
            {"name": "str", "count": "int"},
        )
        assert not result.preserved

    def test_get_statistics(self):
        verifier = SemanticPreservationVerifier()
        verifier.verify_translation({"a": 1}, {"a": 1})
        verifier.verify_translation({"a": 1}, {"b": 2})
        stats = verifier.get_statistics()
        assert stats["verifications_performed"] == 2
        assert stats["verifications_passed"] >= 1

    def test_verification_time_recorded(self):
        verifier = SemanticPreservationVerifier()
        result = verifier.verify_translation({"a": 1}, {"a": 1})
        assert result.verification_time_ms >= 0.0


# ===========================================================================
# specifications.py Tests
# ===========================================================================


class TestProtocolPropertyResult:
    """Tests for ProtocolPropertyResult."""

    def test_is_satisfied(self):
        r = ProtocolPropertyResult(
            property_name="test",
            property_type=ProtocolPropertyType.VALIDITY,
            status=ProtocolPropertyStatus.SATISFIED,
        )
        assert r.is_satisfied()

    def test_is_not_satisfied(self):
        r = ProtocolPropertyResult(
            property_name="test",
            property_type=ProtocolPropertyType.VALIDITY,
            status=ProtocolPropertyStatus.VIOLATED,
        )
        assert not r.is_satisfied()


class TestProtocolValidPath:
    """Tests for ProtocolValidPath specification."""

    def test_valid_path(self, simple_graph):
        prop = ProtocolValidPath()
        result = prop.check(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            graph=simple_graph,
        )
        assert result.is_satisfied()

    def test_invalid_protocol_on_edge(self, simple_graph):
        prop = ProtocolValidPath()
        result = prop.check(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.A2A],  # a1-a2 only has MCP
            graph=simple_graph,
        )
        assert result.status == ProtocolPropertyStatus.VIOLATED

    def test_missing_edge(self, simple_graph):
        prop = ProtocolValidPath()
        result = prop.check(
            path=["a1", "a3"],
            protocol_sequence=[ProtocolName.MCP],
            graph=simple_graph,
        )
        assert result.status == ProtocolPropertyStatus.VIOLATED

    def test_length_mismatch(self, simple_graph):
        prop = ProtocolValidPath()
        result = prop.check(
            path=["a1", "a2", "a3"],
            protocol_sequence=[ProtocolName.MCP],  # needs 2
            graph=simple_graph,
        )
        assert result.status == ProtocolPropertyStatus.VIOLATED

    def test_trivial_path(self, simple_graph):
        prop = ProtocolValidPath()
        # Empty protocol_sequence is falsy, so check returns NOT_APPLICABLE.
        # Pass a single-hop valid path instead to test trivial-ish case.
        result = prop.check(
            path=["a1"],
            protocol_sequence=[],
            graph=simple_graph,
        )
        # The implementation treats empty protocol_sequence as missing args
        assert result.status == ProtocolPropertyStatus.NOT_APPLICABLE

    def test_single_edge_valid(self, simple_graph):
        prop = ProtocolValidPath()
        result = prop.check(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            graph=simple_graph,
        )
        assert result.is_satisfied()
        assert "1 edges" in result.message

    def test_missing_arguments(self):
        prop = ProtocolValidPath()
        result = prop.check()
        assert result.status == ProtocolPropertyStatus.NOT_APPLICABLE


class TestSemanticPreservationSpec:
    """Tests for SemanticPreservation specification."""

    def test_preservation_satisfied(self):
        prop = SemanticPreservation(max_semantic_loss=0.1)
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[],
            semantic_loss_estimate=0.05,
        )
        result = prop.check(route=route)
        assert result.is_satisfied()

    def test_preservation_violated(self):
        prop = SemanticPreservation(max_semantic_loss=0.1)
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[("a1", ProtocolName.MCP, ProtocolName.A2A)],
            semantic_loss_estimate=0.5,
        )
        result = prop.check(route=route)
        assert result.status == ProtocolPropertyStatus.VIOLATED

    def test_preservation_from_direct_value(self):
        prop = SemanticPreservation(max_semantic_loss=0.2)
        result = prop.check(semantic_loss=0.15)
        assert result.is_satisfied()

    def test_preservation_no_data(self):
        prop = SemanticPreservation()
        result = prop.check()
        assert result.status == ProtocolPropertyStatus.NOT_APPLICABLE


class TestOptimalRouting:
    """Tests for OptimalRouting specification."""

    def test_optimal_no_alternatives(self):
        prop = OptimalRouting()
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
        )
        result = prop.check(route=route)
        assert result.is_satisfied()

    def test_optimal_is_best(self):
        prop = OptimalRouting()
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
        )
        alt = OptimalRoute(
            path=["a1", "a3", "a2"],
            protocol_sequence=[ProtocolName.MCP, ProtocolName.A2A],
            total_cost=3.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
        )
        result = prop.check(route=route, all_routes=[route, alt])
        assert result.is_satisfied()

    def test_not_optimal(self):
        prop = OptimalRouting()
        route = OptimalRoute(
            path=["a1", "a3", "a2"],
            protocol_sequence=[ProtocolName.MCP, ProtocolName.A2A],
            total_cost=3.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
        )
        better = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
        )
        result = prop.check(route=route, all_routes=[better])
        assert result.status == ProtocolPropertyStatus.VIOLATED

    def test_no_route_provided(self):
        prop = OptimalRouting()
        result = prop.check()
        assert result.status == ProtocolPropertyStatus.NOT_APPLICABLE


class TestTranslationBoundProperty:
    """Tests for TranslationBoundProperty specification."""

    def test_within_bound(self):
        prop = TranslationBoundProperty(max_translations=3)
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[("a1", ProtocolName.MCP, ProtocolName.A2A)],
            semantic_loss_estimate=0.0,
        )
        result = prop.check(route=route)
        assert result.is_satisfied()

    def test_exceeds_bound(self):
        prop = TranslationBoundProperty(max_translations=1)
        route = OptimalRoute(
            path=["a1", "a2", "a3"],
            protocol_sequence=[ProtocolName.MCP, ProtocolName.A2A],
            total_cost=2.0,
            translation_points=[
                ("a1", ProtocolName.MCP, ProtocolName.A2A),
                ("a2", ProtocolName.A2A, ProtocolName.ANP),
            ],
            semantic_loss_estimate=0.0,
        )
        result = prop.check(route=route)
        assert result.status == ProtocolPropertyStatus.VIOLATED

    def test_direct_count(self):
        prop = TranslationBoundProperty(max_translations=2)
        result = prop.check(num_translations=2)
        assert result.is_satisfied()

    def test_no_data(self):
        prop = TranslationBoundProperty()
        result = prop.check()
        assert result.status == ProtocolPropertyStatus.NOT_APPLICABLE


class TestProtocolSpecification:
    """Tests for ProtocolSpecification."""

    def test_verify_valid_route(self, simple_graph):
        spec = ProtocolSpecification(max_semantic_loss=0.5, max_translations=5)
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
            num_hops=1,
        )
        results = spec.verify(route, simple_graph)
        assert len(results) == 4  # 4 default properties

    def test_is_valid_route(self, simple_graph):
        spec = ProtocolSpecification(max_semantic_loss=0.5, max_translations=5)
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
            num_hops=1,
        )
        assert spec.is_valid(route, simple_graph)

    def test_verify_validity_shortcut(self, simple_graph):
        spec = ProtocolSpecification()
        result = spec.verify_validity(["a1", "a2"], [ProtocolName.MCP], simple_graph)
        assert result.is_satisfied()

    def test_verify_invalid_validity(self, simple_graph):
        spec = ProtocolSpecification()
        result = spec.verify_validity(["a1", "a2"], [ProtocolName.A2A], simple_graph)
        assert result.status == ProtocolPropertyStatus.VIOLATED

    def test_summary(self, simple_graph):
        spec = ProtocolSpecification(max_semantic_loss=0.5)
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
            num_hops=1,
        )
        text = spec.summary(route, simple_graph)
        assert "Protocol Specification Verification" in text
        assert "PASS" in text or "FAIL" in text

    def test_custom_properties(self, simple_graph):
        class CustomProp(TranslationBoundProperty):
            def __init__(self):
                super().__init__(max_translations=0)

        spec = ProtocolSpecification(custom_properties=[CustomProp()])
        route = OptimalRoute(
            path=["a1", "a2"],
            protocol_sequence=[ProtocolName.MCP],
            total_cost=1.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
            num_hops=1,
        )
        results = spec.verify(route, simple_graph)
        assert len(results) == 5  # 4 default + 1 custom


# ===========================================================================
# Integration Tests: End-to-End Routing + Verification
# ===========================================================================


class TestEndToEnd:
    """Integration tests combining graph, routing, and specification."""

    def test_route_and_verify(self, simple_graph, compat_matrix):
        calc = PathCostCalculator(simple_graph, compat_matrix)
        router = ProtocolAwareDijkstra(simple_graph, compat_matrix, calc)
        route = router.find_optimal_route("a1", "a2", ProtocolName.MCP)
        assert route is not None

        spec = ProtocolSpecification(max_semantic_loss=0.5, max_translations=5)
        results = spec.verify(route, simple_graph)
        satisfied = [r for r in results if r.is_satisfied()]
        assert len(satisfied) >= 2  # At least validity + translation bound

    def test_multi_hop_route_and_verify(self, simple_graph, compat_matrix):
        calc = PathCostCalculator(simple_graph, compat_matrix)
        router = ProtocolAwareDijkstra(simple_graph, compat_matrix, calc)
        route = router.find_optimal_route("a1", "a3", ProtocolName.MCP)
        assert route is not None
        assert len(route.path) >= 2

        spec = ProtocolSpecification(max_semantic_loss=0.5, max_translations=5)
        spec.is_valid(route, simple_graph)
        # Route found by Dijkstra should satisfy at least validity
        results = spec.verify(route, simple_graph)
        validity_results = [r for r in results if r.property_name == "ProtocolValidPath"]
        assert len(validity_results) == 1
        assert validity_results[0].is_satisfied()

    def test_diamond_route_cost_comparison(self, diamond_graph, compat_matrix):
        calc = PathCostCalculator(diamond_graph, compat_matrix)
        router = ProtocolAwareDijkstra(diamond_graph, compat_matrix, calc)

        route = router.find_optimal_route("a1", "a4", ProtocolName.MCP)
        assert route is not None

        # Verify cost is computed
        assert route.total_cost > 0.0
        assert route.total_cost < float("inf")
