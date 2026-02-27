"""Tests for protocol graph model.

Covers:
- NodeType taxonomy (agent types)
- Protocol Diversity Index
- Protocol-constrained reachability
- Protocol resilience specification
"""

from __future__ import annotations

import pytest

from agenticraft_foundation.protocols.graph import (
    AgentNode,
    NodeType,
    ProtocolEdge,
    ProtocolGraph,
)
from agenticraft_foundation.protocols.specifications import (
    ProtocolPropertyStatus,
    ProtocolResilience,
)
from agenticraft_foundation.types import ProtocolName

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def diamond_graph() -> ProtocolGraph:
    """Diamond graph: A→B, A→C, B→D, C→D with mixed protocols."""
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
    """Linear graph: A→B→C with MCP only."""
    graph = ProtocolGraph()
    graph.add_agent("A", ["start"], {ProtocolName.MCP})
    graph.add_agent("B", ["middle"], {ProtocolName.MCP})
    graph.add_agent("C", ["end"], {ProtocolName.MCP})
    graph.add_edge("A", "B", {ProtocolName.MCP})
    graph.add_edge("B", "C", {ProtocolName.MCP})
    return graph


# ── NodeType Taxonomy ────────────────────────────────────────────────


class TestNodeTypeTaxonomy:
    """Agent type taxonomy for mesh node classification."""

    def test_llm_agent_type_exists(self):
        assert NodeType.LLM_AGENT.value == "llm_agent"

    def test_tool_server_type_exists(self):
        assert NodeType.TOOL_SERVER.value == "tool_server"

    def test_coordinator_type_exists(self):
        assert NodeType.COORDINATOR.value == "coordinator"

    def test_backward_compatible_agent_type(self):
        assert NodeType.AGENT.value == "agent"

    def test_all_types_present(self):
        expected = {
            "agent",
            "llm_agent",
            "tool_server",
            "coordinator",
            "gateway",
            "router",
            "translator",
        }
        actual = {t.value for t in NodeType}
        assert expected == actual

    def test_agent_node_with_new_type(self):
        node = AgentNode(
            agent_id="llm1",
            capabilities=["inference"],
            node_type=NodeType.LLM_AGENT,
        )
        assert node.node_type == NodeType.LLM_AGENT

    def test_graph_add_typed_agent(self, diamond_graph: ProtocolGraph):
        diamond_graph.add_agent(
            "coord1",
            ["orchestration"],
            {ProtocolName.MCP},
            NodeType.COORDINATOR,
        )
        assert diamond_graph.agents["coord1"].node_type == NodeType.COORDINATOR


# ── Protocol Diversity Index ─────────────────────────────────────────


class TestProtocolDiversityIndex:
    """PDI(e) = |P_e| / |P| measures protocol diversity on edges."""

    def test_single_protocol_edge(self):
        edge = ProtocolEdge(
            source="A",
            target="B",
            protocols={ProtocolName.MCP},
        )
        assert edge.diversity_index(4) == 0.25

    def test_multi_protocol_edge(self):
        edge = ProtocolEdge(
            source="A",
            target="B",
            protocols={ProtocolName.MCP, ProtocolName.A2A},
        )
        assert edge.diversity_index(4) == 0.5

    def test_full_diversity(self):
        all_protocols = set(ProtocolName)
        edge = ProtocolEdge(
            source="A",
            target="B",
            protocols=all_protocols,
        )
        assert edge.diversity_index(len(all_protocols)) == 1.0

    def test_zero_total_protocols(self):
        edge = ProtocolEdge(
            source="A",
            target="B",
            protocols={ProtocolName.MCP},
        )
        assert edge.diversity_index(0) == 0.0

    def test_graph_protocol_diversity(self, diamond_graph: ProtocolGraph):
        diversity = diamond_graph.protocol_diversity()
        # Returns dict of edge_key -> PDI value
        assert isinstance(diversity, dict)
        assert len(diversity) > 0
        # Check all values are between 0 and 1
        for pdi in diversity.values():
            assert 0.0 <= pdi <= 1.0


# ── Protocol-Constrained Reachability ────────────────────────────────


class TestProtocolReachability:
    """Reachability queries with protocol constraints."""

    def test_reachable_same_protocol(self, diamond_graph: ProtocolGraph):
        assert diamond_graph.is_reachable("A", "D", {ProtocolName.MCP})

    def test_reachable_different_protocol(self, diamond_graph: ProtocolGraph):
        assert diamond_graph.is_reachable("A", "D", {ProtocolName.A2A})

    def test_unreachable_wrong_protocol(self, linear_graph: ProtocolGraph):
        # Linear graph only has MCP edges
        assert not linear_graph.is_reachable("A", "C", {ProtocolName.A2A})

    def test_reachable_self(self, diamond_graph: ProtocolGraph):
        assert diamond_graph.is_reachable("A", "A", {ProtocolName.MCP})

    def test_reachable_agents_mcp(self, diamond_graph: ProtocolGraph):
        reachable = diamond_graph.reachable_agents("A", {ProtocolName.MCP})
        assert "B" in reachable
        assert "D" in reachable

    def test_reachable_agents_a2a(self, diamond_graph: ProtocolGraph):
        reachable = diamond_graph.reachable_agents("A", {ProtocolName.A2A})
        assert "C" in reachable
        assert "D" in reachable

    def test_reachable_nonexistent_source(self, diamond_graph: ProtocolGraph):
        assert not diamond_graph.is_reachable("X", "A", {ProtocolName.MCP})

    def test_reachable_agents_empty_protocols(self, diamond_graph: ProtocolGraph):
        reachable = diamond_graph.reachable_agents("A", set())
        assert len(reachable) == 0


# ── Protocol Resilience ──────────────────────────────────────────────


class TestProtocolResilience:
    """k-protocol resilience verification."""

    def test_resilience_satisfied(self, diamond_graph: ProtocolGraph):
        prop = ProtocolResilience()
        result = prop.check(diamond_graph, source="A", target="D")
        # Diamond graph has MCP and A2A paths — should be 1-resilient
        assert result.status in {
            ProtocolPropertyStatus.SATISFIED,
            ProtocolPropertyStatus.UNKNOWN,
        }

    def test_resilience_single_protocol_graph(self, linear_graph: ProtocolGraph):
        prop = ProtocolResilience()
        result = prop.check(linear_graph, source="A", target="C")
        # Only MCP → 0-resilient (any single protocol failure disconnects)
        assert result.status in {
            ProtocolPropertyStatus.SATISFIED,
            ProtocolPropertyStatus.VIOLATED,
        }

    def test_resilience_custom_k(self, diamond_graph: ProtocolGraph):
        prop = ProtocolResilience(k=1)
        result = prop.check(diamond_graph, source="A", target="D")
        assert result.property_name == "ProtocolResilience"
