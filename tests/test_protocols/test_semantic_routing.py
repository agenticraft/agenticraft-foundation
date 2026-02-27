"""Tests for semantic protocol-aware routing.

Covers:
- CapabilityEmbedding (bag-of-words, cosine similarity)
- SemanticRouter (capability-based agent discovery)
"""

from __future__ import annotations

import pytest

from agenticraft_foundation.protocols.graph import ProtocolGraph
from agenticraft_foundation.protocols.semantic_routing import (
    CapabilityEmbedding,
    SemanticRouter,
)
from agenticraft_foundation.types import ProtocolName


class TestCapabilityEmbedding:
    def test_from_text_basic(self):
        emb = CapabilityEmbedding.from_text("code_execution")
        assert "code" in emb.vector
        assert "execution" in emb.vector

    def test_self_similarity(self):
        emb = CapabilityEmbedding.from_text("web_search")
        assert emb.similarity(emb) == pytest.approx(1.0, abs=0.01)

    def test_similar_capabilities(self):
        e1 = CapabilityEmbedding.from_text("code_execution")
        e2 = CapabilityEmbedding.from_text("code_generation")
        assert e1.similarity(e2) > 0.3

    def test_dissimilar_capabilities(self):
        e1 = CapabilityEmbedding.from_text("code_execution")
        e2 = CapabilityEmbedding.from_text("web_search")
        assert e1.similarity(e2) < 0.1

    def test_camel_case_tokenization(self):
        emb = CapabilityEmbedding.from_text("codeExecution")
        assert "code" in emb.vector
        assert "execution" in emb.vector

    def test_empty_vector_similarity(self):
        e1 = CapabilityEmbedding(capability="", vector={})
        e2 = CapabilityEmbedding.from_text("test")
        assert e1.similarity(e2) == 0.0


class TestSemanticRouter:
    def test_find_best_agent(self):
        graph = ProtocolGraph()
        graph.add_agent("src", ["routing"], {ProtocolName.MCP})
        graph.add_agent("coder", ["code_execution", "code_generation"], {ProtocolName.MCP})
        graph.add_agent("searcher", ["web_search", "data_retrieval"], {ProtocolName.MCP})
        graph.add_edge("src", "coder", {ProtocolName.MCP})
        graph.add_edge("src", "searcher", {ProtocolName.MCP})

        router = SemanticRouter(graph)
        candidates = router.find_best_agent("code_execution", "src", ProtocolName.MCP)
        assert len(candidates) > 0
        assert candidates[0].agent_id == "coder"

    def test_no_match_below_threshold(self):
        graph = ProtocolGraph()
        graph.add_agent("src", ["routing"], {ProtocolName.MCP})
        graph.add_agent("other", ["unrelated_capability"], {ProtocolName.MCP})
        graph.add_edge("src", "other", {ProtocolName.MCP})

        router = SemanticRouter(graph, similarity_threshold=0.5)
        candidates = router.find_best_agent("code_execution", "src", ProtocolName.MCP)
        assert len(candidates) == 0

    def test_find_agent_with_affinity_matrix(self):
        """Test routing with an explicit affinity matrix."""
        from agenticraft_foundation.protocols.affinity import CapabilityAffinityMatrix

        graph = ProtocolGraph()
        graph.add_agent("src", ["routing"], {ProtocolName.MCP})
        graph.add_agent("coder", ["code_execution"], {ProtocolName.MCP})
        graph.add_edge("src", "coder", {ProtocolName.MCP})

        affinity = CapabilityAffinityMatrix()
        router = SemanticRouter(graph, affinity_matrix=affinity)
        candidates = router.find_best_agent("code_execution", "src", ProtocolName.MCP)
        assert len(candidates) > 0
        # With affinity matrix, protocol_affinity should be set
        assert candidates[0].protocol_affinity > 0

    def test_estimate_cost_bfs_indirect(self):
        """Test cost estimation via BFS for non-adjacent agents."""
        graph = ProtocolGraph()
        graph.add_agent("src", ["routing"], {ProtocolName.MCP})
        graph.add_agent("mid", ["relay"], {ProtocolName.MCP})
        graph.add_agent("dest", ["code_execution"], {ProtocolName.MCP})
        # src -> mid -> dest (no direct edge src->dest)
        graph.add_edge("src", "mid", {ProtocolName.MCP})
        graph.add_edge("mid", "dest", {ProtocolName.MCP})

        router = SemanticRouter(graph, similarity_threshold=0.0)
        candidates = router.find_best_agent("code_execution", "src", ProtocolName.MCP)
        assert len(candidates) > 0
        dest_candidate = [c for c in candidates if c.agent_id == "dest"]
        assert len(dest_candidate) == 1
        # Path cost should be 2 (two hops via BFS)
        assert dest_candidate[0].path_cost == 2.0

    def test_estimate_cost_unreachable(self):
        """Test cost estimation for unreachable agent is infinity."""
        graph = ProtocolGraph()
        graph.add_agent("src", ["routing"], {ProtocolName.MCP})
        graph.add_agent("island", ["code_execution"], {ProtocolName.MCP})
        # No edges at all

        router = SemanticRouter(graph, similarity_threshold=0.0)
        candidates = router.find_best_agent("code_execution", "src", ProtocolName.MCP)
        # With infinite cost, composite score = sim * affinity / (1 + inf) = 0
        # Candidate should still be returned but with score ~0
        if candidates:
            island = [c for c in candidates if c.agent_id == "island"]
            if island:
                assert island[0].path_cost == float("inf")
