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
