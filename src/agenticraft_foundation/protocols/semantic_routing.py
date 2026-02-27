"""Semantic Protocol-Aware Routing (Algorithm 4).

Routes messages based on semantic similarity of capabilities,
combined with protocol affinity and path cost.

Score: similarity(cap_q, cap_a) × affinity(p, cap_a) / (1 + cost(π))

Zero external dependencies — uses pure Python bag-of-words with
cosine similarity for capability embeddings.

Falls back to ProtocolAwareDijkstra when embeddings are unavailable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agenticraft_foundation.types import ProtocolName

if TYPE_CHECKING:
    from .affinity import CapabilityAffinityMatrix
    from .graph import ProtocolGraph

logger = logging.getLogger(__name__)


@dataclass
class CapabilityEmbedding:
    """Vector representation of a capability for semantic matching.

    Uses bag-of-words (BoW) with optional TF-IDF weighting.
    Zero external dependencies.
    """

    capability: str
    """Original capability string"""

    vector: dict[str, float] = field(default_factory=dict)
    """Sparse vector: token → weight"""

    @classmethod
    def from_text(cls, capability: str) -> CapabilityEmbedding:
        """Create embedding from capability text.

        Tokenizes on underscores, hyphens, spaces, and camelCase boundaries.
        """
        tokens = cls._tokenize(capability)
        vector: dict[str, float] = {}
        for token in tokens:
            token_lower = token.lower()
            vector[token_lower] = vector.get(token_lower, 0.0) + 1.0

        # Normalize
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            vector = {k: v / norm for k, v in vector.items()}

        return cls(capability=capability, vector=vector)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize capability string."""
        # Replace separators with spaces
        for sep in ["_", "-", ".", "/", ":"]:
            text = text.replace(sep, " ")

        # Split camelCase
        result: list[str] = []
        current = ""
        for char in text:
            if char.isupper() and current:
                result.append(current)
                current = char.lower()
            elif char == " ":
                if current:
                    result.append(current)
                current = ""
            else:
                current += char.lower()
        if current:
            result.append(current)

        return [t for t in result if len(t) > 1]  # Filter single chars

    def similarity(self, other: CapabilityEmbedding) -> float:
        """Compute cosine similarity with another embedding."""
        if not self.vector or not other.vector:
            return 0.0

        # Sparse dot product
        dot = sum(self.vector[k] * other.vector.get(k, 0.0) for k in self.vector)

        # Both vectors are already normalized
        return max(0.0, min(1.0, dot))


@dataclass
class SemanticRouteCandidate:
    """A candidate agent for semantic routing."""

    agent_id: str
    """Agent identifier"""

    capability_similarity: float
    """Semantic similarity of agent's capabilities to query"""

    protocol_affinity: float
    """Protocol affinity score"""

    path_cost: float
    """Estimated path cost from source"""

    composite_score: float
    """Combined routing score"""

    matched_capability: str
    """Best matching capability on this agent"""


class SemanticRouter:
    """Semantic protocol-aware routing.

    Finds the best agent to handle a capability query by balancing:
    1. Semantic similarity between requested and available capabilities
    2. Protocol affinity for the agent's protocols
    3. Path cost from source to agent

    Score = similarity × affinity / (1 + cost)
    """

    def __init__(
        self,
        graph: ProtocolGraph,
        affinity_matrix: CapabilityAffinityMatrix | None = None,
        similarity_threshold: float = 0.1,
    ):
        """Initialize semantic router.

        Args:
            graph: Protocol graph
            affinity_matrix: Capability-protocol affinity matrix
            similarity_threshold: Minimum similarity to consider a match
        """
        self._graph = graph
        self._affinity = affinity_matrix
        self._threshold = similarity_threshold

        # Pre-compute capability embeddings for all agents
        self._embeddings: dict[str, list[CapabilityEmbedding]] = {}
        for agent_id, agent in graph.agents.items():
            self._embeddings[agent_id] = [
                CapabilityEmbedding.from_text(cap) for cap in agent.capabilities
            ]

    def find_best_agent(
        self,
        query_capability: str,
        source: str,
        source_protocol: ProtocolName,
        top_k: int = 5,
    ) -> list[SemanticRouteCandidate]:
        """Find best agents for a capability query.

        Args:
            query_capability: Requested capability
            source: Source agent ID
            source_protocol: Source agent's protocol
            top_k: Number of top candidates to return

        Returns:
            Sorted list of SemanticRouteCandidate (best first)
        """
        query_embedding = CapabilityEmbedding.from_text(query_capability)
        candidates: list[SemanticRouteCandidate] = []

        for agent_id, agent in self._graph.agents.items():
            if agent_id == source:
                continue

            # Find best matching capability
            best_sim = 0.0
            best_cap = ""
            for emb in self._embeddings.get(agent_id, []):
                sim = query_embedding.similarity(emb)
                if sim > best_sim:
                    best_sim = sim
                    best_cap = emb.capability

            if best_sim < self._threshold:
                continue

            # Protocol affinity
            affinity = self._compute_affinity(best_cap, agent.protocols)

            # Path cost (simplified: use edge weight if direct, else estimate)
            path_cost = self._estimate_cost(source, agent_id, source_protocol)

            # Composite score
            score = best_sim * affinity / (1.0 + path_cost)

            candidates.append(
                SemanticRouteCandidate(
                    agent_id=agent_id,
                    capability_similarity=best_sim,
                    protocol_affinity=affinity,
                    path_cost=path_cost,
                    composite_score=score,
                    matched_capability=best_cap,
                )
            )

        # Sort by composite score descending
        candidates.sort(key=lambda c: c.composite_score, reverse=True)
        return candidates[:top_k]

    def _compute_affinity(
        self,
        capability: str,
        protocols: set[ProtocolName],
    ) -> float:
        """Compute protocol affinity for a capability."""
        if self._affinity is None:
            return 1.0  # No affinity matrix — neutral score

        max_affinity = 0.0
        for protocol in protocols:
            score = self._affinity.get_affinity(capability, protocol)
            max_affinity = max(max_affinity, score)

        return max_affinity if max_affinity > 0 else 0.5

    def _estimate_cost(
        self,
        source: str,
        target: str,
        source_protocol: ProtocolName,
    ) -> float:
        """Estimate path cost from source to target."""
        # Check for direct edge
        edge = self._graph.edges.get((source, target))
        if edge is not None:
            return edge.get_weight(source_protocol)

        # Estimate based on hop count via reachability
        # Simple BFS for distance
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(source, 0)]

        while queue:
            node, dist = queue.pop(0)
            if node == target:
                return float(dist) * 1.0  # Unit cost per hop
            if node in visited:
                continue
            visited.add(node)

            for (src, tgt), _edge in self._graph.edges.items():
                if src == node and tgt not in visited:
                    queue.append((tgt, dist + 1))
                elif tgt == node and src not in visited:
                    queue.append((src, dist + 1))

        return float("inf")  # Unreachable


__all__ = [
    "CapabilityEmbedding",
    "SemanticRouteCandidate",
    "SemanticRouter",
]
