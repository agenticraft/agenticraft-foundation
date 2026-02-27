"""Path cost function: cost(π, σ).

Computes cost(π, σ) = Σ wₑ(p) + Σ τ(pᵢ, pᵢ₊₁, vᵢ)
where π is the path, σ is the protocol sequence, wₑ(p) is the
edge weight for protocol p, and τ is the translation cost at each node.

Based on Definition 12 (Path Cost Function) from the formal multi-protocol mesh model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticraft_foundation.types import ProtocolName

if TYPE_CHECKING:
    from .compatibility import ProtocolCompatibilityMatrix
    from .graph import ProtocolGraph

logger = logging.getLogger(__name__)


@dataclass
class TranslationCost:
    """
    Cost of translating between protocols at a node.

    Represents τ(pᵢₙ, pₒᵤₜ, v) — the cost incurred when translating
    from pᵢₙ to pₒᵤₜ at node v.
    """

    source_protocol: ProtocolName
    """Incoming protocol (pᵢₙ)"""

    target_protocol: ProtocolName
    """Outgoing protocol (pₒᵤₜ)"""

    base_cost: float = 0.1
    """τ_base — base translation overhead"""

    semantic_loss_penalty: float = 0.0
    """τ_semantic — penalty for semantic loss during translation"""

    latency_ms: float = 10.0
    """Expected latency in milliseconds for this translation"""

    node_id: str | None = None
    """Node where translation occurs (v)"""

    @property
    def total_cost(self) -> float:
        """
        Total translation cost.

        τ(pᵢₙ, pₒᵤₜ, v) = τ_base + τ_semantic
        """
        return self.base_cost + self.semantic_loss_penalty

    @property
    def is_identity(self) -> bool:
        """Check if this is an identity translation (same protocol)."""
        return self.source_protocol == self.target_protocol

    def __repr__(self) -> str:
        return (
            f"TranslationCost({self.source_protocol.value}→{self.target_protocol.value}, "
            f"cost={self.total_cost:.3f}, latency={self.latency_ms}ms)"
        )


@dataclass
class PathCost:
    """
    Complete cost breakdown for a protocol-aware path.

    Represents cost(π, σ) = Σ wₑ(p) + Σ τ(pᵢ, pᵢ₊₁, vᵢ) with detailed breakdown.
    """

    edge_costs: list[float] = field(default_factory=list)
    """wₑ(p) for each edge in the path"""

    translation_costs: list[TranslationCost] = field(default_factory=list)
    """τ(pᵢ, pᵢ₊₁, vᵢ) for each translation point"""

    path: list[str] = field(default_factory=list)
    """Agent IDs in the path"""

    protocol_sequence: list[ProtocolName] = field(default_factory=list)
    """Protocols used on each edge"""

    @property
    def total_edge_cost(self) -> float:
        """Sum of all edge costs: Σ wₑ(p)"""
        return sum(self.edge_costs)

    @property
    def total_translation_cost(self) -> float:
        """Sum of all translation costs: Σ τ(pᵢ, pᵢ₊₁, vᵢ)"""
        return sum(tc.total_cost for tc in self.translation_costs)

    @property
    def total_cost(self) -> float:
        """
        Total path cost.

        cost(π, σ) = Σ wₑ(p) + Σ τ(pᵢ, pᵢ₊₁, vᵢ)
        """
        return self.total_edge_cost + self.total_translation_cost

    @property
    def total_latency_ms(self) -> float:
        """Estimated total latency in milliseconds."""
        return sum(tc.latency_ms for tc in self.translation_costs)

    @property
    def num_translations(self) -> int:
        """Number of protocol translations (excluding identity)."""
        return sum(1 for tc in self.translation_costs if not tc.is_identity)

    @property
    def semantic_loss_estimate(self) -> float:
        """
        Estimated semantic loss across all translations.

        Compounds multiplicatively: (1 - loss1) * (1 - loss2) * ...
        """
        if not self.translation_costs:
            return 0.0

        preservation = 1.0
        for tc in self.translation_costs:
            if tc.semantic_loss_penalty > 0:
                # Semantic loss penalty is proportional to loss
                preservation *= 1 - min(tc.semantic_loss_penalty, 1.0)

        return 1 - preservation

    def get_summary(self) -> dict[str, Any]:
        """Get cost summary dictionary."""
        return {
            "total_cost": self.total_cost,
            "edge_cost": self.total_edge_cost,
            "translation_cost": self.total_translation_cost,
            "num_edges": len(self.edge_costs),
            "num_translations": self.num_translations,
            "total_latency_ms": self.total_latency_ms,
            "semantic_loss_estimate": self.semantic_loss_estimate,
        }


@dataclass
class CostConfig:
    """Configuration for path cost calculation."""

    default_edge_weight: float = 1.0
    """Default weight for edges without explicit weights"""

    default_translation_base_cost: float = 0.1
    """Default base cost for protocol translations"""

    semantic_loss_multiplier: float = 2.0
    """Multiplier for semantic loss in cost calculation"""

    latency_weight: float = 0.01
    """Weight for latency in cost calculation (cost per ms)"""

    identity_translation_cost: float = 0.0
    """Cost for identity translations (same protocol)"""


class PathCostCalculator:
    """
    Calculate path cost for protocol-aware routing.

    Implements Definition 12 (Path Cost Function) from the formal model::

        cost(π, σ) = Σ wₑᵢ(pᵢ) + Σ τ(pᵢ, pᵢ₊₁, vᵢ)

    Where:

    - π is the path (sequence of agents)
    - σ is the protocol sequence
    - wₑᵢ(pᵢ) is the weight of edge i using protocol pᵢ
    - τ(pᵢ, pᵢ₊₁, vᵢ) is the translation cost at intermediate nodes

    Usage:
        calculator = PathCostCalculator(graph, compatibility_matrix)

        # Calculate cost for a path
        cost = calculator.calculate_path_cost(
            path=["agent1", "agent2", "agent3"],
            protocol_sequence=[ProtocolName.MCP, ProtocolName.A2A]
        )

        # Get detailed cost breakdown
        path_cost = calculator.get_path_cost_breakdown(path, protocol_sequence)
        print(f"Total: {path_cost.total_cost}")
        print(f"Translations: {path_cost.num_translations}")
    """

    def __init__(
        self,
        graph: ProtocolGraph,
        compatibility_matrix: ProtocolCompatibilityMatrix,
        config: CostConfig | None = None,
    ):
        """
        Initialize path cost calculator.

        Args:
            graph: Protocol graph with agents and edges
            compatibility_matrix: Protocol compatibility information
            config: Cost calculation configuration
        """
        self._graph = graph
        self._compatibility = compatibility_matrix
        self._config = config or CostConfig()

        # Cache for translation costs
        self._translation_cache: dict[tuple[ProtocolName, ProtocolName], TranslationCost] = {}

    def calculate_path_cost(
        self,
        path: list[str],
        protocol_sequence: list[ProtocolName],
    ) -> float:
        """
        Calculate total cost for a protocol-aware path.

        Args:
            path: Sequence of agent IDs
            protocol_sequence: Protocol to use on each edge

        Returns:
            Total path cost (infinity if path is invalid)

        Raises:
            ValueError: If path and protocol_sequence lengths don't match
        """
        if len(path) < 2:
            return 0.0

        if len(protocol_sequence) != len(path) - 1:
            raise ValueError(
                f"Protocol sequence length ({len(protocol_sequence)}) must equal "
                f"path length - 1 ({len(path) - 1})"
            )

        path_cost = self.get_path_cost_breakdown(path, protocol_sequence)
        return path_cost.total_cost

    def get_path_cost_breakdown(
        self,
        path: list[str],
        protocol_sequence: list[ProtocolName],
    ) -> PathCost:
        """
        Get detailed cost breakdown for a path.

        Args:
            path: Sequence of agent IDs
            protocol_sequence: Protocol to use on each edge

        Returns:
            PathCost with detailed breakdown
        """
        result = PathCost(path=path, protocol_sequence=protocol_sequence)

        if len(path) < 2:
            return result

        # Calculate edge costs
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            protocol = protocol_sequence[i]

            edge_cost = self._get_edge_cost(source, target, protocol)
            result.edge_costs.append(edge_cost)

        # Calculate translation costs at intermediate nodes
        for i in range(len(protocol_sequence) - 1):
            source_protocol = protocol_sequence[i]
            target_protocol = protocol_sequence[i + 1]
            node_id = path[i + 1]  # Translation happens at the intermediate node

            translation_cost = self.get_translation_cost(source_protocol, target_protocol, node_id)
            result.translation_costs.append(translation_cost)

        return result

    def _get_edge_cost(
        self,
        source: str,
        target: str,
        protocol: ProtocolName,
    ) -> float:
        """
        Get edge weight for a specific protocol.

        Args:
            source: Source agent ID
            target: Target agent ID
            protocol: Protocol to use

        Returns:
            Edge weight (infinity if edge doesn't exist or protocol not supported)
        """
        edge = self._graph.get_edge(source, target)
        if edge is None:
            return float("inf")

        return edge.get_weight(protocol)

    def get_translation_cost(
        self,
        source_protocol: ProtocolName,
        target_protocol: ProtocolName,
        at_node: str | None = None,
    ) -> TranslationCost:
        """
        Get cost of translating between protocols at a node.

        Args:
            source_protocol: Incoming protocol
            target_protocol: Outgoing protocol
            at_node: Node where translation occurs

        Returns:
            TranslationCost for this translation
        """
        # Check cache
        cache_key = (source_protocol, target_protocol)
        if cache_key in self._translation_cache:
            cached = self._translation_cache[cache_key]
            # Update node_id for this specific call
            return TranslationCost(
                source_protocol=cached.source_protocol,
                target_protocol=cached.target_protocol,
                base_cost=cached.base_cost,
                semantic_loss_penalty=cached.semantic_loss_penalty,
                latency_ms=cached.latency_ms,
                node_id=at_node,
            )

        # Identity translation (same protocol)
        if source_protocol == target_protocol:
            cost = TranslationCost(
                source_protocol=source_protocol,
                target_protocol=target_protocol,
                base_cost=self._config.identity_translation_cost,
                semantic_loss_penalty=0.0,
                latency_ms=0.0,
                node_id=at_node,
            )
            self._translation_cache[cache_key] = cost
            return cost

        # Get compatibility information
        compatibility = self._compatibility.get_compatibility(source_protocol, target_protocol)

        # Calculate costs based on compatibility
        if not compatibility.is_compatible:
            # No translation possible
            cost = TranslationCost(
                source_protocol=source_protocol,
                target_protocol=target_protocol,
                base_cost=float("inf"),
                semantic_loss_penalty=float("inf"),
                latency_ms=float("inf"),
                node_id=at_node,
            )
        else:
            base_cost = self._config.default_translation_base_cost
            semantic_penalty = compatibility.semantic_loss * self._config.semantic_loss_multiplier
            latency = compatibility.latency_factor * 10.0  # Base 10ms

            cost = TranslationCost(
                source_protocol=source_protocol,
                target_protocol=target_protocol,
                base_cost=base_cost,
                semantic_loss_penalty=semantic_penalty,
                latency_ms=latency,
                node_id=at_node,
            )

        self._translation_cache[cache_key] = cost
        return cost

    def estimate_minimum_cost(
        self,
        source: str,
        target: str,
        source_protocol: ProtocolName,
        target_protocol: ProtocolName | None = None,
    ) -> float:
        """
        Estimate minimum possible cost between two agents.

        Useful for A* heuristic or early termination.

        Args:
            source: Source agent ID
            target: Target agent ID
            source_protocol: Starting protocol
            target_protocol: Required ending protocol (optional)

        Returns:
            Lower bound on path cost
        """
        # Base estimate: minimum edge weight
        min_edge_weight = self._config.default_edge_weight

        # If we need protocol translation at the end
        translation_cost = 0.0
        if target_protocol and target_protocol != source_protocol:
            tc = self.get_translation_cost(source_protocol, target_protocol)
            translation_cost = tc.total_cost

        # Heuristic: at least one edge + possible translation
        return min_edge_weight + translation_cost

    def compare_paths(
        self,
        path1: tuple[list[str], list[ProtocolName]],
        path2: tuple[list[str], list[ProtocolName]],
    ) -> int:
        """
        Compare two paths by cost.

        Args:
            path1: First (path, protocol_sequence) tuple
            path2: Second (path, protocol_sequence) tuple

        Returns:
            -1 if path1 cheaper, 1 if path2 cheaper, 0 if equal
        """
        cost1 = self.calculate_path_cost(path1[0], path1[1])
        cost2 = self.calculate_path_cost(path2[0], path2[1])

        if cost1 < cost2:
            return -1
        elif cost1 > cost2:
            return 1
        return 0

    def clear_cache(self) -> None:
        """Clear the translation cost cache."""
        self._translation_cache.clear()
        logger.debug("Cleared translation cost cache")

    def get_statistics(self) -> dict[str, Any]:
        """Get calculator statistics."""
        return {
            "cache_size": len(self._translation_cache),
            "config": {
                "default_edge_weight": self._config.default_edge_weight,
                "default_translation_base_cost": self._config.default_translation_base_cost,
                "semantic_loss_multiplier": self._config.semantic_loss_multiplier,
            },
        }


__all__ = [
    "TranslationCost",
    "PathCost",
    "CostConfig",
    "PathCostCalculator",
]
