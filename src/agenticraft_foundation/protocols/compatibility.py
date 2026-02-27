"""Protocol compatibility relation.

Defines full, partial, or incompatible translation between protocols
based on Definition 6 (Protocol Compatibility Relation) from the formal model.

Compatibility levels:
- FULL: No semantic loss during translation.
- PARTIAL: Some semantic loss possible.
- NONE: No translation possible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticraft_foundation.types import ProtocolName

logger = logging.getLogger(__name__)


class CompatibilityLevel(str, Enum):
    """
    Protocol compatibility levels.

    Represents the degree of semantic preservation when translating
    between protocols.
    """

    FULL = "full"
    """✓ No semantic loss - complete bidirectional translation"""

    PARTIAL = "partial"
    """⚡ Some semantic loss possible - requires validation"""

    NONE = "none"
    """✗ No translation possible - incompatible protocols"""


@dataclass
class CompatibilityRelation:
    """
    Protocol compatibility relation p₁ ∼_p p₂.

    Based on Definition 6 (Protocol Compatibility Relation) from the formal model.
    For protocols p₁, p₂ ∈ P, p₁ ∼_p p₂ ⟺ ∃ bidirectional translator T_{p₁→p₂}

    Attributes:
        source: Source protocol
        target: Target protocol
        level: Compatibility level (FULL, PARTIAL, NONE)
        semantic_loss: Estimated semantic loss (0.0 = no loss, 1.0 = total loss)
        reversible: Whether T⁻¹(T(m)) ≈ m holds
        latency_factor: Relative latency multiplier for translation
        conditions: Conditions under which translation works
        feature_mapping: Mapping of features between protocols
    """

    source: ProtocolName
    target: ProtocolName
    level: CompatibilityLevel
    semantic_loss: float = 0.0
    reversible: bool = True
    latency_factor: float = 1.0
    conditions: list[str] = field(default_factory=list)
    feature_mapping: dict[str, str] = field(default_factory=dict)

    @property
    def is_full(self) -> bool:
        """Check if this is full compatibility."""
        return self.level == CompatibilityLevel.FULL

    @property
    def is_partial(self) -> bool:
        """Check if this is partial compatibility."""
        return self.level == CompatibilityLevel.PARTIAL

    @property
    def is_compatible(self) -> bool:
        """Check if translation is possible (FULL or PARTIAL)."""
        return self.level != CompatibilityLevel.NONE

    def get_translation_cost(self) -> float:
        """
        Calculate translation cost based on semantic loss and latency.

        Returns:
            Combined cost factor for routing decisions
        """
        if self.level == CompatibilityLevel.NONE:
            return float("inf")

        # Cost = base cost + semantic loss penalty + latency penalty
        base_cost = 0.1 if self.level == CompatibilityLevel.FULL else 0.5
        semantic_penalty = self.semantic_loss * 2.0
        latency_penalty = (self.latency_factor - 1.0) * 0.5

        return base_cost + semantic_penalty + latency_penalty


# Built-in protocol compatibility matrix
# Based on protocol characteristics analysis
PROTOCOL_COMPATIBILITY: dict[tuple[ProtocolName, ProtocolName], CompatibilityRelation] = {
    # MCP -> A2A (Full compatibility)
    (ProtocolName.MCP, ProtocolName.A2A): CompatibilityRelation(
        source=ProtocolName.MCP,
        target=ProtocolName.A2A,
        level=CompatibilityLevel.FULL,
        semantic_loss=0.0,
        reversible=True,
        latency_factor=1.1,
        conditions=[
            "Tools map to AgentSkills",
            "Resources map to Task artifacts",
            "Prompts map to Task instructions",
        ],
        feature_mapping={
            "tools/list": "skills/list",
            "tools/call": "tasks/send",
            "resources/read": "artifacts/read",
            "prompts/get": "tasks/create",
        },
    ),
    # A2A -> MCP (Partial - task status may be simplified)
    (ProtocolName.A2A, ProtocolName.MCP): CompatibilityRelation(
        source=ProtocolName.A2A,
        target=ProtocolName.MCP,
        level=CompatibilityLevel.PARTIAL,
        semantic_loss=0.1,
        reversible=True,
        latency_factor=1.1,
        conditions=[
            "Task status simplified to success/error",
            "Push notifications not supported",
            "Streaming via SSE differences",
        ],
        feature_mapping={
            "skills/list": "tools/list",
            "tasks/send": "tools/call",
            "artifacts/read": "resources/read",
        },
    ),
    # Self-compatibility (identity translation)
    (ProtocolName.MCP, ProtocolName.MCP): CompatibilityRelation(
        source=ProtocolName.MCP,
        target=ProtocolName.MCP,
        level=CompatibilityLevel.FULL,
        semantic_loss=0.0,
        reversible=True,
        latency_factor=1.0,
    ),
    (ProtocolName.A2A, ProtocolName.A2A): CompatibilityRelation(
        source=ProtocolName.A2A,
        target=ProtocolName.A2A,
        level=CompatibilityLevel.FULL,
        semantic_loss=0.0,
        reversible=True,
        latency_factor=1.0,
    ),
    (ProtocolName.CUSTOM, ProtocolName.CUSTOM): CompatibilityRelation(
        source=ProtocolName.CUSTOM,
        target=ProtocolName.CUSTOM,
        level=CompatibilityLevel.FULL,
        semantic_loss=0.0,
        reversible=True,
        latency_factor=1.0,
    ),
    # CUSTOM protocol - partial compatibility with all
    (ProtocolName.MCP, ProtocolName.CUSTOM): CompatibilityRelation(
        source=ProtocolName.MCP,
        target=ProtocolName.CUSTOM,
        level=CompatibilityLevel.PARTIAL,
        semantic_loss=0.2,
        reversible=True,
        latency_factor=1.5,
        conditions=["Custom adapter required"],
    ),
    (ProtocolName.CUSTOM, ProtocolName.MCP): CompatibilityRelation(
        source=ProtocolName.CUSTOM,
        target=ProtocolName.MCP,
        level=CompatibilityLevel.PARTIAL,
        semantic_loss=0.2,
        reversible=True,
        latency_factor=1.5,
        conditions=["Custom adapter required"],
    ),
    (ProtocolName.A2A, ProtocolName.CUSTOM): CompatibilityRelation(
        source=ProtocolName.A2A,
        target=ProtocolName.CUSTOM,
        level=CompatibilityLevel.PARTIAL,
        semantic_loss=0.2,
        reversible=True,
        latency_factor=1.5,
        conditions=["Custom adapter required"],
    ),
    (ProtocolName.CUSTOM, ProtocolName.A2A): CompatibilityRelation(
        source=ProtocolName.CUSTOM,
        target=ProtocolName.A2A,
        level=CompatibilityLevel.PARTIAL,
        semantic_loss=0.2,
        reversible=True,
        latency_factor=1.5,
        conditions=["Custom adapter required"],
    ),
}


class ProtocolCompatibilityMatrix:
    """
    Lookup and query protocol compatibility.

    Provides efficient access to the protocol compatibility matrix
    for routing and translation decisions.
    """

    def __init__(
        self,
        compatibility_data: dict[tuple[ProtocolName, ProtocolName], CompatibilityRelation]
        | None = None,
    ):
        """
        Initialize compatibility matrix.

        Args:
            compatibility_data: Optional custom compatibility data.
                               Uses PROTOCOL_COMPATIBILITY by default.
        """
        self._matrix = compatibility_data or PROTOCOL_COMPATIBILITY.copy()
        self._protocols = {ProtocolName.MCP, ProtocolName.A2A, ProtocolName.CUSTOM}

    def get_compatibility(
        self,
        source: ProtocolName,
        target: ProtocolName,
    ) -> CompatibilityRelation:
        """
        Get compatibility relation between two protocols.

        Args:
            source: Source protocol
            target: Target protocol

        Returns:
            CompatibilityRelation describing the translation
        """
        key = (source, target)
        if key in self._matrix:
            return self._matrix[key]

        # Default to no compatibility for unknown pairs
        return CompatibilityRelation(
            source=source,
            target=target,
            level=CompatibilityLevel.NONE,
            semantic_loss=1.0,
            reversible=False,
            conditions=["No translator available"],
        )

    def can_translate(
        self,
        source: ProtocolName,
        target: ProtocolName,
    ) -> bool:
        """
        Check if translation is possible between protocols.

        Args:
            source: Source protocol
            target: Target protocol

        Returns:
            True if translation is possible (FULL or PARTIAL)
        """
        relation = self.get_compatibility(source, target)
        return relation.is_compatible

    def get_translation_cost(
        self,
        source: ProtocolName,
        target: ProtocolName,
    ) -> float:
        """
        Get the cost of translating between protocols.

        Args:
            source: Source protocol
            target: Target protocol

        Returns:
            Translation cost (infinity if not possible)
        """
        relation = self.get_compatibility(source, target)
        return relation.get_translation_cost()

    def get_semantic_loss(
        self,
        source: ProtocolName,
        target: ProtocolName,
    ) -> float:
        """
        Get the expected semantic loss for a translation.

        Args:
            source: Source protocol
            target: Target protocol

        Returns:
            Semantic loss (0.0 to 1.0, 1.0 if not possible)
        """
        relation = self.get_compatibility(source, target)
        if not relation.is_compatible:
            return 1.0
        return relation.semantic_loss

    def get_all_compatible(
        self,
        source: ProtocolName,
        level: CompatibilityLevel | None = None,
    ) -> list[ProtocolName]:
        """
        Get all protocols compatible with a source protocol.

        Args:
            source: Source protocol
            level: Optional level filter (FULL or PARTIAL)

        Returns:
            List of compatible protocols
        """
        compatible = []
        for (src, tgt), relation in self._matrix.items():
            if src == source and relation.is_compatible:
                if level is None or relation.level == level:
                    compatible.append(tgt)
        return compatible

    def get_best_translation_path(
        self,
        source: ProtocolName,
        target: ProtocolName,
        max_hops: int = 2,
    ) -> list[ProtocolName] | None:
        """
        Find the best translation path (lowest semantic loss).

        Uses breadth-first search with cost optimization.

        Args:
            source: Source protocol
            target: Target protocol
            max_hops: Maximum number of intermediate translations

        Returns:
            List of protocols in the path, or None if no path exists
        """
        if source == target:
            return [source]

        # Direct translation
        direct = self.get_compatibility(source, target)
        if direct.is_compatible:
            return [source, target]

        if max_hops < 2:
            return None

        # Find path through intermediaries
        best_path = None
        best_cost = float("inf")

        for intermediate in self._protocols:
            if intermediate == source or intermediate == target:
                continue

            first_hop = self.get_compatibility(source, intermediate)
            second_hop = self.get_compatibility(intermediate, target)

            if first_hop.is_compatible and second_hop.is_compatible:
                cost = first_hop.get_translation_cost() + second_hop.get_translation_cost()
                if cost < best_cost:
                    best_cost = cost
                    best_path = [source, intermediate, target]

        return best_path

    def register_compatibility(
        self,
        relation: CompatibilityRelation,
    ) -> None:
        """
        Register a new compatibility relation.

        Args:
            relation: Compatibility relation to register
        """
        key = (relation.source, relation.target)
        self._matrix[key] = relation
        self._protocols.add(relation.source)
        self._protocols.add(relation.target)
        logger.debug(
            f"Registered compatibility: {relation.source} -> {relation.target} ({relation.level})"
        )

    def get_statistics(self) -> dict[str, Any]:
        """
        Get matrix statistics.

        Returns:
            Dictionary of statistics
        """
        full_count = sum(1 for r in self._matrix.values() if r.level == CompatibilityLevel.FULL)
        partial_count = sum(
            1 for r in self._matrix.values() if r.level == CompatibilityLevel.PARTIAL
        )
        none_count = sum(1 for r in self._matrix.values() if r.level == CompatibilityLevel.NONE)

        return {
            "num_protocols": len(self._protocols),
            "num_relations": len(self._matrix),
            "full_compatibility": full_count,
            "partial_compatibility": partial_count,
            "no_compatibility": none_count,
            "avg_semantic_loss": (
                sum(r.semantic_loss for r in self._matrix.values()) / len(self._matrix)
                if self._matrix
                else 0.0
            ),
        }


__all__ = [
    "CompatibilityLevel",
    "CompatibilityRelation",
    "PROTOCOL_COMPATIBILITY",
    "ProtocolCompatibilityMatrix",
]
