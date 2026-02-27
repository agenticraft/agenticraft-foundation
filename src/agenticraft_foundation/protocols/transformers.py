"""Composable Protocol Transformers.

Models protocol transformers as composable algebraic objects:

    T_{p→p'}: Mₚ → Mₚ' ∪ {⊥}

With identity and composition laws:
- Identity: T_{p→p} = id (no transformation)
- Composition: T_{p→p''} = T_{p'→p''} ∘ T_{p→p'}

Implements Theorem 6: Lossless/lossy classification.
A transformer is lossless iff T⁻¹(T(m)) = m for all m ∈ Mₚ.

Usage::

    registry = TransformerRegistry()
    registry.register(mcp_to_a2a)
    registry.register(a2a_to_mcp)

    # Auto-compose: MCP → A2A → ANP
    composed = registry.get_transformer(ProtocolName.MCP, ProtocolName.ANP)
    result = composed.transform(message)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticraft_foundation.types import ProtocolName

logger = logging.getLogger(__name__)


class TransformerClassification(str, Enum):
    """Classification of a protocol transformer (Theorem 6)."""

    LOSSLESS = "lossless"
    """T⁻¹(T(m)) = m for all m — perfect round-trip"""

    LOSSY = "lossy"
    """T⁻¹(T(m)) ≈ m — some information loss"""

    DESTRUCTIVE = "destructive"
    """T⁻¹(T(m)) may be ⊥ — significant information loss"""


@dataclass
class TransformResult:
    """Result of a protocol transformation."""

    success: bool
    """Whether the transformation succeeded"""

    message: dict[str, Any] | None = None
    """Transformed message (None if failed, i.e., ⊥)"""

    source_protocol: ProtocolName | None = None
    """Source protocol"""

    target_protocol: ProtocolName | None = None
    """Target protocol"""

    fields_preserved: int = 0
    """Number of semantic fields preserved"""

    fields_lost: int = 0
    """Number of semantic fields lost"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Transformation metadata"""

    @property
    def preservation_ratio(self) -> float:
        """Ratio of preserved fields to total fields."""
        total = self.fields_preserved + self.fields_lost
        if total == 0:
            return 1.0
        return self.fields_preserved / total


class BaseProtocolTransformer(ABC):
    """Base class for protocol transformers.

    T_{p→p'}: Mₚ → Mₚ' ∪ {⊥}

    Subclasses implement the actual transformation logic for
    specific protocol pairs.
    """

    @property
    @abstractmethod
    def source_protocol(self) -> ProtocolName:
        """Source protocol."""
        ...

    @property
    @abstractmethod
    def target_protocol(self) -> ProtocolName:
        """Target protocol."""
        ...

    @property
    def classification(self) -> TransformerClassification:
        """Transformer classification (default: lossy)."""
        return TransformerClassification.LOSSY

    @abstractmethod
    def transform(self, message: dict[str, Any]) -> TransformResult:
        """Transform a message from source to target protocol.

        Args:
            message: Message in source protocol format

        Returns:
            TransformResult with transformed message or failure (⊥)
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.source_protocol.value}→{self.target_protocol.value})"
        )


class IdentityTransformer(BaseProtocolTransformer):
    """Identity transformer: T_{p→p} = id.

    Returns the message unchanged. Always lossless.
    """

    def __init__(self, protocol: ProtocolName):
        self._protocol = protocol

    @property
    def source_protocol(self) -> ProtocolName:
        return self._protocol

    @property
    def target_protocol(self) -> ProtocolName:
        return self._protocol

    @property
    def classification(self) -> TransformerClassification:
        return TransformerClassification.LOSSLESS

    def transform(self, message: dict[str, Any]) -> TransformResult:
        return TransformResult(
            success=True,
            message=dict(message),
            source_protocol=self._protocol,
            target_protocol=self._protocol,
            fields_preserved=len(message),
            fields_lost=0,
        )


class ComposedTransformer(BaseProtocolTransformer):
    """Composition of two transformers: T_{p→p''} = T_{p'→p''} ∘ T_{p→p'}.

    Applies first transformer, then second transformer to the result.
    """

    def __init__(
        self,
        first: BaseProtocolTransformer,
        second: BaseProtocolTransformer,
    ):
        if first.target_protocol != second.source_protocol:
            msg = (
                f"Cannot compose: first target ({first.target_protocol.value}) "
                f"!= second source ({second.source_protocol.value})"
            )
            raise ValueError(msg)

        self._first = first
        self._second = second

    @property
    def source_protocol(self) -> ProtocolName:
        return self._first.source_protocol

    @property
    def target_protocol(self) -> ProtocolName:
        return self._second.target_protocol

    @property
    def classification(self) -> TransformerClassification:
        """Composition classification: worst of the two."""
        order = [
            TransformerClassification.LOSSLESS,
            TransformerClassification.LOSSY,
            TransformerClassification.DESTRUCTIVE,
        ]
        idx1 = order.index(self._first.classification)
        idx2 = order.index(self._second.classification)
        return order[max(idx1, idx2)]

    def transform(self, message: dict[str, Any]) -> TransformResult:
        # Apply first transformer
        intermediate = self._first.transform(message)
        if not intermediate.success or intermediate.message is None:
            return TransformResult(
                success=False,
                source_protocol=self.source_protocol,
                target_protocol=self.target_protocol,
                fields_lost=len(message),
                metadata={"failed_at": "first", "reason": "first transformer failed"},
            )

        # Apply second transformer
        final = self._second.transform(intermediate.message)
        if not final.success:
            return TransformResult(
                success=False,
                source_protocol=self.source_protocol,
                target_protocol=self.target_protocol,
                fields_preserved=0,
                fields_lost=len(message),
                metadata={"failed_at": "second", "reason": "second transformer failed"},
            )

        # Aggregate field stats
        total_lost = intermediate.fields_lost + final.fields_lost
        return TransformResult(
            success=True,
            message=final.message,
            source_protocol=self.source_protocol,
            target_protocol=self.target_protocol,
            fields_preserved=final.fields_preserved,
            fields_lost=total_lost,
            metadata={
                "intermediate_protocol": intermediate.target_protocol.value
                if intermediate.target_protocol
                else None,
                "composed": True,
            },
        )


class TransformerRegistry:
    """Registry of protocol transformers with automatic composition.

    Manages registered transformers and computes composed transformers
    for protocol pairs that don't have direct transformers.

    Usage::

        registry = TransformerRegistry()
        registry.register(mcp_to_a2a_transformer)
        registry.register(a2a_to_anp_transformer)

        # Direct lookup
        t = registry.get_transformer(ProtocolName.MCP, ProtocolName.A2A)

        # Auto-composed (MCP→A2A→ANP)
        t = registry.get_transformer(ProtocolName.MCP, ProtocolName.ANP)
    """

    def __init__(self) -> None:
        # Direct transformers: (source, target) → transformer
        self._transformers: dict[tuple[ProtocolName, ProtocolName], BaseProtocolTransformer] = {}

        # Register identity transformers for all protocols
        for protocol in ProtocolName:
            self._transformers[(protocol, protocol)] = IdentityTransformer(protocol)

    def register(self, transformer: BaseProtocolTransformer) -> None:
        """Register a transformer."""
        key = (transformer.source_protocol, transformer.target_protocol)
        self._transformers[key] = transformer
        logger.debug(
            "Registered transformer: %s → %s",
            transformer.source_protocol.value,
            transformer.target_protocol.value,
        )

    def has_transformer(
        self,
        source: ProtocolName,
        target: ProtocolName,
    ) -> bool:
        """Check if a direct transformer exists."""
        return (source, target) in self._transformers

    def get_direct_transformer(
        self,
        source: ProtocolName,
        target: ProtocolName,
    ) -> BaseProtocolTransformer | None:
        """Get a direct (non-composed) transformer."""
        return self._transformers.get((source, target))

    def get_transformer(
        self,
        source: ProtocolName,
        target: ProtocolName,
    ) -> BaseProtocolTransformer | None:
        """Get a transformer, composing if needed.

        First checks for a direct transformer. If not found, attempts to
        compose via a single intermediate protocol (BFS depth 1).

        Args:
            source: Source protocol
            target: Target protocol

        Returns:
            Transformer or None if no path exists
        """
        # Direct lookup
        direct = self._transformers.get((source, target))
        if direct is not None:
            return direct

        # Try single-hop composition: source → intermediate → target
        best: BaseProtocolTransformer | None = None
        best_class_idx = len(TransformerClassification)

        order = [
            TransformerClassification.LOSSLESS,
            TransformerClassification.LOSSY,
            TransformerClassification.DESTRUCTIVE,
        ]

        for intermediate in ProtocolName:
            if intermediate == source or intermediate == target:
                continue

            first = self._transformers.get((source, intermediate))
            second = self._transformers.get((intermediate, target))

            if first is not None and second is not None:
                composed = ComposedTransformer(first, second)
                class_idx = order.index(composed.classification)
                if class_idx < best_class_idx:
                    best = composed
                    best_class_idx = class_idx

        return best

    def get_all_paths(
        self,
        source: ProtocolName,
        target: ProtocolName,
        max_hops: int = 3,
    ) -> list[list[ProtocolName]]:
        """Find all transformation paths from source to target.

        Uses BFS to find paths up to max_hops length.

        Args:
            source: Source protocol
            target: Target protocol
            max_hops: Maximum number of intermediate transformations

        Returns:
            List of protocol paths (including source and target)
        """
        if source == target:
            return [[source]]

        paths: list[list[ProtocolName]] = []
        queue: list[list[ProtocolName]] = [[source]]

        while queue:
            path = queue.pop(0)
            current = path[-1]

            if len(path) > max_hops + 1:
                continue

            for (src, tgt), _ in self._transformers.items():
                if src == current and tgt not in path:
                    new_path = path + [tgt]
                    if tgt == target:
                        paths.append(new_path)
                    elif len(new_path) <= max_hops + 1:
                        queue.append(new_path)

        return paths

    def compose_path(
        self,
        path: list[ProtocolName],
    ) -> BaseProtocolTransformer | None:
        """Compose a transformer from a protocol path.

        Args:
            path: List of protocols [p1, p2, ..., pn]

        Returns:
            Composed transformer p1→p2→...→pn, or None if any hop is missing
        """
        if len(path) < 2:
            if path:
                return IdentityTransformer(path[0])
            return None

        current = self._transformers.get((path[0], path[1]))
        if current is None:
            return None

        for i in range(2, len(path)):
            hop = self._transformers.get((path[i - 1], path[i]))
            if hop is None:
                return None
            current = ComposedTransformer(current, hop)

        return current

    def verify_round_trip(
        self,
        protocol_a: ProtocolName,
        protocol_b: ProtocolName,
        test_message: dict[str, Any],
    ) -> bool:
        """Verify round-trip preservation: T⁻¹(T(m)) = m.

        Args:
            protocol_a: First protocol
            protocol_b: Second protocol
            test_message: Test message to verify with

        Returns:
            True if round-trip preserves the message
        """
        forward = self.get_transformer(protocol_a, protocol_b)
        reverse = self.get_transformer(protocol_b, protocol_a)

        if forward is None or reverse is None:
            return False

        # Forward transform
        forward_result = forward.transform(test_message)
        if not forward_result.success or forward_result.message is None:
            return False

        # Reverse transform
        reverse_result = reverse.transform(forward_result.message)
        if not reverse_result.success or reverse_result.message is None:
            return False

        # Compare
        return reverse_result.message == test_message

    @property
    def registered_pairs(self) -> list[tuple[ProtocolName, ProtocolName]]:
        """Get all registered transformer pairs (excluding identity)."""
        return [(src, tgt) for (src, tgt) in self._transformers if src != tgt]

    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics."""
        non_identity = [(s, t) for (s, t) in self._transformers if s != t]
        classifications: dict[str, int] = {}
        for key in non_identity:
            cls = self._transformers[key].classification.value
            classifications[cls] = classifications.get(cls, 0) + 1

        return {
            "total_transformers": len(self._transformers),
            "identity_transformers": len(self._transformers) - len(non_identity),
            "direct_transformers": len(non_identity),
            "classifications": classifications,
            "protocols": [p.value for p in ProtocolName],
        }


__all__ = [
    "TransformerClassification",
    "TransformResult",
    "BaseProtocolTransformer",
    "IdentityTransformer",
    "ComposedTransformer",
    "TransformerRegistry",
]
