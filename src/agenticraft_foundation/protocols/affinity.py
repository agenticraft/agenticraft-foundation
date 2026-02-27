"""Capability-protocol affinity matrix: α(c, p) → [0, 1].

Maps capabilities to optimal protocols based on protocol characteristics.
Higher affinity indicates better suitability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agenticraft_foundation.types import ProtocolName

logger = logging.getLogger(__name__)


# Default capability-protocol affinity matrix
# Values range from 0.0 (poor fit) to 1.0 (excellent fit)
CAPABILITY_PROTOCOL_AFFINITY: dict[tuple[str, ProtocolName], float] = {
    # ========================================================================
    # Code Execution Capabilities
    # ========================================================================
    # MCP excels at tool/code execution (designed for this)
    ("code_execution", ProtocolName.MCP): 0.95,
    ("code_execution", ProtocolName.A2A): 0.70,
    ("code_execution", ProtocolName.CUSTOM): 0.50,
    # Shell/command execution
    ("shell_execution", ProtocolName.MCP): 0.95,
    ("shell_execution", ProtocolName.A2A): 0.65,
    # ========================================================================
    # Task Delegation Capabilities
    # ========================================================================
    # A2A excels at task delegation between agents
    ("task_delegation", ProtocolName.A2A): 0.95,
    ("task_delegation", ProtocolName.MCP): 0.50,
    ("task_delegation", ProtocolName.CUSTOM): 0.60,
    # Multi-agent coordination
    ("multi_agent_coordination", ProtocolName.A2A): 0.90,
    ("multi_agent_coordination", ProtocolName.MCP): 0.40,
    # ========================================================================
    # Resource Access Capabilities
    # ========================================================================
    # MCP designed for resource access
    ("resource_access", ProtocolName.MCP): 0.90,
    ("resource_access", ProtocolName.A2A): 0.70,
    ("resource_access", ProtocolName.CUSTOM): 0.50,
    # File operations
    ("file_operations", ProtocolName.MCP): 0.90,
    ("file_operations", ProtocolName.A2A): 0.65,
    # Database access
    ("database_access", ProtocolName.MCP): 0.85,
    ("database_access", ProtocolName.A2A): 0.70,
    # ========================================================================
    # Streaming Capabilities
    # ========================================================================
    # MCP has good streaming support via SSE
    ("streaming", ProtocolName.MCP): 0.90,
    ("streaming", ProtocolName.A2A): 0.80,
    ("streaming", ProtocolName.CUSTOM): 0.60,
    # Real-time updates
    ("real_time", ProtocolName.MCP): 0.85,
    ("real_time", ProtocolName.A2A): 0.80,
    # ========================================================================
    # Discovery Capabilities
    # ========================================================================
    # A2A designed for agent discovery
    ("agent_discovery", ProtocolName.A2A): 0.95,
    ("agent_discovery", ProtocolName.MCP): 0.50,
    ("agent_discovery", ProtocolName.CUSTOM): 0.40,
    # Capability-based discovery
    ("capability_discovery", ProtocolName.A2A): 0.90,
    ("capability_discovery", ProtocolName.MCP): 0.60,
    # ========================================================================
    # Communication Capabilities
    # ========================================================================
    # Asynchronous messaging
    ("async_messaging", ProtocolName.A2A): 0.90,
    ("async_messaging", ProtocolName.MCP): 0.70,
    ("async_messaging", ProtocolName.CUSTOM): 0.60,
    # Synchronous request/response
    ("sync_request", ProtocolName.MCP): 0.90,
    ("sync_request", ProtocolName.A2A): 0.85,
    # Push notifications
    ("push_notifications", ProtocolName.A2A): 0.95,
    ("push_notifications", ProtocolName.MCP): 0.50,
    # ========================================================================
    # Security Capabilities
    # ========================================================================
    # Mutual TLS
    ("mtls", ProtocolName.MCP): 0.90,
    ("mtls", ProtocolName.A2A): 0.85,
    # API key authentication
    ("api_key_auth", ProtocolName.MCP): 0.95,
    ("api_key_auth", ProtocolName.A2A): 0.90,
    # OAuth
    ("oauth", ProtocolName.A2A): 0.90,
    ("oauth", ProtocolName.MCP): 0.85,
    # ========================================================================
    # Data Processing Capabilities
    # ========================================================================
    # Batch processing
    ("batch_processing", ProtocolName.A2A): 0.85,
    ("batch_processing", ProtocolName.MCP): 0.80,
    # Data transformation
    ("data_transformation", ProtocolName.MCP): 0.85,
    ("data_transformation", ProtocolName.A2A): 0.80,
    # ========================================================================
    # AI/ML Capabilities
    # ========================================================================
    # LLM integration
    ("llm_integration", ProtocolName.MCP): 0.95,
    ("llm_integration", ProtocolName.A2A): 0.80,
    # Model inference
    ("model_inference", ProtocolName.MCP): 0.90,
    ("model_inference", ProtocolName.A2A): 0.80,
    # Embedding generation
    ("embedding_generation", ProtocolName.MCP): 0.90,
    ("embedding_generation", ProtocolName.A2A): 0.75,
}


@dataclass
class AffinityConfig:
    """Configuration for affinity calculations."""

    default_affinity: float = 0.5
    """Default affinity for unknown capability-protocol pairs"""

    min_affinity: float = 0.0
    """Minimum affinity value"""

    max_affinity: float = 1.0
    """Maximum affinity value"""

    aggregation_method: str = "weighted_average"
    """Aggregation method: 'average', 'weighted_average', 'max', 'min'."""


class CapabilityAffinityMatrix:
    """
    Lookup capability-protocol affinities.

    The affinity matrix maps capabilities to protocols, indicating how well
    each protocol supports specific capabilities. Higher values (closer to 1.0)
    indicate better fit.

    Usage:
        matrix = CapabilityAffinityMatrix()

        # Get affinity for a single capability
        affinity = matrix.get_affinity("code_execution", ProtocolName.MCP)

        # Get optimal protocol for a capability
        best_protocol = matrix.get_optimal_protocol("task_delegation")

        # Score protocol for multiple capabilities
        score = matrix.score_protocol_for_capabilities(
            ["code_execution", "resource_access"],
            ProtocolName.MCP
        )
    """

    def __init__(
        self,
        affinity_data: dict[tuple[str, ProtocolName], float] | None = None,
        config: AffinityConfig | None = None,
    ):
        """
        Initialize affinity matrix.

        Args:
            affinity_data: Optional custom affinity data.
                          Uses CAPABILITY_PROTOCOL_AFFINITY by default.
            config: Configuration for affinity calculations.
        """
        self._matrix = affinity_data or CAPABILITY_PROTOCOL_AFFINITY.copy()
        self._config = config or AffinityConfig()
        self._protocols = {
            ProtocolName.MCP,
            ProtocolName.A2A,
            ProtocolName.CUSTOM,
        }
        self._capabilities: set[str] = {key[0] for key in self._matrix}

    def get_affinity(
        self,
        capability: str,
        protocol: ProtocolName,
    ) -> float:
        """
        Get affinity score for a capability-protocol pair.

        Args:
            capability: Capability name
            protocol: Protocol name

        Returns:
            Affinity score (0.0 to 1.0)
        """
        key = (capability, protocol)
        if key in self._matrix:
            return self._matrix[key]
        return self._config.default_affinity

    def get_optimal_protocol(
        self,
        capability: str,
    ) -> ProtocolName:
        """
        Get the optimal protocol for a capability.

        Args:
            capability: Capability to optimize for

        Returns:
            Protocol with highest affinity for this capability
        """
        best_protocol = ProtocolName.MCP
        best_affinity = 0.0

        for protocol in self._protocols:
            affinity = self.get_affinity(capability, protocol)
            if affinity > best_affinity:
                best_affinity = affinity
                best_protocol = protocol

        return best_protocol

    def score_protocol_for_capabilities(
        self,
        capabilities: list[str],
        protocol: ProtocolName,
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Calculate aggregate affinity score for multiple capabilities.

        Args:
            capabilities: List of capabilities
            protocol: Protocol to score
            weights: Optional weights for each capability

        Returns:
            Aggregate affinity score (0.0 to 1.0)
        """
        if not capabilities:
            return self._config.default_affinity

        affinities = [self.get_affinity(cap, protocol) for cap in capabilities]

        if self._config.aggregation_method == "max":
            return max(affinities)
        elif self._config.aggregation_method == "min":
            return min(affinities)
        elif self._config.aggregation_method == "weighted_average" and weights:
            total_weight = sum(weights.get(cap, 1.0) for cap in capabilities)
            if total_weight == 0:
                return self._config.default_affinity
            weighted_sum = sum(
                self.get_affinity(cap, protocol) * weights.get(cap, 1.0) for cap in capabilities
            )
            return weighted_sum / total_weight
        else:
            # Default: simple average
            return sum(affinities) / len(affinities)

    def get_best_protocol_for_capabilities(
        self,
        capabilities: list[str],
        weights: dict[str, float] | None = None,
        exclude_protocols: set[ProtocolName] | None = None,
    ) -> tuple[ProtocolName, float]:
        """
        Find the best protocol for a set of capabilities.

        Args:
            capabilities: List of capabilities
            weights: Optional weights for each capability
            exclude_protocols: Protocols to exclude from consideration

        Returns:
            Tuple of (best protocol, aggregate score)
        """
        exclude = exclude_protocols or set()
        best_protocol = None
        best_score = 0.0

        for protocol in self._protocols:
            if protocol in exclude:
                continue

            score = self.score_protocol_for_capabilities(capabilities, protocol, weights)
            if score > best_score:
                best_score = score
                best_protocol = protocol

        if best_protocol is None:
            # All protocols excluded, return first non-excluded
            for protocol in self._protocols:
                if protocol not in exclude:
                    return protocol, self._config.default_affinity
            return ProtocolName.MCP, 0.0

        return best_protocol, best_score

    def rank_protocols_for_capabilities(
        self,
        capabilities: list[str],
        weights: dict[str, float] | None = None,
    ) -> list[tuple[ProtocolName, float]]:
        """
        Rank all protocols for a set of capabilities.

        Args:
            capabilities: List of capabilities
            weights: Optional weights for each capability

        Returns:
            List of (protocol, score) tuples, sorted by score descending
        """
        scores = [
            (protocol, self.score_protocol_for_capabilities(capabilities, protocol, weights))
            for protocol in self._protocols
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def get_capability_coverage(
        self,
        protocol: ProtocolName,
        threshold: float = 0.7,
    ) -> list[str]:
        """
        Get capabilities well-supported by a protocol.

        Args:
            protocol: Protocol to analyze
            threshold: Minimum affinity threshold

        Returns:
            List of capabilities with affinity >= threshold
        """
        return [cap for cap in self._capabilities if self.get_affinity(cap, protocol) >= threshold]

    def register_affinity(
        self,
        capability: str,
        protocol: ProtocolName,
        affinity: float,
    ) -> None:
        """
        Register or update an affinity value.

        Args:
            capability: Capability name
            protocol: Protocol name
            affinity: Affinity value (0.0 to 1.0)
        """
        affinity = max(
            self._config.min_affinity,
            min(self._config.max_affinity, affinity),
        )
        self._matrix[(capability, protocol)] = affinity
        self._capabilities.add(capability)
        self._protocols.add(protocol)
        logger.debug(f"Registered affinity: {capability}/{protocol} = {affinity}")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get matrix statistics.

        Returns:
            Dictionary of statistics
        """
        if not self._matrix:
            return {
                "num_capabilities": 0,
                "num_protocols": len(self._protocols),
                "num_entries": 0,
                "avg_affinity": 0.0,
                "coverage": {},
            }

        affinities = list(self._matrix.values())
        coverage = {
            protocol: len(self.get_capability_coverage(protocol)) for protocol in self._protocols
        }

        return {
            "num_capabilities": len(self._capabilities),
            "num_protocols": len(self._protocols),
            "num_entries": len(self._matrix),
            "avg_affinity": sum(affinities) / len(affinities),
            "min_affinity": min(affinities),
            "max_affinity": max(affinities),
            "coverage": coverage,
        }


__all__ = [
    "CAPABILITY_PROTOCOL_AFFINITY",
    "AffinityConfig",
    "CapabilityAffinityMatrix",
]
