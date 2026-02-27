"""Formal protocol model for multi-protocol mesh.

Implements the formal model with the following components:

- G = (V, E, P, Φ, Γ): Protocol graph model.
- Protocol-Aware Dijkstra: Optimal routing with translation costs.
- Semantic Preservation: Round-trip verification.
- Formal Specifications: Protocol routing property verification.

Example::

    from agenticraft_foundation.protocols import (
        ProtocolGraph, ProtocolAwareDijkstra, SemanticPreservationVerifier,
    )

    graph = ProtocolGraph()
    graph.add_agent("agent1", ["code_execution"], {ProtocolName.MCP})
    graph.add_agent("agent2", ["task_delegation"], {ProtocolName.A2A})
    graph.add_edge("agent1", "agent2", {ProtocolName.MCP, ProtocolName.A2A})
"""

from __future__ import annotations

# Capability-protocol affinity
from agenticraft_foundation.protocols.affinity import (
    CAPABILITY_PROTOCOL_AFFINITY,
    AffinityConfig,
    CapabilityAffinityMatrix,
)

# Compatibility matrix
from agenticraft_foundation.protocols.compatibility import (
    PROTOCOL_COMPATIBILITY,
    CompatibilityLevel,
    CompatibilityRelation,
    ProtocolCompatibilityMatrix,
)

# Path cost calculator
from agenticraft_foundation.protocols.cost import (
    CostConfig,
    PathCost,
    PathCostCalculator,
    TranslationCost,
)

# Graph model
from agenticraft_foundation.protocols.graph import (
    AgentNode,
    NodeType,
    ProtocolEdge,
    ProtocolGraph,
)

# Protocol-Aware routing algorithms
from agenticraft_foundation.protocols.routing import (
    OptimalRoute,
    ProtocolAwareDijkstra,
    ProtocolConstrainedBFS,
    ResilientRouter,
    RoutingConfig,
    RoutingState,
)

# Semantic preservation verification
from agenticraft_foundation.protocols.semantic import (
    ProtocolMessage,
    ProtocolTranslator,
    SemanticPreservationVerifier,
    SemanticVerificationResult,
    SemanticViolation,
    SemanticViolationType,
    VerificationConfig,
)

# Semantic routing
from agenticraft_foundation.protocols.semantic_routing import (
    CapabilityEmbedding,
    SemanticRouteCandidate,
    SemanticRouter,
)

# Formal specifications
from agenticraft_foundation.protocols.specifications import (
    OptimalRouting,
    ProtocolFormalProperty,
    ProtocolPropertyResult,
    ProtocolPropertyStatus,
    ProtocolPropertyType,
    ProtocolResilience,
    ProtocolSpecification,
    ProtocolValidPath,
    SemanticPreservation,
    TranslationBoundProperty,
)

# Composable protocol transformers
from agenticraft_foundation.protocols.transformers import (
    BaseProtocolTransformer,
    ComposedTransformer,
    IdentityTransformer,
    TransformerClassification,
    TransformerRegistry,
    TransformResult,
)

# Protocol workflow model
from agenticraft_foundation.protocols.workflow import (
    OptimalProtocolAssigner,
    ProtocolWorkflow,
    WorkflowTask,
    WorkflowTaskStatus,
    WorkflowValidationResult,
    WorkflowValidator,
)

__all__ = [
    # Graph model
    "NodeType",
    "AgentNode",
    "ProtocolEdge",
    "ProtocolGraph",
    # Compatibility
    "CompatibilityLevel",
    "CompatibilityRelation",
    "PROTOCOL_COMPATIBILITY",
    "ProtocolCompatibilityMatrix",
    # Affinity
    "AffinityConfig",
    "CAPABILITY_PROTOCOL_AFFINITY",
    "CapabilityAffinityMatrix",
    # Cost
    "TranslationCost",
    "PathCost",
    "CostConfig",
    "PathCostCalculator",
    # Routing
    "RoutingState",
    "OptimalRoute",
    "RoutingConfig",
    "ProtocolAwareDijkstra",
    "ProtocolConstrainedBFS",
    "ResilientRouter",
    # Semantic
    "SemanticViolationType",
    "SemanticViolation",
    "SemanticVerificationResult",
    "ProtocolMessage",
    "ProtocolTranslator",
    "VerificationConfig",
    "SemanticPreservationVerifier",
    # Transformers
    "TransformerClassification",
    "TransformResult",
    "BaseProtocolTransformer",
    "IdentityTransformer",
    "ComposedTransformer",
    "TransformerRegistry",
    # Workflow
    "WorkflowTaskStatus",
    "WorkflowTask",
    "ProtocolWorkflow",
    "WorkflowValidationResult",
    "WorkflowValidator",
    "OptimalProtocolAssigner",
    # Semantic routing
    "CapabilityEmbedding",
    "SemanticRouteCandidate",
    "SemanticRouter",
    # Specifications
    "ProtocolPropertyType",
    "ProtocolPropertyStatus",
    "ProtocolPropertyResult",
    "ProtocolFormalProperty",
    "ProtocolValidPath",
    "SemanticPreservation",
    "OptimalRouting",
    "TranslationBoundProperty",
    "ProtocolResilience",
    "ProtocolSpecification",
]
