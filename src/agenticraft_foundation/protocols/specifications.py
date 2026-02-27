"""Formal protocol specifications for verifying routing correctness.

Provides executable formal specifications following the pattern from
``specifications/consensus_spec.py``.

Key properties:
- ProtocolValidPath: all (u,v) in path use supported protocols
- SemanticPreservation: meaning(m) = meaning(T_{p→p'}(m))
- OptimalRouting: no path with lower cost exists
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticraft_foundation.types import ProtocolName

if TYPE_CHECKING:
    from .graph import ProtocolGraph
    from .routing import OptimalRoute


class ProtocolPropertyType(str, Enum):
    """Types of protocol formal properties."""

    VALIDITY = "validity"
    """Path validity constraints"""

    SEMANTIC = "semantic"
    """Semantic preservation constraints"""

    OPTIMALITY = "optimality"
    """Optimality constraints"""

    INVARIANT = "invariant"
    """Invariants that must always hold"""


class ProtocolPropertyStatus(str, Enum):
    """Status of protocol property verification."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ProtocolPropertyResult:
    """Result of protocol property verification.

    Attributes:
        property_name: Name of the property checked
        property_type: Type of property
        status: Verification status
        message: Human-readable message
        counterexample: Counterexample if violated
        details: Additional verification details
        timestamp: When verification was performed
    """

    property_name: str
    property_type: ProtocolPropertyType
    status: ProtocolPropertyStatus
    message: str = ""
    counterexample: Any | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def is_satisfied(self) -> bool:
        """Check if property was satisfied."""
        return self.status == ProtocolPropertyStatus.SATISFIED


class ProtocolFormalProperty(ABC):
    """Base class for protocol formal properties.

    Properties are executable specifications that can be checked
    against protocol routing states.
    """

    def __init__(self, name: str, property_type: ProtocolPropertyType):
        """Initialize property.

        Args:
            name: Property name
            property_type: Type of property
        """
        self.name = name
        self.property_type = property_type

    @abstractmethod
    def check(self, **kwargs: Any) -> ProtocolPropertyResult:
        """Check if property holds.

        Returns:
            ProtocolPropertyResult with verification outcome
        """
        pass


class ProtocolValidPath(ProtocolFormalProperty):
    """
    Protocol-valid path property.

    Constraint: ∀ (u,v) ∈ π: pᵢ ∈ Φ(u,v)

    Verifies that every edge in the path supports the protocol used on it.
    """

    def __init__(self) -> None:
        """Initialize ProtocolValidPath property."""
        super().__init__("ProtocolValidPath", ProtocolPropertyType.VALIDITY)

    def check(
        self,
        path: list[str] | None = None,
        protocol_sequence: list[ProtocolName] | None = None,
        graph: ProtocolGraph | None = None,
        **kwargs: Any,
    ) -> ProtocolPropertyResult:
        """Check if path is protocol-valid.

        Args:
            path: Sequence of agent IDs
            protocol_sequence: Protocol used on each edge
            graph: Protocol graph

        Returns:
            ProtocolPropertyResult indicating validity
        """
        if not path or not protocol_sequence or not graph:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.NOT_APPLICABLE,
                message="Missing required arguments (path, protocol_sequence, graph)",
            )

        if len(path) < 2:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.SATISFIED,
                message="Trivially valid (path length < 2)",
            )

        if len(protocol_sequence) != len(path) - 1:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.VIOLATED,
                message=(
                    f"Protocol sequence length ({len(protocol_sequence)}) "
                    f"does not match path edges ({len(path) - 1})"
                ),
                counterexample={
                    "path_length": len(path),
                    "protocol_sequence_length": len(protocol_sequence),
                },
            )

        # Check each edge
        violations = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            protocol = protocol_sequence[i]

            edge = graph.get_edge(source, target)

            if edge is None:
                violations.append(
                    {
                        "edge": (source, target),
                        "index": i,
                        "error": "Edge does not exist",
                    }
                )
            elif not edge.supports_protocol(protocol):
                violations.append(
                    {
                        "edge": (source, target),
                        "index": i,
                        "protocol": protocol.value,
                        "supported_protocols": [p.value for p in edge.protocols],
                        "error": "Protocol not supported on edge",
                    }
                )

        if violations:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.VIOLATED,
                message=f"Path violates protocol constraints at {len(violations)} edge(s)",
                counterexample={"violations": violations},
                details={
                    "path": path,
                    "protocol_sequence": [p.value for p in protocol_sequence],
                },
            )

        return ProtocolPropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=ProtocolPropertyStatus.SATISFIED,
            message=f"Path is protocol-valid ({len(path) - 1} edges)",
            details={
                "num_edges": len(path) - 1,
                "protocols_used": list({p.value for p in protocol_sequence}),
            },
        )


class SemanticPreservation(ProtocolFormalProperty):
    """
    Semantic preservation property.

    Constraint: meaning(m) = meaning(T_{p→p'}(m))

    Verifies that protocol translations preserve semantic meaning.
    """

    def __init__(self, max_semantic_loss: float = 0.1):
        """Initialize SemanticPreservation property.

        Args:
            max_semantic_loss: Maximum allowed semantic loss (0.0 to 1.0)
        """
        super().__init__("SemanticPreservation", ProtocolPropertyType.SEMANTIC)
        self.max_semantic_loss = max_semantic_loss

    def check(
        self,
        route: OptimalRoute | None = None,
        semantic_loss: float | None = None,
        **kwargs: Any,
    ) -> ProtocolPropertyResult:
        """Check if semantic preservation holds.

        Args:
            route: OptimalRoute with semantic_loss_estimate
            semantic_loss: Direct semantic loss value (alternative to route)

        Returns:
            ProtocolPropertyResult indicating preservation
        """
        # Get semantic loss from route or direct value
        loss = semantic_loss
        if route is not None:
            loss = route.semantic_loss_estimate

        if loss is None:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.NOT_APPLICABLE,
                message="No semantic loss information available",
            )

        if loss <= self.max_semantic_loss:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.SATISFIED,
                message=(
                    f"Semantic preservation holds "
                    f"(loss={loss:.3f} <= threshold={self.max_semantic_loss})"
                ),
                details={
                    "semantic_loss": loss,
                    "threshold": self.max_semantic_loss,
                    "margin": self.max_semantic_loss - loss,
                },
            )
        else:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.VIOLATED,
                message=(
                    f"Semantic loss exceeds threshold "
                    f"(loss={loss:.3f} > threshold={self.max_semantic_loss})"
                ),
                counterexample={
                    "semantic_loss": loss,
                    "threshold": self.max_semantic_loss,
                    "excess": loss - self.max_semantic_loss,
                },
            )


class OptimalRouting(ProtocolFormalProperty):
    """
    Optimal routing property.

    Constraint: No path π' exists with cost(π') < cost(π)

    Verifies that the chosen route is optimal among alternatives.
    """

    def __init__(self, tolerance: float = 0.001):
        """Initialize OptimalRouting property.

        Args:
            tolerance: Tolerance for floating-point comparison
        """
        super().__init__("OptimalRouting", ProtocolPropertyType.OPTIMALITY)
        self.tolerance = tolerance

    def check(
        self,
        route: OptimalRoute | None = None,
        all_routes: list[OptimalRoute] | None = None,
        **kwargs: Any,
    ) -> ProtocolPropertyResult:
        """Check if route is optimal.

        Args:
            route: The route claimed to be optimal
            all_routes: All possible routes for comparison

        Returns:
            ProtocolPropertyResult indicating optimality
        """
        if route is None:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.NOT_APPLICABLE,
                message="No route provided",
            )

        if not all_routes:
            # No alternatives to compare - assume optimal
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.SATISFIED,
                message="Route is optimal (no alternatives)",
                details={"route_cost": route.total_cost},
            )

        # Find minimum cost among all routes
        min_cost = min(r.total_cost for r in all_routes)
        route_cost = route.total_cost

        # Check if route is within tolerance of optimal
        if route_cost <= min_cost + self.tolerance:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.SATISFIED,
                message=(f"Route is optimal (cost={route_cost:.4f}, min={min_cost:.4f})"),
                details={
                    "route_cost": route_cost,
                    "min_cost": min_cost,
                    "num_alternatives": len(all_routes),
                },
            )
        else:
            # Find the better route
            better_routes = [r for r in all_routes if r.total_cost < route_cost - self.tolerance]
            best_alternative = min(better_routes, key=lambda r: r.total_cost)

            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.VIOLATED,
                message=(f"Route is not optimal (cost={route_cost:.4f} > min={min_cost:.4f})"),
                counterexample={
                    "route_cost": route_cost,
                    "better_cost": best_alternative.total_cost,
                    "better_path": best_alternative.path,
                    "cost_difference": route_cost - best_alternative.total_cost,
                },
            )


class TranslationBoundProperty(ProtocolFormalProperty):
    """
    Translation bound property.

    Constraint: Number of protocol translations ≤ max_translations

    Verifies that routing doesn't exceed translation limits.
    """

    def __init__(self, max_translations: int = 3):
        """Initialize TranslationBoundProperty.

        Args:
            max_translations: Maximum allowed translations
        """
        super().__init__("TranslationBound", ProtocolPropertyType.INVARIANT)
        self.max_translations = max_translations

    def check(
        self,
        route: OptimalRoute | None = None,
        num_translations: int | None = None,
        **kwargs: Any,
    ) -> ProtocolPropertyResult:
        """Check if translation count is within bounds.

        Args:
            route: OptimalRoute to check
            num_translations: Direct translation count (alternative)

        Returns:
            ProtocolPropertyResult indicating if bound holds
        """
        count = num_translations
        if route is not None:
            count = route.num_translations

        if count is None:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.NOT_APPLICABLE,
                message="No translation count available",
            )

        if count <= self.max_translations:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.SATISFIED,
                message=(f"Translation bound satisfied ({count} <= {self.max_translations})"),
                details={
                    "translations": count,
                    "max_allowed": self.max_translations,
                },
            )
        else:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.VIOLATED,
                message=(f"Translation bound exceeded ({count} > {self.max_translations})"),
                counterexample={
                    "translations": count,
                    "max_allowed": self.max_translations,
                    "excess": count - self.max_translations,
                },
            )


class ProtocolResilience(ProtocolFormalProperty):
    """
    k-Protocol resilience verification.

    Theorem 1: A mesh achieves at most (|P|-1)-protocol resilience.
    Verifies that removing any k-1 protocols preserves graph connectivity.

    This checks: for each subset S of protocols with |S| = k-1,
    removing all edges that ONLY support protocols in S still leaves
    the graph connected.
    """

    def __init__(self, k: int = 1):
        """Initialize ProtocolResilience property.

        Args:
            k: Resilience level to verify. The mesh is k-protocol-resilient
                if it remains connected when any k-1 protocols fail.
        """
        super().__init__("ProtocolResilience", ProtocolPropertyType.INVARIANT)
        self.k = k

    def check(
        self,
        graph: ProtocolGraph | None = None,
        **kwargs: Any,
    ) -> ProtocolPropertyResult:
        """Check if mesh is k-protocol-resilient.

        Args:
            graph: Protocol graph to verify

        Returns:
            ProtocolPropertyResult indicating resilience status
        """
        if graph is None:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.NOT_APPLICABLE,
                message="No graph provided",
            )

        protocols = list(graph.protocols)
        num_protocols = len(protocols)

        if self.k > num_protocols:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.VIOLATED,
                message=(
                    f"Cannot achieve {self.k}-resilience with only "
                    f"{num_protocols} protocols (max is {num_protocols - 1})"
                ),
                details={
                    "k": self.k,
                    "num_protocols": num_protocols,
                    "max_resilience": num_protocols - 1,
                },
            )

        if len(graph.agents) < 2:
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.SATISFIED,
                message="Trivially resilient (< 2 agents)",
            )

        # Check all subsets of size k-1
        failures_to_try = self.k - 1
        if failures_to_try <= 0:
            # k=1 means 0 failures to tolerate — always resilient if connected
            agents = list(graph.agents.keys())
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    if not graph.is_reachable(agents[i], agents[j]):
                        return ProtocolPropertyResult(
                            property_name=self.name,
                            property_type=self.property_type,
                            status=ProtocolPropertyStatus.VIOLATED,
                            message="Graph is not connected",
                            counterexample={
                                "unreachable_pair": (agents[i], agents[j]),
                            },
                        )
            return ProtocolPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=ProtocolPropertyStatus.SATISFIED,
                message="Graph is connected (1-protocol-resilient)",
            )

        # Generate all subsets of protocols of size failures_to_try
        from itertools import combinations

        agents = list(graph.agents.keys())

        for failed_protocols in combinations(protocols, failures_to_try):
            failed_set = set(failed_protocols)
            surviving = set(protocols) - failed_set

            # Check if all agent pairs are reachable using surviving protocols
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    if not graph.is_reachable(agents[i], agents[j], allowed_protocols=surviving):
                        return ProtocolPropertyResult(
                            property_name=self.name,
                            property_type=self.property_type,
                            status=ProtocolPropertyStatus.VIOLATED,
                            message=(
                                f"Not {self.k}-resilient: removing "
                                f"{[p.value for p in failed_set]} disconnects "
                                f"{agents[i]} from {agents[j]}"
                            ),
                            counterexample={
                                "failed_protocols": [p.value for p in failed_set],
                                "disconnected_pair": (agents[i], agents[j]),
                            },
                        )

        return ProtocolPropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=ProtocolPropertyStatus.SATISFIED,
            message=(
                f"Mesh is {self.k}-protocol-resilient "
                f"(survives any {failures_to_try} protocol failures)"
            ),
            details={
                "k": self.k,
                "num_protocols": num_protocols,
                "protocols_checked": num_protocols,
            },
        )


class ProtocolSpecification:
    """Complete specification for protocol routing.

    Combines all protocol properties and provides verification API.
    """

    def __init__(
        self,
        max_semantic_loss: float = 0.1,
        max_translations: int = 3,
        custom_properties: list[ProtocolFormalProperty] | None = None,
    ):
        """Initialize specification.

        Args:
            max_semantic_loss: Threshold for semantic preservation
            max_translations: Maximum allowed translations
            custom_properties: Additional custom properties
        """
        self.properties: list[ProtocolFormalProperty] = [
            ProtocolValidPath(),
            SemanticPreservation(max_semantic_loss),
            OptimalRouting(),
            TranslationBoundProperty(max_translations),
        ]
        if custom_properties:
            self.properties.extend(custom_properties)

    def verify(
        self,
        route: OptimalRoute,
        graph: ProtocolGraph,
        all_routes: list[OptimalRoute] | None = None,
    ) -> list[ProtocolPropertyResult]:
        """Verify all properties for a route.

        Args:
            route: Route to verify
            graph: Protocol graph
            all_routes: Alternative routes for optimality check

        Returns:
            List of property results
        """
        results = []

        for prop in self.properties:
            result = prop.check(
                route=route,
                path=route.path,
                protocol_sequence=route.protocol_sequence,
                graph=graph,
                all_routes=all_routes,
            )
            results.append(result)

        return results

    def verify_validity(
        self,
        path: list[str],
        protocol_sequence: list[ProtocolName],
        graph: ProtocolGraph,
    ) -> ProtocolPropertyResult:
        """Verify only path validity.

        Args:
            path: Path to verify
            protocol_sequence: Protocols used
            graph: Protocol graph

        Returns:
            Validity property result
        """
        for prop in self.properties:
            if isinstance(prop, ProtocolValidPath):
                return prop.check(
                    path=path,
                    protocol_sequence=protocol_sequence,
                    graph=graph,
                )

        # Create temporary property
        return ProtocolValidPath().check(
            path=path,
            protocol_sequence=protocol_sequence,
            graph=graph,
        )

    def is_valid(
        self,
        route: OptimalRoute,
        graph: ProtocolGraph,
    ) -> bool:
        """Check if route satisfies all properties.

        Args:
            route: Route to check
            graph: Protocol graph

        Returns:
            True if all properties satisfied
        """
        results = self.verify(route, graph)
        return all(r.is_satisfied() for r in results)

    def summary(
        self,
        route: OptimalRoute,
        graph: ProtocolGraph,
        all_routes: list[OptimalRoute] | None = None,
    ) -> str:
        """Generate verification summary.

        Args:
            route: Route to summarize
            graph: Protocol graph
            all_routes: Alternative routes

        Returns:
            Human-readable summary
        """
        results = self.verify(route, graph, all_routes)

        lines = [
            "Protocol Specification Verification",
            "=" * 40,
            f"Path: {' -> '.join(route.path)}",
            f"Protocols: {[p.value for p in route.protocol_sequence]}",
            f"Cost: {route.total_cost:.4f}",
            f"Translations: {route.num_translations}",
            "",
            "Property Results:",
        ]

        for result in results:
            status_icon = {
                ProtocolPropertyStatus.SATISFIED: "✓",
                ProtocolPropertyStatus.VIOLATED: "✗",
                ProtocolPropertyStatus.UNKNOWN: "?",
                ProtocolPropertyStatus.NOT_APPLICABLE: "-",
            }.get(result.status, "?")

            lines.append(f"  {status_icon} {result.property_name}: {result.message}")

            if result.counterexample:
                lines.append(f"      Counterexample: {result.counterexample}")

        # Overall status
        all_valid = all(
            r.is_satisfied() or r.status == ProtocolPropertyStatus.NOT_APPLICABLE for r in results
        )
        lines.append("")
        lines.append(f"Overall: {'PASS' if all_valid else 'FAIL'}")

        return "\n".join(lines)


__all__ = [
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
