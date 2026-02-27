"""Graph Laplacian analysis for network topology optimization.

This module provides spectral analysis tools for understanding and optimizing
distributed system topologies based on graph-theoretic principles.

Key concepts:
- Algebraic Connectivity (λ₂): Second smallest eigenvalue of Laplacian
- Fiedler Vector: Eigenvector corresponding to λ₂
- Consensus Convergence: T = O(n log n / λ₂)

References:
- Fiedler, M. (1973). Algebraic connectivity of graphs.
- Olfati-Saber, R. (2006). Flocking for multi-agent dynamic systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class TopologyType(str, Enum):
    """Common network topology types."""

    RING = "ring"
    STAR = "star"
    MESH = "mesh"
    TREE = "tree"
    COMPLETE = "complete"
    RANDOM = "random"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"


@dataclass
class Node:
    """A node in the network graph.

    Attributes:
        id: Unique identifier
        neighbors: Set of neighbor node IDs
        weight: Optional weight for weighted graphs
        metadata: Additional node metadata
    """

    id: str
    neighbors: set[str] = field(default_factory=set)
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """An edge in the network graph.

    Attributes:
        source: Source node ID
        target: Target node ID
        weight: Edge weight (default 1.0)
    """

    source: str
    target: str
    weight: float = 1.0


@dataclass
class LaplacianAnalysis:
    """Results of Laplacian spectral analysis.

    Attributes:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges in the graph
        algebraic_connectivity: Second smallest eigenvalue (λ₂)
        spectral_gap: Gap between λ₁ and λ₂
        fiedler_vector: Eigenvector for λ₂ (maps node ID to value)
        eigenvalues: All eigenvalues in ascending order
        consensus_bound: Upper bound on consensus convergence time
        diameter_estimate: Estimated graph diameter from spectral properties
        bottleneck_edges: Edges with high Fiedler values (potential bottlenecks)
        suggested_edges: Suggested edges to add for improved connectivity
        is_connected: Whether the graph is connected
    """

    num_nodes: int
    num_edges: int
    algebraic_connectivity: float
    spectral_gap: float
    fiedler_vector: dict[str, float]
    eigenvalues: list[float]
    consensus_bound: float
    diameter_estimate: float
    bottleneck_edges: list[tuple[str, str]]
    suggested_edges: list[tuple[str, str]]
    is_connected: bool

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Graph Spectral Analysis",
            "=" * 40,
            f"Connected: {self.is_connected}",
            f"Algebraic Connectivity (λ₂): {self.algebraic_connectivity:.4f}",
            f"Spectral Gap: {self.spectral_gap:.4f}",
            f"Estimated Diameter: {self.diameter_estimate:.1f}",
            f"Consensus Convergence Bound: O({self.consensus_bound:.1f})",
            "",
        ]

        if self.bottleneck_edges:
            lines.append("Potential Bottleneck Edges:")
            for src, tgt in self.bottleneck_edges[:5]:
                lines.append(f"  {src} -- {tgt}")
            lines.append("")

        if self.suggested_edges:
            lines.append("Suggested Edges to Add:")
            for src, tgt in self.suggested_edges[:5]:
                lines.append(f"  {src} -- {tgt}")

        return "\n".join(lines)


class NetworkGraph:
    """Network graph for topology analysis.

    Provides methods for building and analyzing network topologies
    using graph Laplacian spectral analysis.
    """

    def __init__(self) -> None:
        """Initialize empty graph."""
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    def add_node(
        self,
        node_id: str,
        weight: float = 1.0,
        **metadata: Any,
    ) -> Node:
        """Add a node to the graph.

        Args:
            node_id: Unique node identifier
            weight: Node weight
            **metadata: Additional metadata

        Returns:
            The created Node
        """
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.weight = weight
            node.metadata.update(metadata)
        else:
            node = Node(id=node_id, weight=weight, metadata=metadata)
            self._nodes[node_id] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        bidirectional: bool = True,
    ) -> Edge:
        """Add an edge to the graph.

        Args:
            source: Source node ID
            target: Target node ID
            weight: Edge weight
            bidirectional: If True, add edge in both directions

        Returns:
            The created Edge
        """
        # Ensure nodes exist
        if source not in self._nodes:
            self.add_node(source)
        if target not in self._nodes:
            self.add_node(target)

        # Add edge
        edge = Edge(source=source, target=target, weight=weight)
        self._edges.append(edge)

        # Update neighbor sets
        self._nodes[source].neighbors.add(target)
        if bidirectional:
            self._nodes[target].neighbors.add(source)
            self._edges.append(Edge(source=target, target=source, weight=weight))

        return edge

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges.

        Args:
            node_id: Node to remove

        Returns:
            True if node was removed
        """
        if node_id not in self._nodes:
            return False

        # Remove edges
        self._edges = [e for e in self._edges if e.source != node_id and e.target != node_id]

        # Remove from neighbor sets
        for node in self._nodes.values():
            node.neighbors.discard(node_id)

        del self._nodes[node_id]
        return True

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes(self) -> list[Node]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_edges(self) -> list[Edge]:
        """Get all edges."""
        return self._edges

    @property
    def node_count(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges."""
        return len(self._edges) // 2  # Divide by 2 for bidirectional

    def _build_adjacency_matrix(self) -> tuple[list[list[float]], list[str]]:
        """Build adjacency matrix.

        Returns:
            Tuple of (adjacency matrix, ordered node IDs)
        """
        node_ids = sorted(self._nodes.keys())
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Initialize matrix
        matrix = [[0.0] * n for _ in range(n)]

        for edge in self._edges:
            i = id_to_idx.get(edge.source)
            j = id_to_idx.get(edge.target)
            if i is not None and j is not None:
                matrix[i][j] = edge.weight

        return matrix, node_ids

    def _build_laplacian_matrix(self) -> tuple[list[list[float]], list[str]]:
        """Build graph Laplacian matrix.

        L = D - A, where D is degree matrix and A is adjacency matrix.

        Returns:
            Tuple of (Laplacian matrix, ordered node IDs)
        """
        adj, node_ids = self._build_adjacency_matrix()
        n = len(node_ids)

        # Build Laplacian: L = D - A
        laplacian = [[0.0] * n for _ in range(n)]
        for i in range(n):
            degree = sum(adj[i])
            laplacian[i][i] = degree
            for j in range(n):
                if i != j:
                    laplacian[i][j] = -adj[i][j]

        return laplacian, node_ids

    def _compute_eigenvalues_simple(
        self,
        matrix: list[list[float]],
    ) -> list[float]:
        """Compute eigenvalues of a symmetric matrix.

        Uses numpy's eigh() for symmetric/Hermitian matrices,
        which returns all eigenvalues in ascending order.

        Args:
            matrix: Square symmetric matrix

        Returns:
            List of eigenvalues in ascending order
        """
        import numpy as np

        n = len(matrix)
        if n == 0:
            return []

        arr = np.array(matrix, dtype=np.float64)
        eigenvalues = np.linalg.eigh(arr)[0]
        return sorted(eigenvalues.tolist())

    def _compute_fiedler_vector(
        self,
        laplacian: list[list[float]],
        node_ids: list[str],
    ) -> dict[str, float]:
        """Compute Fiedler vector (eigenvector for λ₂).

        The Fiedler vector is the eigenvector corresponding to the
        second smallest eigenvalue of the Laplacian.

        Args:
            laplacian: Laplacian matrix
            node_ids: Node IDs corresponding to matrix indices

        Returns:
            Dictionary mapping node ID to Fiedler vector component
        """
        import numpy as np

        n = len(laplacian)
        if n < 2:
            return {node_ids[0]: 0.0} if n == 1 else {}

        arr = np.array(laplacian, dtype=np.float64)
        _, eigenvectors = np.linalg.eigh(arr)
        fiedler = eigenvectors[:, 1]  # second column = Fiedler vector
        return {node_ids[i]: float(fiedler[i]) for i in range(n)}

    def analyze(self) -> LaplacianAnalysis:
        """Perform spectral analysis on the graph.

        Returns:
            LaplacianAnalysis with spectral properties
        """
        n = self.node_count
        e = self.edge_count
        if n == 0:
            return LaplacianAnalysis(
                num_nodes=0,
                num_edges=0,
                algebraic_connectivity=0.0,
                spectral_gap=0.0,
                fiedler_vector={},
                eigenvalues=[],
                consensus_bound=float("inf"),
                diameter_estimate=float("inf"),
                bottleneck_edges=[],
                suggested_edges=[],
                is_connected=False,
            )

        if n == 1:
            node_id = list(self._nodes.keys())[0]
            return LaplacianAnalysis(
                num_nodes=1,
                num_edges=0,
                algebraic_connectivity=0.0,
                spectral_gap=0.0,
                fiedler_vector={node_id: 0.0},
                eigenvalues=[0.0],
                consensus_bound=0.0,
                diameter_estimate=0.0,
                bottleneck_edges=[],
                suggested_edges=[],
                is_connected=True,
            )

        # Build Laplacian
        laplacian, node_ids = self._build_laplacian_matrix()

        # Compute eigenvalues
        eigenvalues = self._compute_eigenvalues_simple(laplacian)

        # λ₁ should be 0 (or very close)
        # λ₂ is algebraic connectivity
        lambda1 = eigenvalues[0] if eigenvalues else 0.0
        lambda2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        lambda_max = eigenvalues[-1] if eigenvalues else 0.0

        # Graph is connected iff λ₂ > 0
        is_connected = lambda2 > 1e-10

        # Spectral gap
        spectral_gap = lambda2 - lambda1

        # Compute Fiedler vector
        fiedler = self._compute_fiedler_vector(laplacian, node_ids)

        # Consensus convergence bound: T = O(n log n / λ₂)
        if lambda2 > 1e-10:
            consensus_bound = n * math.log(n) / lambda2
        else:
            consensus_bound = float("inf")

        # Diameter upper bound from Chung (1997, Theorem 1.1):
        # D ≤ ⌈cosh⁻¹(n-1) / cosh⁻¹((λ_max + λ₂) / (λ_max - λ₂))⌉
        if is_connected and abs(lambda_max - lambda2) < 1e-6:
            # Near-complete graph: all non-zero eigenvalues are equal,
            # so the spectral ratio diverges and the graph has diameter 1.
            diameter_estimate = 1.0
        elif is_connected and lambda_max > lambda2:
            ratio = (lambda_max + lambda2) / (lambda_max - lambda2)
            diameter_estimate = math.ceil(math.acosh(n - 1) / math.acosh(ratio))
        else:
            diameter_estimate = float("inf")

        # Find bottleneck edges (edges connecting nodes with very different Fiedler values)
        bottleneck_edges = self._find_bottleneck_edges(fiedler)

        # Suggest edges to improve connectivity
        suggested_edges = self._suggest_edges(fiedler)

        return LaplacianAnalysis(
            num_nodes=n,
            num_edges=e,
            algebraic_connectivity=lambda2,
            spectral_gap=spectral_gap,
            fiedler_vector=fiedler,
            eigenvalues=eigenvalues,
            consensus_bound=consensus_bound,
            diameter_estimate=diameter_estimate,
            bottleneck_edges=bottleneck_edges,
            suggested_edges=suggested_edges,
            is_connected=is_connected,
        )

    def _find_bottleneck_edges(
        self,
        fiedler: dict[str, float],
        threshold: float = 0.5,
    ) -> list[tuple[str, str]]:
        """Find edges that are potential bottlenecks.

        Bottleneck edges connect nodes with very different Fiedler values,
        indicating they bridge different parts of the graph.

        Args:
            fiedler: Fiedler vector mapping node IDs to values
            threshold: Minimum Fiedler value difference to consider bottleneck

        Returns:
            List of (source, target) edge tuples
        """
        bottlenecks: list[tuple[str, str, float]] = []

        seen_edges: set[tuple[str, str]] = set()
        for edge in self._edges:
            # Normalize edge direction
            sorted_pair = sorted([edge.source, edge.target])
            edge_key: tuple[str, str] = (sorted_pair[0], sorted_pair[1])
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            v1 = fiedler.get(edge.source, 0.0)
            v2 = fiedler.get(edge.target, 0.0)
            diff = abs(v1 - v2)

            if diff > threshold:
                bottlenecks.append((edge.source, edge.target, diff))

        # Sort by difference (highest first)
        bottlenecks.sort(key=lambda x: x[2], reverse=True)
        return [(src, tgt) for src, tgt, _ in bottlenecks]

    def _suggest_edges(
        self,
        fiedler: dict[str, float],
        max_suggestions: int = 5,
    ) -> list[tuple[str, str]]:
        """Suggest edges to add to improve connectivity.

        Suggests connecting nodes with very different Fiedler values
        that aren't currently connected.

        Args:
            fiedler: Fiedler vector mapping node IDs to values
            max_suggestions: Maximum number of suggestions

        Returns:
            List of (source, target) suggested edge tuples
        """
        if len(fiedler) < 2:
            return []

        # Sort nodes by Fiedler value
        sorted_nodes = sorted(fiedler.items(), key=lambda x: x[1])

        suggestions: list[tuple[str, str, float]] = []

        # Try connecting nodes from opposite ends of Fiedler spectrum
        n = len(sorted_nodes)
        for i in range(n // 4):  # Bottom quarter
            for j in range(n - 1, n - n // 4 - 1, -1):  # Top quarter
                node_low = sorted_nodes[i][0]
                node_high = sorted_nodes[j][0]

                # Check if not already connected
                if node_high not in self._nodes[node_low].neighbors:
                    diff = abs(sorted_nodes[j][1] - sorted_nodes[i][1])
                    suggestions.append((node_low, node_high, diff))

        # Sort by Fiedler difference (highest impact first)
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return [(src, tgt) for src, tgt, _ in suggestions[:max_suggestions]]

    @classmethod
    def create_ring(cls, n: int, node_prefix: str = "node") -> NetworkGraph:
        """Create a ring topology.

        Args:
            n: Number of nodes
            node_prefix: Prefix for node IDs

        Returns:
            NetworkGraph with ring topology
        """
        graph = cls()
        for i in range(n):
            graph.add_node(f"{node_prefix}_{i}")

        for i in range(n):
            graph.add_edge(f"{node_prefix}_{i}", f"{node_prefix}_{(i + 1) % n}")

        return graph

    @classmethod
    def create_complete(cls, n: int, node_prefix: str = "node") -> NetworkGraph:
        """Create a complete (fully connected) topology.

        Args:
            n: Number of nodes
            node_prefix: Prefix for node IDs

        Returns:
            NetworkGraph with complete topology
        """
        graph = cls()
        for i in range(n):
            graph.add_node(f"{node_prefix}_{i}")

        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(f"{node_prefix}_{i}", f"{node_prefix}_{j}")

        return graph

    @classmethod
    def create_star(cls, n: int, node_prefix: str = "node") -> NetworkGraph:
        """Create a star topology.

        Args:
            n: Number of nodes (including center)
            node_prefix: Prefix for node IDs

        Returns:
            NetworkGraph with star topology
        """
        graph = cls()
        center = f"{node_prefix}_center"
        graph.add_node(center)

        for i in range(n - 1):
            leaf = f"{node_prefix}_{i}"
            graph.add_node(leaf)
            graph.add_edge(center, leaf)

        return graph

    @classmethod
    def create_mesh(cls, rows: int, cols: int, node_prefix: str = "node") -> NetworkGraph:
        """Create a 2D mesh/grid topology.

        Args:
            rows: Number of rows
            cols: Number of columns
            node_prefix: Prefix for node IDs

        Returns:
            NetworkGraph with mesh topology
        """
        graph = cls()

        # Add nodes
        for r in range(rows):
            for c in range(cols):
                graph.add_node(f"{node_prefix}_{r}_{c}")

        # Add horizontal edges
        for r in range(rows):
            for c in range(cols - 1):
                graph.add_edge(f"{node_prefix}_{r}_{c}", f"{node_prefix}_{r}_{c + 1}")

        # Add vertical edges
        for r in range(rows - 1):
            for c in range(cols):
                graph.add_edge(f"{node_prefix}_{r}_{c}", f"{node_prefix}_{r + 1}_{c}")

        return graph


def analyze_consensus_time(
    graph: NetworkGraph,
    epsilon: float = 0.01,
) -> float:
    """Estimate consensus convergence time.

    For linear consensus protocols, convergence time is bounded by:
    T = O(log(1/ε) / λ₂)

    Args:
        graph: Network graph to analyze
        epsilon: Convergence tolerance

    Returns:
        Estimated convergence time
    """
    analysis = graph.analyze()

    if analysis.algebraic_connectivity <= 0:
        return float("inf")

    return math.log(1 / epsilon) / analysis.algebraic_connectivity


def compare_topologies(
    topologies: dict[str, NetworkGraph],
) -> dict[str, LaplacianAnalysis]:
    """Compare multiple network topologies.

    Args:
        topologies: Dictionary mapping topology name to graph

    Returns:
        Dictionary mapping topology name to analysis
    """
    results = {}
    for name, graph in topologies.items():
        results[name] = graph.analyze()
    return results


def optimal_topology_for_consensus(
    n: int,
    max_edges: int | None = None,
) -> tuple[NetworkGraph, LaplacianAnalysis]:
    """Find optimal topology for consensus with edge budget.

    If unconstrained, complete graph is optimal (λ₂ = n).
    With edge constraints, we optimize for maximum λ₂.

    Args:
        n: Number of nodes
        max_edges: Maximum number of edges (None = unconstrained)

    Returns:
        Tuple of (optimal graph, analysis)
    """
    if max_edges is None or max_edges >= n * (n - 1) // 2:
        # Complete graph is optimal
        graph = NetworkGraph.create_complete(n)
        return graph, graph.analyze()

    # With constraints, start with ring and add edges greedily
    graph = NetworkGraph.create_ring(n)
    current_edges = n  # Ring has n edges

    while current_edges < max_edges:
        # Analyze current graph
        analysis = graph.analyze()

        if not analysis.suggested_edges:
            break

        # Add best suggested edge
        src, tgt = analysis.suggested_edges[0]
        graph.add_edge(src, tgt)
        current_edges += 1

    return graph, graph.analyze()


__all__ = [
    "TopologyType",
    "Node",
    "Edge",
    "LaplacianAnalysis",
    "NetworkGraph",
    "analyze_consensus_time",
    "compare_topologies",
    "optimal_topology_for_consensus",
]
