"""Network connectivity analysis for distributed systems.

This module provides tools for analyzing and verifying network connectivity
properties important for distributed consensus and fault tolerance.

Key concepts:
- k-connectivity: Graph remains connected after removing any k-1 nodes
- Vertex cuts: Minimum set of vertices whose removal disconnects the graph
- Robustness: Ability to maintain connectivity under failures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenticraft_foundation.topology.laplacian import NetworkGraph


@dataclass
class ConnectivityAnalysis:
    """Results of connectivity analysis.

    Attributes:
        is_connected: Whether graph is connected
        vertex_connectivity: Minimum nodes to remove to disconnect (κ)
        edge_connectivity: Minimum edges to remove to disconnect (λ)
        articulation_points: Nodes whose removal disconnects the graph
        bridges: Edges whose removal disconnects the graph
        k_connected: Maximum k for which graph is k-connected
        biconnected_components: List of biconnected component node sets
        strongly_connected: Whether graph is strongly connected (for directed)
        weakly_connected: Whether underlying undirected graph is connected
    """

    is_connected: bool
    vertex_connectivity: int
    edge_connectivity: int
    articulation_points: list[str]
    bridges: list[tuple[str, str]]
    k_connected: int
    biconnected_components: list[set[str]]
    strongly_connected: bool = True
    weakly_connected: bool = True

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Connectivity Analysis",
            "=" * 40,
            f"Connected: {self.is_connected}",
            f"Vertex Connectivity (κ): {self.vertex_connectivity}",
            f"Edge Connectivity (λ): {self.edge_connectivity}",
            f"k-Connected: {self.k_connected}",
            "",
        ]

        if self.articulation_points:
            lines.append("Articulation Points (single points of failure):")
            for node in self.articulation_points[:10]:
                lines.append(f"  - {node}")
            lines.append("")

        if self.bridges:
            lines.append("Bridge Edges (single edge failures):")
            for src, tgt in self.bridges[:10]:
                lines.append(f"  - {src} -- {tgt}")
            lines.append("")

        lines.append(f"Biconnected Components: {len(self.biconnected_components)}")

        return "\n".join(lines)

    def can_tolerate_failures(self, f: int) -> bool:
        """Check if graph can tolerate f node failures.

        Args:
            f: Number of failures

        Returns:
            True if graph remains connected after f failures
        """
        return self.vertex_connectivity > f


@dataclass
class FaultToleranceAnalysis:
    """Analysis of fault tolerance capabilities.

    Attributes:
        crash_tolerance: Maximum crash failures tolerable (f < n/2)
        byzantine_tolerance: Maximum Byzantine failures (f < n/3)
        critical_nodes: Nodes whose failure most impacts connectivity
        redundancy_score: Score indicating redundancy level (0-1)
        suggested_redundancy: Suggested edges to improve fault tolerance
    """

    crash_tolerance: int
    byzantine_tolerance: int
    critical_nodes: list[str]
    redundancy_score: float
    suggested_redundancy: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Fault Tolerance Analysis",
            "=" * 40,
            f"Crash Fault Tolerance: f < {self.crash_tolerance}",
            f"Byzantine Fault Tolerance: f < {self.byzantine_tolerance}",
            f"Redundancy Score: {self.redundancy_score:.2%}",
            "",
        ]

        if self.critical_nodes:
            lines.append("Critical Nodes (high-impact failures):")
            for node in self.critical_nodes[:5]:
                lines.append(f"  - {node}")
            lines.append("")

        if self.suggested_redundancy:
            lines.append("Suggested Redundancy Edges:")
            for src, tgt in self.suggested_redundancy[:5]:
                lines.append(f"  - {src} -- {tgt}")

        return "\n".join(lines)


class ConnectivityAnalyzer:
    """Analyzes network connectivity properties."""

    def __init__(self, graph: NetworkGraph):
        """Initialize analyzer with graph.

        Args:
            graph: Network graph to analyze
        """
        self._graph = graph
        self._node_ids = [n.id for n in graph.get_nodes()]
        self._adjacency = self._build_adjacency()

    def _build_adjacency(self) -> dict[str, set[str]]:
        """Build adjacency list representation."""
        adj: dict[str, set[str]] = {nid: set() for nid in self._node_ids}
        for edge in self._graph.get_edges():
            adj[edge.source].add(edge.target)
        return adj

    def _is_connected(self, exclude_nodes: set[str] | None = None) -> bool:
        """Check if graph is connected (excluding specified nodes).

        Args:
            exclude_nodes: Nodes to exclude from consideration

        Returns:
            True if remaining graph is connected
        """
        exclude = exclude_nodes or set()
        remaining = [nid for nid in self._node_ids if nid not in exclude]

        if len(remaining) <= 1:
            return True

        # BFS from first node
        visited: set[str] = set()
        queue = [remaining[0]]
        visited.add(remaining[0])

        while queue:
            node = queue.pop(0)
            for neighbor in self._adjacency.get(node, set()):
                if neighbor not in exclude and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(remaining)

    def _find_articulation_points(self) -> list[str]:
        """Find articulation points using DFS.

        Articulation points are nodes whose removal disconnects the graph.

        Returns:
            List of articulation point node IDs
        """
        if len(self._node_ids) <= 2:
            return []

        articulation_points: list[str] = []

        # Check each node
        for node in self._node_ids:
            if not self._is_connected(exclude_nodes={node}):
                articulation_points.append(node)

        return articulation_points

    def _find_bridges(self) -> list[tuple[str, str]]:
        """Find bridge edges.

        Bridges are edges whose removal disconnects the graph.

        Returns:
            List of bridge edge tuples
        """
        bridges: list[tuple[str, str]] = []
        seen_edges: set[tuple[str, str]] = set()

        for edge in self._graph.get_edges():
            # Normalize edge direction
            edge_key = tuple(sorted([edge.source, edge.target]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)  # type: ignore[arg-type]

            # Temporarily remove edge and check connectivity
            original_neighbors_src = self._adjacency[edge.source].copy()
            original_neighbors_tgt = self._adjacency[edge.target].copy()

            self._adjacency[edge.source].discard(edge.target)
            self._adjacency[edge.target].discard(edge.source)

            if not self._is_connected():
                bridges.append((edge.source, edge.target))

            # Restore edge
            self._adjacency[edge.source] = original_neighbors_src
            self._adjacency[edge.target] = original_neighbors_tgt

        return bridges

    def _compute_vertex_connectivity(self) -> int:
        """Compute vertex connectivity (κ).

        Vertex connectivity is the minimum number of vertices
        whose removal disconnects the graph.

        Returns:
            Vertex connectivity value
        """
        n = len(self._node_ids)
        if n <= 1:
            return 0

        # Check connectivity first
        if not self._is_connected():
            return 0

        # Try removing subsets of increasing size
        from itertools import combinations

        for k in range(1, n):
            for subset in combinations(self._node_ids, k):
                if not self._is_connected(exclude_nodes=set(subset)):
                    return k

        # If we can't disconnect by removing n-1 nodes, it's n-1 connected
        return n - 1

    def _compute_edge_connectivity(self) -> int:
        """Approximate edge connectivity using minimum degree.

        Returns the minimum vertex degree as an upper bound on
        edge connectivity λ(G). By Whitney's theorem, λ(G) <= δ(G)
        where δ(G) is the minimum degree. This is exact for
        complete graphs and many regular graphs.

        Returns:
            Minimum vertex degree (upper bound on edge connectivity).
        """
        min_degree = min(len(neighbors) for neighbors in self._adjacency.values())
        return min_degree

    def _find_biconnected_components(self) -> list[set[str]]:
        """Find biconnected components.

        A biconnected component has no articulation points.

        Returns:
            List of node sets for each component
        """
        # Simple approach: start with full graph,
        # remove articulation points to find components
        articulation = set(self._find_articulation_points())

        if not articulation:
            # Graph is already biconnected
            return [set(self._node_ids)]

        # Find connected components after removing articulation points
        remaining = [nid for nid in self._node_ids if nid not in articulation]
        visited: set[str] = set()
        components: list[set[str]] = []

        for start in remaining:
            if start in visited:
                continue

            component: set[str] = set()
            queue = [start]

            while queue:
                node = queue.pop(0)
                if node in visited or node in articulation:
                    continue
                visited.add(node)
                component.add(node)

                for neighbor in self._adjacency.get(node, set()):
                    if neighbor not in visited and neighbor not in articulation:
                        queue.append(neighbor)

            if component:
                # Add back articulation points that connect to this component
                for ap in articulation:
                    if any(n in self._adjacency.get(ap, set()) for n in component):
                        component.add(ap)
                components.append(component)

        return components

    def analyze(self) -> ConnectivityAnalysis:
        """Perform full connectivity analysis.

        Returns:
            ConnectivityAnalysis with results
        """
        is_connected = self._is_connected()
        articulation_points = self._find_articulation_points()
        bridges = self._find_bridges()
        vertex_connectivity = self._compute_vertex_connectivity()
        edge_connectivity = self._compute_edge_connectivity()
        biconnected = self._find_biconnected_components()

        return ConnectivityAnalysis(
            is_connected=is_connected,
            vertex_connectivity=vertex_connectivity,
            edge_connectivity=edge_connectivity,
            articulation_points=articulation_points,
            bridges=bridges,
            k_connected=vertex_connectivity,
            biconnected_components=biconnected,
            strongly_connected=is_connected,  # For undirected
            weakly_connected=is_connected,
        )

    def analyze_fault_tolerance(self) -> FaultToleranceAnalysis:
        """Analyze fault tolerance capabilities.

        Returns:
            FaultToleranceAnalysis with results
        """
        n = len(self._node_ids)
        connectivity = self.analyze()

        # Crash fault tolerance: can tolerate f crashes if n >= 2f + 1
        # So f < n/2, but also limited by connectivity
        crash_tolerance = min(connectivity.vertex_connectivity, (n - 1) // 2)

        # Byzantine fault tolerance: requires n >= 3f + 1
        # So f < n/3, but also limited by connectivity
        byzantine_tolerance = min(connectivity.vertex_connectivity, (n - 1) // 3)

        # Find critical nodes (articulation points + high-degree nodes)
        critical_nodes = list(connectivity.articulation_points)

        # Add high-degree nodes not in articulation points
        degree_map = {nid: len(neighbors) for nid, neighbors in self._adjacency.items()}
        sorted_by_degree = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)
        for nid, _ in sorted_by_degree[:5]:
            if nid not in critical_nodes:
                critical_nodes.append(nid)

        # Redundancy score: ratio of actual to maximum possible edges
        max_edges = n * (n - 1) // 2
        actual_edges = self._graph.edge_count
        redundancy_score = actual_edges / max_edges if max_edges > 0 else 0

        # Suggest edges to improve fault tolerance
        suggested: list[tuple[str, str]] = []

        # Connect nodes with low degree
        if n > 2:
            low_degree_nodes = [
                nid
                for nid, degree in degree_map.items()
                if degree < 2 and len(self._adjacency[nid]) < n - 1
            ]

            for _i, node1 in enumerate(low_degree_nodes):
                for node2 in self._node_ids:
                    if (
                        node1 != node2
                        and node2 not in self._adjacency[node1]
                        and len(suggested) < 5
                    ):
                        suggested.append((node1, node2))
                        break

        return FaultToleranceAnalysis(
            crash_tolerance=crash_tolerance,
            byzantine_tolerance=byzantine_tolerance,
            critical_nodes=critical_nodes[:10],
            redundancy_score=redundancy_score,
            suggested_redundancy=suggested,
        )


def verify_consensus_requirements(
    graph: NetworkGraph,
    fault_model: str = "crash",
    f: int = 1,
) -> tuple[bool, str]:
    """Verify graph meets consensus requirements for given fault model.

    Args:
        graph: Network graph
        fault_model: Type of failures ("crash" or "byzantine")
        f: Number of failures to tolerate

    Returns:
        Tuple of (meets_requirements, explanation)
    """
    n = graph.node_count
    analyzer = ConnectivityAnalyzer(graph)
    connectivity = analyzer.analyze()

    if fault_model.lower() == "byzantine":
        required_nodes = 3 * f + 1
        if n < required_nodes:
            return (
                False,
                f"Byzantine consensus needs n >= 3f+1 = {required_nodes}, have {n}",
            )
    else:  # crash
        required_nodes = 2 * f + 1
        if n < required_nodes:
            return (
                False,
                f"Crash consensus needs n >= 2f+1 = {required_nodes}, have {n}",
            )

    if connectivity.vertex_connectivity <= f:
        return (
            False,
            f"Graph vertex connectivity ({connectivity.vertex_connectivity}) "
            f"must be > f ({f}) to tolerate failures",
        )

    return (True, f"Graph meets requirements for f={f} {fault_model} failures")


__all__ = [
    "ConnectivityAnalysis",
    "FaultToleranceAnalysis",
    "ConnectivityAnalyzer",
    "verify_consensus_requirements",
]
