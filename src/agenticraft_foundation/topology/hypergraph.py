"""Hypergraph extension for multi-agent group coordination.

Extends the graph model with hyperedges that connect >2 agents,
enabling formal analysis of group communication patterns.

Hypergraph H = (V, E_h) where each hyperedge e ∈ E_h ⊆ 2^V
can connect any subset of vertices.

Key features:
- Hyperedge management for group coordination patterns
- Hypergraph Laplacian: L_H = D_v - H W D_e^{-1} H^T
- Group coordination analysis (consensus time for multi-party patterns)
- Conversion from/to standard graphs

References:
- Zhou et al. (2006) - Learning with Hypergraphs
- Agarwal et al. (2006) - Higher Order Learning with Graphs
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Hyperedge:
    """A hyperedge connecting multiple nodes.

    Unlike standard edges, a hyperedge can connect any number of nodes,
    modeling group communication patterns (broadcast, scatter-gather, consensus).
    """

    edge_id: str
    """Unique hyperedge identifier"""

    nodes: set[str]
    """Set of connected node IDs"""

    weight: float = 1.0
    """Hyperedge weight"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional edge metadata"""

    @property
    def degree(self) -> int:
        """Hyperedge degree (number of nodes it connects)."""
        return len(self.nodes)

    def contains(self, node_id: str) -> bool:
        """Check if a node is in this hyperedge."""
        return node_id in self.nodes

    def __hash__(self) -> int:
        return hash(self.edge_id)


@dataclass
class HypergraphAnalysis:
    """Results of hypergraph spectral analysis."""

    num_nodes: int
    """Number of nodes"""

    num_hyperedges: int
    """Number of hyperedges"""

    avg_hyperedge_degree: float
    """Average number of nodes per hyperedge"""

    max_hyperedge_degree: int
    """Maximum hyperedge degree"""

    algebraic_connectivity: float
    """Algebraic connectivity (λ₂) of hypergraph Laplacian"""

    eigenvalues: list[float]
    """Eigenvalues of hypergraph Laplacian"""

    consensus_bound: float
    """Estimated consensus convergence time"""

    is_connected: bool
    """Whether the hypergraph is connected"""


class HypergraphNetwork:
    """Hypergraph network for group coordination analysis.

    H = (V, E_h) with weighted hyperedges.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._hyperedges: dict[str, Hyperedge] = {}

    @property
    def nodes(self) -> dict[str, dict[str, Any]]:
        """Get all nodes."""
        return dict(self._nodes)

    @property
    def hyperedges(self) -> dict[str, Hyperedge]:
        """Get all hyperedges."""
        return dict(self._hyperedges)

    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a node to the hypergraph."""
        self._nodes[node_id] = metadata or {}

    def add_hyperedge(
        self,
        edge_id: str,
        nodes: set[str],
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Hyperedge:
        """Add a hyperedge connecting multiple nodes.

        All referenced nodes must exist. Missing nodes are auto-added.

        Args:
            edge_id: Unique edge identifier
            nodes: Set of node IDs to connect
            weight: Edge weight
            metadata: Additional metadata

        Returns:
            The created Hyperedge
        """
        # Auto-add missing nodes
        for nid in nodes:
            if nid not in self._nodes:
                self._nodes[nid] = {}

        edge = Hyperedge(
            edge_id=edge_id,
            nodes=set(nodes),
            weight=weight,
            metadata=metadata or {},
        )
        self._hyperedges[edge_id] = edge
        return edge

    def remove_hyperedge(self, edge_id: str) -> None:
        """Remove a hyperedge."""
        self._hyperedges.pop(edge_id, None)

    def node_degree(self, node_id: str) -> int:
        """Get degree of a node (number of hyperedges it belongs to)."""
        return sum(1 for edge in self._hyperedges.values() if node_id in edge.nodes)

    def node_weighted_degree(self, node_id: str) -> float:
        """Get weighted degree: sum of weights of incident hyperedges."""
        return sum(edge.weight for edge in self._hyperedges.values() if node_id in edge.nodes)

    def incident_edges(self, node_id: str) -> list[Hyperedge]:
        """Get hyperedges incident to a node."""
        return [edge for edge in self._hyperedges.values() if node_id in edge.nodes]

    def neighbors(self, node_id: str) -> set[str]:
        """Get all nodes connected to a node via any hyperedge."""
        result: set[str] = set()
        for edge in self._hyperedges.values():
            if node_id in edge.nodes:
                result.update(edge.nodes)
        result.discard(node_id)
        return result

    def is_connected(self) -> bool:
        """Check if the hypergraph is connected via BFS."""
        if not self._nodes:
            return True

        start = next(iter(self._nodes))
        visited: set[str] = set()
        queue = [start]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)

        return len(visited) == len(self._nodes)

    def incidence_matrix(self) -> tuple[list[str], list[str], list[list[float]]]:
        """Compute incidence matrix H.

        ``H[v][e] = 1`` if node v is in hyperedge e, 0 otherwise.

        Returns:
            (node_ids, edge_ids, matrix H)
        """
        node_ids = sorted(self._nodes.keys())
        edge_ids = sorted(self._hyperedges.keys())
        node_idx = {nid: i for i, nid in enumerate(node_ids)}

        n = len(node_ids)
        m = len(edge_ids)

        h_mat = [[0.0] * m for _ in range(n)]

        for j, eid in enumerate(edge_ids):
            for nid in self._hyperedges[eid].nodes:
                if nid in node_idx:
                    h_mat[node_idx[nid]][j] = 1.0

        return node_ids, edge_ids, h_mat

    def laplacian_matrix(self) -> tuple[list[str], list[list[float]]]:
        """Compute hypergraph Laplacian: L_H = D_v - H W D_e^{-1} H^T.

        Where:
        - D_v: diagonal node degree matrix
        - W: diagonal hyperedge weight matrix
        - D_e: diagonal hyperedge degree matrix
        - H: incidence matrix

        Returns:
            (node_ids, Laplacian matrix)
        """
        import numpy as np

        node_ids, edge_ids, h_mat = self.incidence_matrix()
        n = len(node_ids)

        if n == 0:
            return [], []

        # Convert to numpy for efficient matrix operations
        h_arr = np.array(h_mat, dtype=np.float64)  # n × m

        # D_e^{-1}: inverse hyperedge degree
        de_inv = np.array(
            [
                1.0 / self._hyperedges[eid].degree if self._hyperedges[eid].degree > 0 else 0.0
                for eid in edge_ids
            ],
            dtype=np.float64,
        )

        # W: hyperedge weights
        w = np.array(
            [self._hyperedges[eid].weight for eid in edge_ids],
            dtype=np.float64,
        )

        # M = W * D_e^{-1} (element-wise, diagonal)
        m_diag = w * de_inv

        # H W D_e^{-1} H^T via numpy: (H * m_diag) @ H^T
        hwdh = (h_arr * m_diag) @ h_arr.T

        # D_v: weighted node degree = H @ w
        dv = h_arr @ w

        # L = D_v - H W D_e^{-1} H^T
        lap_arr = np.diag(dv) - hwdh

        return node_ids, lap_arr.tolist()

    def analyze(self) -> HypergraphAnalysis:
        """Perform spectral analysis of the hypergraph.

        Computes eigenvalues of the hypergraph Laplacian and derives
        connectivity and consensus convergence metrics.

        Returns:
            HypergraphAnalysis with spectral properties
        """
        if not self._nodes:
            return HypergraphAnalysis(
                num_nodes=0,
                num_hyperedges=0,
                avg_hyperedge_degree=0.0,
                max_hyperedge_degree=0,
                algebraic_connectivity=0.0,
                eigenvalues=[],
                consensus_bound=float("inf"),
                is_connected=True,
            )

        node_ids, lap = self.laplacian_matrix()
        n = len(node_ids)

        if n <= 1:
            return HypergraphAnalysis(
                num_nodes=n,
                num_hyperedges=len(self._hyperedges),
                avg_hyperedge_degree=0.0,
                max_hyperedge_degree=0,
                algebraic_connectivity=0.0,
                eigenvalues=[0.0] if n == 1 else [],
                consensus_bound=0.0,
                is_connected=True,
            )

        # Compute eigenvalues via power iteration on Laplacian
        eigenvalues = self._compute_eigenvalues(lap, n)

        # Algebraic connectivity = second smallest eigenvalue
        sorted_eigs = sorted(eigenvalues)
        lambda2 = sorted_eigs[1] if len(sorted_eigs) > 1 else 0.0

        # Consensus bound: T = O(n log n / λ₂)
        if lambda2 > 1e-10:
            consensus_bound = n * math.log(n) / lambda2
        else:
            consensus_bound = float("inf")

        # Hyperedge statistics
        degrees = [e.degree for e in self._hyperedges.values()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0

        return HypergraphAnalysis(
            num_nodes=n,
            num_hyperedges=len(self._hyperedges),
            avg_hyperedge_degree=avg_degree,
            max_hyperedge_degree=max_degree,
            algebraic_connectivity=lambda2,
            eigenvalues=sorted_eigs,
            consensus_bound=consensus_bound,
            is_connected=lambda2 > 1e-10,
        )

    def _compute_eigenvalues(
        self,
        matrix: list[list[float]],
        n: int,
    ) -> list[float]:
        """Compute eigenvalues of the hypergraph Laplacian.

        Uses numpy's eigh() for exact eigenvalue computation
        of the symmetric Laplacian matrix.

        Args:
            matrix: Square symmetric Laplacian matrix
            n: Matrix dimension

        Returns:
            List of eigenvalues in ascending order
        """
        import numpy as np

        if n == 0:
            return []
        if n == 1:
            return [matrix[0][0]]

        arr = np.array(matrix, dtype=np.float64)
        eigenvalues = np.linalg.eigh(arr)[0]
        return sorted(eigenvalues.tolist())

    @classmethod
    def from_graph(
        cls,
        edges: list[tuple[str, str]],
        weights: dict[tuple[str, str], float] | None = None,
    ) -> HypergraphNetwork:
        """Create a hypergraph from a standard graph.

        Each edge becomes a 2-hyperedge.

        Args:
            edges: List of (source, target) pairs
            weights: Optional edge weights

        Returns:
            HypergraphNetwork with 2-hyperedges
        """
        hg = cls()
        weights = weights or {}

        for i, (src, tgt) in enumerate(edges):
            hg.add_node(src)
            hg.add_node(tgt)
            w = weights.get((src, tgt), 1.0)
            hg.add_hyperedge(f"e{i}", {src, tgt}, weight=w)

        return hg

    @classmethod
    def clique_expansion(
        cls,
        groups: list[set[str]],
        weights: list[float] | None = None,
    ) -> HypergraphNetwork:
        """Create a hypergraph from clique groups.

        Each group of nodes becomes a hyperedge.

        Args:
            groups: List of node groups
            weights: Optional weights per group

        Returns:
            HypergraphNetwork
        """
        hg = cls()
        weights = weights or [1.0] * len(groups)

        for i, group in enumerate(groups):
            w = weights[i] if i < len(weights) else 1.0
            for nid in group:
                hg.add_node(nid)
            hg.add_hyperedge(f"h{i}", group, weight=w)

        return hg

    def analyze_group_coordination(self) -> dict[str, Any]:
        """Analyze group coordination patterns.

        Returns metrics about how effectively the hypergraph supports
        multi-party coordination.
        """
        analysis = self.analyze()

        # Participation ratio: fraction of nodes in at least one hyperedge
        participating = set()
        for edge in self._hyperedges.values():
            participating.update(edge.nodes)
        participation_ratio = len(participating) / len(self._nodes) if self._nodes else 0.0

        # Coverage: average fraction of nodes covered per hyperedge
        coverage = analysis.avg_hyperedge_degree / len(self._nodes) if self._nodes else 0.0

        return {
            "num_nodes": len(self._nodes),
            "num_hyperedges": len(self._hyperedges),
            "algebraic_connectivity": analysis.algebraic_connectivity,
            "consensus_bound": analysis.consensus_bound,
            "is_connected": analysis.is_connected,
            "participation_ratio": participation_ratio,
            "avg_group_size": analysis.avg_hyperedge_degree,
            "max_group_size": analysis.max_hyperedge_degree,
            "coverage": coverage,
        }


__all__ = [
    "Hyperedge",
    "HypergraphAnalysis",
    "HypergraphNetwork",
]
