"""Network topology analysis for distributed systems.

This module provides graph-theoretic tools for analyzing and optimizing
network topologies in distributed systems.

Key features:
- Graph Laplacian spectral analysis
- Algebraic connectivity (λ₂) computation
- Consensus convergence bounds
- Connectivity analysis and fault tolerance verification
"""

from __future__ import annotations

from agenticraft_foundation.topology.connectivity import (
    ConnectivityAnalysis,
    ConnectivityAnalyzer,
    FaultToleranceAnalysis,
    verify_consensus_requirements,
)

# Hypergraph extension
from agenticraft_foundation.topology.hypergraph import (
    Hyperedge,
    HypergraphAnalysis,
    HypergraphNetwork,
)
from agenticraft_foundation.topology.laplacian import (
    Edge,
    LaplacianAnalysis,
    NetworkGraph,
    Node,
    TopologyType,
    analyze_consensus_time,
    compare_topologies,
    optimal_topology_for_consensus,
)

__all__ = [
    # Laplacian analysis
    "TopologyType",
    "Node",
    "Edge",
    "LaplacianAnalysis",
    "NetworkGraph",
    "analyze_consensus_time",
    "compare_topologies",
    "optimal_topology_for_consensus",
    # Connectivity analysis
    "ConnectivityAnalysis",
    "ConnectivityAnalyzer",
    "FaultToleranceAnalysis",
    "verify_consensus_requirements",
    # Hypergraph
    "Hyperedge",
    "HypergraphAnalysis",
    "HypergraphNetwork",
]
