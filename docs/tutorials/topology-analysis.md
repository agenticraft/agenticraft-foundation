# Analyzing Mesh Topologies

**Time:** 15 minutes

In this tutorial, you will create several common agent network topologies, analyze their spectral properties to understand how well they support consensus, and use hypergraphs to model group coordination patterns.

## Prerequisites

- Python 3.10+
- `agenticraft-foundation` installed
- Basic understanding of graph structure (nodes, edges)

## What You'll Build

A topology analysis workbench that compares full mesh, ring, and star topologies using spectral metrics. You will:

1. Create a full mesh topology and compute its spectral properties
2. Create ring and star topologies for comparison
3. Compare topologies using algebraic connectivity and consensus bounds
4. Build a hypergraph for group coordination analysis

## Step 1: Create a Full Mesh Topology

A full mesh is the ideal topology for consensus: every agent is directly connected to every other agent. Information flows in a single hop between any pair of agents.

```python
from agenticraft_foundation.topology import NetworkGraph, LaplacianAnalysis

mesh = NetworkGraph()
n = 5
for i in range(n):
    mesh.add_node(f"agent_{i}")
for i in range(n):
    for j in range(i + 1, n):
        mesh.add_edge(f"agent_{i}", f"agent_{j}")

analysis = LaplacianAnalysis(mesh)
print(f"Nodes: {analysis.num_nodes}")
print(f"Edges: {analysis.num_edges}")
print(f"Algebraic connectivity (lambda_2): {analysis.algebraic_connectivity:.4f}")
print(f"Consensus bound: {analysis.consensus_bound:.2f}")
```

The algebraic connectivity ($\lambda_2$) is the second-smallest eigenvalue of the graph's Laplacian matrix. It is the single most important metric for consensus in distributed systems. A higher $\lambda_2$ means faster consensus convergence. For a full mesh with $n$ nodes, $\lambda_2$ equals $n$, which is the theoretical maximum for that number of nodes.

The consensus bound is derived from $\lambda_2$ and provides an upper bound on the number of communication rounds needed for the network to converge to agreement.

## Step 2: Create Ring and Star Topologies

Real networks are rarely full meshes. Rings and stars are two common alternatives that trade connectivity for simplicity.

```python
# Ring topology
ring = NetworkGraph()
for i in range(n):
    ring.add_node(f"agent_{i}")
for i in range(n):
    ring.add_edge(f"agent_{i}", f"agent_{(i+1) % n}")

ring_analysis = LaplacianAnalysis(ring)
print(f"Ring lambda_2: {ring_analysis.algebraic_connectivity:.4f}")

# Star topology
star = NetworkGraph()
star.add_node("hub")
for i in range(n - 1):
    star.add_node(f"agent_{i}")
    star.add_edge("hub", f"agent_{i}")

star_analysis = LaplacianAnalysis(star)
print(f"Star lambda_2: {star_analysis.algebraic_connectivity:.4f}")
```

The ring topology connects each agent to its two neighbors, forming a circle. Information must travel through multiple hops to reach distant agents, which means slower consensus. $\lambda_2$ for a ring is approximately $2 - 2\cos(2\pi/n)$, which is much smaller than the full mesh value.

The star topology connects all agents through a central hub. While it has the same number of edges as the ring (n edges), its structure allows information to flow from any agent to any other in just 2 hops. However, the hub is a single point of failure.

## Step 3: Compare Topologies

Putting the metrics side by side reveals the trade-offs clearly.

```python
print(f"{'Topology':<12} {'lambda_2':>10} {'Consensus bound':>16}")
print("-" * 40)
for name, a in [("Full mesh", analysis), ("Ring", ring_analysis), ("Star", star_analysis)]:
    print(f"{name:<12} {a.algebraic_connectivity:>10.4f} {a.consensus_bound:>16.2f}")
```

You will see that the full mesh has the highest $\lambda_2$ and the lowest consensus bound (fastest convergence). The star falls in between, and the ring has the lowest $\lambda_2$ and the highest consensus bound (slowest convergence).

This comparison is not just academic. In a multi-agent system with 50 agents, the difference between a ring and a mesh topology can mean the difference between consensus in 5 rounds and consensus in 50 rounds. The spectral analysis gives you a quantitative basis for choosing your network topology.

The trade-off, of course, is the number of connections: a full mesh with $n$ nodes requires $n(n-1)/2$ edges, which becomes impractical for large networks. This is why real systems use topologies that balance connectivity with operational complexity.

## Step 4: Build a Hypergraph for Group Coordination

Standard graphs model pairwise connections. But in multi-agent systems, agents often coordinate in groups -- a team meeting, a broadcast channel, or a shared workspace. Hypergraphs generalize graphs by allowing edges (called hyperedges) to connect any number of nodes simultaneously.

```python
from agenticraft_foundation.topology import HypergraphNetwork

hg = HypergraphNetwork()
hg.add_hyperedge("team_a", {"agent_0", "agent_1", "agent_2"}, weight=2.0)
hg.add_hyperedge("team_b", {"agent_1", "agent_2", "agent_3"}, weight=1.5)
hg.add_hyperedge("all_hands", {"agent_0", "agent_1", "agent_2", "agent_3"})

hg_analysis = hg.analyze()
print(f"Hypergraph lambda_2: {hg_analysis.algebraic_connectivity:.4f}")
print(f"Consensus bound: {hg_analysis.consensus_bound:.2f}")
print(f"Connected: {hg_analysis.is_connected}")

coord = hg.analyze_group_coordination()
print(f"Participation ratio: {coord['participation_ratio']:.2f}")
print(f"Average group size: {coord['avg_group_size']:.1f}")
```

Each hyperedge has a name, a set of member nodes, and an optional weight. The weight represents the coordination capacity of the group -- a team with a weight of 2.0 can propagate information twice as fast as a team with the default weight of 1.0.

The hypergraph analysis computes spectral properties using the hypergraph Laplacian, which generalizes the standard graph Laplacian. The participation ratio measures how evenly agents are distributed across groups (1.0 means perfectly even, lower values indicate some agents participate in many more groups than others). The average group size gives you a sense of the coordination overhead.

This analysis is particularly useful for designing multi-agent systems with team-based coordination, where you need to balance team sizes, overlap between teams, and the overall connectivity of the coordination structure.

## Complete Script

```python
"""Topology Analysis Tutorial - Complete Script

Compares mesh, ring, and star topologies using spectral analysis,
then builds a hypergraph for group coordination analysis.
"""
from agenticraft_foundation.topology import (
    NetworkGraph, LaplacianAnalysis, HypergraphNetwork,
)

# Step 1: Full mesh topology
n = 5
mesh = NetworkGraph()
for i in range(n):
    mesh.add_node(f"agent_{i}")
for i in range(n):
    for j in range(i + 1, n):
        mesh.add_edge(f"agent_{i}", f"agent_{j}")

analysis = LaplacianAnalysis(mesh)
print(f"Nodes: {analysis.num_nodes}")
print(f"Edges: {analysis.num_edges}")
print(f"Algebraic connectivity (lambda_2): {analysis.algebraic_connectivity:.4f}")
print(f"Consensus bound: {analysis.consensus_bound:.2f}")

# Step 2: Ring and star topologies
ring = NetworkGraph()
for i in range(n):
    ring.add_node(f"agent_{i}")
for i in range(n):
    ring.add_edge(f"agent_{i}", f"agent_{(i+1) % n}")

ring_analysis = LaplacianAnalysis(ring)
print(f"Ring lambda_2: {ring_analysis.algebraic_connectivity:.4f}")

star = NetworkGraph()
star.add_node("hub")
for i in range(n - 1):
    star.add_node(f"agent_{i}")
    star.add_edge("hub", f"agent_{i}")

star_analysis = LaplacianAnalysis(star)
print(f"Star lambda_2: {star_analysis.algebraic_connectivity:.4f}")

# Step 3: Compare topologies
print(f"\n{'Topology':<12} {'lambda_2':>10} {'Consensus bound':>16}")
print("-" * 40)
for name, a in [("Full mesh", analysis), ("Ring", ring_analysis), ("Star", star_analysis)]:
    print(f"{name:<12} {a.algebraic_connectivity:>10.4f} {a.consensus_bound:>16.2f}")

# Step 4: Hypergraph for group coordination
hg = HypergraphNetwork()
hg.add_hyperedge("team_a", {"agent_0", "agent_1", "agent_2"}, weight=2.0)
hg.add_hyperedge("team_b", {"agent_1", "agent_2", "agent_3"}, weight=1.5)
hg.add_hyperedge("all_hands", {"agent_0", "agent_1", "agent_2", "agent_3"})

hg_analysis = hg.analyze()
print(f"\nHypergraph lambda_2: {hg_analysis.algebraic_connectivity:.4f}")
print(f"Consensus bound: {hg_analysis.consensus_bound:.2f}")
print(f"Connected: {hg_analysis.is_connected}")

coord = hg.analyze_group_coordination()
print(f"Participation ratio: {coord['participation_ratio']:.2f}")
print(f"Average group size: {coord['avg_group_size']:.1f}")
```

## Next Steps

- Read [Topology Concepts](../concepts/spectral-topology.md) for the mathematical foundations of spectral graph theory
- Explore the [Topology API Reference](../api/topology/index.md) for weighted graphs, directed graphs, and advanced spectral methods
- Continue to [Formal Consensus Verification](consensus-verification.md) to verify consensus properties directly
