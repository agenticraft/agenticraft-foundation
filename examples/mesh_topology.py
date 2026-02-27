"""Spectral Topology Analysis -- Network resilience via Laplacian.

Demonstrates LaplacianAnalysis for evaluating agent mesh topologies.
"""

from agenticraft_foundation.topology import (
    NetworkGraph,
    analyze_consensus_time,
)

# =============================================================
# Build a 5-agent mesh topology
# =============================================================
print("=== 5-Agent Mesh Topology ===")

graph = NetworkGraph()
for i in range(5):
    graph.add_node(f"agent_{i}")

# Full mesh: every agent connected to every other
for i in range(5):
    for j in range(i + 1, 5):
        graph.add_edge(f"agent_{i}", f"agent_{j}")

# Analyze
analysis = graph.analyze()
print(f"Nodes: {analysis.num_nodes}")
print(f"Edges: {analysis.num_edges}")
print(f"Algebraic connectivity (lambda_2): {analysis.algebraic_connectivity:.4f}")
print(f"Spectral gap: {analysis.spectral_gap:.4f}")
print(f"Consensus bound: {analysis.consensus_bound:.4f}")
print(f"Is connected: {analysis.is_connected}")

# =============================================================
# Compare topologies
# =============================================================
print("\n=== Topology Comparison ===")

# Ring topology
ring = NetworkGraph()
for i in range(5):
    ring.add_node(f"agent_{i}")
for i in range(5):
    ring.add_edge(f"agent_{i}", f"agent_{(i + 1) % 5}")

# Star topology
star = NetworkGraph()
for i in range(5):
    star.add_node(f"agent_{i}")
for i in range(1, 5):
    star.add_edge("agent_0", f"agent_{i}")

for name, g in [("Full Mesh", graph), ("Ring", ring), ("Star", star)]:
    a = g.analyze()
    print(f"  {name:12s}: nodes={a.num_nodes}, edges={a.num_edges}, "
          f"lambda_2={a.algebraic_connectivity:.4f}, connected={a.is_connected}")

# Higher lambda_2 = faster consensus convergence
# Full mesh > Ring > Star (for 5 nodes)

# =============================================================
# Consensus time analysis
# =============================================================
print("\n=== Consensus Convergence ===")

mesh_analysis = graph.analyze()
if mesh_analysis.algebraic_connectivity > 0:
    convergence_bound = 1.0 / mesh_analysis.algebraic_connectivity
    print(f"Full mesh convergence bound: {convergence_bound:.4f}")

ring_analysis = ring.analyze()
if ring_analysis.algebraic_connectivity > 0:
    convergence_bound = 1.0 / ring_analysis.algebraic_connectivity
    print(f"Ring convergence bound: {convergence_bound:.4f}")

# Consensus time estimation
mesh_time = analyze_consensus_time(graph)
ring_time = analyze_consensus_time(ring)
print(f"\nConsensus time (full mesh): {mesh_time:.4f}")
print(f"Consensus time (ring): {ring_time:.4f}")

print("\nHigher algebraic connectivity = faster consensus = more resilient mesh.")
