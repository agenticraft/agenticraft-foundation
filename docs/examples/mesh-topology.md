# Mesh Topology

**Source:** `examples/mesh_topology.py`

This example demonstrates spectral topology analysis using `NetworkGraph` and `LaplacianAnalysis`. It builds different agent mesh topologies, compares their algebraic connectivity, and estimates consensus convergence times.

---

## 1. Build a 5-Node Full Mesh

The example starts by constructing a fully connected graph of 5 agents.

```python
from agenticraft_foundation.topology import (
    NetworkGraph,
    analyze_consensus_time,
)

graph = NetworkGraph()
for i in range(5):
    graph.add_node(f"agent_{i}")

# Full mesh: every agent connected to every other
for i in range(5):
    for j in range(i + 1, 5):
        graph.add_edge(f"agent_{i}", f"agent_{j}")
```

A full mesh connects every agent to every other agent. With 5 nodes, this produces 10 edges. This is the densest possible topology and serves as the upper bound for connectivity metrics.

## 2. Run Laplacian Analysis

The `analyze()` method computes spectral properties of the graph Laplacian.

```python
analysis = graph.analyze()
print(f"Nodes: {analysis.num_nodes}")
print(f"Edges: {analysis.num_edges}")
print(f"Algebraic connectivity (lambda_2): {analysis.algebraic_connectivity:.4f}")
print(f"Spectral gap: {analysis.spectral_gap:.4f}")
print(f"Consensus bound: {analysis.consensus_bound:.4f}")
print(f"Is connected: {analysis.is_connected}")
```

The key metric is **algebraic connectivity** ($\lambda_2$) -- the second-smallest eigenvalue of the graph Laplacian matrix $L = D - A$, where $D$ is the degree matrix and $A$ is the adjacency matrix. $\lambda_2$ quantifies how well-connected the graph is:

- $\lambda_2 = 0$ means the graph is disconnected (at least two components).
- **Higher $\lambda_2$** means faster information propagation and more resilient consensus.
- The **spectral gap** (difference between the two smallest eigenvalues) indicates separation between connected and disconnected behavior.
- The **consensus bound** ($1 / \lambda_2$) gives an upper bound on the time for distributed consensus to converge.

## 3. Compare Topologies

Three topologies are compared: full mesh, ring, and star.

```python
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
```

The comparison reveals a hierarchy:

- **Full mesh**: Highest $\lambda_2$. Every agent can communicate directly with every other. Maximum resilience but $O(n^2)$ edges.
- **Ring**: Moderate $\lambda_2$. Each agent connects to two neighbors. Information must traverse up to $n/2$ hops. $\lambda_2$ decreases as the ring grows.
- **Star**: Lower $\lambda_2$. All communication routes through the hub (agent_0). Removing the hub disconnects the entire network. The hub is a single point of failure.

## 4. Interpret $\lambda_2$

The algebraic connectivity directly predicts consensus performance.

```python
mesh_analysis = graph.analyze()
if mesh_analysis.algebraic_connectivity > 0:
    convergence_bound = 1.0 / mesh_analysis.algebraic_connectivity
    print(f"Full mesh convergence bound: {convergence_bound:.4f}")

ring_analysis = ring.analyze()
if ring_analysis.algebraic_connectivity > 0:
    convergence_bound = 1.0 / ring_analysis.algebraic_connectivity
    print(f"Ring convergence bound: {convergence_bound:.4f}")
```

The convergence bound $1 / \lambda_2$ gives an upper estimate on how many rounds of message passing are needed for all agents to reach agreement. A lower bound means faster consensus:

- Full mesh converges fastest because every agent receives updates directly.
- Ring converges slowest because updates must propagate hop-by-hop around the cycle.

## 5. Consensus Convergence Time

The `analyze_consensus_time()` function provides a direct estimate.

```python
mesh_time = analyze_consensus_time(graph)
ring_time = analyze_consensus_time(ring)
print(f"Consensus time (full mesh): {mesh_time:.4f}")
print(f"Consensus time (ring): {ring_time:.4f}")
```

This function computes a convergence time estimate based on the spectral properties. The ratio `ring_time / mesh_time` quantifies the cost of using a sparser topology. For production systems, this tradeoff matters: full mesh has $O(n^2)$ communication overhead but $O(1)$ convergence, while ring has $O(n)$ communication but $O(n^2)$ convergence.

The takeaway: higher algebraic connectivity means faster consensus and a more resilient mesh. Topology selection is a tradeoff between communication cost and convergence speed.

---

??? example "Complete source"
    ```python
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
    ```
