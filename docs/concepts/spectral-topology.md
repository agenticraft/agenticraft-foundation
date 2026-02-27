# Spectral Topology

## Overview

Spectral graph theory provides a mathematical lens for analyzing the structure and dynamics of agent networks. The key insight is that the **eigenvalues of the graph Laplacian** encode fundamental properties of the network -- connectivity, robustness, and the speed at which agents can reach consensus. AgentiCraft Foundation uses spectral analysis to evaluate topology quality, predict consensus convergence times, and extend standard graph models to hypergraphs for group coordination.

## Key Definitions

### Graph Laplacian

For an undirected graph with $n$ nodes, the **graph Laplacian** is the $n \times n$ matrix:

$$L = D - A$$

where $D$ is the diagonal degree matrix ($D_{ii} = \deg(i)$) and $A$ is the adjacency matrix ($A_{ij} = 1$ if nodes $i$ and $j$ are connected).

The Laplacian is positive semidefinite, so its eigenvalues satisfy $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$.

### Algebraic Connectivity

The second-smallest eigenvalue $\lambda_2(L)$, called the **algebraic connectivity** or **Fiedler value**, characterizes the network:

- $\lambda_2 > 0$ if and only if the graph is connected
- Larger $\lambda_2$ means the network is harder to disconnect (more robust)
- $\lambda_2$ directly governs how fast distributed consensus converges

### Consensus Convergence

For a standard linear consensus protocol where agents iteratively average their values with neighbors, the convergence time is bounded by:

$$T = O\!\left(\frac{n \log n}{\lambda_2}\right)$$

This means networks with higher algebraic connectivity reach agreement faster. A full mesh ($\lambda_2 = n$) converges in $O(\log n)$ rounds, while a ring ($\lambda_2 \approx 2(1 - \cos(2\pi/n))$) may require $O(n^2 \log n)$ rounds.

## Topology Comparison

| Topology | $\lambda_2$ (typical) | Consensus Speed | Resilience | Edge Count |
|----------|----------------------|-----------------|------------|------------|
| Full mesh | $n$ | Fastest | Highest -- tolerates up to $n-2$ node failures | $O(n^2)$ |
| Ring | $\approx 2(1 - \cos(2\pi/n))$ | Slow -- $O(n^2)$ rounds | Low -- single failure disconnects | $O(n)$ |
| Star | $1$ | Medium -- hub bottleneck | Hub-dependent -- hub failure is catastrophic | $O(n)$ |
| Grid ($\sqrt{n} \times \sqrt{n}$) | $\approx 2(1 - \cos(\pi/\sqrt{n}))$ | Moderate | Moderate -- local failures contained | $O(n)$ |
| Random ($p = \log n / n$) | $\approx np - 2\sqrt{np(1-p)}$ | Fast | High -- probabilistically robust | $O(n \log n)$ |
| Expander | $\Theta(1)$ | Fast -- constant spectral gap | High -- by definition | $O(n)$ |

## Hypergraph Extension

Standard graphs model **pairwise** connections. In multi-agent systems, many coordination patterns involve **group** interactions -- a broadcast to a team, a committee vote, or a shared workspace. Hypergraphs generalize graphs by allowing edges (called **hyperedges**) to connect arbitrary subsets of nodes.

### Hypergraph Laplacian

For a hypergraph with incidence matrix $H$ (rows = nodes, columns = hyperedges), the hypergraph Laplacian is:

$$L_H = D_v - H W D_e^{-1} H^T$$

where:

- $H \in \{0, 1\}^{|V| \times |E_H|}$ is the incidence matrix ($H_{ve} = 1$ if node $v$ belongs to hyperedge $e$)
- $W \in \mathbb{R}^{|E_H| \times |E_H|}$ is a diagonal weight matrix for hyperedges
- $D_v \in \mathbb{R}^{|V| \times |V|}$ is the diagonal node degree matrix ($D_{v,ii} = \sum_e W_{ee} H_{ie}$)
- $D_e \in \mathbb{R}^{|E_H| \times |E_H|}$ is the diagonal hyperedge degree matrix ($D_{e,jj} = \sum_v H_{vj}$)

The spectral properties of $L_H$ generalize those of the standard Laplacian: $\lambda_2(L_H) > 0$ implies the hypergraph is connected, and larger values indicate tighter group coordination.

### Clique Expansion

A hypergraph can be approximated by a standard graph through **clique expansion**: each hyperedge $e = \{v_1, \ldots, v_k\}$ is replaced by a clique (complete subgraph) on the same $k$ nodes. This allows standard spectral tools to be applied, though it loses the distinction between pairwise and group interactions.

## How It Maps to Code

```python
from agenticraft_foundation.topology import (
    NetworkGraph, HypergraphNetwork,
)

# Build a standard network graph (ring topology)
graph = NetworkGraph()
for i in range(5):
    graph.add_node(str(i))
graph.add_edge("0", "1")
graph.add_edge("1", "2")
graph.add_edge("2", "3")
graph.add_edge("3", "4")
graph.add_edge("4", "0")

# Compute spectral properties via analyze()
analysis = graph.analyze()
print(f"Algebraic connectivity: {analysis.algebraic_connectivity:.4f}")
print(f"Summary: {analysis.summary}")

# Hypergraph for group coordination
hg = HypergraphNetwork()
for i in range(6):
    hg.add_node(str(i))
hg.add_hyperedge(["0", "1", "2"], weight=1.0)  # team A
hg.add_hyperedge(["3", "4", "5"], weight=1.0)  # team B
hg.add_hyperedge(["2", "3"], weight=0.5)        # bridge

# Compute hypergraph Laplacian
L_H = hg.laplacian_matrix()

# Convert to standard graph for spectral analysis
standard_graph = hg.clique_expansion()
```

### Factory Constructors

The `HypergraphNetwork` class provides factory constructors for common conversions:

- `HypergraphNetwork.from_graph(graph)` -- promotes a standard graph to a hypergraph where each edge becomes a 2-node hyperedge
- `HypergraphNetwork.clique_expansion(hypergraph)` -- converts a hypergraph to a standard graph via clique expansion

## Further Reading

- **API Reference**: [topology/laplacian](../api/topology/laplacian.md), [topology/hypergraph](../api/topology/hypergraph.md)
- **Tutorial**: [Topology Analysis for Agent Networks](../tutorials/topology-analysis.md)
