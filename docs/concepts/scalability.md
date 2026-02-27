# Scalability & Limits

This page documents the practical scalability of each module and what to expect as system size grows.

## LTS State Space

Building an LTS (Labeled Transition System) from a CSP process explores all reachable states. The state space grows exponentially with the number of concurrent processes:

| Configuration | States (worst case) | Practical limit |
|---|---|---|
| Sequential pipeline of $n$ stages | $O(n)$ | Thousands of stages |
| $n$ processes, each with $k$ states, in parallel | $O(k^n)$ | 10--20 concurrent processes |
| $n$ processes with hiding | $O(k^n)$ (may reduce) | Similar to parallel |

**`build_lts()` default limit**: 10,000 states (`max_states` parameter). If the LTS exceeds this, `build_lts` raises an error. This is a safety guard, not a hard algorithmic limit.

**Mitigation**: Use **hierarchical composition** -- verify subsystems independently, then compose verified components. For example, verify a 3-agent team, then treat the verified team as a single unit in a larger system.

## Topology Analysis

Spectral analysis via `NetworkGraph.analyze()` computes the graph Laplacian eigenvalues using power iteration.

| Agent count ($n$) | Time | Memory |
|---|---|---|
| 10 | < 1 ms | Negligible |
| 50 | ~5 ms | Negligible |
| 100 | ~50 ms | ~80 KB |
| 500 | ~2 s | ~2 MB |
| 1,000 | ~15 s | ~8 MB |

These are approximate figures from the `tests/benchmarks/test_topology_benchmarks.py` suite. Actual performance depends on graph density -- sparse graphs are faster.

**Hypergraph analysis** (`HypergraphNetwork.analyze()`) has similar scaling, with additional overhead proportional to hyperedge size.

## Probabilistic Verification (DTMC)

DTMC reachability uses Gaussian elimination on the transition probability matrix:

| States | Time complexity | Practical limit |
|---|---|---|
| $n$ | $O(n^3)$ | ~1,000 states |
| Steady-state (power iteration) | $O(n^2 \cdot k)$ for $k$ iterations | ~5,000 states |

For larger systems, consider decomposing the DTMC into strongly connected components and solving each independently.

## CTL Model Checking

The CTL model checker runs in $O(|S| \cdot |\to| \cdot |\phi|)$ where $|S|$ is the number of states, $|\to|$ is the number of transitions, and $|\phi|$ is the formula size. This is polynomial in the model size but bounded by the LTS state space (see above).

## Protocol Routing

Routing algorithms operate on the protocol graph, not the LTS:

| Algorithm | Complexity | Practical scale |
|---|---|---|
| `ProtocolAwareDijkstra` | $O((V+E) \log V)$ | Thousands of agents |
| `ProtocolConstrainedBFS` | $O(V + E)$ | Thousands of agents |
| `ResilientRouter` | $O((V+E) \log V)$ per failover | Thousands of agents |
| `SemanticRouter` | $O(V \cdot C)$ where $C$ = capabilities | Hundreds of agents |

Protocol routing scales well because the protocol graph is typically much smaller than the LTS state space.

## What This Library Does NOT Replace

agenticraft-foundation is designed for **agent-scale** formal verification -- systems with 2--50 concurrent agents where correctness guarantees matter more than raw state space coverage.

For **million-state** model checking (hardware verification, full protocol exhaustion), use dedicated tools:

- **SPIN** / **FDR4** -- Industrial model checkers with BDD-based state compression
- **TLA+** / **PlusCal** -- Specification languages with TLC model checker
- **PRISM** -- Probabilistic model checker with symbolic engines

These tools trade domain-specific agent abstractions for raw scale. agenticraft-foundation trades scale for agent-specific operators (Timeout, Interrupt, Guard) and direct Python integration.

## Guidelines

1. **Start small**: Verify critical subsystems (2--5 agents) before scaling up.
2. **Compose hierarchically**: Verify components independently, then compose.
3. **Use the right tool**: LTS for agent behavior, protocol graph for routing, DTMC for stochastic analysis.
4. **Monitor state space**: If `build_lts` hits the 10,000-state limit, decompose the system.
5. **Benchmark your topology**: Run `tests/benchmarks/test_topology_benchmarks.py` with `--benchmark` to get numbers for your specific hardware.
