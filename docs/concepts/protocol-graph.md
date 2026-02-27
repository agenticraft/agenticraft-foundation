# Protocol Graph Model

## Overview

The protocol graph models a multi-protocol agent mesh as a weighted, labeled graph. Each agent is a node with capabilities, each connection is an edge annotated with supported protocols, and routing algorithms navigate the graph while respecting protocol constraints and minimizing translation costs. The formal model is:

$$G = (V, E, P, \Phi, \Gamma)$$

This abstraction enables protocol-aware routing, automatic protocol translation, and workflow optimization across heterogeneous agent networks where different agents speak different protocols.

## Key Definitions

- $V$: **Agent nodes**. Each node $v \in V$ has a set of capabilities $\text{caps}(v)$ and a set of supported protocols $\text{protos}(v) \subseteq P$.

- $E$: **Edges**. Each edge $(u, v) \in E$ represents a communication link between agents $u$ and $v$, with an associated weight $w(u, v) \in \mathbb{R}^+$ representing latency or cost.

- $P$: **Protocol universe**. The set of all protocols in the mesh, e.g., $P = \{\text{MCP}, \text{A2A}, \text{ANP}, \ldots\}$.

- $\Phi$: **Protocol assignment**. A function $\Phi : E \to 2^P$ mapping each edge to the set of protocols available on that link.

- $\Gamma$: **Protocol affinity**. A function $\Gamma : V \times P \to [0, 1]$ scoring how well each agent supports each protocol. An affinity of $1.0$ means native support; lower values indicate degraded or translated support.

## Node Types

| Type | Description | Typical Protocols |
|------|-------------|-------------------|
| `LLM_AGENT` | Language model agent with reasoning capabilities | MCP, A2A |
| `TOOL_SERVER` | Stateless tool execution endpoint | MCP |
| `COORDINATOR` | Orchestration node managing multi-agent workflows | A2A, ANP |
| `GATEWAY` | Protocol translation bridge between network segments | All protocols |

## Routing Algorithms

Four routing algorithms are provided, each optimizing for a different objective.

| Algorithm | Class | Strategy | Complexity |
|-----------|-------|----------|------------|
| Dijkstra | `ProtocolAwareDijkstra` | Minimum-cost path with automatic protocol translation at boundaries | $O((V + E) \log V)$ |
| BFS | `ProtocolConstrainedBFS` | Minimum-hop path restricted to edges supporting a required protocol | $O(V + E)$ |
| Resilient | `ResilientRouter` | Failover routing with degraded topology -- reroutes around failed nodes | $O((V + E) \log V)$ |
| Semantic | `SemanticRouter` | Capability-similarity matching weighted by protocol affinity $\Gamma$ | $O(V \cdot C)$ where $C$ is capability dimension |

All routers return a `Route` object containing the path, total cost, protocol transitions, and any translations required.

### Path Cost Function

The cost of a route is defined by the **path cost function** (Definition 12):

$$\text{cost}(\pi, \sigma) = \sum_{i} w_{e_i}(p_i) + \sum_{i} \tau(p_i, p_{i+1}, v_i)$$

where:

- $\pi = (v_0, v_1, \ldots, v_k)$ is the path through agents
- $\sigma = (p_0, p_1, \ldots, p_{k-1})$ is the protocol used on each edge
- $w_{e_i}(p_i)$ is the weight of edge $e_i = (v_i, v_{i+1})$ using protocol $p_i$
- $\tau(p_i, p_{i+1}, v_i)$ is the **translation cost** incurred at node $v_i$ when switching from protocol $p_i$ to $p_{i+1}$

The translation cost decomposes as $\tau = \tau_{\text{base}} + \tau_{\text{semantic}}$, where $\tau_{\text{base}}$ is the fixed overhead and $\tau_{\text{semantic}}$ penalizes information loss. When $p_i = p_{i+1}$ (no protocol change), $\tau = 0$.

## Protocol Transformers

When a message must cross a protocol boundary (e.g., from an MCP tool server to an A2A coordinator), a **protocol transformer** converts the message format.

A transformer is a partial function:

$$T : M_p \to M_{p'} \cup \{\bot\}$$

where $M_p$ is the message space of protocol $p$, $M_{p'}$ is the message space of protocol $p'$, and $\bot$ indicates a failed transformation (the message cannot be represented in the target protocol).

### Transformer Classifications

| Classification | Description | Information Loss |
|----------------|-------------|------------------|
| **Lossless** | Bijective mapping -- no information lost or gained | None |
| **Lossy** | Surjective mapping -- some metadata or structure is dropped | Bounded |
| **Destructive** | Significant structural changes -- may alter semantics | Unbounded |

### Transformer Composition

Transformers compose sequentially. A path requiring two protocol hops uses:

$$T_{p \to p''} = T_{p' \to p''} \circ T_{p \to p'}$$

The composition is valid only if neither component returns $\bot$. The total information loss is the sum of individual losses, making shorter protocol paths preferable.

## Workflow Model

A workflow over the protocol graph is defined as:

$$W = (T, \prec, \rho)$$

where:

- $T = \{t_1, t_2, \ldots, t_n\}$ is a set of tasks
- $\prec$ is a partial order (precedence relation) forming a DAG over $T$
- $\rho : T \to P$ is a protocol assignment mapping each task to a required protocol

### Workflow Analysis

- **WorkflowValidator**: Checks that every task in $W$ can be executed by at least one reachable agent supporting protocol $\rho(t)$. Reports unreachable tasks and missing capabilities.

- **OptimalProtocolAssigner**: Finds the protocol assignment $\rho^*$ that minimizes total translation cost across the workflow:

  $$\rho^* = \arg\min_\rho \sum_{(t_i, t_j) \in \prec} \text{cost}(\rho(t_i), \rho(t_j))$$

  where $\text{cost}(p, p')$ is the translation cost between protocols $p$ and $p'$ (zero if $p = p'$).

## How It Maps to Code

```python
from agenticraft_foundation.types import ProtocolName
from agenticraft_foundation.protocols import (
    ProtocolGraph, AgentNode, NodeType, ProtocolEdge,
    ProtocolAwareDijkstra, ProtocolConstrainedBFS,
    ResilientRouter, SemanticRouter,
)

# Build a protocol graph
graph = ProtocolGraph()
graph.add_agent(AgentNode(
    agent_id="llm-1", node_type=NodeType.LLM_AGENT,
    protocols={ProtocolName.MCP, ProtocolName.A2A},
))
graph.add_agent(AgentNode(
    agent_id="tool-1", node_type=NodeType.TOOL_SERVER,
    protocols={ProtocolName.MCP},
))
graph.add_agent(AgentNode(
    agent_id="coord-1", node_type=NodeType.COORDINATOR,
    protocols={ProtocolName.A2A},
))
graph.add_edge(ProtocolEdge(
    "llm-1", "tool-1", protocols={ProtocolName.MCP},
))
graph.add_edge(ProtocolEdge(
    "llm-1", "coord-1", protocols={ProtocolName.A2A},
))

# Route with protocol awareness
router = ProtocolAwareDijkstra(graph)
route = router.find_route(source="tool-1", target="coord-1")
print(f"Path: {route.path}, Cost: {route.total_cost}")
```

## Further Reading

- **API Reference**: [protocols/graph](../api/protocols/graph.md), [protocols/routing](../api/protocols/routing.md), [protocols/transforms](../api/protocols/transformers.md)
- **Tutorial**: [Multi-Protocol Routing](../tutorials/protocol-routing.md)
