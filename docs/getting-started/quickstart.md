# Quick Start

Four self-contained examples demonstrating the core modules. Each runs independently with no external dependencies.

---

## 1. CSP Deadlock Detection

The `algebra` module implements CSP (Communicating Sequential Processes) for modeling agent coordination. This example builds an interruptible agent task, constructs its labeled transition system, and checks for deadlock freedom.

```python
from agenticraft_foundation import (
    Event, Prefix, Stop, Parallel,
    Interrupt, Timeout, Guard, TIMEOUT_EVENT,
    build_lts, traces, detect_deadlock, is_deadlock_free,
)

process_data = Event("process_data")
handle_priority = Event("handle_priority")
return_result = Event("return_result")

task = Prefix(process_data, Prefix(return_result, Stop()))
handler = Prefix(handle_priority, Stop())
agent = Interrupt(primary=task, handler=handler)

assert process_data in agent.initials()
assert handle_priority in agent.initials()

fallback = Prefix(Event("return_cached"), Stop())
bounded = Timeout(process=agent, duration=30.0, fallback=fallback)

lts = build_lts(bounded)
print(f"States: {len(lts.states)}")
print(f"Deadlock-free: {is_deadlock_free(bounded)}")
```

The `Interrupt` operator lets a priority handler pre-empt a running task -- a common pattern in multi-agent systems where urgent messages must take precedence. The `Timeout` wraps the process with a fallback path if the agent does not complete within the given duration. `build_lts` expands the process into a finite-state labeled transition system, and `is_deadlock_free` checks that no reachable state has zero outgoing transitions.

---

## 2. MPST Protocol Projection

The `mpst` module implements multiparty session types for specifying and verifying multi-agent communication protocols. This example defines a client-server request-response protocol, projects it to per-role local types, and checks well-formedness.

```python
from agenticraft_foundation.mpst import (
    GlobalInteraction, GlobalType, Projector,
    SessionTypeChecker,
)

protocol = GlobalType(interactions=[
    GlobalInteraction(sender="client", receiver="server", message_type="Request"),
    GlobalInteraction(sender="server", receiver="client", message_type="Response"),
])

projector = Projector()
local_client = projector.project(protocol, "client")
local_server = projector.project(protocol, "server")

checker = SessionTypeChecker()
result = checker.check_well_formedness(protocol)
print(f"Well-formed: {result.is_valid}")
```

A `GlobalType` specifies the entire protocol from a bird's-eye view. The `Projector` derives each participant's local view -- what they send and what they expect to receive. `SessionTypeChecker` verifies that the global protocol is well-formed: every send has a matching receive, no role is left waiting indefinitely, and the protocol terminates.

---

## 3. Protocol-Aware Routing

The `protocols` module provides protocol-aware routing across heterogeneous agent networks. This example builds a protocol graph with agents supporting different protocols and finds the optimal route using Dijkstra's algorithm with protocol translation costs.

```python
from agenticraft_foundation.protocols import (
    ProtocolGraph, ProtocolAwareDijkstra,
    PathCostCalculator, ProtocolCompatibilityMatrix,
    PROTOCOL_COMPATIBILITY,
)
from agenticraft_foundation.types import ProtocolName

graph = ProtocolGraph()
graph.add_agent("gateway", ["routing"], {ProtocolName.MCP, ProtocolName.A2A})
graph.add_agent("analyzer", ["analysis"], {ProtocolName.MCP})
graph.add_agent("reporter", ["output"], {ProtocolName.MCP, ProtocolName.A2A})
graph.add_edge("gateway", "analyzer", {ProtocolName.MCP})
graph.add_edge("analyzer", "reporter", {ProtocolName.MCP})

compat = ProtocolCompatibilityMatrix(PROTOCOL_COMPATIBILITY)
calc = PathCostCalculator(graph, compat)
dijkstra = ProtocolAwareDijkstra(graph, compat, calc)
route = dijkstra.find_optimal_route("gateway", "reporter", ProtocolName.MCP)
print(f"Path: {route.path}, Cost: {route.total_cost:.2f}")
```

The `ProtocolGraph` models agents as nodes annotated with their supported protocols and capabilities. `ProtocolCompatibilityMatrix` encodes the cost of translating between protocols (e.g., MCP to A2A). `ProtocolAwareDijkstra` finds the lowest-cost path that respects protocol constraints, factoring in both hop distance and translation overhead.

---

## 4. Topology Spectral Analysis

The `topology` module provides spectral analysis of agent network topologies. This example builds a fully connected 5-agent network and computes its algebraic connectivity -- a measure of how resilient the network is to agent failures.

```python
from agenticraft_foundation.topology import NetworkGraph, LaplacianAnalysis

graph = NetworkGraph()
for i in range(5):
    graph.add_node(f"agent_{i}")
for i in range(5):
    for j in range(i + 1, 5):
        graph.add_edge(f"agent_{i}", f"agent_{j}")

analysis = LaplacianAnalysis(graph)
print(f"Nodes: {analysis.num_nodes}")
print(f"Algebraic connectivity (lambda_2): {analysis.algebraic_connectivity:.4f}")
print(f"Consensus bound: {analysis.consensus_bound:.2f}")
```

The Laplacian matrix of a graph encodes its connectivity structure. Its second-smallest eigenvalue ($\lambda_2$, the algebraic connectivity) quantifies how well-connected the network is: higher values mean faster consensus convergence and greater resilience to node removal. The consensus bound gives the theoretical upper limit on rounds needed for distributed agreement. A fully connected graph of 5 nodes has $\lambda_2 = 5.0$ -- the maximum for that size.
