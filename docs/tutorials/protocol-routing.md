# Multi-Protocol Routing

**Time:** 20 minutes

In this tutorial, you will build a protocol graph representing agents with different protocol capabilities, use multiple routing algorithms to find optimal paths, handle agent failures with resilient routing, and validate a multi-step workflow against the graph.

## Prerequisites

- Python 3.10+
- `agenticraft-foundation` installed
- Familiarity with graph concepts (nodes, edges, paths)

## What You'll Build

A protocol-aware routing system with 4 agents that support different combinations of MCP and A2A protocols. You will:

1. Build a protocol graph with agents and edges
2. Set up protocol compatibility and cost calculation
3. Find optimal routes using Dijkstra (minimum cost)
4. Find shortest routes using BFS (minimum hops)
5. Route around failed agents with resilient routing
6. Define and validate a multi-step workflow

## Step 1: Build the Protocol Graph

A protocol graph models your agent network. Each agent has capabilities (what it can do) and supported protocols (how it communicates). Edges represent direct communication links between agents, annotated with the protocols those links support.

```python
from agenticraft_foundation.protocols import ProtocolGraph
from agenticraft_foundation.types import ProtocolName

graph = ProtocolGraph()
graph.add_agent("gateway", ["routing"], {ProtocolName.MCP, ProtocolName.A2A})
graph.add_agent("analyzer", ["analysis"], {ProtocolName.MCP})
graph.add_agent("executor", ["execution"], {ProtocolName.A2A})
graph.add_agent("reporter", ["output"], {ProtocolName.MCP, ProtocolName.A2A})

graph.add_edge("gateway", "analyzer", {ProtocolName.MCP})
graph.add_edge("gateway", "executor", {ProtocolName.A2A})
graph.add_edge("analyzer", "reporter", {ProtocolName.MCP})
graph.add_edge("executor", "reporter", {ProtocolName.A2A})
```

This creates a diamond-shaped graph. The gateway can reach the reporter via two paths: through the analyzer (using MCP) or through the executor (using A2A). The analyzer only speaks MCP, while the executor only speaks A2A. The gateway and reporter are bilingual.

This topology is common in production systems where different subsystems use different protocols -- for example, an analysis pipeline using structured MCP tool calls and an execution pipeline using agent-to-agent messaging.

## Step 2: Set Up Routing Infrastructure

Before routing, you need two components: a compatibility matrix that defines the cost of protocol transitions, and a path cost calculator that evaluates routes.

```python
from agenticraft_foundation.protocols import (
    ProtocolCompatibilityMatrix, PathCostCalculator,
    PROTOCOL_COMPATIBILITY,
)

compat = ProtocolCompatibilityMatrix(PROTOCOL_COMPATIBILITY)
calc = PathCostCalculator(graph, compat)
```

`PROTOCOL_COMPATIBILITY` is a built-in dictionary defining the base compatibility between protocol pairs. Same-protocol communication has zero overhead. Cross-protocol communication (e.g., translating from MCP to A2A) incurs a cost that reflects the complexity and potential information loss of the translation.

The `PathCostCalculator` uses the compatibility matrix and graph structure to compute the total cost of any path. The cost function is:

$$\text{cost}(\pi, \sigma) = \sum_{i} w_{e_i}(p_i) + \sum_{i} \tau(p_i, p_{i+1}, v_i)$$

The first term sums edge weights $w_{e_i}(p_i)$ — the hop cost of each link using its assigned protocol. The second term sums translation costs $\tau(p_i, p_{i+1}, v_i)$ — the overhead of switching protocols at intermediate nodes. Same-protocol hops contribute zero translation cost.

## Step 3: Dijkstra Routing (Minimum Cost)

Dijkstra's algorithm finds the path with the lowest total cost. In a protocol-aware context, this means finding the route that minimizes both the number of hops and the cost of any protocol translations required.

```python
from agenticraft_foundation.protocols import ProtocolAwareDijkstra

dijkstra = ProtocolAwareDijkstra(graph, compat, calc)
route = dijkstra.find_optimal_route("gateway", "reporter", ProtocolName.MCP)
print(f"Path: {route.path}")
print(f"Total cost: {route.total_cost:.2f}")
print(f"Protocols used: {route.protocols_used}")
```

When requesting an MCP route from gateway to reporter, Dijkstra will prefer the path through the analyzer (gateway -> analyzer -> reporter) because it stays on MCP the entire way, avoiding any protocol translation cost.

If you change the preferred protocol to A2A, the algorithm will route through the executor instead.

## Step 4: BFS Routing (Minimum Hops)

Sometimes you want the shortest path regardless of cost -- for example, when latency matters more than protocol overhead. BFS finds the path with the fewest hops.

```python
from agenticraft_foundation.protocols import ProtocolConstrainedBFS

bfs = ProtocolConstrainedBFS(graph, compat, calc)
shortest = bfs.find_shortest_path("gateway", "reporter", ProtocolName.MCP)
print(f"Path: {shortest.path}")
print(f"Hops: {shortest.num_hops}")
```

In this graph, both paths have the same number of hops (2), so BFS may return either one. The difference between BFS and Dijkstra becomes significant in larger graphs where some short paths have high protocol translation costs.

## Step 5: Resilient Routing with Failures

Production systems experience failures. An agent might crash, or a protocol endpoint might become unavailable. Resilient routing finds alternative paths that avoid failed components.

```python
from agenticraft_foundation.protocols import ResilientRouter

router = ResilientRouter(graph, compat, calc)
resilient = router.find_resilient_route(
    "gateway", "reporter", ProtocolName.A2A,
    failed_protocols=set(), failed_agents={"analyzer"},
)
print(f"Failover path: {resilient.path}")
```

With the analyzer down, the only path from gateway to reporter is through the executor. The resilient router finds this path automatically. You can also specify failed protocols -- for example, if the MCP endpoint is down across the entire network, the router will only use A2A links.

This is particularly useful in self-healing mesh architectures where the routing table must be recomputed dynamically as agents come and go.

## Step 6: Workflow Validation

A workflow defines a sequence of tasks with dependencies, where each task requires specific capabilities and supports specific protocols. Workflow validation checks whether the agent graph can actually execute the workflow.

```python
from agenticraft_foundation.protocols import (
    ProtocolWorkflow, WorkflowTask, WorkflowValidator,
    OptimalProtocolAssigner,
)

workflow = ProtocolWorkflow(workflow_id="data_pipeline")
workflow.add_task(WorkflowTask(
    task_id="extract", capabilities=["data_access"],
    compatible_protocols={ProtocolName.MCP, ProtocolName.A2A},
))
workflow.add_task(WorkflowTask(
    task_id="transform", capabilities=["processing"],
    compatible_protocols={ProtocolName.MCP},
))
workflow.add_task(WorkflowTask(
    task_id="load", capabilities=["storage"],
    compatible_protocols={ProtocolName.MCP, ProtocolName.A2A},
))
workflow.add_edge("extract", "transform")
workflow.add_edge("transform", "load")

validator = WorkflowValidator()
result = validator.validate(workflow, graph, compat)
print(f"Valid: {result.is_valid}")

assigner = OptimalProtocolAssigner()
assignment = assigner.assign(workflow, graph, compat, calc)
print(f"Protocol assignment: {assignment}")
```

The `WorkflowValidator` checks that every task in the workflow can be assigned to at least one agent in the graph with the required capabilities and a compatible protocol. It also checks that the data flow between tasks has a valid protocol path.

The `OptimalProtocolAssigner` goes further: it finds the assignment of protocols to tasks that minimizes the total cost across the entire workflow. Since the `transform` task only supports MCP, the assigner will choose MCP for the whole pipeline to avoid protocol transitions.

## Complete Script

```python
"""Protocol Routing Tutorial - Complete Script

Builds a protocol graph, routes with Dijkstra and BFS,
handles failures with resilient routing, and validates workflows.
"""
from agenticraft_foundation.protocols import (
    ProtocolGraph,
    ProtocolCompatibilityMatrix, PathCostCalculator,
    PROTOCOL_COMPATIBILITY,
    ProtocolAwareDijkstra, ProtocolConstrainedBFS,
    ResilientRouter,
    ProtocolWorkflow, WorkflowTask, WorkflowValidator,
    OptimalProtocolAssigner,
)
from agenticraft_foundation.types import ProtocolName

# Step 1: Build protocol graph
graph = ProtocolGraph()
graph.add_agent("gateway", ["routing"], {ProtocolName.MCP, ProtocolName.A2A})
graph.add_agent("analyzer", ["analysis"], {ProtocolName.MCP})
graph.add_agent("executor", ["execution"], {ProtocolName.A2A})
graph.add_agent("reporter", ["output"], {ProtocolName.MCP, ProtocolName.A2A})

graph.add_edge("gateway", "analyzer", {ProtocolName.MCP})
graph.add_edge("gateway", "executor", {ProtocolName.A2A})
graph.add_edge("analyzer", "reporter", {ProtocolName.MCP})
graph.add_edge("executor", "reporter", {ProtocolName.A2A})

# Step 2: Set up routing infrastructure
compat = ProtocolCompatibilityMatrix(PROTOCOL_COMPATIBILITY)
calc = PathCostCalculator(graph, compat)

# Step 3: Dijkstra routing
dijkstra = ProtocolAwareDijkstra(graph, compat, calc)
route = dijkstra.find_optimal_route("gateway", "reporter", ProtocolName.MCP)
print(f"Path: {route.path}")
print(f"Total cost: {route.total_cost:.2f}")
print(f"Protocols used: {route.protocols_used}")

# Step 4: BFS routing
bfs = ProtocolConstrainedBFS(graph, compat, calc)
shortest = bfs.find_shortest_path("gateway", "reporter", ProtocolName.MCP)
print(f"Path: {shortest.path}")
print(f"Hops: {shortest.num_hops}")

# Step 5: Resilient routing
router = ResilientRouter(graph, compat, calc)
resilient = router.find_resilient_route(
    "gateway", "reporter", ProtocolName.A2A,
    failed_protocols=set(), failed_agents={"analyzer"},
)
print(f"Failover path: {resilient.path}")

# Step 6: Workflow validation
workflow = ProtocolWorkflow(workflow_id="data_pipeline")
workflow.add_task(WorkflowTask(
    task_id="extract", capabilities=["data_access"],
    compatible_protocols={ProtocolName.MCP, ProtocolName.A2A},
))
workflow.add_task(WorkflowTask(
    task_id="transform", capabilities=["processing"],
    compatible_protocols={ProtocolName.MCP},
))
workflow.add_task(WorkflowTask(
    task_id="load", capabilities=["storage"],
    compatible_protocols={ProtocolName.MCP, ProtocolName.A2A},
))
workflow.add_edge("extract", "transform")
workflow.add_edge("transform", "load")

validator = WorkflowValidator()
result = validator.validate(workflow, graph, compat)
print(f"Valid: {result.is_valid}")

assigner = OptimalProtocolAssigner()
assignment = assigner.assign(workflow, graph, compat, calc)
print(f"Protocol assignment: {assignment}")
```

## Next Steps

- Read [Protocol Concepts](../concepts/protocol-graph.md) for the theory behind protocol-aware routing
- Explore the [Protocols API Reference](../api/protocols/index.md) for advanced graph operations and custom cost functions
- Continue to [Analyzing Mesh Topologies](topology-analysis.md) to learn about spectral analysis of agent networks
