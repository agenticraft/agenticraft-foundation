# protocols

Multi-protocol mesh model with formal graph representation, routing algorithms, workflow validation, and composable protocol transformers.

::: agenticraft_foundation.protocols
    options:
      show_root_heading: false
      members: false

## Submodules

| Module | Description |
|--------|-------------|
| [`graph`](graph.md) | Protocol graph $G = (V, E, P, \Phi, \Gamma)$ |
| [`routing`](routing.md) | Dijkstra, BFS, and resilient routing algorithms |
| [`semantic_routing`](semantic-routing.md) | Capability-based semantic routing |
| [`compatibility`](compatibility.md) | Protocol compatibility matrix and translation costs |
| [`cost`](cost.md) | Path cost calculation with configurable edge cost models |
| [`affinity`](affinity.md) | Protocol affinity scoring per agent-protocol pair |
| [`workflow`](workflow.md) | Workflow model $W = (T, \prec, \rho)$ with validation |
| [`transformers`](transformers.md) | Composable protocol message transformers |
| [`semantic`](semantic.md) | Semantic routing implementations |
| [`specifications`](specifications.md) | Formal protocol specifications |
