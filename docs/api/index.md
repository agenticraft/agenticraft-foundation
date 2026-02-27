# API Reference

Auto-generated reference documentation for all `agenticraft-foundation` modules.

## Package Structure

| Module | Description | Source Files |
|--------|-------------|--------------|
| [`types`](types.md) | Shared type definitions (`ProtocolName` enum) | 1 |
| [`algebra`](algebra/index.md) | CSP process algebra — 13 operators, LTS semantics, equivalence, refinement | 8 |
| [`mpst`](mpst/index.md) | Multiparty Session Types — global/local types, projection, monitoring | 7 |
| [`protocols`](protocols/index.md) | Protocol graph model — routing, workflows, transformers, compatibility | 10 |
| [`topology`](topology/index.md) | Spectral graph analysis — Laplacian, connectivity, hypergraph | 4 |
| [`specifications`](specifications/index.md) | Consensus properties, weighted consensus, MAS theory mappings | 3 |
| [`complexity`](complexity/index.md) | Complexity bounds, fault models, annotations | 2 |
| [`verification`](verification/index.md) | CTL temporal logic, DTMC probabilistic analysis, invariant checking, counterexamples | 4 |
| [`integration`](integration/index.md) | Protocol session type and workflow verification adapters | 2 |

## Top-Level Exports

The package re-exports the most commonly used symbols from `agenticraft_foundation`:

```python
from agenticraft_foundation import (
    # CSP Core
    Event, Process, ProcessKind,
    Stop, Skip, Prefix, ExternalChoice, InternalChoice,
    Parallel, Sequential, Hiding,
    Recursion, Variable, substitute,
    # Agent Extensions
    Interrupt, Timeout, Guard, Rename, Pipe, TIMEOUT_EVENT,
    # Semantics
    traces, build_lts, detect_deadlock, is_deadlock_free,
    # Topology
    LaplacianAnalysis,
    # Session Types
    SessionMonitor,
)
```

For full subpackage APIs, import from the specific module:

```python
from agenticraft_foundation.algebra import trace_refines, failures_refines
from agenticraft_foundation.mpst import Projector, SessionTypeChecker
from agenticraft_foundation.protocols import ProtocolGraph, ProtocolAwareDijkstra
from agenticraft_foundation.topology import NetworkGraph, HypergraphNetwork
```
