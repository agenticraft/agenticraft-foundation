# algebra

CSP process algebra for modeling multi-agent coordination.

This module provides 13 CSP operators (8 core primitives + 5 agent-specific extensions), operational semantics (LTS construction, trace extraction, deadlock detection), process equivalence checking, refinement verification, and 6 coordination patterns.

::: agenticraft_foundation.algebra
    options:
      show_root_heading: false
      members: false

## Submodules

| Module | Description |
|--------|-------------|
| [`csp`](csp.md) | 13 process classes — Stop, Skip, Prefix, ExternalChoice, InternalChoice, Parallel, Sequential, Hiding, Interrupt, Timeout, Guard, Rename, Pipe, Recursion, Variable |
| [`semantics`](semantics.md) | LTS construction, trace extraction, deadlock detection |
| [`equivalence`](equivalence.md) | Trace equivalence, bisimulation, failures equivalence |
| [`refinement`](refinement.md) | Trace, failures, and failures-divergence refinement |
| [`patterns`](patterns.md) | 6 coordination patterns |
