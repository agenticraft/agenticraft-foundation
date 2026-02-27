# Changelog

All notable changes to agenticraft-foundation will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-28

Initial release of the formally verified mathematical foundations for multi-agent AI coordination.

### Added

- **13 CSP operators**: 8 core primitives (Stop, Skip, Prefix, ExternalChoice, InternalChoice, Parallel, Sequential, Hiding) + 5 agent-specific extensions (Interrupt, Timeout, Guard, Rename, Pipe)
- **Recursion support**: `Recursion`, `Variable`, `substitute()` with full support for all 13 operators
- **Operational semantics**: `build_lts()`, `traces()`, `detect_deadlock()`, `is_deadlock_free()`
- **Process equivalence**: trace equivalence, strong/weak bisimulation, failures equivalence
- **Refinement checking**: trace refinement, failures refinement, failures-divergence refinement
- **Coordination patterns**: request-response, pipeline, scatter-gather, barrier, mutex, producer-consumer
- **Multiparty Session Types (MPST)**: global types, local types, projection, well-formedness checking, session monitoring, 4 communication patterns
- **Protocol graph model**: `ProtocolGraph`, Dijkstra/BFS/resilient/semantic routing, compatibility matrix, workflow validation, composable transformers
- **Spectral topology**: Laplacian analysis, algebraic connectivity, bridge detection, hypergraph group coordination
- **Formal specifications**: consensus properties (agreement, validity, integrity, termination), weighted quorum consensus, MAS theory mappings (BDI, Joint Intentions, SharedPlans, Contract Net)
- **Complexity analysis**: 30+ bounds, 8 fault models (4 classical + 4 LLM-specific), impossibility results (FLP, Byzantine)
- **Verification**: invariant checker, CTL temporal logic model checking (`AG`, `AF`, `EF`, `EG`, `AU`, `EU`, `AX`, `EX`), probabilistic verification (DTMC reachability, steady-state, expected steps), counterexample generation
- **Integration**: MPST bridge adapter (MCP/A2A session types), CSP orchestration adapter (DAG-to-CSP)
- Structural `_state_key()` on all 15 Process subclasses for efficient LTS construction
- 9 runnable examples including end-to-end RAG pipeline verification
- Scalability & limits documentation, comparison with SPIN/FDR4/TLA+/LangGraph/CrewAI
- 1,300+ tests with 93%+ coverage, 90% minimum enforced
- Minimal dependencies (NumPy only), Python 3.10+
- Type-checked with mypy strict mode
- Apache 2.0 license

[Unreleased]: https://github.com/agenticraft/agenticraft-foundation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/agenticraft/agenticraft-foundation/releases/tag/v0.1.0
