# Changelog

All notable changes to agenticraft-foundation will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SPDX-License-Identifier: Apache-2.0` as the first comment line of every
  Python file under `src/`, `tests/`, `examples/` and `scripts/`, so tooling
  that reads one file -- SBOM generators, vendoring tools -- sees the license
  the packaging metadata declares. No behaviour changes.
- `scripts/check_spdx_headers.py`, a self-testing gate that fails on a Python
  file with no identifier, an identifier that is not the first comment line,
  one naming another license, or bytes it cannot read as UTF-8. It reads the
  expected identifier from the `[project]` table, so the headers and the
  declared license cannot drift apart. Given no paths it walks the tree,
  pruning a directory on what it is rather than on what it is called: a tool's
  own cache wherever it sits, a virtual environment by its `pyvenv.cfg`, a
  dependency tree by the manifest beside it, build output only at the root. A
  package or fixture named `venv`, `node_modules` or `build` is judged as the
  source it is, and a run that judged no file reports that it took no
  measurement rather than certifying a tree it never read.
  Runs in the `lint` job of CI, over the staged files at commit time from
  `.pre-commit-config.yaml`, and stands in the pre-PR checklist. The `typecheck`
  job and the lint job now cover `scripts/`, so the gate is held to the same
  strict rules as the library.

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
