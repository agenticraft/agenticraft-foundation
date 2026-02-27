# Changelog

All notable changes to agenticraft-foundation are documented here.

## 0.1.0 (2026-02-28)

Initial release of the formally verified mathematical foundations for multi-agent AI coordination.

### Added

#### Process Algebra (`algebra`)
- **13 CSP operators**: 8 core primitives (Stop, Skip, Prefix, ExternalChoice, InternalChoice, Parallel, Sequential, Hiding) + 5 agent-specific extensions (Interrupt, Timeout, Guard, Rename, Pipe)
- **Recursion support**: `Recursion`, `Variable`, `substitute()` with full operator compatibility
- **Operational semantics**: `build_lts()` for labeled transition systems, `traces()` for trace extraction, `detect_deadlock()` and `is_deadlock_free()` for deadlock analysis
- **Process equivalence**: trace equivalence, strong/weak bisimulation, failures equivalence
- **Refinement checking**: trace refinement, failures refinement, failures-divergence refinement
- **6 coordination patterns**: request-response, pipeline, scatter-gather, barrier, mutex, producer-consumer

#### Multiparty Session Types (`mpst`)
- **Global types**: protocol specification with multi-party interactions
- **Local types**: per-role projected views via `Projector`
- **Well-formedness checking**: `SessionTypeChecker` validates protocol structure
- **Session monitoring**: `SessionMonitor` for runtime conformance checking
- **4 communication patterns**: request-response, pipeline, scatter-gather, consensus

#### Protocol Graph Model (`protocols`)
- **Protocol graph**: `ProtocolGraph` with $G = (V, E, P, \Phi, \Gamma)$ model
- **Routing algorithms**: `ProtocolAwareDijkstra` (minimum-cost), `ProtocolConstrainedBFS` (minimum-hop), `ResilientRouter` (failover), `SemanticRouter` (capability-based)
- **Protocol compatibility**: `ProtocolCompatibilityMatrix` with translation cost modeling
- **Path cost calculation**: `PathCostCalculator` with configurable edge cost models
- **Protocol transformers**: composable $T: M_p \to M_{p'}$ with lossless/lossy/destructive classification
- **Workflow validation**: `WorkflowValidator` and `OptimalProtocolAssigner` for $W = (T, \prec, \rho)$ workflows
- **Formal specifications**: protocol property definitions

#### Spectral Topology (`topology`)
- **Laplacian analysis**: `LaplacianAnalysis` with algebraic connectivity ($\lambda_2$), consensus convergence bounds
- **Connectivity**: vertex/edge connectivity, bridge detection, component analysis
- **Hypergraph topology**: `HypergraphNetwork` with $L_H = D_v - H W D_e^{-1} H^T$, group coordination metrics, factory constructors (`from_graph`, `clique_expansion`)

#### Formal Specifications (`specifications`)
- **Consensus properties**: Agreement, Validity, Integrity, Termination
- **Weighted consensus**: `WeightedConsensusState`, `WeightedAgreement`, `WeightedQuorum` with $2W/3$ threshold
- **MAS theory mappings**: `BDIMapping`, `JointIntentionMapping`, `SharedPlanMapping`, `ContractNetMapping` with bidirectional preservation

#### Complexity Analysis (`complexity`)
- **30+ complexity bounds**: consensus, gossip, leader election, broadcast, mesh communication
- **Complexity annotations**: decorator-based tracking with `ComplexityClass`
- **8 fault models**: 4 classical (crash-stop, crash-recovery, Byzantine, omission) + 4 LLM-specific (hallucination, prompt injection, non-determinism, context overflow)
- **Impossibility results**: FLP theorem, Byzantine fault tolerance bounds

#### Verification (`verification`)
- **Invariant checker**: `InvariantChecker` for runtime state assertions, transition monitoring, violation tracking
- **CTL temporal logic**: `model_check()` with full CTL formula AST (`AG`, `AF`, `EF`, `EG`, `AU`, `EU`, `AX`, `EX`), backward fixpoint algorithms, counterexample traces
- **Probabilistic verification**: `DTMC` model with `check_reachability()`, `steady_state()`, `expected_steps()`, Gaussian elimination solver
- **Counterexample generation**: `explain_refinement_failure()`, `explain_equivalence_failure()` for structured failure diagnostics
- **Convenience checkers**: `check_safety()`, `check_liveness()`, `check_reachability()`, `check_temporal_invariant()`

#### Integration (`integration`)
- **MPST bridge**: adapter applying session types to protocol verification (MCP/A2A)
- **CSP orchestration**: adapter applying CSP processes to workflow verification (DAG-to-CSP)

#### Package
- Minimal dependencies (NumPy only), Python 3.10+
- Python 3.10, 3.11, 3.12, 3.13 support
- 1,300+ tests with 93%+ coverage
- Apache 2.0 license
- Type-checked with mypy strict mode
