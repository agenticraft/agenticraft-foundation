# Changelog

## Unreleased

### Added

- **Counterexample generation**: Structured failure explanations for refinement and equivalence checking -- `explain_refinement_failure()`, `explain_equivalence_failure()`, `find_minimal_counterexample()` with annotated traces showing exact divergence points
- **CTL temporal logic model checking**: Formula AST (Atomic, Not, And, Or, Implies, EX, EF, EG, EU, AX, AF, AG, AU) with backward fixpoint computation -- `model_check()`, `check_safety()`, `check_liveness()`
- **Probabilistic verification (DTMC)**: Discrete-Time Markov Chain representation, reachability probability via Gaussian elimination, steady-state distribution via power iteration, expected steps analysis -- `DTMC`, `check_reachability()`, `steady_state()`, `expected_steps()`
- 3 runnable examples: `temporal_verification.py`, `probabilistic_verification.py`, `counterexample_generation.py`
- 172 new tests (37 counterexamples + 81 temporal + 54 probabilistic)

### Changed

- Test count: 1012 → 1165
- Verification module: 1 submodule (invariant checker) → 4 submodules
- Integration module: removed all proprietary imports, now fully standalone with zero external dependencies

### Fixed

- 9 mypy strict-mode errors in `invariant_checker.py` (missing return type annotations, ParamSpec incompatibility)

## 0.1.0 (2026-02-26)

Initial release.

### Added

- **13 CSP operators**: 8 core primitives (Stop, Skip, Prefix, ExternalChoice, InternalChoice, Parallel, Sequential, Hiding) + 5 agent-specific extensions (Interrupt, Timeout, Guard, Rename, Pipe)
- **Recursion support**: `Recursion`, `Variable`, `substitute()` with full support for all 13 operators
- **Operational semantics**: `build_lts()`, `traces()`, `detect_deadlock()`, `is_deadlock_free()`
- **Process equivalence**: trace equivalence, strong/weak bisimulation, failures equivalence
- **Refinement checking**: trace refinement, failures refinement, failures-divergence refinement
- **Coordination patterns**: request-response, pipeline, scatter-gather, barrier, mutex, producer-consumer
- **Multiparty Session Types (MPST)**: global types, local types, projection, well-formedness checking, session monitoring
- **Spectral topology analysis**: graph Laplacian, algebraic connectivity, consensus convergence bounds
- **Protocol analysis**: formal protocol definitions and verification
- **1012 tests** with 85%+ coverage
- Zero runtime dependencies, pure Python, Python 3.10+
