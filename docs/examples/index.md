# Examples

Annotated walkthroughs of the example scripts in the `examples/` directory. Each example demonstrates a key module with detailed explanations.

| Example | Module | Description |
|---------|--------|-------------|
| [CSP Composition](csp-composition.md) | algebra | All 13 CSP operators, LTS analysis, deadlock detection |
| [Interrupt & Timeout](interrupt-timeout.md) | algebra | 5 agent-specific extensions with practical scenarios |
| [Consensus Specification](consensus-spec.md) | algebra + specifications | 3-agent consensus modeled in CSP |
| [Mesh Topology](mesh-topology.md) | topology | Spectral analysis, topology comparison |
| [Protocol Verification](protocol-verification.md) | mpst | MPST global types, projection, well-formedness |
| [Temporal Verification](temporal-verification.md) | verification | CTL model checking: safety, liveness, mutual exclusion |
| [Probabilistic Verification](probabilistic-verification.md) | verification | DTMC reachability, steady-state, expected steps |
| [Counterexample Generation](counterexample-generation.md) | verification | Structured failure explanations for refinement/equivalence |

All examples are runnable:

```bash
python examples/<name>.py
```
