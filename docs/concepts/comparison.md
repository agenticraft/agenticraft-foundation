# Comparison with Other Tools

How agenticraft-foundation relates to other formal methods tools and agent frameworks.

## Formal Verification Tools

| Feature | agenticraft-foundation | SPIN | FDR4 | TLA+ |
|---|---|---|---|---|
| **Focus** | Multi-agent AI coordination | General concurrent systems | CSP refinement checking | Distributed algorithms |
| **Process algebra** | CSP (13 operators) | Promela (custom language) | CSP (full language) | -- |
| **Session types** | MPST (global + local) | -- | -- | -- |
| **Temporal logic** | CTL model checking | LTL model checking | -- | TLC model checking |
| **Probabilistic** | DTMC (reachability, steady-state) | -- | -- | -- |
| **Language** | Python (native API) | Promela → C verifier | CSP_M → Haskell verifier | TLA+ → Java verifier |
| **Agent-specific operators** | Timeout, Interrupt, Guard, Pipe, Rename | -- | -- | -- |
| **Topology analysis** | Spectral (Laplacian, $\lambda_2$) | -- | -- | -- |
| **State space scale** | ~10K states (agent-scale) | Millions (BDD compression) | Millions (BDD compression) | Millions (TLC) |
| **Integration** | Direct Python, pip install | External tool, separate workflow | External tool, separate workflow | External tool, separate workflow |

### When to use which

- **agenticraft-foundation**: You're building a multi-agent system in Python and want to verify coordination properties (deadlock freedom, protocol conformance, topology resilience) directly in your codebase. State spaces up to ~10K.

- **SPIN**: You need to verify concurrent protocols with millions of states. You're willing to write specifications in Promela and run an external verifier.

- **FDR4**: You need industrial-strength CSP refinement checking. You're working with the full CSP language and need BDD-based state compression for large models.

- **TLA+**: You're designing distributed algorithms and need to specify and verify high-level protocols. You're comfortable with mathematical notation and the TLC model checker.

## Agent Frameworks

| Feature | agenticraft-foundation | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| **Focus** | Formal verification of agent coordination | Agent workflow orchestration | Role-based agent teams | Multi-agent conversation |
| **Deadlock detection** | LTS-based, exhaustive | -- | -- | -- |
| **Protocol verification** | MPST well-formedness, projection | -- | -- | -- |
| **Temporal properties** | CTL model checking | -- | -- | -- |
| **Probabilistic analysis** | DTMC reachability | -- | -- | -- |
| **Topology analysis** | Spectral graph theory | -- | -- | -- |
| **LLM integration** | -- (verification layer only) | LangChain ecosystem | LLM-powered agents | LLM-powered agents |
| **Runtime execution** | -- (design-time verification) | Workflow execution | Agent execution | Conversation execution |

### Complementary, not competing

agenticraft-foundation is a **verification library**, not an agent runtime. It sits alongside agent frameworks:

1. **Design** your multi-agent system using LangGraph, CrewAI, or any framework
2. **Model** the coordination logic as CSP processes
3. **Verify** deadlock freedom, protocol correctness, and temporal properties
4. **Analyze** the agent topology for resilience
5. **Deploy** with confidence that coordination bugs were caught at design time

The library does not execute agents, call LLMs, or manage agent state -- it provides mathematical guarantees about the coordination structure.
