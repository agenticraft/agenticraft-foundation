# Concepts Overview

AgentiCraft Foundation provides a **formal methods substrate** for building verifiable multi-agent systems. Rather than treating agents as opaque black boxes, the foundation layer models agent interactions using established formalisms from concurrency theory, session types, spectral graph theory, and distributed systems. Every coordination pattern, routing decision, and consensus protocol maps to a mathematically grounded primitive with provable properties.

This section introduces the core theoretical concepts that underpin the library. Each page defines the key abstractions, shows how they map to code, and links to the corresponding API reference and tutorials.

## Concept Areas

| Concept | Description | Page |
|---------|-------------|------|
| **Process Algebra (CSP)** | Hoare's Communicating Sequential Processes with 13 operators for modeling concurrent agent coordination | [process-algebra.md](process-algebra.md) |
| **Multiparty Session Types** | Honda-Yoshida-Carbone formalism for specifying and verifying multi-party communication protocols | [session-types.md](session-types.md) |
| **Protocol Graph Model** | Multi-protocol mesh modeled as a weighted graph with protocol-aware routing and transformation | [protocol-graph.md](protocol-graph.md) |
| **Spectral Topology** | Spectral graph theory applied to agent network analysis, consensus convergence, and hypergraph extensions | [spectral-topology.md](spectral-topology.md) |
| **Consensus & MAS Mappings** | Formal consensus properties and bidirectional mappings to classical multi-agent systems theory | [consensus.md](consensus.md) |
| **Complexity & Fault Models** | Catalogued complexity bounds for distributed algorithms and fault models including LLM-specific failure modes | [complexity.md](complexity.md) |
| **Verification** | Invariant checking, counterexample generation, CTL temporal logic, and probabilistic (DTMC) verification | [verification.md](verification.md) |

## How to Read These Pages

Each concept page follows a consistent structure:

1. **Overview** -- what the formalism is and why it matters for multi-agent systems
2. **Key Definitions** -- precise mathematical definitions with notation
3. **How It Maps to Code** -- import paths and usage examples
4. **Diagrams** -- Mermaid visualizations of key workflows and hierarchies
5. **Further Reading** -- links to API reference, tutorials, and foundational papers
