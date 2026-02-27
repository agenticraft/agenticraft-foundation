# AgentiCraft Foundation

![CI](https://img.shields.io/github/actions/workflow/status/agenticraft/agenticraft-foundation/ci.yml?style=flat-square&label=CI)
![Coverage](https://img.shields.io/codecov/c/github/agenticraft/agenticraft-foundation/main?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)

**Formally verified mathematical foundations for multi-agent AI coordination.** AgentiCraft Foundation provides 8 modules spanning CSP process algebra, multiparty session types, protocol-aware routing, spectral topology analysis, and formal specification -- all with zero runtime dependencies. The library delivers the mathematical guarantees that production multi-agent systems need: deadlock freedom, trace refinement, protocol well-formedness, and topology resilience bounds.

---

## Architecture

```mermaid
graph TB
    subgraph algebra["algebra"]
        csp["CSP Operators<br/><i>13 process primitives</i>"]
        sem["Semantics<br/><i>LTS, traces, deadlock</i>"]
        eq["Equivalence<br/><i>bisimulation, failures</i>"]
        ref["Refinement<br/><i>trace, failures, FD</i>"]
        pat["Patterns<br/><i>coordination templates</i>"]
        csp --> sem
        sem --> eq
        sem --> ref
        csp --> pat
    end

    subgraph mpst["mpst"]
        gt["Global Types<br/><i>protocol specification</i>"]
        lt["Local Types<br/><i>projected per-role</i>"]
        proj["Projector<br/><i>global to local</i>"]
        mon["Session Monitor<br/><i>runtime checking</i>"]
        gt --> proj --> lt --> mon
    end

    subgraph topology["topology"]
        lap["Laplacian Analysis<br/><i>spectral decomposition</i>"]
        conn["Connectivity<br/><i>vertex/edge, bridges</i>"]
        hyper["Hypergraph<br/><i>group coordination</i>"]
        lap --- conn
        lap --- hyper
    end

    subgraph protocols["protocols"]
        pg["Protocol Graph<br/><i>G = (V, E, P, Φ, Γ)</i>"]
        route["Routing<br/><i>Dijkstra, BFS, resilient</i>"]
        semr["Semantic Routing<br/><i>capability embeddings</i>"]
        compat_node["Compatibility<br/><i>translation costs</i>"]
        wf["Workflows<br/><i>W = (T, ≺, ρ)</i>"]
        tx["Transformers<br/><i>composable T: M → M'</i>"]
        pg --> route
        pg --> semr
        compat_node --> route
        compat_node --> tx
        pg --> wf
    end

    subgraph verification["verification"]
        inv["Invariant Checker<br/><i>runtime assertions</i>"]
        ctl["CTL Model Checker<br/><i>AG, AF, EF, AU</i>"]
        dtmc["Probabilistic (DTMC)<br/><i>reachability, steady-state</i>"]
        cex["Counterexamples<br/><i>structured explanations</i>"]
        inv --- ctl
        ctl --- dtmc
        ctl --- cex
    end

    subgraph specs["specifications"]
        formal["Consensus Properties<br/><i>agreement, validity</i>"]
        wcon["Weighted Consensus<br/><i>quality-weighted quorum</i>"]
        mas["MAS Mappings<br/><i>BDI, Contract Net</i>"]
        formal --- wcon
        formal --- mas
    end

    subgraph complexity_mod["complexity"]
        bounds["Complexity Bounds<br/><i>30+ theoretical limits</i>"]
        faults["Fault Models<br/><i>classical + LLM-specific</i>"]
        bounds --- faults
    end

    pat -.->|"coordination<br/>patterns"| gt
    ref -.->|"property<br/>checking"| formal
    lap -.->|"topology<br/>metrics"| pg
    hyper -.->|"group<br/>structure"| wf

    style algebra fill:#0d94881a,stroke:#0D9488
    style mpst fill:#0d94881a,stroke:#0D9488
    style topology fill:#0d94881a,stroke:#0D9488
    style protocols fill:#0d94881a,stroke:#0D9488
    style verification fill:#0d94881a,stroke:#0D9488
    style specs fill:#0d94881a,stroke:#0D9488
    style complexity_mod fill:#0d94881a,stroke:#0D9488
```

---

## Modules at a Glance

| Module | Description | Tests |
|--------|-------------|-------|
| **algebra** | CSP process algebra -- 13 operators, LTS semantics, trace/failures refinement, deadlock detection | 219 |
| **mpst** | Multiparty session types -- global/local types, projection, session monitoring, well-formedness checking | 270 |
| **protocols** | Protocol-aware routing -- protocol graphs, Dijkstra/BFS/resilient routing, compatibility matrices, workflows | 259 |
| **topology** | Spectral topology analysis -- Laplacian decomposition, algebraic connectivity, hypergraph group coordination | 57 |
| **specifications** | Formal specifications -- consensus properties, weighted quorum, BDI and Contract Net mappings | 65 |
| **complexity** | Complexity theory -- 30+ theoretical bounds, classical and LLM-specific fault models | 44 |
| **verification** | Verification -- CTL temporal logic model checking, DTMC probabilistic analysis, invariant checking, counterexample generation | 199 |
| **integration** | Integration adapters -- MPST bridge for protocol session types, CSP orchestration for workflow verification | 52 |

**Total: 8 modules, 1,165 tests, ~20K LOC, zero dependencies.**

---

## Quick Links

- [Getting Started: Installation](getting-started/installation.md) -- install the package and set up a development environment
- [Getting Started: Quick Start](getting-started/quickstart.md) -- four self-contained examples covering the core modules
- [Getting Started: Why Formal Methods?](getting-started/why-formal-methods.md) -- when and why to use formal verification for multi-agent systems
- [API Reference](api/index.md) -- complete module and class reference
- [Concepts](concepts/index.md) -- detailed explanations of the mathematical foundations
- [Tutorials](tutorials/index.md) -- step-by-step walkthroughs for common tasks
