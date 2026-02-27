# Tutorials

These are hands-on tutorials that walk you through building real verification workflows with `agenticraft-foundation`. Each tutorial is self-contained and produces working code you can extend for your own multi-agent systems.

All tutorials use only `agenticraft-foundation` -- no external dependencies required.

## Prerequisites

- Python 3.10 or later
- `agenticraft-foundation` installed:

```bash
pip install agenticraft-foundation
```

## Tutorial Index

| Tutorial | What You'll Learn | Time |
|----------|-------------------|------|
| [Modeling Agent Coordination with CSP](csp-coordination.md) | Build CSP models, analyze LTS, detect deadlocks | 15 min |
| [Verifying Protocols with Session Types](mpst-verification.md) | Define global types, project to local types, verify well-formedness | 15 min |
| [Multi-Protocol Routing](protocol-routing.md) | Build protocol graphs, routing algorithms, workflow validation | 20 min |
| [Analyzing Mesh Topologies](topology-analysis.md) | Spectral analysis, topology comparison, hypergraph coordination | 15 min |
| [Formal Consensus Verification](consensus-verification.md) | Consensus properties, weighted consensus, MAS theory mappings | 15 min |
| [Checking Temporal Properties with CTL](temporal-verification.md) | CTL formulas, safety, liveness, response properties, counterexamples | 15 min |
| [Modeling Stochastic Agents with DTMC](probabilistic-analysis.md) | Markov chains, reachability probability, expected steps, steady-state | 15 min |

## Suggested Order

The tutorials are designed to be read in order, but each is self-contained. If you are new to formal methods for multi-agent systems, start with the CSP tutorial to build intuition for process-algebraic modeling, then proceed through session types and protocol routing. If you are primarily interested in topology or consensus, you can skip directly to those tutorials. The verification tutorials (CTL and DTMC) build on CSP concepts, so read the CSP tutorial first if you haven't already.

## Getting Help

If you run into issues, check the [API Reference](../api/index.md) for detailed method signatures, or the [Concepts](../concepts/index.md) section for background on the formal foundations.
