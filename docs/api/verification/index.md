# verification

Verification tools for distributed systems: runtime invariant checking, structured
counterexample generation, CTL temporal logic model checking, and probabilistic
verification (DTMC).

::: agenticraft_foundation.verification
    options:
      show_root_heading: false
      members: false

## Submodules

| Module | Description |
|--------|-------------|
| [`invariant_checker`](invariant-checker.md) | Runtime state assertions, transition monitoring, violation tracking |
| [`counterexamples`](counterexamples.md) | Structured counterexample generation for refinement/equivalence failures |
| [`temporal`](temporal.md) | CTL formula AST and backward fixpoint model checking |
| [`probabilistic`](probabilistic.md) | DTMC reachability, steady-state distribution, expected steps |
