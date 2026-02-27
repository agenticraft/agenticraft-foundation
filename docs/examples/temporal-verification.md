# Temporal Verification

**Source:** `examples/temporal_verification.py`

This example uses CTL (Computation Tree Logic) to verify temporal properties of a multi-agent task assignment system. CTL formulas express properties like "always", "eventually", and "until" over labeled transition systems.

---

## 1. Build the Agent Process

A task assignment agent that receives tasks, processes them, and either succeeds or retries on failure.

```python
from agenticraft_foundation import (
    Event, ExternalChoice, Prefix, Recursion, Stop, Variable, build_lts,
)

receive = Event("receive_task")
process = Event("process")
succeed = Event("succeed")
fail = Event("fail")
retry = Event("retry")

agent = Recursion(
    variable="X",
    body=Prefix(
        receive,
        Prefix(
            process,
            ExternalChoice(
                left=Prefix(succeed, Stop()),
                right=Prefix(fail, Prefix(retry, Variable("X"))),
            ),
        ),
    ),
)

lts = build_lts(agent, max_states=50)
```

The `Recursion` operator creates a loop: on failure, the agent retries the entire task. This means the LTS has a cycle through the fail/retry path.

## 2. Label States with Atomic Propositions

CTL model checking requires a labeling function that maps each state to a set of atomic propositions.

```python
labeling = {s: set() for s in lts.states}
labeling[lts.initial_state].add("idle")

for s in lts.states:
    outgoing = {e for e, _ in lts.successors(s)}
    if receive in outgoing:
        labeling[s].add("idle")
    if process in outgoing:
        labeling[s].add("received")
    if succeed in outgoing or fail in outgoing:
        labeling[s].add("processing")
    if retry in outgoing:
        labeling[s].add("failed")
    if not outgoing:
        labeling[s].add("done")
```

Each state gets labeled based on its outgoing events. Terminal states (no outgoing transitions) are labeled `done`.

## 3. Safety: No Error States

**AG(not error)** -- "In all reachable states, error never holds."

```python
from agenticraft_foundation.verification import check_safety

result = check_safety(lts, "error", labeling)
print(f"No error states: {result.satisfied}")  # True
```

Since we never labeled any state `error`, this trivially holds. In practice, you'd label error states and verify the system avoids them.

## 4. Liveness: Always Terminates?

**AF(done)** -- "From every state, done is eventually reached."

```python
from agenticraft_foundation.verification import check_liveness

result = check_liveness(lts, "done", labeling)
print(f"Always terminates: {result.satisfied}")  # False
```

This is **False** -- the retry loop means the agent can cycle `receive -> process -> fail -> retry` indefinitely. CTL correctly identifies that there exists an infinite path that never reaches `done`. This is a key insight: the agent *can* succeed but isn't *guaranteed* to.

## 5. Existential Reachability

**EF(done)** -- "There exists a path to a done state."

```python
from agenticraft_foundation.verification import model_check, EF, Atomic

result = model_check(lts, EF(Atomic("done")), labeling)
print(f"Can reach done: {result.satisfied}")  # True
```

While termination isn't guaranteed (AF fails), success is *possible* (EF holds). The distinction between AF and EF is central to CTL.

## 6. Mutual Exclusion

A classic verification example: two processes should never be in their critical sections simultaneously.

```python
from agenticraft_foundation.verification import AG, Not, And

# AG(not (critical_a AND critical_b))
result = model_check(
    mutex_lts,
    AG(Not(And(Atomic("critical_a"), Atomic("critical_b")))),
    mutex_labels,
)
print(f"Mutual exclusion holds: {result.satisfied}")  # True
```

The model checker exhaustively verifies that no reachable state has both agents in their critical sections.

## Key CTL Operators

| Operator | Meaning |
|----------|---------|
| `AG(f)` | f holds in **all** states on **all** paths (safety) |
| `AF(f)` | f **eventually** holds on **all** paths (liveness) |
| `EF(f)` | f is **reachable** on **some** path |
| `EG(f)` | f **persists** along **some** infinite path |
| `AU(f, g)` | f holds **until** g on **all** paths |
| `EU(f, g)` | f holds **until** g on **some** path |

## Output

```
=== Task Assignment Agent ===
States: 5
Transitions: 5

=== Safety: errors never occur ===
  No error states: True

=== Liveness: task always eventually completes ===
  Always terminates: False

=== Existential reachability: success is possible ===
  Can reach done: True

=== Mutual Exclusion Model ===
  Mutual exclusion holds: True
```
