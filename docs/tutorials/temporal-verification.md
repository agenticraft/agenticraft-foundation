# Checking Temporal Properties with CTL

**Time:** 15 minutes

In this tutorial, you will build an LTS from a CSP process, define temporal properties using CTL formulas, and use the model checker to verify safety, liveness, and response properties.

## Prerequisites

- Python 3.10+
- `agenticraft-foundation` installed
- Familiarity with CSP basics (see [Modeling Agent Coordination with CSP](csp-coordination.md))

## What You'll Build

A task-processing agent modeled in CSP that can succeed, fail, or retry. You will:

1. Build an LTS from a CSP process
2. Label states with atomic propositions
3. Check safety properties with AG ("always globally")
4. Check liveness properties with AF ("always eventually")
5. Check response properties with AU ("always until")
6. Interpret counterexamples when properties fail

## Step 1: Build the Agent Process

Start with a simple agent that receives a request, processes it, and either succeeds or encounters an error. On error, the agent can retry.

```python
from agenticraft_foundation import (
    Event, Prefix, ExternalChoice, Stop, Skip,
    Recursion, Variable, build_lts,
)

req = Event("request")
process = Event("process")
success = Event("success")
error = Event("error")
retry = Event("retry")

# Agent: request → process → (success | error → retry → recurse)
body = Prefix(req, Prefix(process, ExternalChoice(
    Prefix(success, Skip()),
    Prefix(error, Prefix(retry, Variable("X"))),
)))
agent = Recursion(variable="X", body=body)

lts = build_lts(agent, max_states=50)
print(f"States: {len(lts.states)}, Transitions: {len(lts.transitions)}")
```

The agent loops: on error it retries the entire sequence. On success it terminates. The `max_states` parameter bounds exploration for recursive processes.

## Step 2: Label States with Atomic Propositions

CTL formulas reference *atomic propositions* -- labels attached to states. You define a labeling function that maps each state ID to a set of proposition names.

```python
labeling = {s: set() for s in lts.states}

# Label the initial state
labeling[lts.initial_state].add("init")

# Label states based on their outgoing transitions
for s in lts.states:
    outgoing = {e for e, _ in lts.successors(s)}
    if not outgoing:
        labeling[s].add("terminal")
    if Event("success") in outgoing:
        labeling[s].add("can_succeed")
    if Event("error") in outgoing:
        labeling[s].add("can_fail")

print(f"Labeled {sum(len(v) for v in labeling.values())} propositions across {len(lts.states)} states")
```

The labeling connects the LTS structure to the properties you want to verify. You can label states however you like -- by outgoing events, by state identity, or by any computed property.

## Step 3: Check Safety -- "Errors Don't Get Stuck"

A safety property says something bad *never* happens. Use `AG(Not(...))` to express "in all reachable states, the bad thing doesn't hold."

```python
from agenticraft_foundation.verification import (
    model_check, AG, AF, Not, Atomic, check_safety,
)

# Safety: terminal states are never reached from error
# (i.e., errors always lead to retry, not deadlock)
result = model_check(lts, AG(Not(Atomic("terminal"))), labeling)
print(f"No terminal states: {result.satisfied}")
print(f"States satisfying AG(¬terminal): {len(result.satisfying_states)}")
```

If the property fails, `result.counterexample` contains a trace leading to a violating state. In this agent, terminal states exist (after success), so this property will fail -- which is correct. Let's ask a better question:

```python
# Convenience function: check_safety checks AG(¬prop)
result = check_safety(lts, "can_fail", labeling)
print(f"Never in can_fail state: {result.satisfied}")
# This will be False -- the agent CAN reach error states, by design
```

## Step 4: Check Liveness -- "Success Is Always Reachable"

A liveness property says something good *eventually* happens. Use `AF(...)` to express "on all paths, the good thing eventually holds."

```python
from agenticraft_foundation.verification import check_liveness

# Liveness: from the initial state, can the agent always eventually succeed?
result = check_liveness(lts, "can_succeed", labeling)
print(f"Always eventually can succeed: {result.satisfied}")
```

This checks whether every execution path eventually reaches a state labeled `can_succeed`. For the retry agent, this depends on whether the recursion always makes progress toward success.

## Step 5: Check Reachability -- "Success Is Possible"

Sometimes you want a weaker property: not that something *always* happens, but that it *can* happen. Use `EF(...)` for existential reachability.

```python
from agenticraft_foundation.verification import EF

# Reachability: is there at least one path to success?
result = model_check(lts, EF(Atomic("can_succeed")), labeling)
print(f"Success reachable: {result.satisfied}")
print(f"States that can reach success: {len(result.satisfying_states)}")
```

`EF(φ)` asks: "does there exist a path from this state that eventually reaches a state where φ holds?" This is useful for checking that success is *possible*, even if not guaranteed.

## Step 6: Check Response -- "Every Request Gets a Response"

Response properties combine two formulas with `AU` (always until): "φ holds at every step until ψ becomes true, and ψ eventually does become true."

```python
from agenticraft_foundation.verification import AU, Or

# First, add more labels for this check
for s in lts.states:
    outgoing = {e for e, _ in lts.successors(s)}
    if Event("request") in outgoing:
        labeling[s].add("waiting")
    if Event("success") in outgoing or Event("error") in outgoing:
        labeling[s].add("responded")

# Response: after init, the agent is either waiting or has responded,
# until it reaches a terminal or can_succeed state
result = model_check(
    lts,
    AU(
        Or(Atomic("waiting"), Atomic("responded")),
        Atomic("can_succeed"),
    ),
    labeling,
)
print(f"Response property: {result.satisfied}")
```

## Step 7: Interpret Counterexamples

When a property fails, the model checker provides a counterexample trace -- a concrete path demonstrating the violation.

```python
# This safety property will fail (terminal states exist after success)
result = model_check(lts, AG(Not(Atomic("terminal"))), labeling)
if not result.satisfied:
    print(f"Property violated!")
    if result.counterexample:
        print(f"Counterexample trace: {result.counterexample}")
    print(f"States violating property: {len(lts.states) - len(result.satisfying_states)}")
```

The counterexample is a sequence of events leading from the initial state to a state that violates the property. Use it to understand *why* a property fails and whether the failure is a genuine bug or an expected behavior.

## Complete Script

```python
"""Temporal Verification Tutorial - Complete Script

Builds a retry agent in CSP, then checks safety, liveness,
and response properties using CTL model checking.
"""
from agenticraft_foundation import (
    Event, Prefix, ExternalChoice, Stop, Skip,
    Recursion, Variable, build_lts,
)
from agenticraft_foundation.verification import (
    model_check, AG, AF, EF, AU, Not, Or, Atomic,
    check_safety, check_liveness,
)

# Step 1: Build the agent
req = Event("request")
process = Event("process")
success = Event("success")
error = Event("error")
retry = Event("retry")

body = Prefix(req, Prefix(process, ExternalChoice(
    Prefix(success, Skip()),
    Prefix(error, Prefix(retry, Variable("X"))),
)))
agent = Recursion(variable="X", body=body)
lts = build_lts(agent, max_states=50)
print(f"States: {len(lts.states)}, Transitions: {len(lts.transitions)}")

# Step 2: Label states
labeling = {s: set() for s in lts.states}
labeling[lts.initial_state].add("init")
for s in lts.states:
    outgoing = {e for e, _ in lts.successors(s)}
    if not outgoing:
        labeling[s].add("terminal")
    if Event("success") in outgoing:
        labeling[s].add("can_succeed")
    if Event("error") in outgoing:
        labeling[s].add("can_fail")

# Step 3: Safety
result = check_safety(lts, "can_fail", labeling)
print(f"Never can_fail: {result.satisfied}")

# Step 4: Liveness
result = check_liveness(lts, "can_succeed", labeling)
print(f"Always eventually can_succeed: {result.satisfied}")

# Step 5: Reachability
result = model_check(lts, EF(Atomic("can_succeed")), labeling)
print(f"Success reachable: {result.satisfied}")

# Step 6: Response
for s in lts.states:
    outgoing = {e for e, _ in lts.successors(s)}
    if Event("request") in outgoing:
        labeling[s].add("waiting")
    if Event("success") in outgoing or Event("error") in outgoing:
        labeling[s].add("responded")

result = model_check(
    lts,
    AU(Or(Atomic("waiting"), Atomic("responded")), Atomic("can_succeed")),
    labeling,
)
print(f"Response property: {result.satisfied}")

# Step 7: Counterexample
result = model_check(lts, AG(Not(Atomic("terminal"))), labeling)
if not result.satisfied and result.counterexample:
    print(f"Counterexample: {result.counterexample}")
```

## Next Steps

- Read [Verification Concepts](../concepts/verification.md) for the formal foundations behind CTL model checking
- Continue to [Modeling Stochastic Agents with DTMC](probabilistic-analysis.md) to analyze probabilistic agent behavior
- See the [Temporal Logic API Reference](../api/verification/temporal.md) for the complete formula AST and model checker API
