# Modeling Agent Coordination with CSP

**Time:** 15 minutes

In this tutorial, you will model a multi-agent consensus protocol using Communicating Sequential Processes (CSP), verify it for deadlock freedom, and extend it with fault tolerance using Timeout and Interrupt constructs.

## Prerequisites

- Python 3.10+
- `agenticraft-foundation` installed
- Basic Python knowledge

## What You'll Build

A 3-agent consensus protocol modeled in CSP where agents propose values, vote, and reach a decision. You will:

1. Define agent events and processes
2. Compose agents in parallel with synchronization
3. Build a Labeled Transition System (LTS) and analyze it
4. Add fault tolerance with Timeout and Interrupt
5. Verify refinement between the original and fault-tolerant versions

## Step 1: Define Agent Events

Every CSP model starts with events -- the atomic actions that agents can perform. In a consensus protocol, agents propose values, vote on proposals, decide on a final value, and acknowledge the decision.

```python
from agenticraft_foundation import Event, Prefix, Stop, Skip

propose = Event("propose")
vote = Event("vote")
decide = Event("decide")
ack = Event("acknowledge")
```

`Event` creates a named event. `Prefix`, `Stop`, and `Skip` are CSP process constructors: `Prefix(e, P)` performs event `e` then behaves like process `P`, `Stop` is the deadlocked process (does nothing), and `Skip` is the successfully terminated process.

## Step 2: Build Individual Agent Processes

Each agent follows the same protocol: propose, then vote, then decide, then terminate successfully.

```python
agent = Prefix(propose, Prefix(vote, Prefix(decide, Skip())))
```

Reading this from left to right: the agent first performs `propose`, then `vote`, then `decide`, then terminates. This is a sequential composition -- the agent must complete each step before moving to the next.

You can think of this as a finite state machine with 4 states and 3 transitions:

```
[start] --propose--> [voted] --vote--> [decided] --decide--> [done]
```

## Step 3: Compose Agents in Parallel

Real multi-agent systems have multiple agents running concurrently. CSP's parallel composition operator lets you model this, with a synchronization set that defines which events require all agents to participate.

```python
from agenticraft_foundation import Parallel

sync_events = {propose, vote, decide}
agent1 = Prefix(propose, Prefix(vote, Prefix(decide, Skip())))
agent2 = Prefix(propose, Prefix(vote, Prefix(decide, Skip())))
system = Parallel(agent1, agent2, sync_set=sync_events)
```

The `sync_set` parameter is critical. Events in the synchronization set can only occur when both agents are ready to perform them simultaneously. This models the requirement that all agents must participate in each phase of the consensus protocol. Events not in the sync set can be performed independently by either agent.

## Step 4: Build the LTS and Analyze

With the system composed, you can build its Labeled Transition System -- a complete state graph showing every reachable state and every possible transition. This is the foundation for all verification.

```python
from agenticraft_foundation import build_lts, traces, detect_deadlock, is_deadlock_free

lts = build_lts(system)
print(f"States: {len(lts.states)}")
print(f"Transitions: {len(lts.transitions)}")
print(f"Deadlock-free: {is_deadlock_free(system)}")

deadlocks = detect_deadlock(system)
print(f"Deadlock states: {len(deadlocks.deadlock_states)}")

all_traces = list(traces(lts, max_length=6))
print(f"Traces (up to length 6): {len(all_traces)}")
```

The `build_lts` function exhaustively explores the state space of the process. `is_deadlock_free` checks whether the system can reach any state where no further progress is possible (other than successful termination). `traces` enumerates all possible event sequences up to a given length.

For this simple two-agent system, you should see a small number of states (since both agents are synchronized on all events, they move in lockstep). In a system with unsynchronized events, the state space grows as agents interleave independently.

## Step 5: Add Fault Tolerance with Timeout

Production multi-agent systems need fault tolerance. What happens if an agent hangs during the vote phase? CSP's Timeout construct lets you model bounded waiting with a fallback.

```python
from agenticraft_foundation import Timeout, Interrupt

fallback = Prefix(Event("use_cached"), Stop())
bounded_agent = Timeout(process=agent, duration=30.0, fallback=fallback)
```

`Timeout(process, duration, fallback)` behaves like `process` if it completes within `duration`, otherwise it switches to `fallback`. Here, the fallback uses a cached value and then stops. This models a real-world pattern where agents have SLA deadlines and must degrade gracefully.

## Step 6: Add Interrupt Handling

Beyond timeouts, agents need to handle external interrupts -- for example, a cancellation signal from a supervisor.

```python
cancel = Event("cancel")
cancel_handler = Prefix(cancel, Prefix(Event("cleanup"), Stop()))
interruptible = Interrupt(primary=bounded_agent, handler=cancel_handler)

lts_final = build_lts(interruptible)
print(f"Final states: {len(lts_final.states)}")
```

`Interrupt(primary, handler)` runs the primary process, but at any point, the handler can take over. Here, if a `cancel` event occurs, the agent performs cleanup and stops. The LTS for this interruptible process will be larger because it includes the interrupt paths at every state.

## Step 7: Check Refinement

Refinement is a core concept in CSP verification. Process `P` refines process `Q` (written `Q [= P`) if every observable behavior of `P` is also a behavior of `Q`. This lets you verify that your fault-tolerant implementation is a valid refinement of the original specification.

```python
from agenticraft_foundation.algebra import trace_refines

# Verify the fault-tolerant version refines the original spec
print(f"Refines original: {trace_refines(agent, bounded_agent)}")
```

If `trace_refines(spec, impl)` returns `True`, it means the implementation never exhibits a trace that the specification does not allow. The fault-tolerant version may have additional behaviors (like the timeout fallback), so trace refinement may not hold in that direction -- and that is expected. What matters is whether the refinement relationship matches your design intent.

## Complete Script

```python
"""CSP Coordination Tutorial - Complete Script

Models a multi-agent consensus protocol, verifies deadlock freedom,
and adds fault tolerance with Timeout and Interrupt.
"""
from agenticraft_foundation import (
    Event, Prefix, Stop, Skip, Parallel,
    Timeout, Interrupt,
    build_lts, traces, detect_deadlock, is_deadlock_free,
)
from agenticraft_foundation.algebra import trace_refines

# Step 1: Define events
propose = Event("propose")
vote = Event("vote")
decide = Event("decide")
ack = Event("acknowledge")

# Step 2: Build agent processes
agent = Prefix(propose, Prefix(vote, Prefix(decide, Skip())))

# Step 3: Compose in parallel
sync_events = {propose, vote, decide}
agent1 = Prefix(propose, Prefix(vote, Prefix(decide, Skip())))
agent2 = Prefix(propose, Prefix(vote, Prefix(decide, Skip())))
system = Parallel(agent1, agent2, sync_set=sync_events)

# Step 4: Build LTS and analyze
lts = build_lts(system)
print(f"States: {len(lts.states)}")
print(f"Transitions: {len(lts.transitions)}")
print(f"Deadlock-free: {is_deadlock_free(system)}")

deadlocks = detect_deadlock(system)
print(f"Deadlock states: {len(deadlocks.deadlock_states)}")

all_traces = list(traces(lts, max_length=6))
print(f"Traces (up to length 6): {len(all_traces)}")

# Step 5: Add timeout fault tolerance
fallback = Prefix(Event("use_cached"), Stop())
bounded_agent = Timeout(process=agent, duration=30.0, fallback=fallback)

# Step 6: Add interrupt handling
cancel = Event("cancel")
cancel_handler = Prefix(cancel, Prefix(Event("cleanup"), Stop()))
interruptible = Interrupt(primary=bounded_agent, handler=cancel_handler)

lts_final = build_lts(interruptible)
print(f"Final states: {len(lts_final.states)}")

# Step 7: Check refinement
print(f"Refines original: {trace_refines(agent, bounded_agent)}")
```

## Next Steps

- Read [Process Algebra Concepts](../concepts/process-algebra.md) for the formal foundations behind CSP
- Explore the [Algebra API Reference](../api/algebra/index.md) for the complete set of process constructors and operations
- Continue to [Verifying Protocols with Session Types](mpst-verification.md) to learn how to verify communication protocols
