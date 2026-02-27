# Consensus Specification

**Source:** `examples/consensus_spec.py`

This example uses CSP operators to formally model a 3-agent consensus protocol, verify its properties, and add fault tolerance with timeouts and interrupts.

---

## 1. Define Consensus Events

The protocol uses per-agent proposal and voting events, plus shared decision events.

```python
propose = [Event(f"propose_{i}") for i in range(3)]
vote = [Event(f"vote_{i}") for i in range(3)]
commit = Event("commit")
abort = Event("abort")
```

Each agent has its own `propose` and `vote` events (indexed by agent ID), but `commit` and `abort` are shared -- all agents must agree on the outcome. This separation is deliberate: proposals and votes are local actions, while the decision is a synchronized global event.

## 2. Build Individual Agent Processes

Each agent follows the same protocol: propose, vote, then either commit or abort.

```python
def make_agent(i: int):
    """Create a consensus participant process."""
    return Prefix(
        propose[i],
        Prefix(
            vote[i],
            ExternalChoice(
                left=Prefix(commit, Stop()),
                right=Prefix(abort, Stop()),
            ),
        ),
    )

agent_0 = make_agent(0)
agent_1 = make_agent(1)
agent_2 = make_agent(2)
```

The `ExternalChoice` at the end models the decision point: the environment (or the coordinator) determines whether the group commits or aborts. Each agent can participate in either outcome.

## 3. Compose with Synchronization

Agents run in parallel, synchronizing on the decision events.

```python
sync_events = frozenset({commit, abort})
two_agents = Parallel(left=agent_0, right=agent_1, sync_set=sync_events)
consensus = Parallel(left=two_agents, right=agent_2, sync_set=sync_events)
```

The `sync_set` contains `commit` and `abort`. This enforces a key consensus property: all agents must participate in the same decision event simultaneously. Agent 0 cannot commit while Agent 1 aborts -- the synchronization set prevents this. Proposals and votes are not in the sync set, so agents can propose and vote independently.

## 4. Analyze LTS and Detect Deadlocks

The composed system is analyzed for correctness.

```python
lts = build_lts(consensus)
print(f"States in consensus LTS: {len(lts.states)}")

dl = detect_deadlock(lts)
print(f"Has deadlock: {dl.has_deadlock}")
if dl.has_deadlock:
    print(f"Deadlock states: {len(dl.deadlock_states)}")
```

Building the LTS explores all reachable states of the composed system. Deadlock analysis reveals whether there are states where no agent can make progress. In this model, deadlock states exist after the decision is made (the system reaches `Stop`), which is expected terminal behavior rather than a liveness violation.

## 5. Trace Analysis

Traces enumerate all possible execution paths.

```python
t = list(traces(lts, max_length=10))
print(f"Total traces (max length 10): {len(t)}")

commit_traces = [tr for tr in t if commit in tr]
abort_traces = [tr for tr in t if abort in tr]
print(f"Traces reaching commit: {len(commit_traces)}")
print(f"Traces reaching abort: {len(abort_traces)}")
```

This verifies that both outcomes are reachable -- the protocol does not force a particular decision. The traces show all interleavings of agent proposals and votes, followed by either a commit or abort. The number of traces grows with the interleavings, but the decision events appear exactly once per trace.

## 6. Add Fault Tolerance with Timeout

A timeout ensures progress even if an agent stalls.

```python
fault_tolerant_agent = Timeout(
    process=agent_0,
    duration=10.0,
    fallback=Prefix(abort, Stop()),
)

print(f"Fault-tolerant agent initials: {fault_tolerant_agent.initials()}")
# Can propose OR timeout
```

Wrapping Agent 0 with a `Timeout` adds a fallback path: if the agent does not propose within the duration, the system falls back to an abort. This models crash-recovery behavior -- a non-responsive agent triggers an abort rather than blocking the entire consensus indefinitely.

## 7. Add Interrupt Handling

A coordinator can force an abort at any point.

```python
coordinator_abort = Event("coordinator_abort")
interruptible = Interrupt(
    primary=fault_tolerant_agent,
    handler=Prefix(coordinator_abort, Prefix(abort, Stop())),
)

print(f"Interruptible agent initials: {interruptible.initials()}")
```

The `Interrupt` wraps the already-timeout-protected agent, adding a coordinator override. At any point during the agent's execution, the coordinator can fire `coordinator_abort` to force an abort. This layers two fault-tolerance mechanisms: timeouts for unresponsive agents, interrupts for coordinator-initiated cancellation.

## 8. Verify the Fault-Tolerant Version

```python
ft_lts = build_lts(fault_tolerant_agent)
ft_traces = list(traces(ft_lts, max_length=10))
print(f"Fault-tolerant traces: {len(ft_traces)}")

print(f"Fault-tolerant is deadlock-free: {is_deadlock_free(fault_tolerant_agent)}")
```

The fault-tolerant version has more traces than the original (because the timeout introduces additional paths), but every trace leads to a terminal action. The analysis confirms that the timeout achieves its goal: the agent always makes progress toward either a commit or an abort.

---

??? example "Complete source"
    ```python
    """Formal Consensus Specification -- CSP model of distributed consensus.

    Demonstrates using CSP operators to model and verify consensus properties.
    """

    from agenticraft_foundation import (
        Event,
        ExternalChoice,
        Interrupt,
        Parallel,
        Prefix,
        Stop,
        Timeout,
        build_lts,
        detect_deadlock,
        is_deadlock_free,
        traces,
    )

    # =============================================================
    # Model a 3-agent consensus protocol
    # =============================================================
    print("=== 3-Agent Consensus Model ===")

    # Events for agent communication
    propose = [Event(f"propose_{i}") for i in range(3)]
    vote = [Event(f"vote_{i}") for i in range(3)]
    commit = Event("commit")
    abort = Event("abort")

    # Each agent: propose, then vote, then commit or abort
    def make_agent(i: int):
        """Create a consensus participant process."""
        return Prefix(
            propose[i],
            Prefix(
                vote[i],
                ExternalChoice(
                    left=Prefix(commit, Stop()),
                    right=Prefix(abort, Stop()),
                ),
            ),
        )

    agent_0 = make_agent(0)
    agent_1 = make_agent(1)
    agent_2 = make_agent(2)

    print(f"Agent 0 initials: {agent_0.initials()}")
    print(f"Agent 0 alphabet: {agent_0.alphabet()}")

    # Compose agents -- sync on commit/abort (all must agree)
    sync_events = frozenset({commit, abort})
    two_agents = Parallel(left=agent_0, right=agent_1, sync_set=sync_events)
    consensus = Parallel(left=two_agents, right=agent_2, sync_set=sync_events)

    print(f"\nConsensus system alphabet: {consensus.alphabet()}")
    print(f"Consensus system initials: {consensus.initials()}")

    # =============================================================
    # Verify properties
    # =============================================================
    print("\n=== Verification ===")

    # Build LTS and analyze
    lts = build_lts(consensus)
    print(f"States in consensus LTS: {len(lts.states)}")

    # Check for deadlock
    dl = detect_deadlock(lts)
    print(f"Has deadlock: {dl.has_deadlock}")
    if dl.has_deadlock:
        print(f"Deadlock states: {len(dl.deadlock_states)}")

    # Enumerate traces
    t = list(traces(lts, max_length=10))
    print(f"Total traces (max length 10): {len(t)}")

    # Check if commit and abort are both reachable
    commit_traces = [tr for tr in t if commit in tr]
    abort_traces = [tr for tr in t if abort in tr]
    print(f"Traces reaching commit: {len(commit_traces)}")
    print(f"Traces reaching abort: {len(abort_traces)}")

    # =============================================================
    # Add fault tolerance: timeout on proposals
    # =============================================================
    print("\n=== Fault-Tolerant Consensus ===")

    # Wrap agent 0 with a timeout: if it doesn't propose in time, abort
    fault_tolerant_agent = Timeout(
        process=agent_0,
        duration=10.0,
        fallback=Prefix(abort, Stop()),
    )

    print(f"Fault-tolerant agent initials: {fault_tolerant_agent.initials()}")
    # Can propose OR timeout

    # Add interrupt: coordinator can force abort
    coordinator_abort = Event("coordinator_abort")
    interruptible = Interrupt(
        primary=fault_tolerant_agent,
        handler=Prefix(coordinator_abort, Prefix(abort, Stop())),
    )

    print(f"Interruptible agent initials: {interruptible.initials()}")

    # Verify the fault-tolerant version
    ft_lts = build_lts(fault_tolerant_agent)
    ft_traces = list(traces(ft_lts, max_length=10))
    print(f"Fault-tolerant traces: {len(ft_traces)}")

    # The timeout ensures the agent always makes progress
    print(f"Fault-tolerant is deadlock-free: {is_deadlock_free(fault_tolerant_agent)}")

    print("\nConsensus specification complete.")
    ```
