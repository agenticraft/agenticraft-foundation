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
