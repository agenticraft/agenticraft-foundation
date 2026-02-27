# Complexity & Fault Models

## Overview

Distributed multi-agent systems operate under fundamental theoretical constraints. This page catalogues the **complexity bounds** for core distributed algorithms -- consensus, gossip, leader election, broadcast, and mesh communication -- and defines the **fault models** that govern failure behavior, including four LLM-specific failure modes that extend the classical taxonomy.

Understanding these bounds is essential for choosing the right coordination strategy: some problems have provable lower bounds that no algorithm can beat, and some failure modes require specific architectural mitigations.

## Complexity Bounds

### Consensus

| Setting | Metric | Bound | Notes |
|---------|--------|-------|-------|
| Synchronous, crash failures | Rounds | $O(f + 1)$ | $f$ = max crash failures; tight |
| Synchronous, Byzantine | Message complexity | $\Theta(n^2)$ | Quadratic message overhead unavoidable |
| Asynchronous, crash (FLP) | Deterministic | **Impossible** | No deterministic protocol solves consensus with even 1 crash failure |
| Asynchronous, randomized | Expected rounds | $O(n)$ | Ben-Or's protocol; exponential in worst case |
| Partially synchronous | Rounds after GST | $O(f + 1)$ | DLS/PBFT model; requires $n \geq 3f + 1$ |

### Gossip

| Setting | Metric | Bound | Notes |
|---------|--------|-------|-------|
| Random gossip (push) | Rounds to converge | $O(\log n)$ | With high probability |
| Random gossip (push-pull) | Rounds to converge | $O(\log \log n)$ | Doubly logarithmic -- very fast |
| Total messages | Messages | $O(n \log n)$ | Each round: $n$ messages |
| Diameter-bounded | Rounds | $O(D \log n)$ | $D$ = network diameter |
| Byzantine gossip | Messages | $O(n^2 \log n)$ | Needs authentication or signatures |

### Leader Election

| Setting | Metric | Bound | Notes |
|---------|--------|-------|-------|
| Synchronous ring | Messages | $O(n)$ | LCR algorithm |
| Asynchronous ring | Messages | $\Omega(n \log n)$ | Proven lower bound (Burns 1980) |
| General network, async | Messages | $O(n \log n)$ | GHS algorithm |
| With unique IDs | Messages | $O(n)$ | If topology is known |

### Broadcast

| Setting | Metric | Bound | Notes |
|---------|--------|-------|-------|
| Flooding | Messages | $O(n^2)$ | Every node forwards to all neighbors |
| Spanning tree | Messages | $O(n)$ | After tree construction |
| Spanning tree | Rounds | $O(\log n)$ for balanced tree | $O(n)$ worst case (path graph) |
| Reliable broadcast (async) | Messages | $O(n^2)$ | Bracha's protocol |
| Byzantine reliable broadcast | Messages | $O(n^2)$ | Requires $n \geq 3f + 1$ |

### Mesh Communication

| Topology | Messages per broadcast | Rounds per broadcast | Notes |
|----------|----------------------|---------------------|-------|
| Full mesh | $O(n)$ | $O(1)$ | Direct delivery |
| Tree | $O(n)$ | $O(\log n)$ balanced, $O(n)$ worst | Height-dependent |
| Ring | $O(n)$ | $O(n)$ | Sequential propagation |
| Hypercube | $O(n \log n)$ | $O(\log n)$ | Dimension-by-dimension |
| $k$-regular expander | $O(kn)$ | $O(\log n)$ | Constant spectral gap |

## Fault Models

### Classical Fault Models

| Model | Description | Behavior |
|-------|-------------|----------|
| **Crash-Stop** | Process halts permanently and does not recover | Simplest model; process is either correct or stopped |
| **Crash-Recovery** | Process halts and may restart with stable storage intact | Models transient failures; recovered process rejoins with persisted state |
| **Byzantine** | Process exhibits arbitrary behavior -- may send conflicting messages, lie, or collude | Worst-case adversarial model; requires $n \geq 3f + 1$ to tolerate $f$ faults |
| **Omission** | Process fails to send or receive some messages but otherwise behaves correctly | Models network partitions and message loss |

### LLM-Specific Fault Models

LLM-based agents exhibit failure modes that do not fit neatly into the classical taxonomy. AgentiCraft Foundation defines four additional fault models:

| Model | Description | Classical Analog | Mitigation |
|-------|-------------|------------------|------------|
| **Hallucination** | Agent produces confident but factually incorrect output | Byzantine -- the output is wrong but indistinguishable from correct without external verification | Ensemble voting, factual grounding, cross-agent verification |
| **Prompt Injection** | External input manipulates the agent's behavior, overriding instructions | Byzantine -- the agent acts against its specification due to adversarial input | Input sanitization, instruction hierarchy, session isolation |
| **Non-Determinism** | Same input produces different outputs across invocations | Omission -- from the system's perspective, the expected output may not arrive | Temperature control, majority voting, deterministic decoding |
| **Context Overflow** | Input exceeds the model's context window, causing truncation or failure | Crash-Recovery -- the agent effectively crashes on that input but can recover on shorter inputs | Context windowing, summarization, hierarchical decomposition |

### Fault Model Hierarchy

The classical fault models form a strict hierarchy of increasing severity:

$$\text{Crash-Stop} \subset \text{Crash-Recovery} \subset \text{Omission} \subset \text{Byzantine}$$

Any algorithm that tolerates Byzantine faults also tolerates all weaker fault models. The LLM-specific models map into this hierarchy at their analog positions but may require domain-specific detection mechanisms.

## Impossibility Results

### FLP Impossibility (Fischer, Lynch, Paterson 1985)

**No deterministic consensus protocol can guarantee termination in an asynchronous system with even one crash failure.**

Formally: there is no deterministic algorithm that satisfies Agreement, Validity, and Termination simultaneously in an asynchronous message-passing system where at least one process may crash.

This is a fundamental result. Practical systems work around FLP through:

- **Randomization**: Ben-Or's protocol achieves consensus with probability 1
- **Failure detectors**: Chandra-Toueg's unreliable failure detector $\diamond S$ is sufficient
- **Partial synchrony**: DLS model assumes the system is eventually synchronous after an unknown Global Stabilization Time (GST)

### Byzantine Fault Tolerance Bound

**Tolerating $f$ Byzantine faults requires at least $n \geq 3f + 1$ processes.**

This means:
- 4 agents can tolerate 1 Byzantine fault
- 7 agents can tolerate 2 Byzantine faults
- In general, more than two-thirds of agents must be correct

For LLM agents with hallucination faults (Byzantine analog), this translates to requiring at least 3 agents to detect 1 hallucinating agent via majority vote.

## How It Maps to Code

```python
from agenticraft_foundation.complexity import (
    CONSENSUS_BOUNDS, GOSSIP_BOUNDS, IMPOSSIBILITY_RESULTS,
    FaultModel, validate_fault_tolerance,
    consensus_complexity, get_impossibility,
)

# Query consensus complexity bounds
for bound in CONSENSUS_BOUNDS:
    print(f"{bound.algorithm}: {bound.time_complexity} ({bound.synchrony_model})")

# Analyze consensus complexity for a specific configuration
analysis = consensus_complexity(n_agents=7, fault_model=FaultModel.BYZANTINE)
print(f"Message complexity: {analysis.message_complexity}")
print(f"Round complexity: {analysis.round_complexity}")

# Validate fault tolerance requirements
result = validate_fault_tolerance(
    n_agents=7,
    max_faults=2,
    fault_model=FaultModel.BYZANTINE,
)
print(f"Tolerant: {result}")  # True: 7 >= 3*2 + 1

# Check impossibility results
flp = get_impossibility("FLP")
print(f"{flp.name}: {flp.description}")
```

## Further Reading

- **API Reference**: [complexity/bounds](../api/complexity/bounds.md)
- **Tutorial**: [Consensus Verification](../tutorials/consensus-verification.md)

### References

- M.J. Fischer, N.A. Lynch, M.S. Paterson, "Impossibility of Distributed Consensus with One Faulty Process," *Journal of the ACM*, 32(2), 1985.
- M. Castro, B. Liskov, "Practical Byzantine Fault Tolerance," *OSDI*, 1999.
