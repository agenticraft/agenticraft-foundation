# Probabilistic Verification

**Source:** `examples/probabilistic_verification.py`

This example uses Discrete-Time Markov Chains (DTMC) to model and verify probabilistic properties of stochastic agent systems -- LLM retry logic, provider fallback chains, and consensus convergence.

---

## 1. LLM Agent Retry Model

An LLM agent that calls an API, handling transient and permanent errors.

```python
from agenticraft_foundation.verification import (
    DTMC, check_reachability, expected_steps, steady_state,
)

dtmc = DTMC()
dtmc.add_state(0, labels={"idle"})
dtmc.add_state(1, labels={"calling_llm"})
dtmc.add_state(2, labels={"success"})
dtmc.add_state(3, labels={"transient_error"})
dtmc.add_state(4, labels={"permanent_error"})

dtmc.add_transition(0, 1, probability=1.0)
dtmc.add_transition(1, 2, probability=0.85)   # 85% success
dtmc.add_transition(1, 3, probability=0.12)   # 12% transient
dtmc.add_transition(1, 4, probability=0.03)   # 3% permanent
dtmc.add_transition(3, 1, probability=1.0)    # retry
dtmc.add_transition(2, 2, probability=1.0)    # absorbing
dtmc.add_transition(4, 4, probability=1.0)    # absorbing

dtmc.validate()
```

Transition probabilities from each state must sum to 1.0. The `validate()` method checks this.

### Reachability Probability

"What's the probability of eventually reaching success?"

```python
result = check_reachability(dtmc, target_labels={"success"})
print(f"P(success) = {result.probability:.6f}")  # 0.965909
```

The system solves a linear system: $p(s) = \sum_{s'} P(s, s') \cdot p(s')$ with boundary conditions $p(\text{success}) = 1$ and $p(\text{permanent\_error}) = 0$.

The retry loop amplifies the per-attempt 85% rate to 96.6% overall -- but the 3% permanent error rate caps it.

### Expected Steps

```python
steps = expected_steps(dtmc, target_labels={"success"})
print(f"E[steps] = {steps.expected:.4f}")  # 2.2727
```

On average, 2.27 steps to reach success (accounting for retries).

## 2. Provider Fallback Chain

Three providers with decreasing reliability as fallbacks.

```python
fallback = DTMC()
fallback.add_state(0, labels={"start"})
fallback.add_state(1, labels={"primary"})       # 95% success
fallback.add_state(2, labels={"secondary"})     # 90% success
fallback.add_state(3, labels={"tertiary"})      # 80% success
fallback.add_state(4, labels={"success"})
fallback.add_state(5, labels={"total_failure"})

fallback.add_transition(0, 1, probability=1.0)
fallback.add_transition(1, 4, probability=0.95)
fallback.add_transition(1, 2, probability=0.05)
fallback.add_transition(2, 4, probability=0.90)
fallback.add_transition(2, 3, probability=0.10)
fallback.add_transition(3, 4, probability=0.80)
fallback.add_transition(3, 5, probability=0.20)
fallback.add_transition(4, 4, probability=1.0)
fallback.add_transition(5, 5, probability=1.0)

result = check_reachability(fallback, target_labels={"success"})
print(f"P(success) = {result.probability:.6f}")  # 0.999000
```

The cascade: $0.95 + 0.05 \times 0.90 + 0.05 \times 0.10 \times 0.80 = 0.999$. Three nines of reliability from providers that individually offer far less.

## 3. Consensus Convergence

A gossip-based consensus model where convergence probability increases per round.

```python
consensus = DTMC()
consensus.add_state(0, labels={"round_0", "diverged"})
consensus.add_state(1, labels={"round_1", "diverged"})
consensus.add_state(2, labels={"round_2", "diverged"})
consensus.add_state(3, labels={"converged"})

consensus.add_transition(0, 1, probability=0.6)
consensus.add_transition(0, 3, probability=0.4)
consensus.add_transition(1, 2, probability=0.3)
consensus.add_transition(1, 3, probability=0.7)
consensus.add_transition(2, 3, probability=1.0)
consensus.add_transition(3, 3, probability=1.0)
```

```python
steps = expected_steps(consensus, target_labels={"converged"})
print(f"E[rounds] = {steps.expected:.4f}")  # 1.7800
```

Guaranteed convergence within 3 rounds, expected in under 2.

## 4. Ergodic Steady-State

An agent cycling through think/act/observe states indefinitely.

```python
cycle = DTMC()
cycle.add_state(0, labels={"thinking"})
cycle.add_state(1, labels={"acting"})
cycle.add_state(2, labels={"observing"})

cycle.add_transition(0, 1, probability=0.8)
cycle.add_transition(0, 2, probability=0.2)
cycle.add_transition(1, 2, probability=1.0)
cycle.add_transition(2, 0, probability=1.0)

dist = steady_state(cycle)
```

The steady-state distribution $\pi$ satisfies $\pi = \pi P$:

| State | Long-run fraction |
|-------|-------------------|
| thinking | 35.7% |
| acting | 28.6% |
| observing | 35.7% |

This tells you the long-run resource allocation: the agent spends about a third of its time thinking, a third observing, and slightly less acting.

## Output

```
=== LLM Agent Retry Model ===
P(eventually success) = 0.965909
E[steps to success] = 2.2727

=== Provider Fallback Chain ===
P(success with fallback chain) = 0.999000
P(total failure) = 0.001000

=== Consensus Convergence ===
P(eventually converge) = 1.0000
E[rounds to converge] = 1.7800

=== Ergodic Agent Cycle ===
  thinking: 35.7%
  acting: 28.6%
  observing: 35.7%
```
