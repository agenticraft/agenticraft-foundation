# Modeling Stochastic Agents with DTMC

**Time:** 15 minutes

In this tutorial, you will build Discrete-Time Markov Chain models of stochastic LLM agents, compute reachability probabilities, analyze steady-state distributions, and calculate expected completion times.

## Prerequisites

- Python 3.10+
- `agenticraft-foundation` installed
- Basic probability (what a transition probability means)

## What You'll Build

A probabilistic model of an LLM agent with retry logic and provider fallback. You will:

1. Build a DTMC for a single-provider agent with retries
2. Compute reachability probability ("will it succeed?")
3. Compute expected steps to success
4. Extend to a multi-provider fallback model
5. Analyze steady-state distribution for an ergodic agent
6. Compare designs quantitatively

## Step 1: Build a Single-Provider Agent

An LLM agent sends a request to a provider. With probability 0.9, the call succeeds. With probability 0.1, it fails transiently. On failure, the agent retries.

```python
from agenticraft_foundation.verification import DTMC

dtmc = DTMC()
dtmc.add_state(0, labels={"start"})
dtmc.add_state(1, labels={"calling"})
dtmc.add_state(2, labels={"success"})
dtmc.add_state(3, labels={"failed"})

dtmc.add_transition(0, 1, probability=1.0)    # start → calling
dtmc.add_transition(1, 2, probability=0.9)    # 90% success
dtmc.add_transition(1, 3, probability=0.1)    # 10% transient failure
dtmc.add_transition(3, 1, probability=1.0)    # retry on failure
dtmc.add_transition(2, 2, probability=1.0)    # success is absorbing

dtmc.validate()  # checks probabilities sum to 1.0 per state
print(f"States: {len(dtmc.states)}, Transitions: {dtmc.transition_count()}")
```

The `validate()` method raises `ValueError` if any state's outgoing probabilities don't sum to 1.0. Always call it after building a DTMC.

## Step 2: Compute Reachability Probability

The fundamental question: "what is the probability that the agent eventually reaches success?"

```python
from agenticraft_foundation.verification import check_reachability

result = check_reachability(dtmc, target_labels={"success"})
print(f"P(eventually success) = {result.probability:.4f}")
print(f"Per-state probabilities:")
for state_id, prob in sorted(result.per_state.items()):
    labels = dtmc.states[state_id].labels
    print(f"  State {state_id} {labels}: {prob:.4f}")
```

With retry on every failure, the probability of *eventually* succeeding is 1.0 -- the agent will keep trying until it works. The math: $P = 0.9 + 0.1 \cdot 0.9 + 0.1^2 \cdot 0.9 + \cdots = \sum_{k=0}^{\infty} 0.1^k \cdot 0.9 = 1.0$.

## Step 3: Compute Expected Steps

How many steps does it take on average to reach success?

```python
from agenticraft_foundation.verification import expected_steps

result = expected_steps(dtmc, target_labels={"success"})
print(f"E[steps to success] = {result.expected:.2f}")
print(f"Per-state expected steps:")
for state_id, steps in sorted(result.per_state.items()):
    labels = dtmc.states[state_id].labels
    print(f"  State {state_id} {labels}: {steps:.2f}")
```

From the start state, the expected number of steps is $1 + \frac{1}{0.9} \approx 2.11$ -- one step to reach `calling`, then on average $\frac{1}{0.9} \approx 1.11$ attempts to succeed (geometric distribution).

## Step 4: Multi-Provider Fallback

Now model a more realistic agent that falls back to a secondary provider when the primary fails.

```python
fallback = DTMC()
fallback.add_state(0, labels={"start"})
fallback.add_state(1, labels={"primary"})
fallback.add_state(2, labels={"secondary"})
fallback.add_state(3, labels={"success"})
fallback.add_state(4, labels={"total_failure"})

fallback.add_transition(0, 1, probability=1.0)     # start → try primary
fallback.add_transition(1, 3, probability=0.9)     # primary succeeds (90%)
fallback.add_transition(1, 2, probability=0.1)     # primary fails → try secondary
fallback.add_transition(2, 3, probability=0.95)    # secondary succeeds (95%)
fallback.add_transition(2, 4, probability=0.05)    # secondary also fails
fallback.add_transition(3, 3, probability=1.0)     # absorbing
fallback.add_transition(4, 4, probability=1.0)     # absorbing

fallback.validate()

result = check_reachability(fallback, target_labels={"success"})
print(f"P(success) = {result.probability:.4f}")

failure = check_reachability(fallback, target_labels={"total_failure"})
print(f"P(total failure) = {failure.probability:.4f}")
```

With fallback, the success probability is $0.9 + 0.1 \cdot 0.95 = 0.995$. The total failure probability is $0.1 \cdot 0.05 = 0.005$. This is a non-absorbing chain with two absorbing states, so probabilities don't need to sum to 1.0 per target -- they sum across targets.

## Step 5: Steady-State Distribution

For an agent that runs continuously (not absorbing), the steady-state distribution tells you the long-run fraction of time spent in each state.

```python
from agenticraft_foundation.verification import steady_state

# Ergodic agent: cycles through states continuously
cyclic = DTMC()
cyclic.add_state(0, labels={"idle"})
cyclic.add_state(1, labels={"thinking"})
cyclic.add_state(2, labels={"acting"})
cyclic.add_state(3, labels={"reviewing"})

cyclic.add_transition(0, 1, probability=1.0)     # idle → thinking
cyclic.add_transition(1, 2, probability=0.7)     # think → act (70%)
cyclic.add_transition(1, 1, probability=0.3)     # think more (30%)
cyclic.add_transition(2, 3, probability=1.0)     # act → review
cyclic.add_transition(3, 0, probability=0.8)     # review → idle (80%)
cyclic.add_transition(3, 1, probability=0.2)     # review → rethink (20%)

cyclic.validate()

dist = steady_state(cyclic)
print("Steady-state distribution:")
for state_id, prob in sorted(dist.items()):
    labels = cyclic.states[state_id].labels
    print(f"  State {state_id} {labels}: {prob:.4f}")
```

The steady-state distribution $\pi$ satisfies $\pi = \pi P$ and $\sum_s \pi(s) = 1$. It tells you: in the long run, what fraction of time does the agent spend thinking vs. acting vs. reviewing?

## Step 6: Compare Designs

Use probabilistic verification to make quantitative design decisions. Compare the single-provider retry vs. multi-provider fallback:

```python
# Single-provider with retry
retry_steps = expected_steps(dtmc, target_labels={"success"})
retry_prob = check_reachability(dtmc, target_labels={"success"})

# Multi-provider fallback
fallback_steps = expected_steps(fallback, target_labels={"success"})
fallback_prob = check_reachability(fallback, target_labels={"success"})

print("Design Comparison:")
print(f"  {'Metric':<30} {'Retry':>10} {'Fallback':>10}")
print(f"  {'P(success)':<30} {retry_prob.probability:>10.4f} {fallback_prob.probability:>10.4f}")
print(f"  {'E[steps]':<30} {retry_steps.expected:>10.2f} {fallback_steps.expected:>10.2f}")
```

The retry model guarantees eventual success ($P = 1.0$) but may take many steps. The fallback model has a small failure probability ($P = 0.005$) but always finishes in exactly 3 steps. Which is better depends on your requirements.

## Complete Script

```python
"""Probabilistic Analysis Tutorial - Complete Script

Models LLM agents as DTMCs and computes reachability,
expected steps, and steady-state distributions.
"""
from agenticraft_foundation.verification import (
    DTMC, check_reachability, steady_state, expected_steps,
)

# Step 1: Single-provider with retry
dtmc = DTMC()
dtmc.add_state(0, labels={"start"})
dtmc.add_state(1, labels={"calling"})
dtmc.add_state(2, labels={"success"})
dtmc.add_state(3, labels={"failed"})
dtmc.add_transition(0, 1, probability=1.0)
dtmc.add_transition(1, 2, probability=0.9)
dtmc.add_transition(1, 3, probability=0.1)
dtmc.add_transition(3, 1, probability=1.0)
dtmc.add_transition(2, 2, probability=1.0)
dtmc.validate()

# Step 2: Reachability
result = check_reachability(dtmc, target_labels={"success"})
print(f"Retry model P(success) = {result.probability:.4f}")

# Step 3: Expected steps
steps = expected_steps(dtmc, target_labels={"success"})
print(f"Retry model E[steps] = {steps.expected:.2f}")

# Step 4: Multi-provider fallback
fallback = DTMC()
fallback.add_state(0, labels={"start"})
fallback.add_state(1, labels={"primary"})
fallback.add_state(2, labels={"secondary"})
fallback.add_state(3, labels={"success"})
fallback.add_state(4, labels={"total_failure"})
fallback.add_transition(0, 1, probability=1.0)
fallback.add_transition(1, 3, probability=0.9)
fallback.add_transition(1, 2, probability=0.1)
fallback.add_transition(2, 3, probability=0.95)
fallback.add_transition(2, 4, probability=0.05)
fallback.add_transition(3, 3, probability=1.0)
fallback.add_transition(4, 4, probability=1.0)
fallback.validate()

result = check_reachability(fallback, target_labels={"success"})
print(f"Fallback model P(success) = {result.probability:.4f}")

# Step 5: Steady-state
cyclic = DTMC()
cyclic.add_state(0, labels={"idle"})
cyclic.add_state(1, labels={"thinking"})
cyclic.add_state(2, labels={"acting"})
cyclic.add_state(3, labels={"reviewing"})
cyclic.add_transition(0, 1, probability=1.0)
cyclic.add_transition(1, 2, probability=0.7)
cyclic.add_transition(1, 1, probability=0.3)
cyclic.add_transition(2, 3, probability=1.0)
cyclic.add_transition(3, 0, probability=0.8)
cyclic.add_transition(3, 1, probability=0.2)
cyclic.validate()

dist = steady_state(cyclic)
print("Steady-state:", {cyclic.states[s].labels: f"{p:.3f}" for s, p in dist.items()})

# Step 6: Compare
retry_steps = expected_steps(dtmc, target_labels={"success"})
fallback_steps = expected_steps(fallback, target_labels={"success"})
print(f"Retry E[steps]={retry_steps.expected:.2f}, Fallback E[steps]={fallback_steps.expected:.2f}")
```

## Next Steps

- Read [Verification Concepts](../concepts/verification.md) for the theory behind DTMC model checking
- See the [Probabilistic API Reference](../api/verification/probabilistic.md) for the complete DTMC API
- Explore [Checking Temporal Properties with CTL](temporal-verification.md) for non-probabilistic verification
