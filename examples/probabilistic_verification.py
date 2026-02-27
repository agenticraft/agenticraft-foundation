"""Probabilistic Verification -- DTMC model checking for stochastic agents.

Demonstrates DTMC construction, reachability probability computation,
steady-state distribution, and expected steps analysis for LLM agent
reliability modeling.
"""

from agenticraft_foundation.verification import (
    DTMC,
    check_reachability,
    expected_steps,
    steady_state,
)

# =============================================================
# Model 1: LLM agent with retry logic
# =============================================================
print("=== LLM Agent Retry Model ===")

dtmc = DTMC()
dtmc.add_state(0, labels={"idle"})
dtmc.add_state(1, labels={"calling_llm"})
dtmc.add_state(2, labels={"success"})
dtmc.add_state(3, labels={"transient_error"})
dtmc.add_state(4, labels={"permanent_error"})

# idle → calling LLM (always)
dtmc.add_transition(0, 1, probability=1.0)
# LLM call: 85% success, 12% transient error, 3% permanent error
dtmc.add_transition(1, 2, probability=0.85)
dtmc.add_transition(1, 3, probability=0.12)
dtmc.add_transition(1, 4, probability=0.03)
# Transient error: retry (back to calling LLM)
dtmc.add_transition(3, 1, probability=1.0)
# Absorbing states
dtmc.add_transition(2, 2, probability=1.0)
dtmc.add_transition(4, 4, probability=1.0)

dtmc.validate()
print(f"States: {len(dtmc.states)}")

# Reachability: what's the probability of success?
result = check_reachability(dtmc, target_labels={"success"})
print(f"P(eventually success) = {result.probability:.6f}")
print(f"P(eventually permanent error) = {1 - result.probability:.6f}")

# Per-state reachability
print("\nPer-state success probability:")
for s in sorted(result.per_state):
    labels = dtmc.states[s].labels
    print(f"  State {s} {labels}: {result.per_state[s]:.6f}")

# Expected steps to success
steps = expected_steps(dtmc, target_labels={"success"})
print(f"\nE[steps to success | success] = {steps.expected:.4f}")

# =============================================================
# Model 2: Provider fallback chain
# =============================================================
print("\n=== Provider Fallback Chain ===")

fallback = DTMC()
fallback.add_state(0, labels={"start"})
fallback.add_state(1, labels={"primary"})       # OpenAI
fallback.add_state(2, labels={"secondary"})     # Anthropic
fallback.add_state(3, labels={"tertiary"})      # Local model
fallback.add_state(4, labels={"success"})
fallback.add_state(5, labels={"total_failure"})

fallback.add_transition(0, 1, probability=1.0)
# Primary: 95% success, 5% fail → secondary
fallback.add_transition(1, 4, probability=0.95)
fallback.add_transition(1, 2, probability=0.05)
# Secondary: 90% success, 10% fail → tertiary
fallback.add_transition(2, 4, probability=0.90)
fallback.add_transition(2, 3, probability=0.10)
# Tertiary: 80% success, 20% total failure
fallback.add_transition(3, 4, probability=0.80)
fallback.add_transition(3, 5, probability=0.20)
# Absorbing
fallback.add_transition(4, 4, probability=1.0)
fallback.add_transition(5, 5, probability=1.0)

fallback.validate()

result = check_reachability(fallback, target_labels={"success"})
print(f"P(success with fallback chain) = {result.probability:.6f}")
# Expected: 0.95 + 0.05*0.90 + 0.05*0.10*0.80 = 0.999

failure = check_reachability(fallback, target_labels={"total_failure"})
print(f"P(total failure) = {failure.probability:.6f}")
# Expected: 0.05*0.10*0.20 = 0.001

steps = expected_steps(fallback, target_labels={"success"})
print(f"E[steps to success] = {steps.expected:.4f}")

# =============================================================
# Model 3: Consensus convergence
# =============================================================
print("\n=== Consensus Convergence ===")

# 3-agent gossip-based consensus: at each round, with some probability
# agents converge or stay diverged
consensus = DTMC()
consensus.add_state(0, labels={"round_0", "diverged"})
consensus.add_state(1, labels={"round_1", "diverged"})
consensus.add_state(2, labels={"round_2", "diverged"})
consensus.add_state(3, labels={"converged"})

# Each round: probability of converging increases
consensus.add_transition(0, 1, probability=0.6)  # stay diverged
consensus.add_transition(0, 3, probability=0.4)  # converge
consensus.add_transition(1, 2, probability=0.3)  # stay diverged
consensus.add_transition(1, 3, probability=0.7)  # converge
consensus.add_transition(2, 3, probability=1.0)  # guaranteed by round 3
consensus.add_transition(3, 3, probability=1.0)  # absorbing

consensus.validate()

result = check_reachability(consensus, target_labels={"converged"})
print(f"P(eventually converge) = {result.probability:.4f}")

steps = expected_steps(consensus, target_labels={"converged"})
print(f"E[rounds to converge] = {steps.expected:.4f}")

# Steady-state: everything ends up converged
dist = steady_state(consensus)
print(f"Steady-state distribution:")
for s in sorted(dist.distribution):
    labels = consensus.states[s].labels
    if dist.distribution[s] > 1e-10:
        print(f"  State {s} {labels}: {dist.distribution[s]:.6f}")

# =============================================================
# Model 4: Ergodic agent behavior
# =============================================================
print("\n=== Ergodic Agent Cycle ===")

# Agent cycles through states indefinitely
cycle = DTMC()
cycle.add_state(0, labels={"thinking"})
cycle.add_state(1, labels={"acting"})
cycle.add_state(2, labels={"observing"})

cycle.add_transition(0, 1, probability=0.8)
cycle.add_transition(0, 2, probability=0.2)
cycle.add_transition(1, 2, probability=1.0)
cycle.add_transition(2, 0, probability=1.0)

cycle.validate()

dist = steady_state(cycle)
print("Long-run time allocation:")
for s in sorted(dist.distribution):
    labels = cycle.states[s].labels
    print(f"  {labels}: {dist.distribution[s]*100:.1f}%")

print("\nDone.")
