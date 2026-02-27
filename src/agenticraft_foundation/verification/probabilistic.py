"""Probabilistic verification for Discrete-Time Markov Chains (DTMC).

This module provides:
- DTMC representation with labeled states and probabilistic transitions
- Reachability probability computation (linear system solving)
- Steady-state distribution (power iteration)
- Expected steps to reach target states
- DTMC validation (probability sum = 1.0 per state)

Theory: Hansson & Jonsson (1994), PCTL model checking.
Algorithms from Baier & Katoen, "Principles of Model Checking", Chapter 10.

Key insight for LLM agents: agents are inherently stochastic. The LTS/CSP
framework handles non-determinism (what CAN happen), but DTMC handles
probability (how LIKELY things are). This module bridges the gap.

Implementation uses pure Python (no numpy/scipy dependency):
- Gaussian elimination for small systems (< 100 states)
- Value iteration fallback for larger systems
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

# =============================================================================
# Core Types
# =============================================================================

_EPSILON = 1e-10
"""Tolerance for floating-point comparisons."""

_MAX_ITERATIONS = 10000
"""Maximum iterations for value iteration / power iteration."""

_CONVERGENCE_THRESHOLD = 1e-12
"""Convergence threshold for iterative methods."""

_PROBABILITY_SUM_TOLERANCE = 1e-6
"""Tolerance for checking probability sums equal 1.0."""

_GAUSSIAN_ELIMINATION_THRESHOLD = 100
"""Use Gaussian elimination for systems smaller than this."""


@dataclass(frozen=True)
class ProbabilisticTransition:
    """A probabilistic transition: source ─p→ target.

    Attributes:
        source: Source state ID.
        target: Target state ID.
        probability: Transition probability (0 < p <= 1).
    """

    source: int
    target: int
    probability: float

    def __repr__(self) -> str:
        return f"{self.source} --{self.probability:.4f}--> {self.target}"


@dataclass
class DTMCState:
    """A state in a DTMC.

    Attributes:
        id: State identifier.
        labels: Set of atomic propositions holding in this state.
    """

    id: int
    labels: set[str] = field(default_factory=set)

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DTMCState):
            return self.id == other.id
        return False


@dataclass
class DTMC:
    """Discrete-Time Markov Chain.

    A DTMC is a tuple (S, P, ι, AP, L) where:
    - S: finite set of states
    - P: S × S → [0,1] transition probability matrix
    - ι: initial distribution (here: single initial state)
    - AP: set of atomic propositions
    - L: S → 2^AP labeling function

    Attributes:
        states: Mapping from state IDs to DTMCState objects.
        transitions: List of probabilistic transitions.
        initial_state: ID of the initial state.
    """

    states: dict[int, DTMCState] = field(default_factory=dict)
    transitions: list[ProbabilisticTransition] = field(default_factory=list)
    initial_state: int = 0

    def add_state(self, state_id: int, labels: set[str] | None = None) -> DTMCState:
        """Add a state to the DTMC.

        Args:
            state_id: Unique state identifier.
            labels: Set of atomic propositions for this state.

        Returns:
            The created DTMCState.
        """
        state = DTMCState(id=state_id, labels=labels or set())
        self.states[state_id] = state
        return state

    def add_transition(
        self,
        source: int,
        target: int,
        probability: float,
    ) -> ProbabilisticTransition:
        """Add a probabilistic transition.

        Args:
            source: Source state ID.
            target: Target state ID.
            probability: Transition probability (must be in (0, 1]).

        Returns:
            The created ProbabilisticTransition.

        Raises:
            ValueError: If probability is not in (0, 1] or states don't exist.
        """
        if probability <= 0 or probability > 1.0 + _EPSILON:
            raise ValueError(f"Probability must be in (0, 1], got {probability}")
        if source not in self.states:
            raise ValueError(f"Source state {source} does not exist")
        if target not in self.states:
            raise ValueError(f"Target state {target} does not exist")

        trans = ProbabilisticTransition(
            source=source, target=target, probability=probability
        )
        self.transitions.append(trans)
        return trans

    def get_transitions_from(self, state_id: int) -> list[ProbabilisticTransition]:
        """Get all transitions from a state.

        Args:
            state_id: The source state.

        Returns:
            List of transitions from the state.
        """
        return [t for t in self.transitions if t.source == state_id]

    def get_transition_probability(self, source: int, target: int) -> float:
        """Get the probability of transitioning from source to target.

        Args:
            source: Source state ID.
            target: Target state ID.

        Returns:
            Transition probability, or 0.0 if no transition exists.
        """
        for t in self.transitions:
            if t.source == source and t.target == target:
                return t.probability
        return 0.0

    def successors(self, state_id: int) -> list[tuple[int, float]]:
        """Get successor states with their probabilities.

        Args:
            state_id: The source state.

        Returns:
            List of (target_state, probability) tuples.
        """
        return [(t.target, t.probability) for t in self.get_transitions_from(state_id)]

    def states_with_label(self, label: str) -> set[int]:
        """Get all states with a given label.

        Args:
            label: The atomic proposition to search for.

        Returns:
            Set of state IDs with the given label.
        """
        return {s.id for s in self.states.values() if label in s.labels}

    def states_with_labels(self, labels: set[str]) -> set[int]:
        """Get all states with any of the given labels.

        Args:
            labels: Set of atomic propositions.

        Returns:
            Set of state IDs with at least one of the given labels.
        """
        return {s.id for s in self.states.values() if s.labels & labels}

    @property
    def num_states(self) -> int:
        """Number of states."""
        return len(self.states)

    @property
    def num_transitions(self) -> int:
        """Number of transitions."""
        return len(self.transitions)

    def is_absorbing(self, state_id: int) -> bool:
        """Check if a state is absorbing (self-loop with probability 1).

        Args:
            state_id: The state to check.

        Returns:
            True if the state is absorbing.
        """
        transitions = self.get_transitions_from(state_id)
        return (
            len(transitions) == 1
            and transitions[0].target == state_id
            and abs(transitions[0].probability - 1.0) < _EPSILON
        )

    def validate(self) -> None:
        """Validate the DTMC.

        Checks that:
        1. All states have outgoing transitions
        2. Outgoing probabilities sum to 1.0 for each state
        3. All transition targets exist

        Raises:
            ValueError: If the DTMC is malformed.
        """
        for state_id in self.states:
            transitions = self.get_transitions_from(state_id)
            if not transitions:
                raise ValueError(f"State {state_id} has no outgoing transitions")

            prob_sum = sum(t.probability for t in transitions)
            if abs(prob_sum - 1.0) > _PROBABILITY_SUM_TOLERANCE:
                raise ValueError(
                    f"State {state_id}: outgoing probabilities sum to "
                    f"{prob_sum:.6f}, expected 1.0"
                )

            for t in transitions:
                if t.target not in self.states:
                    raise ValueError(
                        f"Transition {t.source} -> {t.target}: "
                        f"target state {t.target} does not exist"
                    )


# =============================================================================
# Results
# =============================================================================


@dataclass
class ProbabilisticResult:
    """Result of a probabilistic verification query.

    Attributes:
        probability: The probability for the initial state.
        per_state: Probability for each state.
        is_certain: True if probability is 1.0.
        is_impossible: True if probability is 0.0.
    """

    probability: float
    per_state: dict[int, float] = field(default_factory=dict)

    @property
    def is_certain(self) -> bool:
        """True if probability is 1.0 (within tolerance)."""
        return abs(self.probability - 1.0) < _EPSILON

    @property
    def is_impossible(self) -> bool:
        """True if probability is 0.0 (within tolerance)."""
        return abs(self.probability) < _EPSILON


@dataclass
class SteadyStateResult:
    """Result of steady-state distribution computation.

    Attributes:
        distribution: Mapping from state ID to steady-state probability.
        converged: Whether the computation converged.
        iterations: Number of iterations used.
    """

    distribution: dict[int, float] = field(default_factory=dict)
    converged: bool = True
    iterations: int = 0


@dataclass
class ExpectedStepsResult:
    """Result of expected steps computation.

    Attributes:
        expected: Expected number of steps from the initial state.
        per_state: Expected steps from each state.
        is_infinite: True if target is unreachable (infinite expected steps).
    """

    expected: float
    per_state: dict[int, float] = field(default_factory=dict)

    @property
    def is_infinite(self) -> bool:
        """True if expected steps is infinite."""
        return math.isinf(self.expected)


# =============================================================================
# Linear System Solving (Pure Python)
# =============================================================================


def _gaussian_elimination(
    a_matrix: list[list[float]],
    b_vector: list[float],
) -> list[float]:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Args:
        a_matrix: Coefficient matrix (n × n), modified in place.
        b_vector: Right-hand side vector (n), modified in place.

    Returns:
        Solution vector x.

    Raises:
        ValueError: If the system is singular.
    """
    n = len(b_vector)

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(a_matrix[col][col])
        for row in range(col + 1, n):
            if abs(a_matrix[row][col]) > max_val:
                max_val = abs(a_matrix[row][col])
                max_row = row

        # Swap rows
        if max_row != col:
            a_matrix[col], a_matrix[max_row] = a_matrix[max_row], a_matrix[col]
            b_vector[col], b_vector[max_row] = b_vector[max_row], b_vector[col]

        pivot = a_matrix[col][col]
        if abs(pivot) < _EPSILON:
            # Singular — set this variable to 0
            continue

        # Eliminate below
        for row in range(col + 1, n):
            factor = a_matrix[row][col] / pivot
            for j in range(col, n):
                a_matrix[row][j] -= factor * a_matrix[col][j]
            b_vector[row] -= factor * b_vector[col]

    # Back substitution
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        if abs(a_matrix[row][row]) < _EPSILON:
            x[row] = 0.0
            continue
        x[row] = b_vector[row]
        for col in range(row + 1, n):
            x[row] -= a_matrix[row][col] * x[col]
        x[row] /= a_matrix[row][row]

    return x


def _value_iteration(
    dtmc: DTMC,
    target_states: set[int],
    zero_states: set[int],
    unknown_states: list[int],
    max_iterations: int = _MAX_ITERATIONS,
) -> dict[int, float]:
    """Solve reachability probabilities via value iteration.

    Args:
        dtmc: The DTMC.
        target_states: States with probability 1.
        zero_states: States with probability 0.
        unknown_states: States to solve for.
        max_iterations: Maximum iterations.

    Returns:
        Mapping from state ID to reachability probability.
    """
    prob: dict[int, float] = {}
    for s in dtmc.states:
        if s in target_states:
            prob[s] = 1.0
        elif s in zero_states:
            prob[s] = 0.0
        else:
            prob[s] = 0.0  # Initial guess

    for _ in range(max_iterations):
        max_diff = 0.0
        for s in unknown_states:
            new_val = sum(
                p * prob.get(t, 0.0) for t, p in dtmc.successors(s)
            )
            diff = abs(new_val - prob[s])
            if diff > max_diff:
                max_diff = diff
            prob[s] = new_val

        if max_diff < _CONVERGENCE_THRESHOLD:
            break

    return prob


# =============================================================================
# Reachability Analysis
# =============================================================================


def _compute_reachable(dtmc: DTMC, start: int) -> set[int]:
    """Compute all states reachable from a starting state.

    Args:
        dtmc: The DTMC.
        start: Starting state ID.

    Returns:
        Set of reachable state IDs.
    """
    reachable: set[int] = set()
    queue = deque([start])

    while queue:
        state_id = queue.popleft()
        if state_id in reachable:
            continue
        reachable.add(state_id)
        for target, _ in dtmc.successors(state_id):
            if target not in reachable:
                queue.append(target)

    return reachable


def _can_reach_any(dtmc: DTMC, start: int, targets: set[int]) -> bool:
    """Check if any target state is reachable from start.

    Args:
        dtmc: The DTMC.
        start: Starting state ID.
        targets: Target state IDs.

    Returns:
        True if at least one target is reachable.
    """
    reachable = _compute_reachable(dtmc, start)
    return bool(reachable & targets)


def check_reachability(
    dtmc: DTMC,
    target_labels: set[str] | None = None,
    target_states: set[int] | None = None,
) -> ProbabilisticResult:
    """Compute reachability probabilities for target states.

    Computes Pr(◇ target) — the probability of eventually reaching a
    state with the given labels, from each state.

    The algorithm:
    1. Partition states into:
       - target states: prob = 1.0
       - states that can't reach target: prob = 0.0
       - remaining: solve linear system p(s) = Σ P(s,s') × p(s')
    2. Solve via Gaussian elimination (small) or value iteration (large).

    Args:
        dtmc: The DTMC to analyze.
        target_labels: Labels identifying target states (at least one must match).
        target_states: Explicit set of target state IDs. If both are given,
            the union is used.

    Returns:
        ProbabilisticResult with probabilities per state.

    Raises:
        ValueError: If neither target_labels nor target_states is provided.
    """
    if target_labels is None and target_states is None:
        raise ValueError("Must specify target_labels or target_states")

    # Identify target states
    targets: set[int] = set()
    if target_labels:
        targets |= dtmc.states_with_labels(target_labels)
    if target_states:
        targets |= target_states

    if not targets:
        # No target states — probability is 0 everywhere
        per_state = {s: 0.0 for s in dtmc.states}
        return ProbabilisticResult(
            probability=0.0,
            per_state=per_state,
        )

    # Partition states
    # States that cannot reach target have prob = 0
    zero_states: set[int] = set()
    for s in dtmc.states:
        if s not in targets and not _can_reach_any(dtmc, s, targets):
            zero_states.add(s)

    # Unknown states need solving
    unknown = [s for s in dtmc.states if s not in targets and s not in zero_states]

    if not unknown:
        # All states are either target or zero
        per_state = {}
        for s in dtmc.states:
            per_state[s] = 1.0 if s in targets else 0.0
        return ProbabilisticResult(
            probability=per_state.get(dtmc.initial_state, 0.0),
            per_state=per_state,
        )

    # Solve the linear system
    if len(unknown) <= _GAUSSIAN_ELIMINATION_THRESHOLD:
        per_state = _solve_reachability_gaussian(dtmc, targets, zero_states, unknown)
    else:
        per_state = _value_iteration(dtmc, targets, zero_states, unknown)

    return ProbabilisticResult(
        probability=per_state.get(dtmc.initial_state, 0.0),
        per_state=per_state,
    )


def _solve_reachability_gaussian(
    dtmc: DTMC,
    target_states: set[int],
    zero_states: set[int],
    unknown_states: list[int],
) -> dict[int, float]:
    """Solve reachability via Gaussian elimination.

    System: p(s) = Σ_{s'} P(s,s') × p(s')
    Rearranged: p(s) - Σ_{s' in unknown} P(s,s') × p(s') = Σ_{s' in target} P(s,s')

    Args:
        dtmc: The DTMC.
        target_states: States with prob = 1.
        zero_states: States with prob = 0.
        unknown_states: States to solve for.

    Returns:
        Mapping from state ID to probability.
    """
    n = len(unknown_states)
    state_to_idx = {s: i for i, s in enumerate(unknown_states)}

    # Build system: (I - P_unknown) × x = b
    a_matrix = [[0.0] * n for _ in range(n)]
    b_vector = [0.0] * n

    for i, s in enumerate(unknown_states):
        a_matrix[i][i] = 1.0  # Identity diagonal
        for target, prob in dtmc.successors(s):
            if target in state_to_idx:
                j = state_to_idx[target]
                a_matrix[i][j] -= prob
            elif target in target_states:
                b_vector[i] += prob
            # If target in zero_states, contributes 0 — skip

    solution = _gaussian_elimination(a_matrix, b_vector)

    # Build full result
    per_state: dict[int, float] = {}
    for s in dtmc.states:
        if s in target_states:
            per_state[s] = 1.0
        elif s in zero_states:
            per_state[s] = 0.0
        else:
            idx = state_to_idx[s]
            per_state[s] = max(0.0, min(1.0, solution[idx]))  # Clamp to [0,1]

    return per_state


# =============================================================================
# Steady-State Distribution
# =============================================================================


def steady_state(
    dtmc: DTMC,
    max_iterations: int = _MAX_ITERATIONS,
) -> SteadyStateResult:
    """Compute the steady-state (stationary) distribution.

    For ergodic chains, computes π such that π = πP.
    For absorbing chains, computes the absorption probabilities.

    Uses power iteration: π_{k+1} = π_k × P.

    Args:
        dtmc: The DTMC.
        max_iterations: Maximum iterations for convergence.

    Returns:
        SteadyStateResult with distribution and convergence info.
    """
    n = dtmc.num_states
    state_ids = sorted(dtmc.states.keys())
    state_to_idx = {s: i for i, s in enumerate(state_ids)}

    # Initial distribution: all mass on initial state
    dist = [0.0] * n
    dist[state_to_idx[dtmc.initial_state]] = 1.0

    converged = False
    iterations = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1
        new_dist = [0.0] * n

        for i, s in enumerate(state_ids):
            for target, prob in dtmc.successors(s):
                j = state_to_idx[target]
                new_dist[j] += dist[i] * prob

        # Check convergence
        max_diff = max(abs(new_dist[i] - dist[i]) for i in range(n))
        dist = new_dist

        if max_diff < _CONVERGENCE_THRESHOLD:
            converged = True
            break

    # Build result
    distribution = {state_ids[i]: dist[i] for i in range(n)}

    return SteadyStateResult(
        distribution=distribution,
        converged=converged,
        iterations=iterations,
    )


# =============================================================================
# Expected Steps
# =============================================================================


def expected_steps(
    dtmc: DTMC,
    target_labels: set[str] | None = None,
    target_states: set[int] | None = None,
) -> ExpectedStepsResult:
    """Compute expected number of steps to reach target states.

    Solves: e(s) = 1 + Σ P(s,s') × e(s') for non-target states.
    Target states: e(s) = 0.
    States that can't reach target: e(s) = ∞.

    Args:
        dtmc: The DTMC.
        target_labels: Labels identifying target states.
        target_states: Explicit set of target state IDs.

    Returns:
        ExpectedStepsResult with expected steps per state.

    Raises:
        ValueError: If neither target_labels nor target_states is provided.
    """
    if target_labels is None and target_states is None:
        raise ValueError("Must specify target_labels or target_states")

    # Identify target states
    targets: set[int] = set()
    if target_labels:
        targets |= dtmc.states_with_labels(target_labels)
    if target_states:
        targets |= target_states

    if not targets:
        per_state = {s: float("inf") for s in dtmc.states}
        return ExpectedStepsResult(
            expected=float("inf"),
            per_state=per_state,
        )

    # Partition: target (e=0), unreachable (e=∞), unknown (solve)
    inf_states: set[int] = set()
    for s in dtmc.states:
        if s not in targets and not _can_reach_any(dtmc, s, targets):
            inf_states.add(s)

    unknown = [s for s in dtmc.states if s not in targets and s not in inf_states]

    if not unknown:
        all_expected: dict[int, float] = {}
        for s in dtmc.states:
            if s in targets:
                all_expected[s] = 0.0
            else:
                all_expected[s] = float("inf")
        return ExpectedStepsResult(
            expected=all_expected.get(dtmc.initial_state, float("inf")),
            per_state=all_expected,
        )

    # Solve: e(s) - Σ_{s' in unknown} P(s,s') × e(s') = 1 + Σ_{s' in inf} P(s,s') × ∞
    # Since ∞ terms make it unsolvable, we only include reachable-to-target states
    n = len(unknown)
    state_to_idx = {s: i for i, s in enumerate(unknown)}

    a_matrix = [[0.0] * n for _ in range(n)]
    b_vector = [1.0] * n  # e(s) = 1 + ...

    for i, s in enumerate(unknown):
        a_matrix[i][i] = 1.0
        for target, prob in dtmc.successors(s):
            if target in state_to_idx:
                j = state_to_idx[target]
                a_matrix[i][j] -= prob
            # target states contribute 0 (e(target) = 0)
            # inf states: if we reach inf state, expected is ∞
            # But we already filtered — unknown states can reach target

    solution = _gaussian_elimination(a_matrix, b_vector)

    per_state_result: dict[int, float] = {}
    for s in dtmc.states:
        if s in targets:
            per_state_result[s] = 0.0
        elif s in inf_states:
            per_state_result[s] = float("inf")
        else:
            idx = state_to_idx[s]
            per_state_result[s] = max(0.0, solution[idx])

    return ExpectedStepsResult(
        expected=per_state_result.get(dtmc.initial_state, float("inf")),
        per_state=per_state_result,
    )


# =============================================================================
# DTMC Builder
# =============================================================================


def build_dtmc_from_lts(
    lts_states: dict[int, set[str]],
    lts_transitions: dict[int, list[tuple[int, float]]],
    initial_state: int = 0,
) -> DTMC:
    """Build a DTMC from explicit state/transition descriptions.

    Convenience function for creating DTMCs from dictionaries.

    Args:
        lts_states: Mapping from state ID to label sets.
        lts_transitions: Mapping from state ID to list of (target, probability) tuples.
        initial_state: The initial state ID.

    Returns:
        A validated DTMC.
    """
    dtmc = DTMC(initial_state=initial_state)

    for state_id, labels in lts_states.items():
        dtmc.add_state(state_id, labels)

    for source, targets in lts_transitions.items():
        for target, prob in targets:
            dtmc.add_transition(source, target, prob)

    dtmc.validate()
    return dtmc


__all__ = [
    # Core types
    "ProbabilisticTransition",
    "DTMCState",
    "DTMC",
    # Results
    "ProbabilisticResult",
    "SteadyStateResult",
    "ExpectedStepsResult",
    # Functions
    "check_reachability",
    "steady_state",
    "expected_steps",
    # Builder
    "build_dtmc_from_lts",
]
