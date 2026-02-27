"""Process equivalence checking for CSP.

This module provides:
- Trace equivalence: P =_T Q iff traces(P) = traces(Q)
- Strong bisimulation: P ∼ Q (finest equivalence)
- Weak bisimulation: P ≈ Q (observational equivalence)
- Failures equivalence: P =_F Q (CSP standard)

Equivalence hierarchy (finest to coarsest)::

    Strong bisimulation (∼)
        ⊂ Weak bisimulation (≈)
            ⊂ Failures equivalence (=_F)
                ⊂ Trace equivalence (=_T)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .csp import TAU, TICK, Event, Process
from .semantics import LTS, LTSBuilder, Trace, _tau_closure, traces

# =============================================================================
# Equivalence Results
# =============================================================================


@dataclass
class EquivalenceResult:
    """Result of an equivalence check."""

    equivalent: bool
    witness: Trace | None = None  # Distinguishing trace if not equivalent
    relation: set[tuple[int, int]] | None = None  # Bisimulation relation if equivalent

    @property
    def is_equivalent(self) -> bool:
        return self.equivalent


# =============================================================================
# Trace Equivalence
# =============================================================================


def trace_equivalent(
    p: Process | LTS,
    q: Process | LTS,
    max_trace_length: int = 100,
) -> EquivalenceResult:
    """Check trace equivalence: traces(P) = traces(Q).

    Two processes are trace equivalent if they can perform
    exactly the same sequences of visible events.

    Args:
        p: First process or LTS
        q: Second process or LTS
        max_trace_length: Maximum trace length to consider

    Returns:
        EquivalenceResult with equivalence status
    """
    lts_p = _to_lts(p)
    lts_q = _to_lts(q)

    # Collect traces from both
    traces_p = set(traces(lts_p, max_length=max_trace_length))
    traces_q = set(traces(lts_q, max_length=max_trace_length))

    if traces_p == traces_q:
        return EquivalenceResult(equivalent=True)

    # Find distinguishing trace
    diff_p = traces_p - traces_q
    diff_q = traces_q - traces_p

    witness = None
    if diff_p:
        witness = min(diff_p, key=len)  # Shortest distinguishing trace
    elif diff_q:
        witness = min(diff_q, key=len)

    return EquivalenceResult(equivalent=False, witness=witness)


# =============================================================================
# Strong Bisimulation
# =============================================================================


def strong_bisimilar(
    p: Process | LTS,
    q: Process | LTS,
) -> EquivalenceResult:
    """Check strong bisimulation: P ∼ Q.

    Strong bisimulation is the finest process equivalence.
    P ∼ Q iff there exists a relation R such that:
    - (P, Q) ∈ R
    - If (P', Q') ∈ R and P' ─a→ P'', then Q' ─a→ Q'' and (P'', Q'') ∈ R
    - Symmetrically for Q'

    Args:
        p: First process or LTS
        q: Second process or LTS

    Returns:
        EquivalenceResult with bisimulation relation if equivalent
    """
    lts_p = _to_lts(p)
    lts_q = _to_lts(q)

    # Offset Q's state IDs to avoid collision
    offset = lts_p.num_states

    # Build partition refinement
    # Initial partition: all states in one block
    partition = _initial_partition(lts_p, lts_q, offset)

    # Refine until stable
    changed = True
    while changed:
        changed = False
        new_partition: list[set[int]] = []

        for block in partition:
            # Split block based on transition signatures
            splits = _split_block(block, partition, lts_p, lts_q, offset)
            if len(splits) > 1:
                changed = True
            new_partition.extend(splits)

        partition = new_partition

    # Check if initial states are in the same block
    initial_p = lts_p.initial_state
    initial_q = lts_q.initial_state + offset

    for block in partition:
        if initial_p in block and initial_q in block:
            # Build bisimulation relation
            relation = _extract_relation(partition, offset, lts_q.num_states)
            return EquivalenceResult(equivalent=True, relation=relation)

    # Find distinguishing trace
    witness = _find_distinguishing_trace(lts_p, lts_q, partition, offset)
    return EquivalenceResult(equivalent=False, witness=witness)


def _initial_partition(lts_p: LTS, lts_q: LTS, offset: int) -> list[set[int]]:
    """Create initial partition with all states."""
    all_states: set[int] = set()

    for state_id in lts_p.states:
        all_states.add(state_id)
    for state_id in lts_q.states:
        all_states.add(state_id + offset)

    return [all_states]


def _split_block(
    block: set[int],
    partition: list[set[int]],
    lts_p: LTS,
    lts_q: LTS,
    offset: int,
) -> list[set[int]]:
    """Split a block based on transition signatures."""
    signatures: dict[tuple[tuple[Event, int], ...], set[int]] = {}

    for state_id in block:
        sig = _compute_signature(state_id, partition, lts_p, lts_q, offset)
        if sig not in signatures:
            signatures[sig] = set()
        signatures[sig].add(state_id)

    return list(signatures.values())


def _compute_signature(
    state_id: int,
    partition: list[set[int]],
    lts_p: LTS,
    lts_q: LTS,
    offset: int,
) -> tuple[tuple[Event, int], ...]:
    """Compute transition signature for a state."""
    # Get transitions
    if state_id < offset:
        transitions = list(lts_p.successors(state_id))
    else:
        transitions = [(e, t + offset) for e, t in lts_q.successors(state_id - offset)]

    # Signature: frozenset of (event, block_index) pairs
    sig_items = []
    for event, target in transitions:
        for i, block in enumerate(partition):
            if target in block:
                sig_items.append((event, i))
                break

    return tuple(sorted(sig_items))


def _extract_relation(
    partition: list[set[int]],
    offset: int,
    q_size: int,
) -> set[tuple[int, int]]:
    """Extract bisimulation relation from partition."""
    relation: set[tuple[int, int]] = set()

    for block in partition:
        p_states = [s for s in block if s < offset]
        q_states = [s - offset for s in block if s >= offset]

        for p in p_states:
            for q in q_states:
                relation.add((p, q))

    return relation


def _find_distinguishing_trace(
    lts_p: LTS,
    lts_q: LTS,
    partition: list[set[int]],
    offset: int,
) -> Trace | None:
    """Find a trace that distinguishes the two processes."""
    # BFS to find distinguishing trace
    queue: deque[tuple[int, int, Trace]] = deque()
    queue.append((lts_p.initial_state, lts_q.initial_state, ()))
    seen: set[tuple[int, int]] = set()

    while queue:
        state_p, state_q, trace = queue.popleft()

        if (state_p, state_q) in seen:
            continue
        seen.add((state_p, state_q))

        # Get events from both states
        events_p = {e for e, _ in lts_p.successors(state_p)}
        events_q = {e for e, _ in lts_q.successors(state_q)}

        # Check for distinguishing event
        if events_p != events_q:
            diff = events_p.symmetric_difference(events_q)
            return trace + (min(diff),)

        # Explore further
        for event in events_p:
            targets_p = [t for e, t in lts_p.successors(state_p) if e == event]
            targets_q = [t for e, t in lts_q.successors(state_q) if e == event]

            for tp in targets_p:
                for tq in targets_q:
                    queue.append((tp, tq, trace + (event,)))

    return None


# =============================================================================
# Weak Bisimulation
# =============================================================================


def weak_bisimilar(
    p: Process | LTS,
    q: Process | LTS,
) -> EquivalenceResult:
    """Check weak bisimulation: P ≈ Q.

    Weak bisimulation ignores τ (internal) actions.
    P ≈ Q iff there exists a relation R such that:
    - (P, Q) ∈ R
    - If (P', Q') ∈ R and P' ─a→ P'', then Q' ═a⇒ Q'' and (P'', Q'') ∈ R
      where ═a⇒ means τ*·a·τ* (zero or more τ, then a, then zero or more τ)
    - Symmetrically for Q'

    Args:
        p: First process or LTS
        q: Second process or LTS

    Returns:
        EquivalenceResult with bisimulation relation if equivalent
    """
    lts_p = _to_lts(p)
    lts_q = _to_lts(q)

    offset = lts_p.num_states

    # Compute weak transitions (τ-closures)
    weak_trans_p = _compute_weak_transitions(lts_p)
    weak_trans_q = _compute_weak_transitions(lts_q)

    # Build partition refinement with weak transitions
    partition = _initial_partition(lts_p, lts_q, offset)

    changed = True
    while changed:
        changed = False
        new_partition: list[set[int]] = []

        for block in partition:
            splits = _split_block_weak(block, partition, weak_trans_p, weak_trans_q, offset)
            if len(splits) > 1:
                changed = True
            new_partition.extend(splits)

        partition = new_partition

    # Check if initial states are in the same block (with τ-closure)
    initial_p_closure = _tau_closure(lts_p, lts_p.initial_state)
    initial_q_closure = {s + offset for s in _tau_closure(lts_q, lts_q.initial_state)}

    for block in partition:
        if initial_p_closure & block and initial_q_closure & block:
            relation = _extract_relation(partition, offset, lts_q.num_states)
            return EquivalenceResult(equivalent=True, relation=relation)

    return EquivalenceResult(equivalent=False)


def _compute_weak_transitions(
    lts: LTS,
) -> dict[int, dict[Event, set[int]]]:
    """Compute weak transitions (τ*-a-τ*) for each state."""
    weak_trans: dict[int, dict[Event, set[int]]] = {}

    for state_id in lts.states:
        weak_trans[state_id] = {}

        # τ-closure of current state
        tau_before = _tau_closure(lts, state_id)

        for start in tau_before:
            for event, mid in lts.successors(start):
                if event == TAU:
                    continue

                # τ-closure after the event
                tau_after = _tau_closure(lts, mid)

                if event not in weak_trans[state_id]:
                    weak_trans[state_id][event] = set()
                weak_trans[state_id][event].update(tau_after)

    return weak_trans


def _split_block_weak(
    block: set[int],
    partition: list[set[int]],
    weak_p: dict[int, dict[Event, set[int]]],
    weak_q: dict[int, dict[Event, set[int]]],
    offset: int,
) -> list[set[int]]:
    """Split block using weak transition signatures."""
    signatures: dict[tuple[tuple[Event, int], ...], set[int]] = {}

    for state_id in block:
        sig = _compute_weak_signature(state_id, partition, weak_p, weak_q, offset)
        if sig not in signatures:
            signatures[sig] = set()
        signatures[sig].add(state_id)

    return list(signatures.values())


def _compute_weak_signature(
    state_id: int,
    partition: list[set[int]],
    weak_p: dict[int, dict[Event, set[int]]],
    weak_q: dict[int, dict[Event, set[int]]],
    offset: int,
) -> tuple[tuple[Event, int], ...]:
    """Compute weak transition signature."""
    if state_id < offset:
        weak_trans = weak_p.get(state_id, {})
    else:
        orig_trans = weak_q.get(state_id - offset, {})
        weak_trans = {e: {t + offset for t in ts} for e, ts in orig_trans.items()}

    sig_items = []
    for event, targets in weak_trans.items():
        for target in targets:
            for i, block in enumerate(partition):
                if target in block:
                    sig_items.append((event, i))
                    break

    return tuple(sorted(set(sig_items)))


# =============================================================================
# Failures Equivalence
# =============================================================================


@dataclass(frozen=True)
class Failure:
    """A failure: (trace, refusal set)."""

    trace: Trace
    refusals: frozenset[Event]


def failures(lts: LTS, max_trace_length: int = 50) -> Iterator[Failure]:
    """Generate failures of an LTS.

    A failure is a pair (s, X) where:
    - s is a trace the process can perform
    - X is a set of events the process can refuse after s

    Args:
        lts: The LTS to analyze
        max_trace_length: Maximum trace length

    Yields:
        Failure objects
    """
    queue: deque[tuple[int, Trace]] = deque()
    queue.append((lts.initial_state, ()))
    seen: set[tuple[int, Trace]] = set()

    while queue:
        state_id, trace = queue.popleft()

        if (state_id, trace) in seen:
            continue
        seen.add((state_id, trace))

        # Compute refusal set (events not in initials)
        available = {e for e, _ in lts.successors(state_id)}
        refusals = lts.alphabet - available - {TAU, TICK}

        if refusals:
            yield Failure(trace=trace, refusals=frozenset(refusals))

        if len(trace) >= max_trace_length:
            continue

        for event, target in lts.successors(state_id):
            if event == TAU:
                queue.append((target, trace))
            else:
                queue.append((target, trace + (event,)))


def failures_equivalent(
    p: Process | LTS,
    q: Process | LTS,
    max_trace_length: int = 50,
) -> EquivalenceResult:
    """Check failures equivalence: P =_F Q.

    Two processes are failures equivalent if they have
    exactly the same failures.

    Args:
        p: First process or LTS
        q: Second process or LTS
        max_trace_length: Maximum trace length

    Returns:
        EquivalenceResult with equivalence status
    """
    lts_p = _to_lts(p)
    lts_q = _to_lts(q)

    failures_p = set(failures(lts_p, max_trace_length))
    failures_q = set(failures(lts_q, max_trace_length))

    if failures_p == failures_q:
        return EquivalenceResult(equivalent=True)

    # Find distinguishing failure
    diff = failures_p.symmetric_difference(failures_q)
    if diff:
        witness_failure = min(diff, key=lambda f: len(f.trace))
        return EquivalenceResult(equivalent=False, witness=witness_failure.trace)

    return EquivalenceResult(equivalent=False)


# =============================================================================
# Helper Functions
# =============================================================================


def _to_lts(p: Process | LTS) -> LTS:
    """Convert process to LTS if needed."""
    if isinstance(p, LTS):
        return p
    return LTSBuilder().build(p)


# =============================================================================
# Convenience Functions
# =============================================================================


def are_equivalent(
    p: Process,
    q: Process,
    equivalence: str = "trace",
) -> bool:
    """Check if two processes are equivalent.

    Args:
        p: First process
        q: Second process
        equivalence: Type of equivalence:
            - "trace": Trace equivalence
            - "strong": Strong bisimulation
            - "weak": Weak bisimulation
            - "failures": Failures equivalence

    Returns:
        True if processes are equivalent
    """
    checkers: dict[str, Callable[[Process, Process], EquivalenceResult]] = {
        "trace": trace_equivalent,
        "strong": strong_bisimilar,
        "weak": weak_bisimilar,
        "failures": failures_equivalent,
    }

    checker = checkers.get(equivalence)
    if checker is None:
        raise ValueError(f"Unknown equivalence type: {equivalence}")

    return checker(p, q).is_equivalent


__all__ = [
    # Result types
    "EquivalenceResult",
    "Failure",
    # Equivalence checks
    "trace_equivalent",
    "strong_bisimilar",
    "weak_bisimilar",
    "failures",
    "failures_equivalent",
    # Convenience
    "are_equivalent",
]
