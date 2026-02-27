"""Structured counterexample generation for refinement and equivalence failures.

This module provides:
- Annotated trace explanations showing exactly where and why verification fails
- Divergence point identification in refinement failures
- Distinguishing event analysis for equivalence failures

Theory: Synchronized product of spec/impl LTS + BFS divergence search
(Roscoe, 1994 - Theory and Practice of Concurrency).

Given a refinement or equivalence failure with a raw counterexample trace,
this module walks the trace through both LTS step-by-step, recording what
each side can do at each point, and identifies the exact divergence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from agenticraft_foundation.algebra.csp import TAU, Event, Process
from agenticraft_foundation.algebra.equivalence import EquivalenceResult, Failure
from agenticraft_foundation.algebra.refinement import RefinementResult
from agenticraft_foundation.algebra.semantics import LTS, LTSBuilder, Trace, _tau_closure

# =============================================================================
# Types
# =============================================================================


class StepStatus(Enum):
    """Status of a single step in an annotated trace."""

    OK = auto()
    VIOLATION = auto()


@dataclass(frozen=True)
class AnnotatedStep:
    """A single step in an annotated trace.

    Attributes:
        event: The event performed at this step.
        status: Whether this step was OK or a VIOLATION.
        spec_available: Events available in the spec LTS at this point.
        impl_available: Events available in the impl LTS at this point.
        spec_states: Set of spec state IDs after tau-closure at this point.
        impl_states: Set of impl state IDs after tau-closure at this point.
    """

    event: Event
    status: StepStatus
    spec_available: frozenset[Event]
    impl_available: frozenset[Event]
    spec_states: frozenset[int] = field(default_factory=frozenset)
    impl_states: frozenset[int] = field(default_factory=frozenset)


@dataclass
class CounterexampleExplanation:
    """Structured explanation of why a verification check failed.

    Provides a human-readable summary, an annotated trace showing each step's
    status, and the precise divergence point where spec and impl disagree.

    Attributes:
        summary: Human-readable summary of the failure.
        annotated_trace: List of annotated steps showing the trace walkthrough.
        divergence_point: Index in the trace where divergence occurs, or -1 if
            the trace itself is the issue (e.g., empty trace divergence).
        spec_allowed: Events the spec allows at the divergence point.
        impl_attempted: The event the impl attempted (or None for refusal issues).
        failure_kind: The kind of failure ("trace", "failures", "equivalence").
        raw_counterexample: The original raw counterexample from the checker.
    """

    summary: str
    annotated_trace: list[AnnotatedStep]
    divergence_point: int
    spec_allowed: frozenset[Event]
    impl_attempted: Event | None
    failure_kind: str = ""
    raw_counterexample: Trace | Failure | None = None


# =============================================================================
# Internal Helpers
# =============================================================================


def _to_lts(p: Process | LTS) -> LTS:
    """Convert process to LTS if needed."""
    if isinstance(p, LTS):
        return p
    return LTSBuilder().build(p)


def _visible_events_from_states(lts: LTS, state_ids: set[int]) -> frozenset[Event]:
    """Get all visible events available from a set of states (after tau-closure).

    Args:
        lts: The LTS to query.
        state_ids: Set of current state IDs (already tau-closed).

    Returns:
        Frozenset of visible events available.
    """
    events: set[Event] = set()
    for sid in state_ids:
        for event, _ in lts.successors(sid):
            if event != TAU:
                events.add(event)
    return frozenset(events)


def _tau_closed_states(lts: LTS, state_ids: set[int]) -> set[int]:
    """Compute the tau-closure of a set of states.

    Args:
        lts: The LTS.
        state_ids: Starting state IDs.

    Returns:
        Union of tau-closures for all input states.
    """
    result: set[int] = set()
    for sid in state_ids:
        result |= _tau_closure(lts, sid)
    return result


def _advance_states(lts: LTS, state_ids: set[int], event: Event) -> set[int]:
    """Advance a set of states by performing a visible event.

    First computes tau-closure, then follows the event, then computes
    tau-closure again (standard weak transition semantics).

    Args:
        lts: The LTS.
        state_ids: Current state IDs.
        event: The visible event to perform.

    Returns:
        Set of states reachable after performing the event, or empty set
        if the event cannot be performed.
    """
    closed = _tau_closed_states(lts, state_ids)
    next_states: set[int] = set()
    for sid in closed:
        for e, target in lts.successors(sid):
            if e == event:
                next_states.add(target)
    # Tau-close the result
    if next_states:
        next_states = _tau_closed_states(lts, next_states)
    return next_states


def _trace_str(trace: Trace) -> str:
    """Format trace for display."""
    if not trace:
        return "\u27e8\u27e9"
    return f"\u27e8{', '.join(str(e) for e in trace)}\u27e9"


def _walk_lts_with_trace(
    lts: LTS,
    trace: Trace,
) -> list[tuple[set[int], frozenset[Event]]]:
    """Walk an LTS through a trace, recording state sets and available events.

    At each step, records the tau-closed state set and the visible events
    available from those states.

    Args:
        lts: The LTS to walk.
        trace: The trace to follow.

    Returns:
        List of (state_set, available_events) tuples, one per trace prefix.
        The first entry is for the empty prefix (initial states).
        Length is len(trace) + 1.
    """
    result: list[tuple[set[int], frozenset[Event]]] = []
    current = _tau_closed_states(lts, {lts.initial_state})
    available = _visible_events_from_states(lts, current)
    result.append((current, available))

    for event in trace:
        next_states = _advance_states(lts, current, event)
        if not next_states:
            # Can't perform this event — record empty state
            result.append((set(), frozenset()))
            # All subsequent steps will also be empty
            current = set()
        else:
            current = next_states
            available = _visible_events_from_states(lts, current)
            result.append((current, available))

    return result


# =============================================================================
# Refinement Counterexample Explanation
# =============================================================================


def explain_refinement_failure(
    spec: Process | LTS,
    impl: Process | LTS,
    result: RefinementResult,
) -> CounterexampleExplanation:
    """Generate a structured explanation for a refinement failure.

    Takes the raw counterexample from a refinement check and walks it through
    both spec and impl LTS, identifying the exact divergence point.

    Args:
        spec: The specification process or LTS.
        impl: The implementation process or LTS.
        result: The RefinementResult containing the raw counterexample.

    Returns:
        CounterexampleExplanation with annotated trace and divergence details.

    Raises:
        ValueError: If result indicates refinement holds (no failure to explain).
    """
    if result.is_valid:
        raise ValueError("Cannot explain a passing refinement check")

    lts_spec = _to_lts(spec)
    lts_impl = _to_lts(impl)

    # Handle Failure counterexamples (from failures/FD refinement)
    if isinstance(result.counterexample, Failure):
        return _explain_failure_refinement(lts_spec, lts_impl, result.counterexample, result)

    # Handle Trace counterexamples
    trace = result.counterexample if result.counterexample is not None else ()

    return _explain_trace_refinement(lts_spec, lts_impl, trace, result)


def _explain_trace_refinement(
    lts_spec: LTS,
    lts_impl: LTS,
    trace: Trace,
    result: RefinementResult,
) -> CounterexampleExplanation:
    """Explain a trace refinement failure."""
    spec_walk = _walk_lts_with_trace(lts_spec, trace)
    impl_walk = _walk_lts_with_trace(lts_impl, trace)

    annotated: list[AnnotatedStep] = []
    divergence_point = -1
    spec_allowed_at_div = frozenset[Event]()
    impl_attempted_at_div: Event | None = None

    for i, event in enumerate(trace):
        # spec_walk[i] = state before this event, spec_walk[i+1] = state after
        spec_states_before = spec_walk[i][0]
        spec_available_before = spec_walk[i][1]
        impl_states_before = impl_walk[i][0]
        impl_available_before = impl_walk[i][1]

        # Check if spec can perform this event
        spec_after = _advance_states(lts_spec, spec_states_before, event)

        if not spec_after:
            # Spec cannot perform this event — this is the divergence point
            status = StepStatus.VIOLATION
            if divergence_point == -1:
                divergence_point = i
                spec_allowed_at_div = spec_available_before
                impl_attempted_at_div = event
        else:
            status = StepStatus.OK

        annotated.append(
            AnnotatedStep(
                event=event,
                status=status,
                spec_available=spec_available_before,
                impl_available=impl_available_before,
                spec_states=frozenset(spec_states_before),
                impl_states=frozenset(impl_states_before),
            )
        )

    # Build summary
    if divergence_point >= 0:
        prefix = trace[:divergence_point]
        summary = (
            f"Impl violates spec at trace {_trace_str(trace)}: "
            f"After {_trace_str(prefix)}, spec allows "
            f"{{{', '.join(sorted(str(e) for e in spec_allowed_at_div))}}} "
            f"but impl performs {impl_attempted_at_div}"
        )
    else:
        summary = f"Trace {_trace_str(trace)} is not accepted by the specification"

    return CounterexampleExplanation(
        summary=summary,
        annotated_trace=annotated,
        divergence_point=divergence_point,
        spec_allowed=spec_allowed_at_div,
        impl_attempted=impl_attempted_at_div,
        failure_kind="trace",
        raw_counterexample=trace,
    )


def _explain_failure_refinement(
    lts_spec: LTS,
    lts_impl: LTS,
    failure: Failure,
    result: RefinementResult,
) -> CounterexampleExplanation:
    """Explain a failures refinement failure."""
    trace = failure.trace
    spec_walk = _walk_lts_with_trace(lts_spec, trace)
    impl_walk = _walk_lts_with_trace(lts_impl, trace)

    annotated: list[AnnotatedStep] = []
    trace_divergence = -1

    for i, event in enumerate(trace):
        spec_states_before = spec_walk[i][0]
        spec_available_before = spec_walk[i][1]
        impl_states_before = impl_walk[i][0]
        impl_available_before = impl_walk[i][1]

        spec_after = _advance_states(lts_spec, spec_states_before, event)
        status = StepStatus.VIOLATION if not spec_after else StepStatus.OK
        if status == StepStatus.VIOLATION and trace_divergence == -1:
            trace_divergence = i

        annotated.append(
            AnnotatedStep(
                event=event,
                status=status,
                spec_available=spec_available_before,
                impl_available=impl_available_before,
                spec_states=frozenset(spec_states_before),
                impl_states=frozenset(impl_states_before),
            )
        )

    # After the trace, check what spec allows vs what impl refuses
    spec_final = spec_walk[-1] if spec_walk else (set(), frozenset())

    spec_available_after = spec_final[1]

    # The refusal set is what impl cannot do
    refused = failure.refusals

    if trace_divergence >= 0:
        summary = (
            f"Impl trace {_trace_str(trace)} diverges from spec at step {trace_divergence}"
        )
        divergence_point = trace_divergence
        spec_allowed = spec_walk[trace_divergence][1]
        impl_attempted = trace[trace_divergence]
    else:
        refused_str = ", ".join(sorted(str(e) for e in refused))
        summary = (
            f"After trace {_trace_str(trace)}, impl refuses "
            f"{{{refused_str}}} but spec does not refuse all of these"
        )
        divergence_point = len(trace)
        spec_allowed = spec_available_after
        impl_attempted = None

    return CounterexampleExplanation(
        summary=summary,
        annotated_trace=annotated,
        divergence_point=divergence_point,
        spec_allowed=spec_allowed,
        impl_attempted=impl_attempted,
        failure_kind="failures",
        raw_counterexample=failure,
    )


# =============================================================================
# Equivalence Counterexample Explanation
# =============================================================================


def explain_equivalence_failure(
    p: Process | LTS,
    q: Process | LTS,
    result: EquivalenceResult,
) -> CounterexampleExplanation:
    """Generate a structured explanation for an equivalence failure.

    Takes the raw witness trace from an equivalence check and walks it through
    both LTS, identifying where the two processes first disagree.

    Args:
        p: The first process or LTS.
        q: The second process or LTS.
        result: The EquivalenceResult containing the raw witness.

    Returns:
        CounterexampleExplanation with annotated trace and divergence details.

    Raises:
        ValueError: If result indicates equivalence holds (no failure to explain).
    """
    if result.is_equivalent:
        raise ValueError("Cannot explain a passing equivalence check")

    lts_p = _to_lts(p)
    lts_q = _to_lts(q)

    trace = result.witness if result.witness is not None else ()

    p_walk = _walk_lts_with_trace(lts_p, trace)
    q_walk = _walk_lts_with_trace(lts_q, trace)

    annotated: list[AnnotatedStep] = []
    divergence_point = -1
    p_allowed_at_div = frozenset[Event]()
    q_allowed_at_div = frozenset[Event]()
    attempted: Event | None = None

    for i, event in enumerate(trace):
        p_states_before = p_walk[i][0]
        p_available = p_walk[i][1]
        q_states_before = q_walk[i][0]
        q_available = q_walk[i][1]

        p_after = _advance_states(lts_p, p_states_before, event)
        q_after = _advance_states(lts_q, q_states_before, event)

        if (not p_after or not q_after) and divergence_point == -1:
            # One side can't perform this event
            status = StepStatus.VIOLATION
            divergence_point = i
            p_allowed_at_div = p_available
            q_allowed_at_div = q_available
            attempted = event
        elif p_available != q_available and divergence_point == -1:
            # Both can do this event but their menu differs — potential divergence
            status = StepStatus.OK
        else:
            status = StepStatus.OK

        annotated.append(
            AnnotatedStep(
                event=event,
                status=status,
                spec_available=p_available,
                impl_available=q_available,
                spec_states=frozenset(p_states_before),
                impl_states=frozenset(q_states_before),
            )
        )

    # If no divergence found in the trace, check the final states
    if divergence_point == -1:
        p_final_available = p_walk[-1][1] if p_walk else frozenset()
        q_final_available = q_walk[-1][1] if q_walk else frozenset()

        if p_final_available != q_final_available:
            divergence_point = len(trace)
            p_allowed_at_div = p_final_available
            q_allowed_at_div = q_final_available
            diff = p_final_available.symmetric_difference(q_final_available)
            attempted = min(diff) if diff else None

    # Build summary
    if divergence_point >= 0:
        prefix = trace[:divergence_point]
        p_only = p_allowed_at_div - q_allowed_at_div
        q_only = q_allowed_at_div - p_allowed_at_div

        parts = []
        if p_only:
            parts.append(f"P can do {{{', '.join(sorted(str(e) for e in p_only))}}}")
        if q_only:
            parts.append(f"Q can do {{{', '.join(sorted(str(e) for e in q_only))}}}")

        detail = " but ".join(parts) if parts else "event sets differ"
        summary = f"After {_trace_str(prefix)}, {detail}"
    else:
        summary = f"Processes differ on trace {_trace_str(trace)}"

    # Use symmetric union for spec_allowed
    all_allowed = p_allowed_at_div | q_allowed_at_div

    return CounterexampleExplanation(
        summary=summary,
        annotated_trace=annotated,
        divergence_point=divergence_point,
        spec_allowed=all_allowed,
        impl_attempted=attempted,
        failure_kind="equivalence",
        raw_counterexample=trace,
    )


# =============================================================================
# Convenience: Find Minimal Counterexample
# =============================================================================


def find_minimal_counterexample(
    spec: Process | LTS,
    impl: Process | LTS,
    max_trace_length: int = 50,
) -> CounterexampleExplanation | None:
    """Find the shortest counterexample trace and explain it.

    Performs a BFS over the synchronized product of spec and impl LTS
    to find the shortest trace where they diverge.

    Args:
        spec: The specification process or LTS.
        impl: The implementation process or LTS.
        max_trace_length: Maximum trace length to search.

    Returns:
        CounterexampleExplanation for the shortest diverging trace,
        or None if impl refines spec up to max_trace_length.
    """
    lts_spec = _to_lts(spec)
    lts_impl = _to_lts(impl)

    # BFS over product state space
    spec_init = frozenset(_tau_closed_states(lts_spec, {lts_spec.initial_state}))
    impl_init = frozenset(_tau_closed_states(lts_impl, {lts_impl.initial_state}))

    queue: deque[tuple[frozenset[int], frozenset[int], Trace]] = deque()
    queue.append((spec_init, impl_init, ()))
    seen: set[tuple[frozenset[int], frozenset[int]]] = set()

    while queue:
        spec_states, impl_states, trace = queue.popleft()

        state_pair = (spec_states, impl_states)
        if state_pair in seen:
            continue
        seen.add(state_pair)

        if len(trace) >= max_trace_length:
            continue

        # Get events from impl
        impl_events = _visible_events_from_states(lts_impl, set(impl_states))
        spec_events = _visible_events_from_states(lts_spec, set(spec_states))

        # Check for impl events not in spec
        for event in impl_events:
            if event not in spec_events:
                # Found a counterexample
                ce_trace = trace + (event,)
                from agenticraft_foundation.algebra.refinement import RefinementResult

                fake_result = RefinementResult(
                    refines=False,
                    counterexample=ce_trace,
                    message=f"Implementation trace {_trace_str(ce_trace)} not in specification",
                )
                return explain_refinement_failure(lts_spec, lts_impl, fake_result)

        # Explore common events
        for event in impl_events & spec_events:
            next_spec = frozenset(_advance_states(lts_spec, set(spec_states), event))
            next_impl = frozenset(_advance_states(lts_impl, set(impl_states), event))
            if next_spec and next_impl:
                queue.append((next_spec, next_impl, trace + (event,)))

    return None


__all__ = [
    "StepStatus",
    "AnnotatedStep",
    "CounterexampleExplanation",
    "explain_refinement_failure",
    "explain_equivalence_failure",
    "find_minimal_counterexample",
]
