"""Labeled Transition System (LTS) for CSP operational semantics.

This module provides:
- LTS representation of process behavior
- State space exploration
- Trace generation
- Deadlock detection

An LTS is a tuple (S, Act, →, s₀) where:
- S: set of states
- Act: set of actions/events
- →: transition relation S × Act × S
- s₀: initial state
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from .csp import TAU, TICK, Event, Process

# =============================================================================
# Core LTS Types
# =============================================================================


@dataclass(frozen=True)
class Transition:
    """A transition in an LTS: source ─event→ target."""

    source: int  # State ID
    event: Event
    target: int  # State ID

    def __repr__(self) -> str:
        return f"{self.source} --{self.event}--> {self.target}"


@dataclass
class LTSState:
    """A state in the LTS, wrapping a process."""

    id: int
    process: Process
    is_terminal: bool = False
    is_deadlock: bool = False

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LTSState):
            return self.id == other.id
        return False


@dataclass
class LTS:
    """Labeled Transition System.

    Represents the operational semantics of a CSP process as a graph.

    Attributes:
        states: All states in the LTS
        transitions: All transitions
        initial_state: The starting state ID
        alphabet: All events in the LTS
    """

    states: dict[int, LTSState] = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)
    initial_state: int = 0
    alphabet: frozenset[Event] = field(default_factory=frozenset)

    def add_state(self, state: LTSState) -> None:
        """Add a state to the LTS."""
        self.states[state.id] = state

    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the LTS."""
        self.transitions.append(transition)
        self.alphabet = self.alphabet | frozenset({transition.event})

    def get_transitions_from(self, state_id: int) -> list[Transition]:
        """Get all transitions from a state."""
        return [t for t in self.transitions if t.source == state_id]

    def get_transitions_to(self, state_id: int) -> list[Transition]:
        """Get all transitions to a state."""
        return [t for t in self.transitions if t.target == state_id]

    def successors(self, state_id: int) -> Iterator[tuple[Event, int]]:
        """Yield (event, target_state) pairs for transitions from state."""
        for t in self.get_transitions_from(state_id):
            yield t.event, t.target

    def predecessors(self, state_id: int) -> Iterator[tuple[Event, int]]:
        """Yield (event, source_state) pairs for transitions to state."""
        for t in self.get_transitions_to(state_id):
            yield t.event, t.source

    @property
    def num_states(self) -> int:
        """Number of states in the LTS."""
        return len(self.states)

    @property
    def num_transitions(self) -> int:
        """Number of transitions in the LTS."""
        return len(self.transitions)

    def deadlock_states(self) -> list[int]:
        """Return IDs of deadlock states."""
        return [state_id for state_id, state in self.states.items() if state.is_deadlock]

    def terminal_states(self) -> list[int]:
        """Return IDs of terminal (successfully terminated) states."""
        return [state_id for state_id, state in self.states.items() if state.is_terminal]


# =============================================================================
# LTS Builder
# =============================================================================


class LTSBuilder:
    """Builds an LTS from a CSP process through state space exploration.

    Uses breadth-first exploration to discover all reachable states
    and transitions. State identity is determined by ``_state_key()``
    of the process term -- structurally identical terms produce the
    same tuple key and map to the same state. This is conservative:
    semantically equivalent but structurally different terms may
    create duplicate states, which increases the state space but
    never misses behaviors.

    Note: Processes containing callables (e.g., Guard) use object
    identity (``id()``) in their state key, so distinct Guard
    instances with equivalent conditions will be treated as
    different states.
    """

    def __init__(self, max_states: int = 10000, max_depth: int = 1000):
        """Initialize the builder.

        Args:
            max_states: Maximum number of states to explore
            max_depth: Maximum depth of exploration
        """
        self.max_states = max_states
        self.max_depth = max_depth

    def build(self, process: Process) -> LTS:
        """Build an LTS from a process.

        Args:
            process: The CSP process to analyze

        Returns:
            The resulting LTS
        """
        lts = LTS()
        seen_processes: dict[tuple[object, ...], int] = {}  # _state_key -> state_id
        queue: deque[tuple[Process, int, int]] = deque()  # (process, state_id, depth)

        next_id = 0

        # Create initial state
        initial_key = process._state_key()
        initial_state = LTSState(
            id=next_id,
            process=process,
            is_terminal=process.is_terminated(),
            is_deadlock=process.is_deadlocked(),
        )
        lts.add_state(initial_state)
        seen_processes[initial_key] = next_id
        queue.append((process, next_id, 0))
        next_id += 1

        # Explore state space
        while queue and lts.num_states < self.max_states:
            current_process, current_id, depth = queue.popleft()

            if depth >= self.max_depth:
                continue

            # Explore all possible transitions
            for event in current_process.initials():
                try:
                    next_process = current_process.after(event)
                    next_key = next_process._state_key()

                    # If we reached this state via TICK, it's terminal
                    # A state can be BOTH terminal AND deadlock if reachable both ways
                    reached_via_tick = event == TICK
                    is_deadlock_path = not reached_via_tick and next_process.is_deadlocked()

                    # Check if we've seen this state
                    if next_key in seen_processes:
                        target_id = seen_processes[next_key]
                        existing = lts.states[target_id]
                        # Update terminal/deadlock flags based on this path
                        if reached_via_tick:
                            existing.is_terminal = True
                        if is_deadlock_path:
                            existing.is_deadlock = True
                    else:
                        target_id = next_id
                        next_state = LTSState(
                            id=target_id,
                            process=next_process,
                            is_terminal=reached_via_tick or next_process.is_terminated(),
                            is_deadlock=is_deadlock_path,
                        )
                        lts.add_state(next_state)
                        seen_processes[next_key] = target_id
                        queue.append((next_process, target_id, depth + 1))
                        next_id += 1

                    # Add transition
                    lts.add_transition(Transition(source=current_id, event=event, target=target_id))

                except ValueError:
                    # Cannot perform this event (shouldn't happen if initials is correct)
                    pass

        return lts


# =============================================================================
# Trace Operations
# =============================================================================


Trace = tuple[Event, ...]


def traces(
    lts: LTS,
    max_length: int = 100,
    include_tau: bool = False,
) -> Iterator[Trace]:
    """Generate all traces of the LTS up to a maximum length.

    A trace is a sequence of visible events (τ is typically hidden).

    Args:
        lts: The LTS to analyze
        max_length: Maximum trace length
        include_tau: Whether to include τ (internal) events

    Yields:
        Traces as tuples of events
    """
    queue: deque[tuple[int, Trace]] = deque()
    queue.append((lts.initial_state, ()))

    seen: set[tuple[int, Trace]] = set()

    while queue:
        state_id, trace = queue.popleft()

        if (state_id, trace) in seen:
            continue
        seen.add((state_id, trace))

        yield trace

        if len(trace) >= max_length:
            continue

        for event, target in lts.successors(state_id):
            if event == TAU and not include_tau:
                # τ is hidden - extend trace without adding event
                queue.append((target, trace))
            else:
                new_trace = trace + (event,)
                queue.append((target, new_trace))


def maximal_traces(lts: LTS, max_length: int = 100) -> Iterator[Trace]:
    """Generate maximal traces (traces that cannot be extended).

    A maximal trace ends in either:
    - A deadlock state
    - A terminal state (successful termination)
    - Maximum length reached

    Args:
        lts: The LTS to analyze
        max_length: Maximum trace length

    Yields:
        Maximal traces
    """
    queue: deque[tuple[int, Trace]] = deque()
    queue.append((lts.initial_state, ()))

    seen: set[tuple[int, Trace]] = set()

    while queue:
        state_id, trace = queue.popleft()

        if (state_id, trace) in seen:
            continue
        seen.add((state_id, trace))

        state = lts.states[state_id]

        # Check if this is a maximal trace
        if state.is_deadlock or state.is_terminal or len(trace) >= max_length:
            yield trace
            continue

        for event, target in lts.successors(state_id):
            if event == TAU:
                queue.append((target, trace))
            else:
                new_trace = trace + (event,)
                queue.append((target, new_trace))


def accepts(lts: LTS, trace: Trace) -> bool:
    """Check if the LTS can perform the given trace.

    Args:
        lts: The LTS to check
        trace: The trace to verify

    Returns:
        True if the trace can be performed
    """
    current_states = {lts.initial_state}

    for event in trace:
        next_states: set[int] = set()

        for state_id in current_states:
            # Consider τ-closure (states reachable via τ)
            reachable = _tau_closure(lts, state_id)
            for reachable_id in reachable:
                for trans_event, target in lts.successors(reachable_id):
                    if trans_event == event:
                        next_states.add(target)

        if not next_states:
            return False
        current_states = next_states

    return True


def _tau_closure(lts: LTS, state_id: int) -> set[int]:
    """Compute τ-closure: all states reachable via τ transitions."""
    closure = {state_id}
    queue = deque([state_id])

    while queue:
        current = queue.popleft()
        for event, target in lts.successors(current):
            if event == TAU and target not in closure:
                closure.add(target)
                queue.append(target)

    return closure


# =============================================================================
# Deadlock Detection
# =============================================================================


@dataclass
class DeadlockAnalysis:
    """Result of deadlock analysis."""

    has_deadlock: bool
    deadlock_states: list[int]
    deadlock_traces: list[Trace]


def detect_deadlock(lts: LTS, max_trace_length: int = 50) -> DeadlockAnalysis:
    """Detect deadlock states and find traces leading to them.

    Args:
        lts: The LTS to analyze
        max_trace_length: Maximum trace length to consider

    Returns:
        DeadlockAnalysis with results
    """
    deadlock_states = lts.deadlock_states()

    if not deadlock_states:
        return DeadlockAnalysis(
            has_deadlock=False,
            deadlock_states=[],
            deadlock_traces=[],
        )

    # Find shortest traces to deadlock states
    deadlock_traces = []
    for state_id in deadlock_states:
        trace = _find_trace_to(lts, state_id, max_trace_length)
        if trace is not None:
            deadlock_traces.append(trace)

    return DeadlockAnalysis(
        has_deadlock=True,
        deadlock_states=deadlock_states,
        deadlock_traces=deadlock_traces,
    )


def _find_trace_to(
    lts: LTS,
    target_state: int,
    max_length: int,
) -> Trace | None:
    """Find a trace from initial state to target state."""
    queue: deque[tuple[int, Trace]] = deque()
    queue.append((lts.initial_state, ()))
    seen: set[int] = set()

    while queue:
        state_id, trace = queue.popleft()

        if state_id in seen:
            continue
        seen.add(state_id)

        if state_id == target_state:
            return trace

        if len(trace) >= max_length:
            continue

        for event, target in lts.successors(state_id):
            if event == TAU:
                queue.append((target, trace))
            else:
                queue.append((target, trace + (event,)))

    return None


# =============================================================================
# Liveness Analysis
# =============================================================================


@dataclass
class LivenessAnalysis:
    """Result of liveness analysis."""

    is_live: bool
    stuck_states: list[int]  # States from which no progress is possible
    live_events: dict[Event, bool]  # Whether each event is eventually possible


def analyze_liveness(lts: LTS, events_of_interest: set[Event] | None = None) -> LivenessAnalysis:
    """Analyze liveness properties of the LTS.

    Args:
        lts: The LTS to analyze
        events_of_interest: Events to check for liveness

    Returns:
        LivenessAnalysis with results
    """
    if events_of_interest is None:
        events_of_interest = set(lts.alphabet) - {TAU, TICK}

    # Find stuck states (neither deadlock nor can progress)
    stuck_states = []
    for state_id, state in lts.states.items():
        if not state.is_terminal and not state.is_deadlock:
            # Check if state has outgoing transitions
            transitions = lts.get_transitions_from(state_id)
            if not transitions:
                stuck_states.append(state_id)

    # Check which events are live (eventually possible from initial state)
    live_events: dict[Event, bool] = {}
    reachable = _compute_reachable(lts)

    for event in events_of_interest:
        # Event is live if there exists a reachable state that can perform it
        event_live = any(
            any(e == event for e, _ in lts.successors(state_id)) for state_id in reachable
        )
        live_events[event] = event_live

    return LivenessAnalysis(
        is_live=len(stuck_states) == 0 and all(live_events.values()),
        stuck_states=stuck_states,
        live_events=live_events,
    )


def _compute_reachable(lts: LTS) -> set[int]:
    """Compute all reachable states from initial state."""
    reachable = set()
    queue = deque([lts.initial_state])

    while queue:
        state_id = queue.popleft()
        if state_id in reachable:
            continue
        reachable.add(state_id)

        for _, target in lts.successors(state_id):
            if target not in reachable:
                queue.append(target)

    return reachable


# =============================================================================
# Convenience Functions
# =============================================================================


def build_lts(process: Process, max_states: int = 10000) -> LTS:
    """Build an LTS from a process.

    Args:
        process: The CSP process
        max_states: Maximum states to explore

    Returns:
        The LTS representation
    """
    builder = LTSBuilder(max_states=max_states)
    return builder.build(process)


def is_deadlock_free(process: Process) -> bool:
    """Check if a process is deadlock-free.

    Args:
        process: The CSP process to check

    Returns:
        True if the process cannot deadlock
    """
    lts = build_lts(process)
    analysis = detect_deadlock(lts)
    return not analysis.has_deadlock


__all__ = [
    # Core types
    "Transition",
    "LTSState",
    "LTS",
    "LTSBuilder",
    # Trace operations
    "Trace",
    "traces",
    "maximal_traces",
    "accepts",
    # Deadlock detection
    "DeadlockAnalysis",
    "detect_deadlock",
    # Liveness analysis
    "LivenessAnalysis",
    "analyze_liveness",
    # Convenience
    "build_lts",
    "is_deadlock_free",
]
