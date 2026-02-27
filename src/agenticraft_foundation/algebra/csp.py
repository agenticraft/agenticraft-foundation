"""CSP (Communicating Sequential Processes) implementation.

Based on Hoare (1985) - Communicating Sequential Processes.

This module provides:
- Core process primitives (STOP, SKIP, Prefix)
- Choice operators (external □, internal ⊓)
- Parallel composition (interleaving ||, synchronized |[A]|)
- Sequential composition (;)
- Hiding (P \\ H)

Key Concepts:
- Process: Entity that can engage in events
- Event: Atomic unit of communication/action
- Alphabet: Set of events a process can engage in
- Trace: Sequence of events performed by a process
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

# =============================================================================
# Core Types
# =============================================================================


class Event(str):
    """An event in CSP.

    Events are atomic units of communication or action.
    The special event "✓" (TICK) indicates successful termination.
    """

    pass


# Special events
TICK = Event("✓")  # Successful termination
TAU = Event("τ")  # Internal/silent action


class ProcessKind(Enum):
    """Classification of process types."""

    STOP = auto()
    SKIP = auto()
    PREFIX = auto()
    EXTERNAL_CHOICE = auto()
    INTERNAL_CHOICE = auto()
    PARALLEL = auto()
    SEQUENTIAL = auto()
    HIDING = auto()
    RECURSION = auto()
    VARIABLE = auto()
    # Agent-specific extensions
    INTERRUPT = auto()
    TIMEOUT = auto()
    GUARD = auto()
    RENAME = auto()
    PIPE = auto()


# =============================================================================
# Process Base Class
# =============================================================================


class Process(ABC):
    """Abstract base class for CSP processes.

    A process is characterized by:
    - alphabet: The set of events it can engage in
    - initials: The initial events it can perform
    - after(e): The process after performing event e
    """

    @property
    @abstractmethod
    def kind(self) -> ProcessKind:
        """Process classification."""
        pass

    @abstractmethod
    def alphabet(self) -> frozenset[Event]:
        """Return the alphabet (set of events) this process can engage in."""
        pass

    @abstractmethod
    def initials(self) -> frozenset[Event]:
        """Return the set of initial events the process can perform."""
        pass

    @abstractmethod
    def after(self, event: Event) -> Process:
        """Return the process after performing the given event.

        Args:
            event: The event to perform

        Returns:
            The resulting process after the event

        Raises:
            ValueError: If the event cannot be performed
        """
        pass

    def can_perform(self, event: Event) -> bool:
        """Check if the process can perform the given event."""
        return event in self.initials()

    def is_deadlocked(self) -> bool:
        """Check if the process is deadlocked (no possible events)."""
        return len(self.initials()) == 0 and not self.is_terminated()

    def is_terminated(self) -> bool:
        """Check if the process has successfully terminated."""
        return TICK in self.initials()

    @abstractmethod
    def __repr__(self) -> str:
        pass


# =============================================================================
# Primitive Processes
# =============================================================================


@dataclass(frozen=True)
class Stop(Process):
    """STOP - The deadlocked process.

    STOP cannot perform any events. It represents deadlock.

    Properties:
    - alphabet(STOP) = {}
    - initials(STOP) = {}
    - traces(STOP) = {⟨⟩}
    """

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.STOP

    def alphabet(self) -> frozenset[Event]:
        return frozenset()

    def initials(self) -> frozenset[Event]:
        return frozenset()

    def after(self, event: Event) -> Process:
        raise ValueError(f"STOP cannot perform any event, including {event}")

    def __repr__(self) -> str:
        return "STOP"


@dataclass(frozen=True)
class Skip(Process):
    """SKIP - Successful termination.

    SKIP can only perform the tick event (✓) and then becomes STOP.

    Properties:
    - alphabet(SKIP) = {✓}
    - initials(SKIP) = {✓}
    - traces(SKIP) = {⟨⟩, ⟨✓⟩}
    """

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.SKIP

    def alphabet(self) -> frozenset[Event]:
        return frozenset({TICK})

    def initials(self) -> frozenset[Event]:
        return frozenset({TICK})

    def after(self, event: Event) -> Process:
        if event == TICK:
            return Stop()
        raise ValueError(f"SKIP can only perform ✓, not {event}")

    def __repr__(self) -> str:
        return "SKIP"


@dataclass(frozen=True)
class Prefix(Process):
    """a → P (Prefix/Guarding).

    Prefix performs event 'a' and then behaves as process P.

    Properties:
    - alphabet(a → P) = {a} ∪ alphabet(P)
    - initials(a → P) = {a}
    - (a → P) ─a→ P
    """

    event: Event
    continuation: Process

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.PREFIX

    def alphabet(self) -> frozenset[Event]:
        return frozenset({self.event}) | self.continuation.alphabet()

    def initials(self) -> frozenset[Event]:
        return frozenset({self.event})

    def after(self, event: Event) -> Process:
        if event == self.event:
            return self.continuation
        raise ValueError(f"Cannot perform {event}, expected {self.event}")

    def __repr__(self) -> str:
        return f"{self.event} → {self.continuation}"


# =============================================================================
# Choice Operators
# =============================================================================


@dataclass(frozen=True)
class ExternalChoice(Process):
    """P □ Q (External Choice).

    The environment chooses between P and Q based on the initial event.

    Properties:
    - alphabet(P □ Q) = alphabet(P) ∪ alphabet(Q)
    - initials(P □ Q) = initials(P) ∪ initials(Q)
    - P □ Q ─a→ P' if P ─a→ P' and a ∉ initials(Q)
    - P □ Q ─a→ Q' if Q ─a→ Q' and a ∉ initials(P)
    - P □ Q ─a→ P' □ Q' if P ─a→ P' and Q ─a→ Q' (deterministic)
    """

    left: Process
    right: Process

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.EXTERNAL_CHOICE

    def alphabet(self) -> frozenset[Event]:
        return self.left.alphabet() | self.right.alphabet()

    def initials(self) -> frozenset[Event]:
        return self.left.initials() | self.right.initials()

    def after(self, event: Event) -> Process:
        """Resolve external choice by performing an event.

        When both branches offer the event, the left branch is
        chosen by convention (deterministic resolution).

        Args:
            event: The event to perform.

        Returns:
            Continuation process after performing the event.

        Raises:
            ValueError: If neither branch can perform the event.
        """
        left_can = event in self.left.initials()
        right_can = event in self.right.initials()

        if left_can and right_can:
            # Both can perform - resolved to left branch by convention
            return self.left.after(event)
        elif left_can:
            return self.left.after(event)
        elif right_can:
            return self.right.after(event)
        else:
            raise ValueError(f"Neither branch can perform {event}")

    def __repr__(self) -> str:
        return f"({self.left} □ {self.right})"


@dataclass(frozen=True)
class InternalChoice(Process):
    """P ⊓ Q (Internal/Nondeterministic Choice).

    The process internally chooses between P and Q.
    The environment has no control over which branch is taken.

    Properties:
    - alphabet(P ⊓ Q) = alphabet(P) ∪ alphabet(Q)
    - initials(P ⊓ Q) = initials(P) ∪ initials(Q)
    - P ⊓ Q ─τ→ P (internal transition to P)
    - P ⊓ Q ─τ→ Q (internal transition to Q)
    """

    left: Process
    right: Process

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.INTERNAL_CHOICE

    def alphabet(self) -> frozenset[Event]:
        return self.left.alphabet() | self.right.alphabet()

    def initials(self) -> frozenset[Event]:
        # Internal choice can perform τ to resolve, or any initial from either
        return self.left.initials() | self.right.initials() | frozenset({TAU})

    def after(self, event: Event) -> Process:
        if event == TAU:
            # Nondeterministically choose left (could be right)
            # In a real implementation, this would branch
            return self.left
        # Otherwise, try to perform from either branch
        if event in self.left.initials():
            return self.left.after(event)
        if event in self.right.initials():
            return self.right.after(event)
        raise ValueError(f"Cannot perform {event}")

    def __repr__(self) -> str:
        return f"({self.left} ⊓ {self.right})"


# =============================================================================
# Parallel Composition
# =============================================================================


@dataclass(frozen=True)
class Parallel(Process):
    """P |[A]| Q (Generalized Parallel Composition).

    P and Q run in parallel, synchronizing on events in A.
    - Events in A require both processes to participate
    - Events not in A can be performed independently

    Special cases:
    - P ||| Q (A = {}) : Interleaving (no synchronization)
    - P || Q (A = alphabet(P) ∩ alphabet(Q)) : Alphabetized parallel

    Properties:
    - alphabet(P |[A]| Q) = alphabet(P) ∪ alphabet(Q)
    - For a ∈ A: P |[A]| Q ─a→ P' |[A]| Q' iff P ─a→ P' and Q ─a→ Q'
    - For a ∉ A: P |[A]| Q ─a→ P' |[A]| Q iff P ─a→ P'
    """

    left: Process
    right: Process
    sync_set: frozenset[Event] = field(default_factory=frozenset)

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.PARALLEL

    def alphabet(self) -> frozenset[Event]:
        return self.left.alphabet() | self.right.alphabet()

    def initials(self) -> frozenset[Event]:
        left_init = self.left.initials()
        right_init = self.right.initials()

        result: set[Event] = set()

        # Events not in sync_set can be performed independently
        for e in left_init:
            if e not in self.sync_set:
                result.add(e)
        for e in right_init:
            if e not in self.sync_set:
                result.add(e)

        # Events in sync_set require both to be ready
        for e in left_init & right_init & self.sync_set:
            result.add(e)

        return frozenset(result)

    def after(self, event: Event) -> Process:
        if event in self.sync_set:
            # Synchronized event: both must participate
            if event in self.left.initials() and event in self.right.initials():
                return Parallel(
                    left=self.left.after(event),
                    right=self.right.after(event),
                    sync_set=self.sync_set,
                )
            raise ValueError(f"Synchronized event {event} requires both processes")
        else:
            # Independent event: either can perform
            if event in self.left.initials():
                return Parallel(
                    left=self.left.after(event),
                    right=self.right,
                    sync_set=self.sync_set,
                )
            elif event in self.right.initials():
                return Parallel(
                    left=self.left,
                    right=self.right.after(event),
                    sync_set=self.sync_set,
                )
            raise ValueError(f"Neither process can perform {event}")

    def __repr__(self) -> str:
        if not self.sync_set:
            return f"({self.left} ||| {self.right})"
        sync_str = ", ".join(sorted(self.sync_set))
        return f"({self.left} |[{{{sync_str}}}]| {self.right})"


# =============================================================================
# Sequential Composition
# =============================================================================


@dataclass(frozen=True)
class Sequential(Process):
    """P ; Q (Sequential Composition).

    P runs first, and when it terminates (performs ✓), Q starts.

    Properties:
    - alphabet(P ; Q) = (alphabet(P) - {✓}) ∪ alphabet(Q)
    - initials(P ; Q) = initials(P) - {✓}  if ✓ ∉ initials(P)
                      = (initials(P) - {✓}) ∪ initials(Q)  if ✓ ∈ initials(P)
    - P ; Q ─a→ P' ; Q if P ─a→ P' and a ≠ ✓
    - P ; Q ─τ→ Q if P ─✓→ (starts Q after P terminates)
    """

    first: Process
    second: Process

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.SEQUENTIAL

    def alphabet(self) -> frozenset[Event]:
        # Tick from first is hidden, replaced by second's events
        first_alpha = self.first.alphabet() - frozenset({TICK})
        return first_alpha | self.second.alphabet()

    def initials(self) -> frozenset[Event]:
        first_init = self.first.initials()
        # If first can terminate, second's initials are also available
        if TICK in first_init:
            return (first_init - frozenset({TICK})) | self.second.initials()
        return first_init

    def after(self, event: Event) -> Process:
        first_init = self.first.initials()

        # If first can terminate and event is from second
        if TICK in first_init and event in self.second.initials():
            return self.second.after(event)

        # Otherwise, perform in first
        if event in first_init and event != TICK:
            new_first = self.first.after(event)
            return Sequential(first=new_first, second=self.second)

        raise ValueError(f"Cannot perform {event}")

    def __repr__(self) -> str:
        return f"({self.first} ; {self.second})"


# =============================================================================
# Hiding
# =============================================================================


@dataclass(frozen=True)
class Hiding(Process):
    """P \\ H (Hiding).

    Events in H are hidden (become internal τ actions).

    Properties:
    - alphabet(P \\ H) = alphabet(P) - H
    - P \\ H ─a→ P' \\ H if P ─a→ P' and a ∉ H
    - P \\ H ─τ→ P' \\ H if P ─a→ P' and a ∈ H
    """

    process: Process
    hidden: frozenset[Event]

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.HIDING

    def alphabet(self) -> frozenset[Event]:
        return self.process.alphabet() - self.hidden

    def initials(self) -> frozenset[Event]:
        proc_init = self.process.initials()
        visible = proc_init - self.hidden
        # Hidden initials become τ
        if proc_init & self.hidden:
            visible = visible | frozenset({TAU})
        return visible

    def after(self, event: Event) -> Process:
        if event == TAU:
            # Perform any hidden event
            for h in self.hidden:
                if h in self.process.initials():
                    return Hiding(
                        process=self.process.after(h),
                        hidden=self.hidden,
                    )
            raise ValueError("No hidden events available")
        elif event not in self.hidden:
            return Hiding(
                process=self.process.after(event),
                hidden=self.hidden,
            )
        raise ValueError(f"Event {event} is hidden")

    def __repr__(self) -> str:
        hidden_str = ", ".join(sorted(self.hidden))
        return f"({self.process} \\\\ {{{hidden_str}}})"


# =============================================================================
# Recursion
# =============================================================================


@dataclass(frozen=True)
class Recursion(Process):
    """μX.P (Recursion).

    Recursive process definition where X is bound in P.
    """

    variable: str
    body: Process

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.RECURSION

    def alphabet(self) -> frozenset[Event]:
        return self.body.alphabet()

    def initials(self) -> frozenset[Event]:
        # Unfold one level to get initials
        return self.unfold().initials()

    def after(self, event: Event) -> Process:
        return self.unfold().after(event)

    def unfold(self) -> Process:
        """Unfold one level of recursion."""
        return substitute(self.body, self.variable, self)

    def __repr__(self) -> str:
        return f"μ{self.variable}.{self.body}"


@dataclass(frozen=True)
class Variable(Process):
    """X (Recursion Variable).

    A reference to a bound recursion variable.
    """

    name: str

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.VARIABLE

    def alphabet(self) -> frozenset[Event]:
        # Variables have no alphabet until substituted
        return frozenset()

    def initials(self) -> frozenset[Event]:
        return frozenset()

    def after(self, event: Event) -> Process:
        raise ValueError(f"Cannot perform events on unbound variable {self.name}")

    def __repr__(self) -> str:
        return self.name


def substitute(process: Process, var_name: str, replacement: Process) -> Process:
    """Substitute all occurrences of a named variable with a process.

    Recursively traverses the process tree, replacing each ``Variable``
    node whose name matches *var_name* with *replacement*. Respects
    variable shadowing in ``Recursion`` nodes.

    Args:
        process: The process tree to transform.
        var_name: Name of the variable to replace.
        replacement: Process to substitute in place of the variable.

    Returns:
        A new process tree with all matching variables replaced.
    """
    if isinstance(process, Variable):
        if process.name == var_name:
            return replacement
        return process
    elif isinstance(process, Prefix):
        return Prefix(
            event=process.event,
            continuation=substitute(process.continuation, var_name, replacement),
        )
    elif isinstance(process, ExternalChoice):
        return ExternalChoice(
            left=substitute(process.left, var_name, replacement),
            right=substitute(process.right, var_name, replacement),
        )
    elif isinstance(process, InternalChoice):
        return InternalChoice(
            left=substitute(process.left, var_name, replacement),
            right=substitute(process.right, var_name, replacement),
        )
    elif isinstance(process, Parallel):
        return Parallel(
            left=substitute(process.left, var_name, replacement),
            right=substitute(process.right, var_name, replacement),
            sync_set=process.sync_set,
        )
    elif isinstance(process, Sequential):
        return Sequential(
            first=substitute(process.first, var_name, replacement),
            second=substitute(process.second, var_name, replacement),
        )
    elif isinstance(process, Hiding):
        return Hiding(
            process=substitute(process.process, var_name, replacement),
            hidden=process.hidden,
        )
    elif isinstance(process, Recursion):
        if process.variable == var_name:
            # Variable is shadowed, don't substitute in body
            return process
        return Recursion(
            variable=process.variable,
            body=substitute(process.body, var_name, replacement),
        )
    else:
        # Handle extended operators (late import avoids circular dep)
        try:
            from agenticraft_foundation.algebra.operators import (
                Guard,
                Interrupt,
                Pipe,
                Rename,
                Timeout,
            )

            if isinstance(process, Interrupt):
                return Interrupt(
                    primary=substitute(process.primary, var_name, replacement),
                    handler=substitute(process.handler, var_name, replacement),
                )
            elif isinstance(process, Timeout):
                return Timeout(
                    process=substitute(process.process, var_name, replacement),
                    duration=process.duration,
                    fallback=substitute(process.fallback, var_name, replacement),
                )
            elif isinstance(process, Guard):
                return Guard(
                    condition=process.condition,
                    process=substitute(process.process, var_name, replacement),
                )
            elif isinstance(process, Rename):
                return Rename(
                    process=substitute(process.process, var_name, replacement),
                    mapping=process.mapping,
                )
            elif isinstance(process, Pipe):
                return Pipe(
                    producer=substitute(process.producer, var_name, replacement),
                    consumer=substitute(process.consumer, var_name, replacement),
                    channel=process.channel,
                )
        except ImportError:
            pass  # operators module not available (standalone csp.py usage)
        # Stop, Skip, or unknown -- return unchanged
        return process


# =============================================================================
# Convenience Constructors
# =============================================================================


def stop() -> Stop:
    """Create STOP process."""
    return Stop()


def skip() -> Skip:
    """Create SKIP process."""
    return Skip()


def prefix(event: str | Event, continuation: Process) -> Prefix:
    """Create prefix process: event → continuation."""
    return Prefix(event=Event(event), continuation=continuation)


def choice(left: Process, right: Process) -> ExternalChoice:
    """Create external choice: left □ right."""
    return ExternalChoice(left=left, right=right)


def internal_choice(left: Process, right: Process) -> InternalChoice:
    """Create internal choice: left ⊓ right."""
    return InternalChoice(left=left, right=right)


def parallel(
    left: Process,
    right: Process,
    sync: frozenset[str] | set[str] | None = None,
) -> Parallel:
    """Create parallel composition: left |[sync]| right."""
    sync_set = frozenset(Event(e) for e in (sync or set()))
    return Parallel(left=left, right=right, sync_set=sync_set)


def interleave(left: Process, right: Process) -> Parallel:
    """Create interleaving: left ||| right."""
    return Parallel(left=left, right=right, sync_set=frozenset())


def sequential(first: Process, second: Process) -> Sequential:
    """Create sequential composition: first ; second."""
    return Sequential(first=first, second=second)


def hide(process: Process, hidden: set[str] | frozenset[str]) -> Hiding:
    """Create hiding: process \\ hidden."""
    return Hiding(process=process, hidden=frozenset(Event(e) for e in hidden))


def rec(variable: str, body: Process) -> Recursion:
    """Create recursion: μvariable.body."""
    return Recursion(variable=variable, body=body)


def var(name: str) -> Variable:
    """Create recursion variable."""
    return Variable(name=name)


__all__ = [
    # Types
    "Event",
    "TICK",
    "TAU",
    "ProcessKind",
    "Process",
    # Primitives
    "Stop",
    "Skip",
    "Prefix",
    # Choice
    "ExternalChoice",
    "InternalChoice",
    # Parallel
    "Parallel",
    # Sequential
    "Sequential",
    # Hiding
    "Hiding",
    # Recursion
    "Recursion",
    "Variable",
    "substitute",
    # Constructors
    "stop",
    "skip",
    "prefix",
    "choice",
    "internal_choice",
    "parallel",
    "interleave",
    "sequential",
    "hide",
    "rec",
    "var",
]
