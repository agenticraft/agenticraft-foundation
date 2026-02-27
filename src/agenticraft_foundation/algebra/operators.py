"""
Extended CSP operators for agent coordination.

These operators extend the core CSP algebra (csp.py) with
agent-specific constructs for preemption, bounded execution,
conditional activation, event abstraction, and pipeline composition.

All operators implement the full Process contract:
- kind: ProcessKind property
- alphabet(): frozenset[Event] method
- initials(): frozenset[Event] method
- after(event): Process state transition method
- __repr__(): string representation

Traces, deadlock detection, and refinement checking work automatically
via the LTS builder in semantics.py, which uses after() to explore
the state space. No traces() method is needed on these classes.

Recursion support: substitute() in csp.py has late-import cases for
all 5 operators, so rec("X", Interrupt(var("X"), handler)) works.

Operators:
    Interrupt (triangle)  -- Task preemption / priority override
    Timeout        -- Bounded execution with fallback
    Guard          -- Conditional process activation
    Rename         -- Event vocabulary mapping
    Pipe           -- Producer-consumer pipeline
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agenticraft_foundation.algebra.csp import (
    Event,
    Process,
    ProcessKind,
    Stop,
)

__all__ = [
    "TIMEOUT_EVENT",
    "Guard",
    "Interrupt",
    "Pipe",
    "Rename",
    "Timeout",
]


# --- Interrupt (triangle) ---


@dataclass(frozen=True)
class Interrupt(Process):
    """
    Interrupt operator (triangle): Run primary process until an interrupt
    event occurs, then switch to handler process.

    P triangle Q means: execute P, but if any event in Q's initial
    alphabet occurs, abandon P and continue with Q.after(event).

    Note: This is a CSP extension for agent coordination, not part of
    the standard CSP algebra (Hoare 1985). Based on Roscoe's interrupt
    (§5.3) but simplified for agent preemption.

    Agent use case: An agent processing a long task (P) gets
    interrupted by a higher-priority request (Q). The interrupt
    triggers on any event in Q's initial set.

    Example:
        task = Prefix(Event("process_data"), Prefix(Event("return_result"), Stop()))
        handler = Prefix(Event("handle_priority"), Prefix(Event("return_urgent"), Stop()))
        interruptible = Interrupt(primary=task, handler=handler)

        # interruptible can do task events OR be interrupted by handler events
        assert Event("process_data") in interruptible.initials()
        assert Event("handle_priority") in interruptible.initials()
    """

    primary: Process
    handler: Process

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.INTERRUPT

    def alphabet(self) -> frozenset[Event]:
        return self.primary.alphabet() | self.handler.alphabet()

    def initials(self) -> frozenset[Event]:
        return self.primary.initials() | self.handler.initials()

    def after(self, event: Event) -> Process:
        """
        If event is in handler's initials -> switch to handler.after(event).
        If event is in primary's initials -> continue with Interrupt(primary.after(event), handler).
        If event is in both -> nondeterministic (handler takes priority).
        """
        handler_initials = self.handler.initials()
        primary_initials = self.primary.initials()

        if event in handler_initials:
            # Interrupt fires -- switch to handler
            return self.handler.after(event)
        elif event in primary_initials:
            # Primary continues -- remain interruptible
            return Interrupt(primary=self.primary.after(event), handler=self.handler)
        else:
            raise ValueError(f"Event {event!r} not in alphabet of {self!r}")

    def _state_key(self) -> tuple[object, ...]:
        return ("Interrupt", self.primary._state_key(), self.handler._state_key())

    def __repr__(self) -> str:
        return f"({self.primary!r} \u25b3 {self.handler!r})"


# --- Timeout ---


# Sentinel event for timeout expiry
TIMEOUT_EVENT = Event("\u03c4_timeout")


@dataclass(frozen=True)
class Timeout(Process):
    """
    Timeout operator: Run process with a time bound; if not complete,
    switch to fallback.

    Timeout(P, duration, Q) means: execute P for at most `duration`.
    If P completes (reaches Skip/Stop), done. If the timeout event
    occurs, switch to Q.

    In the formal model, duration is an abstract positive value for
    ordering and composition. The tau_timeout event models expiry.
    At runtime, the orchestrator maps duration to wall-clock time.

    Note: This is a CSP extension. The τ_timeout sentinel event is
    not part of standard CSP; it models abstract time bounds for
    agent orchestration.

    Agent use case: LLM call with bounded execution -- if no response
    within timeout, activate fallback (retry, cached response, escalate).

    Example:
        llm_call = Prefix(Event("call_gpt4"), Prefix(Event("parse"), Stop()))
        fallback = Prefix(Event("return_cached"), Stop())
        bounded = Timeout(process=llm_call, duration=30.0, fallback=fallback)
    """

    process: Process
    duration: float
    fallback: Process

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"Timeout duration must be positive, got {self.duration}")

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.TIMEOUT

    def alphabet(self) -> frozenset[Event]:
        return self.process.alphabet() | self.fallback.alphabet() | frozenset({TIMEOUT_EVENT})

    def initials(self) -> frozenset[Event]:
        # Can do process events OR timeout can fire
        return self.process.initials() | frozenset({TIMEOUT_EVENT})

    def after(self, event: Event) -> Process:
        """
        If tau_timeout -> switch to fallback.
        If process event -> continue process (still under timeout).
        """
        if event == TIMEOUT_EVENT:
            return self.fallback
        elif event in self.process.initials():
            return Timeout(
                process=self.process.after(event),
                duration=self.duration,
                fallback=self.fallback,
            )
        else:
            raise ValueError(f"Event {event!r} not in alphabet of {self!r}")

    def _state_key(self) -> tuple[object, ...]:
        return ("Timeout", self.process._state_key(), self.duration, self.fallback._state_key())

    def __repr__(self) -> str:
        return f"Timeout({self.process!r}, {self.duration}, {self.fallback!r})"


# --- Guard ---


@dataclass(frozen=True)
class Guard(Process):
    """
    Guard operator: Conditional process activation.

    Guard(condition, P) behaves as P if condition() returns True,
    and as Stop if condition() returns False.

    The condition is a callable evaluated at query time (alphabet,
    initials, after). For static guards, pass a lambda.

    Note: This is a CSP extension. Standard CSP uses boolean guards
    in replicated operators; this operator extends that to
    runtime-evaluated conditions for agent activation.

    Agent use case: Only activate expensive agent if budget remains,
    safety checks pass, or load is below threshold.

    Example:
        expensive = Prefix(Event("call_gpt4o"), Prefix(Event("process"), Stop()))
        guarded = Guard(condition=lambda: budget > 0, process=expensive)
    """

    condition: Callable[[], bool]
    process: Process

    # frozen=True + Callable requires eq/hash override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Guard):
            return NotImplemented
        return self.process == other.process and self.condition is other.condition

    def __hash__(self) -> int:
        return hash((id(self.condition), self.process))

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.GUARD

    def _active(self) -> Process:
        """Resolve the guard to its active process."""
        return self.process if self.condition() else Stop()

    def alphabet(self) -> frozenset[Event]:
        return self.process.alphabet()

    def initials(self) -> frozenset[Event]:
        return self._active().initials()

    def after(self, event: Event) -> Process:
        active = self._active()
        if event in active.initials():
            return active.after(event)
        raise ValueError(f"Event {event!r} not in initials of {self!r}")

    def _state_key(self) -> tuple[object, ...]:
        return ("Guard", id(self.condition), self.process._state_key())

    def __repr__(self) -> str:
        return f"Guard(<condition>, {self.process!r})"


# --- Rename ---


@dataclass(frozen=True)
class Rename(Process):
    """
    Rename operator: Map event names for process composability.

    Rename(P, mapping) behaves as P but with specified events
    renamed in all transitions. The mapping is a dict from
    old Event to new Event.

    Agent use case: Two agents use different event vocabularies
    for the same logical operation. Rename bridges them without
    modifying either agent's internal logic.

    Example:
        agent_a = Prefix(Event("task_done"), Stop())
        compatible = Rename(
            process=agent_a,
            mapping={Event("task_done"): Event("work_complete")},
        )
        assert Event("work_complete") in compatible.initials()
    """

    process: Process
    mapping: tuple[tuple[Event, Event], ...]  # frozen-compatible (dict -> tuple of pairs)

    @classmethod
    def from_dict(cls, process: Process, mapping: dict[Event, Event]) -> Rename:
        """Convenience constructor from a dict."""
        return cls(process=process, mapping=tuple(mapping.items()))

    @property
    def _map(self) -> dict[Event, Event]:
        return dict(self.mapping)

    @property
    def _reverse_map(self) -> dict[Event, Event]:
        return {v: k for k, v in self.mapping}

    def _rename(self, event: Event) -> Event:
        return self._map.get(event, event)

    def _unrename(self, event: Event) -> Event:
        return self._reverse_map.get(event, event)

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.RENAME

    def alphabet(self) -> frozenset[Event]:
        return frozenset(self._rename(e) for e in self.process.alphabet())

    def initials(self) -> frozenset[Event]:
        return frozenset(self._rename(e) for e in self.process.initials())

    def after(self, event: Event) -> Process:
        """Translate the external event back to internal, step, then re-wrap."""
        internal_event = self._unrename(event)
        if internal_event in self.process.initials():
            return Rename(
                process=self.process.after(internal_event),
                mapping=self.mapping,
            )
        raise ValueError(f"Event {event!r} not in initials of {self!r}")

    def _state_key(self) -> tuple[object, ...]:
        return ("Rename", self.process._state_key(), tuple(sorted(self.mapping)))

    def __repr__(self) -> str:
        mapping_str = ", ".join(f"{k}<-{v}" for k, v in self.mapping)
        return f"({self.process!r})[[ {mapping_str} ]]"


# --- Pipe ---


@dataclass(frozen=True)
class Pipe(Process):
    """
    Pipe operator: Connect output of one process to input of another.

    Pipe(P, Q, channel) runs P and Q, synchronizing on channel events
    (P's outputs that are Q's inputs), hiding the channel from the
    external alphabet.

    Semantically: Pipe(P, Q, c) ~ Hiding(Parallel(P, Q), c)
    with specific synchronization rules.

    Agent use case: Retrieval -> Processing pipeline where one agent's
    output feeds the next agent's input. Each stage only sees its
    own external interface.

    Example:
        retriever = Prefix(Event("query"), Prefix(Event("emit_docs"), Stop()))
        processor = Prefix(Event("emit_docs"), Prefix(Event("summarize"), Stop()))
        pipeline = Pipe(
            producer=retriever,
            consumer=processor,
            channel=frozenset({Event("emit_docs")}),
        )
        # External: query -> summarize (emit_docs is hidden internal channel)
    """

    producer: Process
    consumer: Process
    channel: frozenset[Event]

    @property
    def kind(self) -> ProcessKind:
        return ProcessKind.PIPE

    def alphabet(self) -> frozenset[Event]:
        return (self.producer.alphabet() | self.consumer.alphabet()) - self.channel

    def initials(self) -> frozenset[Event]:
        # Producer's non-channel initials + channel sync events (if both ready)
        producer_initials = self.producer.initials()
        consumer_initials = self.consumer.initials()

        # External events from producer
        external = producer_initials - self.channel
        # Channel sync: if producer can emit and consumer can receive
        channel_ready = producer_initials & consumer_initials & self.channel

        result = external
        # If channel event is available, the sync happens silently
        # and we expose consumer's next non-channel initials
        if channel_ready:
            for ch_event in channel_ready:
                next_consumer = self.consumer.after(ch_event)
                next_producer = self.producer.after(ch_event)
                inner_pipe = Pipe(
                    producer=next_producer,
                    consumer=next_consumer,
                    channel=self.channel,
                )
                result = result | inner_pipe.initials()
        return result

    def after(self, event: Event) -> Process:
        if event in self.channel:
            raise ValueError(f"Channel event {event!r} is hidden in {self!r}")

        producer_initials = self.producer.initials()
        if event in producer_initials:
            return Pipe(
                producer=self.producer.after(event),
                consumer=self.consumer,
                channel=self.channel,
            )

        consumer_initials = self.consumer.initials()
        if event in (consumer_initials - self.channel):
            return Pipe(
                producer=self.producer,
                consumer=self.consumer.after(event),
                channel=self.channel,
            )

        raise ValueError(f"Event {event!r} not in initials of {self!r}")

    def _state_key(self) -> tuple[object, ...]:
        return ("Pipe", self.producer._state_key(), self.consumer._state_key(), self.channel)

    def __repr__(self) -> str:
        ch = ", ".join(str(e) for e in sorted(self.channel))
        return f"({self.producer!r} |>{{{ch}}}> {self.consumer!r})"
