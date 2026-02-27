"""Coordination patterns for multi-agent systems.

These patterns express common coordination scenarios in CSP,
enabling formal verification of multi-agent interactions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..csp import (
    Event,
    Process,
    parallel,
    prefix,
    rec,
    skip,
    var,
)
from ..refinement import RefinementResult, trace_refines
from ..semantics import is_deadlock_free

# =============================================================================
# Pattern Base
# =============================================================================


class CoordinationPattern(ABC):
    """Base class for coordination patterns."""

    @abstractmethod
    def participants(self) -> set[str]:
        """Return set of participant names."""
        pass

    @abstractmethod
    def events(self) -> set[Event]:
        """Return set of events in this pattern."""
        pass

    @abstractmethod
    def global_process(self) -> Process:
        """Return the global (system) process representing the pattern."""
        pass

    @abstractmethod
    def local_process(self, participant: str) -> Process:
        """Return the local process for a specific participant."""
        pass

    def compose(self, sync_events: set[str] | None = None) -> Process:
        """Compose all local processes into the global system.

        Args:
            sync_events: Events to synchronize on. Defaults to pattern events.

        Returns:
            Composed parallel process
        """
        parts = list(self.participants())
        if not parts:
            return skip()

        if sync_events is None:
            sync_events = {str(e) for e in self.events()}

        result = self.local_process(parts[0])
        for part in parts[1:]:
            result = parallel(result, self.local_process(part), sync_events)

        return result

    def is_deadlock_free(self) -> bool:
        """Check if the pattern is deadlock-free."""
        return is_deadlock_free(self.global_process())


# =============================================================================
# Request-Response Pattern
# =============================================================================


@dataclass
class RequestResponsePattern(CoordinationPattern):
    """Request-Response pattern for client-server interaction.

    Interaction:
        Client ─request→ Server
        Server ─response→ Client

    CSP Model:
        CLIENT = request → response → SKIP
        SERVER = request → response → SKIP
        SYSTEM = CLIENT |[{request, response}]| SERVER
    """

    client: str = "client"
    server: str = "server"
    request_event: str = "request"
    response_event: str = "response"
    repeatable: bool = False

    def participants(self) -> set[str]:
        return {self.client, self.server}

    def events(self) -> set[Event]:
        return {Event(self.request_event), Event(self.response_event)}

    def global_process(self) -> Process:
        body = prefix(
            self.request_event,
            prefix(self.response_event, skip()),
        )
        if self.repeatable:
            return rec("X", prefix(self.request_event, prefix(self.response_event, var("X"))))
        return body

    def local_process(self, participant: str) -> Process:
        if participant == self.client:
            body = prefix(self.request_event, prefix(self.response_event, skip()))
            if self.repeatable:
                return rec("X", prefix(self.request_event, prefix(self.response_event, var("X"))))
            return body
        elif participant == self.server:
            body = prefix(self.request_event, prefix(self.response_event, skip()))
            if self.repeatable:
                return rec("X", prefix(self.request_event, prefix(self.response_event, var("X"))))
            return body
        else:
            raise ValueError(f"Unknown participant: {participant}")


def request_response(
    client: str = "client",
    server: str = "server",
    request_event: str = "request",
    response_event: str = "response",
    repeatable: bool = False,
) -> Process:
    """Create a request-response pattern process.

    Args:
        client: Client participant name
        server: Server participant name
        request_event: Request event name
        response_event: Response event name
        repeatable: Whether the pattern repeats

    Returns:
        The global process representing the pattern
    """
    pattern = RequestResponsePattern(
        client=client,
        server=server,
        request_event=request_event,
        response_event=response_event,
        repeatable=repeatable,
    )
    return pattern.global_process()


# =============================================================================
# Pipeline Pattern
# =============================================================================


@dataclass
class PipelinePattern(CoordinationPattern):
    """Pipeline pattern for sequential multi-stage processing.

    Interaction:
        Stage1 ─data→ Stage2 ─data→ Stage3 → ...

    CSP Model:
        STAGE_1 = data_1_2 → SKIP
        STAGE_2 = data_1_2 → data_2_3 → SKIP
        STAGE_N = data_{n-1}_n → SKIP
        SYSTEM = STAGE_1 ||| STAGE_2 ||| ... ||| STAGE_N (synchronized on data events)
    """

    stages: Sequence[str]
    data_event_prefix: str = "data"
    repeatable: bool = False

    def __post_init__(self) -> None:
        if len(self.stages) < 2:
            raise ValueError("Pipeline requires at least 2 stages")

    def participants(self) -> set[str]:
        return set(self.stages)

    def events(self) -> set[Event]:
        events = set()
        for i in range(len(self.stages) - 1):
            events.add(Event(f"{self.data_event_prefix}_{self.stages[i]}_{self.stages[i + 1]}"))
        return events

    def global_process(self) -> Process:
        # Build chain: data_1_2 → data_2_3 → ... → SKIP
        result: Process = skip()
        for i in range(len(self.stages) - 2, -1, -1):
            event = f"{self.data_event_prefix}_{self.stages[i]}_{self.stages[i + 1]}"
            result = prefix(event, result)

        if self.repeatable:
            # Wrap in recursion
            inner: Process = var("X")
            for i in range(len(self.stages) - 2, -1, -1):
                event = f"{self.data_event_prefix}_{self.stages[i]}_{self.stages[i + 1]}"
                inner = prefix(event, inner)
            return rec("X", inner)

        return result

    def local_process(self, participant: str) -> Process:
        try:
            idx = list(self.stages).index(participant)
        except ValueError as e:
            raise ValueError(f"Unknown participant: {participant}") from e

        actions: list[str] = []

        # Receive from previous stage
        if idx > 0:
            actions.append(f"{self.data_event_prefix}_{self.stages[idx - 1]}_{participant}")

        # Send to next stage
        if idx < len(self.stages) - 1:
            actions.append(f"{self.data_event_prefix}_{participant}_{self.stages[idx + 1]}")

        # Build process
        if not actions:
            return skip()

        result: Process = skip() if not self.repeatable else var("X")
        for action in reversed(actions):
            result = prefix(action, result)

        if self.repeatable:
            return rec("X", result)
        return result


def pipeline(
    stages: Sequence[str],
    data_event_prefix: str = "data",
    repeatable: bool = False,
) -> Process:
    """Create a pipeline pattern process.

    Args:
        stages: Ordered list of stage names
        data_event_prefix: Prefix for data events
        repeatable: Whether the pattern repeats

    Returns:
        The global process representing the pattern
    """
    pattern = PipelinePattern(
        stages=stages,
        data_event_prefix=data_event_prefix,
        repeatable=repeatable,
    )
    return pattern.global_process()


# =============================================================================
# Scatter-Gather Pattern
# =============================================================================


@dataclass
class ScatterGatherPattern(CoordinationPattern):
    """Scatter-Gather pattern for parallel task distribution.

    Interaction:
        Coordinator ─task→ Worker1, Worker2, ...
        Worker1, Worker2, ... ─result→ Coordinator

    CSP Model:
        COORD = task_w1 → task_w2 → result_w1 → result_w2 → SKIP
        WORKER_i = task_wi → result_wi → SKIP
    """

    coordinator: str = "coordinator"
    workers: Sequence[str] = field(default_factory=list)
    task_event_prefix: str = "task"
    result_event_prefix: str = "result"
    repeatable: bool = False

    def participants(self) -> set[str]:
        return {self.coordinator} | set(self.workers)

    def events(self) -> set[Event]:
        events = set()
        for worker in self.workers:
            events.add(Event(f"{self.task_event_prefix}_{worker}"))
            events.add(Event(f"{self.result_event_prefix}_{worker}"))
        return events

    def global_process(self) -> Process:
        if not self.workers:
            return skip()

        # Build: task_w1 → task_w2 → ... → result_w1 → result_w2 → ... → SKIP
        result: Process = skip() if not self.repeatable else var("X")

        # Results in reverse order
        for worker in reversed(self.workers):
            result = prefix(f"{self.result_event_prefix}_{worker}", result)

        # Tasks in reverse order
        for worker in reversed(self.workers):
            result = prefix(f"{self.task_event_prefix}_{worker}", result)

        if self.repeatable:
            return rec("X", result)
        return result

    def local_process(self, participant: str) -> Process:
        if participant == self.coordinator:
            return self.global_process()
        elif participant in self.workers:
            body = prefix(
                f"{self.task_event_prefix}_{participant}",
                prefix(f"{self.result_event_prefix}_{participant}", skip()),
            )
            if self.repeatable:
                return rec(
                    "X",
                    prefix(
                        f"{self.task_event_prefix}_{participant}",
                        prefix(f"{self.result_event_prefix}_{participant}", var("X")),
                    ),
                )
            return body
        else:
            raise ValueError(f"Unknown participant: {participant}")


def scatter_gather(
    coordinator: str = "coordinator",
    workers: Sequence[str] = (),
    task_event_prefix: str = "task",
    result_event_prefix: str = "result",
    repeatable: bool = False,
) -> Process:
    """Create a scatter-gather pattern process.

    Args:
        coordinator: Coordinator name
        workers: List of worker names
        task_event_prefix: Prefix for task events
        result_event_prefix: Prefix for result events
        repeatable: Whether the pattern repeats

    Returns:
        The global process representing the pattern
    """
    pattern = ScatterGatherPattern(
        coordinator=coordinator,
        workers=list(workers),
        task_event_prefix=task_event_prefix,
        result_event_prefix=result_event_prefix,
        repeatable=repeatable,
    )
    return pattern.global_process()


# =============================================================================
# Barrier Pattern
# =============================================================================


@dataclass
class BarrierPattern(CoordinationPattern):
    """Barrier pattern for multi-party synchronization.

    All participants must reach the barrier before any can proceed.

    CSP Model:
        PARTICIPANT_i = arrive → depart → SKIP
        BARRIER = arrive_1 → arrive_2 → ... → depart_1 → depart_2 → ... → SKIP
    """

    participants_list: Sequence[str]
    arrive_event: str = "arrive"
    depart_event: str = "depart"
    repeatable: bool = False

    def participants(self) -> set[str]:
        return set(self.participants_list)

    def events(self) -> set[Event]:
        events = set()
        for p in self.participants_list:
            events.add(Event(f"{self.arrive_event}_{p}"))
            events.add(Event(f"{self.depart_event}_{p}"))
        return events

    def global_process(self) -> Process:
        if not self.participants_list:
            return skip()

        result: Process = skip() if not self.repeatable else var("X")

        # All depart in any order (external choice)
        for p in reversed(self.participants_list):
            result = prefix(f"{self.depart_event}_{p}", result)

        # All arrive (any order - external choice)
        for p in reversed(self.participants_list):
            result = prefix(f"{self.arrive_event}_{p}", result)

        if self.repeatable:
            return rec("X", result)
        return result

    def local_process(self, participant: str) -> Process:
        if participant not in self.participants_list:
            raise ValueError(f"Unknown participant: {participant}")

        body = prefix(
            f"{self.arrive_event}_{participant}",
            prefix(f"{self.depart_event}_{participant}", skip()),
        )

        if self.repeatable:
            return rec(
                "X",
                prefix(
                    f"{self.arrive_event}_{participant}",
                    prefix(f"{self.depart_event}_{participant}", var("X")),
                ),
            )
        return body


def barrier(
    participants: Sequence[str],
    arrive_event: str = "arrive",
    depart_event: str = "depart",
    repeatable: bool = False,
) -> Process:
    """Create a barrier synchronization pattern.

    Args:
        participants: List of participant names
        arrive_event: Prefix for arrival events
        depart_event: Prefix for departure events
        repeatable: Whether the pattern repeats

    Returns:
        The global process representing the pattern
    """
    pattern = BarrierPattern(
        participants_list=list(participants),
        arrive_event=arrive_event,
        depart_event=depart_event,
        repeatable=repeatable,
    )
    return pattern.global_process()


# =============================================================================
# Mutex Pattern
# =============================================================================


@dataclass
class MutexPattern(CoordinationPattern):
    """Mutual exclusion pattern.

    Only one process can be in the critical section at a time.

    CSP Model:
        MUTEX = acquire → release → MUTEX
        PROCESS_i = acquire → critical_i → release → SKIP
    """

    processes: Sequence[str]
    acquire_event: str = "acquire"
    release_event: str = "release"
    critical_event_prefix: str = "critical"

    def participants(self) -> set[str]:
        return set(self.processes)

    def events(self) -> set[Event]:
        events = {Event(self.acquire_event), Event(self.release_event)}
        for p in self.processes:
            events.add(Event(f"{self.critical_event_prefix}_{p}"))
        return events

    def global_process(self) -> Process:
        """The mutex resource process."""
        # μX. acquire → release → X
        return rec("X", prefix(self.acquire_event, prefix(self.release_event, var("X"))))

    def local_process(self, participant: str) -> Process:
        if participant not in self.processes:
            raise ValueError(f"Unknown participant: {participant}")

        # acquire → critical_p → release → SKIP
        return prefix(
            self.acquire_event,
            prefix(
                f"{self.critical_event_prefix}_{participant}",
                prefix(self.release_event, skip()),
            ),
        )

    def system_process(self) -> Process:
        """The full system with mutex and all processes."""
        mutex_proc = self.global_process()
        sync = {self.acquire_event, self.release_event}

        # Compose all processes with mutex
        result = mutex_proc
        for p in self.processes:
            result = parallel(result, self.local_process(p), sync)

        return result


def mutex(
    processes: Sequence[str],
    acquire_event: str = "acquire",
    release_event: str = "release",
    critical_event_prefix: str = "critical",
) -> Process:
    """Create a mutual exclusion pattern.

    Args:
        processes: List of process names
        acquire_event: Event to acquire mutex
        release_event: Event to release mutex
        critical_event_prefix: Prefix for critical section events

    Returns:
        The mutex resource process
    """
    pattern = MutexPattern(
        processes=list(processes),
        acquire_event=acquire_event,
        release_event=release_event,
        critical_event_prefix=critical_event_prefix,
    )
    return pattern.global_process()


# =============================================================================
# Producer-Consumer Pattern
# =============================================================================


@dataclass
class ProducerConsumerPattern(CoordinationPattern):
    """Producer-Consumer pattern with bounded buffer.

    CSP Model:
        PRODUCER = produce → put → PRODUCER
        CONSUMER = get → consume → CONSUMER
        BUFFER = put → get → BUFFER (simplified unbounded)
    """

    producer: str = "producer"
    consumer: str = "consumer"
    produce_event: str = "produce"
    consume_event: str = "consume"
    put_event: str = "put"
    get_event: str = "get"
    buffer_size: int = 1

    def participants(self) -> set[str]:
        return {self.producer, self.consumer, "buffer"}

    def events(self) -> set[Event]:
        return {
            Event(self.produce_event),
            Event(self.consume_event),
            Event(self.put_event),
            Event(self.get_event),
        }

    def global_process(self) -> Process:
        """Simplified: produce → put → get → consume → recurse."""
        return rec(
            "X",
            prefix(
                self.produce_event,
                prefix(
                    self.put_event,
                    prefix(
                        self.get_event,
                        prefix(self.consume_event, var("X")),
                    ),
                ),
            ),
        )

    def local_process(self, participant: str) -> Process:
        if participant == self.producer:
            return rec("X", prefix(self.produce_event, prefix(self.put_event, var("X"))))
        elif participant == self.consumer:
            return rec("X", prefix(self.get_event, prefix(self.consume_event, var("X"))))
        elif participant == "buffer":
            return rec("X", prefix(self.put_event, prefix(self.get_event, var("X"))))
        else:
            raise ValueError(f"Unknown participant: {participant}")


def producer_consumer(
    producer: str = "producer",
    consumer: str = "consumer",
    produce_event: str = "produce",
    consume_event: str = "consume",
    put_event: str = "put",
    get_event: str = "get",
) -> Process:
    """Create a producer-consumer pattern.

    Args:
        producer: Producer name
        consumer: Consumer name
        produce_event: Event when producer creates item
        consume_event: Event when consumer uses item
        put_event: Event to put item in buffer
        get_event: Event to get item from buffer

    Returns:
        The global process representing the pattern
    """
    pattern = ProducerConsumerPattern(
        producer=producer,
        consumer=consumer,
        produce_event=produce_event,
        consume_event=consume_event,
        put_event=put_event,
        get_event=get_event,
    )
    return pattern.global_process()


# =============================================================================
# Utilities
# =============================================================================


def compose_agents(
    agents: dict[str, Process],
    sync_events: set[str] | None = None,
) -> Process:
    """Compose multiple agent processes in parallel.

    Args:
        agents: Dict mapping agent names to their processes
        sync_events: Events to synchronize on (all if None)

    Returns:
        Parallel composition of all agents
    """
    if not agents:
        return skip()

    processes = list(agents.values())
    if len(processes) == 1:
        return processes[0]

    if sync_events is None:
        # Compute common alphabet
        alphabets = [p.alphabet() for p in processes]
        common = set(alphabets[0])
        for alpha in alphabets[1:]:
            common = common.intersection(alpha)
        sync_events = {str(e) for e in common}

    result = processes[0]
    for proc in processes[1:]:
        result = parallel(result, proc, sync_events)

    return result


@dataclass
class PatternVerification:
    """Result of pattern verification."""

    pattern_name: str
    is_deadlock_free: bool
    refines_spec: bool
    refinement_result: RefinementResult | None = None


def verify_pattern(
    pattern: CoordinationPattern,
    spec: Process | None = None,
) -> PatternVerification:
    """Verify properties of a coordination pattern.

    Args:
        pattern: The pattern to verify
        spec: Optional specification to check refinement against

    Returns:
        PatternVerification with results
    """
    global_proc = pattern.global_process()

    # Check deadlock freedom
    deadlock_free = is_deadlock_free(global_proc)

    # Check refinement if spec provided
    refines = True
    refinement_result = None
    if spec is not None:
        refinement_result = trace_refines(spec, global_proc)
        refines = refinement_result.is_valid

    return PatternVerification(
        pattern_name=pattern.__class__.__name__,
        is_deadlock_free=deadlock_free,
        refines_spec=refines,
        refinement_result=refinement_result,
    )


__all__ = [
    # Base
    "CoordinationPattern",
    # Request-Response
    "RequestResponsePattern",
    "request_response",
    # Pipeline
    "PipelinePattern",
    "pipeline",
    # Scatter-Gather
    "ScatterGatherPattern",
    "scatter_gather",
    # Barrier
    "BarrierPattern",
    "barrier",
    # Mutex
    "MutexPattern",
    "mutex",
    # Producer-Consumer
    "ProducerConsumerPattern",
    "producer_consumer",
    # Utilities
    "compose_agents",
    "PatternVerification",
    "verify_pattern",
]
