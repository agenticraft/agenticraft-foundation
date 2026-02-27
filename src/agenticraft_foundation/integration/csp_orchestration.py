"""CSP Process Algebra integration for workflow orchestration.

This module provides workflow verification using CSP refinement checking.

Key Features:
- Workflow specification using CSP process algebra
- Refinement checking of workflow implementations
- Deadlock freedom verification
- Liveness analysis for workflow progress
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..algebra import (
    TICK,
    Event,
    Process,
    analyze_liveness,
    build_lts,
    choice,
    detect_deadlock,
    failures_refines,
    fd_refines,
    is_deadlock_free,
    parallel,
    prefix,
    sequential,
    skip,
    trace_refines,
    traces,
)

# =============================================================================
# Workflow Specification Types
# =============================================================================


class WorkflowNodeType(str, Enum):
    """Types of nodes in a workflow."""

    TASK = "task"
    PARALLEL = "parallel"
    CHOICE = "choice"
    SEQUENCE = "sequence"
    LOOP = "loop"
    CONDITIONAL = "conditional"


@dataclass
class WorkflowSpec:
    """Specification of a workflow using CSP.

    Attributes:
        name: Workflow name
        process: The CSP process representing the workflow
        events_of_interest: Events to check for liveness
        invariants: Custom invariant predicates
    """

    name: str
    process: Process
    events_of_interest: frozenset[Event] = field(default_factory=frozenset)
    invariants: list[Callable[[Process], bool]] = field(default_factory=list)

    @classmethod
    def from_dag(
        cls,
        name: str,
        nodes: dict[str, WorkflowNodeType],
        edges: list[tuple[str, str]],
        sync_events: set[str] | None = None,
    ) -> WorkflowSpec:
        """Create a workflow spec from a DAG definition.

        Args:
            name: Workflow name
            nodes: Node ID to type mapping
            edges: List of (source, target) edges
            sync_events: Events that require synchronization

        Returns:
            WorkflowSpec instance
        """
        process = _dag_to_process(nodes, edges, sync_events or set())
        events = frozenset(
            Event(node_id) for node_id in nodes if nodes[node_id] == WorkflowNodeType.TASK
        )
        return cls(name=name, process=process, events_of_interest=events)

    @classmethod
    def sequential_tasks(cls, name: str, task_ids: list[str]) -> WorkflowSpec:
        """Create a sequential workflow spec.

        Args:
            name: Workflow name
            task_ids: Ordered list of task IDs

        Returns:
            WorkflowSpec instance
        """
        if not task_ids:
            return cls(name=name, process=skip())

        process: Process = skip()
        for task_id in reversed(task_ids):
            process = prefix(task_id, process)

        return cls(
            name=name,
            process=process,
            events_of_interest=frozenset(Event(tid) for tid in task_ids),
        )

    @classmethod
    def parallel_tasks(
        cls,
        name: str,
        task_ids: list[str],
        sync_on_complete: bool = True,
    ) -> WorkflowSpec:
        """Create a parallel workflow spec.

        Args:
            name: Workflow name
            task_ids: List of task IDs to run in parallel
            sync_on_complete: Whether tasks must all complete (sync on TICK)

        Returns:
            WorkflowSpec instance
        """
        if not task_ids:
            return cls(name=name, process=skip())

        processes = [prefix(task_id, skip()) for task_id in task_ids]
        sync_set = frozenset({TICK}) if sync_on_complete else frozenset()

        process: Process = processes[0]
        for p in processes[1:]:
            process = parallel(process, p, sync_set)

        return cls(
            name=name,
            process=process,
            events_of_interest=frozenset(Event(tid) for tid in task_ids),
        )

    @classmethod
    def choice_tasks(cls, name: str, task_ids: list[str]) -> WorkflowSpec:
        """Create a choice (branching) workflow spec.

        Args:
            name: Workflow name
            task_ids: List of alternative task IDs

        Returns:
            WorkflowSpec instance
        """
        if not task_ids:
            return cls(name=name, process=skip())

        processes = [prefix(task_id, skip()) for task_id in task_ids]

        process: Process = processes[0]
        for p in processes[1:]:
            process = choice(process, p)

        return cls(
            name=name,
            process=process,
            events_of_interest=frozenset(Event(tid) for tid in task_ids),
        )


def _dag_to_process(
    nodes: dict[str, WorkflowNodeType],
    edges: list[tuple[str, str]],
    sync_events: set[str] | None = None,
) -> Process:
    """Convert a DAG to a CSP process.

    Uses topological sort to build process from leaves up.

    Args:
        nodes: Node ID to type mapping.
        edges: List of (source, target) edges.
        sync_events: Events to synchronize on (reserved for future use).

    Returns:
        A CSP process representing the DAG workflow.
    """
    _ = sync_events  # Reserved for future synchronization support
    # Build adjacency lists
    successors: dict[str, list[str]] = {node: [] for node in nodes}
    predecessors: dict[str, list[str]] = {node: [] for node in nodes}

    for src, tgt in edges:
        successors[src].append(tgt)
        predecessors[tgt].append(src)

    # Find root nodes (no predecessors)
    roots = [n for n in nodes if not predecessors[n]]

    # Build process for each node, memoized
    cache: dict[str, Process] = {}

    def build_node(node_id: str) -> Process:
        if node_id in cache:
            return cache[node_id]

        node_type = nodes[node_id]
        succs = successors[node_id]

        # Build continuation from successors
        continuation: Process
        if not succs:
            continuation = skip()
        elif len(succs) == 1:
            continuation = build_node(succs[0])
        else:
            # Multiple successors - parallel composition
            succ_processes = [build_node(s) for s in succs]
            continuation = succ_processes[0]
            for p in succ_processes[1:]:
                continuation = parallel(continuation, p, frozenset())

        # Build this node's process
        process: Process
        if node_type == WorkflowNodeType.TASK:
            process = prefix(node_id, continuation)
        elif node_type == WorkflowNodeType.PARALLEL:
            # Parallel node starts all successors in parallel
            process = continuation
        elif node_type == WorkflowNodeType.CHOICE:
            # Choice node offers external choice between successors
            if succs:
                succ_processes = [build_node(s) for s in succs]
                process = succ_processes[0]
                for p in succ_processes[1:]:
                    process = choice(process, p)
            else:
                process = skip()
        elif node_type == WorkflowNodeType.SEQUENCE:
            process = continuation
        else:
            process = prefix(node_id, continuation)

        cache[node_id] = process
        return process

    # Build from roots
    if not roots:
        return skip()
    elif len(roots) == 1:
        return build_node(roots[0])
    else:
        root_processes = [build_node(r) for r in roots]
        result = root_processes[0]
        for p in root_processes[1:]:
            result = parallel(result, p, frozenset())
        return result


# =============================================================================
# Workflow Verification Result
# =============================================================================


@dataclass
class WorkflowVerificationResult:
    """Result of workflow verification.

    Attributes:
        is_valid: Whether the workflow passes all checks
        is_deadlock_free: Whether the workflow cannot deadlock
        is_live: Whether all events of interest are live
        refines_spec: Whether implementation refines specification
        violations: List of verification violations
        traces_sample: Sample of workflow traces
        counterexample: Counterexample trace if refinement fails
    """

    is_valid: bool
    is_deadlock_free: bool = True
    is_live: bool = True
    refines_spec: bool = True
    violations: list[str] = field(default_factory=list)
    traces_sample: list[tuple[Event, ...]] = field(default_factory=list)
    counterexample: tuple[Event, ...] | None = None

    @classmethod
    def passed(
        cls,
        traces_sample: list[tuple[Event, ...]] | None = None,
    ) -> WorkflowVerificationResult:
        """Create a passing verification result."""
        return cls(
            is_valid=True,
            traces_sample=traces_sample or [],
        )

    @classmethod
    def failed(
        cls,
        violations: list[str],
        is_deadlock_free: bool = True,
        is_live: bool = True,
        refines_spec: bool = True,
        counterexample: tuple[Event, ...] | None = None,
    ) -> WorkflowVerificationResult:
        """Create a failing verification result."""
        return cls(
            is_valid=False,
            is_deadlock_free=is_deadlock_free,
            is_live=is_live,
            refines_spec=refines_spec,
            violations=violations,
            counterexample=counterexample,
        )


# =============================================================================
# CSP Orchestration Adapter
# =============================================================================


class CSPOrchestrationAdapter:
    """Adapter integrating CSP process algebra with OrchestrationService.

    This adapter provides workflow verification using CSP refinement
    checking. It can verify that workflow implementations conform to
    their specifications and detect potential issues like deadlocks.

    Usage:
        adapter = CSPOrchestrationAdapter()

        # Register a workflow specification
        spec = WorkflowSpec.sequential_tasks("my_workflow", ["task1", "task2"])
        adapter.register_spec(spec)

        # Verify a workflow implementation
        result = adapter.verify_workflow("my_workflow", implementation_process)

        # Check deadlock freedom
        is_safe = adapter.check_deadlock_freedom("my_workflow")

        # Analyze liveness
        liveness = adapter.analyze_liveness("my_workflow")
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._specs: dict[str, WorkflowSpec] = {}
        self._implementations: dict[str, Process] = {}

    def register_spec(self, spec: WorkflowSpec) -> None:
        """Register a workflow specification.

        Args:
            spec: The workflow specification
        """
        self._specs[spec.name] = spec

    def get_spec(self, name: str) -> WorkflowSpec | None:
        """Get a registered workflow specification.

        Args:
            name: Workflow name

        Returns:
            The specification, or None if not registered
        """
        return self._specs.get(name)

    def register_implementation(self, name: str, process: Process) -> None:
        """Register a workflow implementation.

        Args:
            name: Workflow name
            process: The CSP process implementation
        """
        self._implementations[name] = process

    def verify_spec(self, spec: WorkflowSpec) -> WorkflowVerificationResult:
        """Verify a workflow specification is valid.

        Checks:
        - Deadlock freedom
        - Liveness of events of interest
        - Custom invariants

        Args:
            spec: The workflow specification

        Returns:
            Verification result
        """
        violations = []

        # Check deadlock freedom
        lts = build_lts(spec.process)
        deadlock_result = detect_deadlock(lts)

        if deadlock_result.has_deadlock:
            violations.append(
                f"Specification has potential deadlock at states: {deadlock_result.deadlock_states}"
            )

        # Check liveness
        if spec.events_of_interest:
            liveness_result = analyze_liveness(lts, set(spec.events_of_interest))

            for event, is_live in liveness_result.live_events.items():
                if not is_live:
                    violations.append(f"Event '{event}' is not live (never possible)")

        # Check custom invariants
        for i, invariant in enumerate(spec.invariants):
            try:
                if not invariant(spec.process):
                    violations.append(f"Custom invariant {i} failed")
            except Exception as e:
                violations.append(f"Custom invariant {i} raised exception: {e}")

        # Get sample traces
        sample_traces = list(traces(lts, max_length=10))[:5]

        if violations:
            result = WorkflowVerificationResult.failed(
                violations=violations,
                is_deadlock_free=not deadlock_result.has_deadlock,
            )
            result.traces_sample = sample_traces
            return result

        return WorkflowVerificationResult.passed(traces_sample=sample_traces)

    def verify_workflow(
        self,
        name: str,
        implementation: Process | None = None,
        mode: str = "trace",
    ) -> WorkflowVerificationResult:
        """Verify a workflow implementation against its specification.

        Args:
            name: Workflow name
            implementation: Optional implementation process (uses registered if None)
            mode: Refinement mode ("trace", "failures", "fd")

        Returns:
            Verification result
        """
        spec = self._specs.get(name)
        if spec is None:
            return WorkflowVerificationResult.failed(
                violations=[f"No specification registered for workflow '{name}'"],
            )

        impl = implementation or self._implementations.get(name)
        if impl is None:
            return WorkflowVerificationResult.failed(
                violations=[f"No implementation provided for workflow '{name}'"],
            )

        # First verify the spec itself
        spec_result = self.verify_spec(spec)
        if not spec_result.is_valid:
            return spec_result

        # Then check refinement
        if mode == "trace":
            ref_result = trace_refines(spec.process, impl)
        elif mode == "failures":
            ref_result = failures_refines(spec.process, impl)
        elif mode == "fd":
            ref_result = fd_refines(spec.process, impl)
        else:
            return WorkflowVerificationResult.failed(
                violations=[f"Unknown refinement mode: {mode}"],
            )

        if not ref_result.is_valid:
            # Narrow counterexample to tuple[Event, ...] | None (exclude Failure)
            counterexample = (
                ref_result.counterexample
                if isinstance(ref_result.counterexample, tuple)
                else None
            )
            return WorkflowVerificationResult.failed(
                violations=[f"Implementation does not refine specification ({mode})"],
                refines_spec=False,
                counterexample=counterexample,
            )

        # Check implementation-specific properties
        impl_lts = build_lts(impl)
        impl_deadlock = detect_deadlock(impl_lts)

        violations = []
        if impl_deadlock.has_deadlock:
            violations.append(
                f"Implementation has potential deadlock: {impl_deadlock.deadlock_traces[:3]}"
            )

        if violations:
            return WorkflowVerificationResult.failed(
                violations=violations,
                is_deadlock_free=False,
                refines_spec=True,
            )

        return WorkflowVerificationResult.passed()

    def check_deadlock_freedom(self, name: str) -> bool:
        """Check if a workflow specification is deadlock-free.

        Args:
            name: Workflow name

        Returns:
            True if deadlock-free
        """
        spec = self._specs.get(name)
        if spec is None:
            return False

        return is_deadlock_free(spec.process)

    def analyze_workflow_liveness(
        self,
        name: str,
        events: set[Event] | None = None,
    ) -> dict[Event, bool]:
        """Analyze liveness of events in a workflow.

        Args:
            name: Workflow name
            events: Events to check (uses spec's events_of_interest if None)

        Returns:
            Mapping of event to liveness status
        """
        spec = self._specs.get(name)
        if spec is None:
            return {}

        check_events = events or set(spec.events_of_interest)
        lts = build_lts(spec.process)
        result = analyze_liveness(lts, check_events)

        return result.live_events

    def get_workflow_traces(
        self,
        name: str,
        max_length: int = 20,
        max_count: int = 100,
    ) -> list[tuple[Event, ...]]:
        """Get sample traces from a workflow.

        Args:
            name: Workflow name
            max_length: Maximum trace length
            max_count: Maximum number of traces

        Returns:
            List of traces
        """
        spec = self._specs.get(name)
        if spec is None:
            return []

        lts = build_lts(spec.process)
        all_traces = list(traces(lts, max_length=max_length))

        return all_traces[:max_count]

    def compose_workflows(
        self,
        name: str,
        workflow_names: list[str],
        composition_type: str = "parallel",
        sync_events: set[str] | None = None,
    ) -> WorkflowSpec | None:
        """Compose multiple workflows into a new workflow.

        Args:
            name: Name for the composed workflow
            workflow_names: Names of workflows to compose
            composition_type: "parallel", "sequential", or "choice"
            sync_events: Events to synchronize on (for parallel)

        Returns:
            Composed workflow spec, or None if any workflow not found
        """
        processes = []
        all_events: set[Event] = set()

        for wf_name in workflow_names:
            spec = self._specs.get(wf_name)
            if spec is None:
                return None
            processes.append(spec.process)
            all_events.update(spec.events_of_interest)

        if not processes:
            return WorkflowSpec(name=name, process=skip())

        if composition_type == "parallel":
            sync_set = frozenset(Event(e) for e in (sync_events or set()))
            composed = processes[0]
            for p in processes[1:]:
                composed = parallel(composed, p, sync_set)
        elif composition_type == "sequential":
            composed = processes[0]
            for p in processes[1:]:
                composed = sequential(composed, p)
        elif composition_type == "choice":
            composed = processes[0]
            for p in processes[1:]:
                composed = choice(composed, p)
        else:
            return None

        result = WorkflowSpec(
            name=name,
            process=composed,
            events_of_interest=frozenset(all_events),
        )

        # Auto-register the composed workflow
        self.register_spec(result)

        return result

    def workflow_to_process(
        self,
        dag: dict[str, Any],
    ) -> Process:
        """Convert an orchestration DAG to a CSP process.

        This method can be used to convert the OrchestrationService's
        internal DAG representation to a CSP process for verification.

        Args:
            dag: DAG definition with nodes and edges

        Returns:
            CSP process representation
        """
        nodes = {}
        edges = []

        # Parse DAG structure
        for node_id, node_data in dag.get("nodes", {}).items():
            node_type_str = node_data.get("type", "task")
            try:
                node_type = WorkflowNodeType(node_type_str)
            except ValueError:
                node_type = WorkflowNodeType.TASK
            nodes[node_id] = node_type

        for edge in dag.get("edges", []):
            src = edge.get("source") or edge.get("from")
            tgt = edge.get("target") or edge.get("to")
            if src and tgt:
                edges.append((src, tgt))

        sync_events = set(dag.get("sync_events", []))

        return _dag_to_process(nodes, edges, sync_events)


__all__ = [
    "WorkflowNodeType",
    "WorkflowSpec",
    "WorkflowVerificationResult",
    "CSPOrchestrationAdapter",
]
