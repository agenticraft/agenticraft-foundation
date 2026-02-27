"""Protocol Workflow Model: W = (T, ≺, ρ).

Formal model for protocol-aware workflow composition.

- T: Set of tasks (nodes in the workflow DAG)
- ≺: Precedence relation (directed edges, forming a DAG)
- ρ: T → P assigns each task to a protocol

Implements:
- Workflow executability verification (Theorem 9 conditions)
- Optimal protocol assignment via DP (tree-structured) and greedy (general DAGs)
- Translation cost minimization across task boundaries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticraft_foundation.types import ProtocolName

if TYPE_CHECKING:
    from .compatibility import ProtocolCompatibilityMatrix
    from .graph import ProtocolGraph

logger = logging.getLogger(__name__)


class WorkflowTaskStatus(str, Enum):
    """Status of a workflow task."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowTask:
    """A task in the protocol workflow.

    Each task has an ID, required capabilities, and an assigned protocol.
    """

    task_id: str
    """Unique task identifier"""

    capabilities: list[str] = field(default_factory=list)
    """Capabilities required to execute this task"""

    assigned_protocol: ProtocolName | None = None
    """Protocol assigned to this task (ρ(t))"""

    compatible_protocols: set[ProtocolName] = field(default_factory=set)
    """Protocols that can execute this task"""

    assigned_agent: str | None = None
    """Agent assigned to execute this task"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional task metadata"""

    status: WorkflowTaskStatus = WorkflowTaskStatus.PENDING
    """Current task status"""

    def __hash__(self) -> int:
        return hash(self.task_id)


@dataclass
class WorkflowEdge:
    """Precedence edge in the workflow DAG."""

    source: str
    """Predecessor task ID"""

    target: str
    """Successor task ID"""

    data_dependency: bool = True
    """Whether the edge represents a data dependency"""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolWorkflow:
    """Protocol workflow model: W = (T, ≺, ρ).

    Represents a directed acyclic graph of tasks with protocol assignments.
    """

    workflow_id: str
    """Unique workflow identifier"""

    tasks: dict[str, WorkflowTask] = field(default_factory=dict)
    """Task set T, keyed by task_id"""

    edges: list[WorkflowEdge] = field(default_factory=list)
    """Precedence relation ≺"""

    metadata: dict[str, Any] = field(default_factory=dict)

    def add_task(
        self,
        task_id: str,
        capabilities: list[str] | None = None,
        compatible_protocols: set[ProtocolName] | None = None,
        assigned_protocol: ProtocolName | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowTask:
        """Add a task to the workflow."""
        task = WorkflowTask(
            task_id=task_id,
            capabilities=capabilities or [],
            compatible_protocols=compatible_protocols or set(),
            assigned_protocol=assigned_protocol,
            metadata=metadata or {},
        )
        self.tasks[task_id] = task
        return task

    def add_precedence(
        self,
        source: str,
        target: str,
        data_dependency: bool = True,
    ) -> WorkflowEdge:
        """Add a precedence edge: source ≺ target (source must complete before target)."""
        if source not in self.tasks:
            msg = f"Source task '{source}' not found in workflow"
            raise ValueError(msg)
        if target not in self.tasks:
            msg = f"Target task '{target}' not found in workflow"
            raise ValueError(msg)

        edge = WorkflowEdge(
            source=source,
            target=target,
            data_dependency=data_dependency,
        )
        self.edges.append(edge)
        return edge

    def predecessors(self, task_id: str) -> list[str]:
        """Get predecessor task IDs."""
        return [e.source for e in self.edges if e.target == task_id]

    def successors(self, task_id: str) -> list[str]:
        """Get successor task IDs."""
        return [e.target for e in self.edges if e.source == task_id]

    def roots(self) -> list[str]:
        """Get tasks with no predecessors (entry points)."""
        all_targets = {e.target for e in self.edges}
        return [tid for tid in self.tasks if tid not in all_targets]

    def leaves(self) -> list[str]:
        """Get tasks with no successors (exit points)."""
        all_sources = {e.source for e in self.edges}
        return [tid for tid in self.tasks if tid not in all_sources]

    def topological_sort(self) -> list[str]:
        """Return tasks in topological order. Raises ValueError if cycle detected."""
        in_degree: dict[str, int] = {tid: 0 for tid in self.tasks}
        for edge in self.edges:
            in_degree[edge.target] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result: list[str] = []

        while queue:
            # Sort for deterministic ordering
            queue.sort()
            node = queue.pop(0)
            result.append(node)

            for succ in self.successors(node):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(result) != len(self.tasks):
            msg = "Workflow contains a cycle — not a valid DAG"
            raise ValueError(msg)

        return result

    def is_tree(self) -> bool:
        """Check if the workflow DAG is a tree (each node has at most one predecessor)."""
        for tid in self.tasks:
            if len(self.predecessors(tid)) > 1:
                return False
        return True

    def translation_points(self) -> list[tuple[str, str, ProtocolName, ProtocolName]]:
        """Find edges where protocol translation is needed.

        Returns list of (source_task, target_task, source_protocol, target_protocol).
        """
        points = []
        for edge in self.edges:
            src_task = self.tasks[edge.source]
            tgt_task = self.tasks[edge.target]
            if (
                src_task.assigned_protocol is not None
                and tgt_task.assigned_protocol is not None
                and src_task.assigned_protocol != tgt_task.assigned_protocol
            ):
                points.append(
                    (
                        edge.source,
                        edge.target,
                        src_task.assigned_protocol,
                        tgt_task.assigned_protocol,
                    )
                )
        return points

    def total_translation_cost(
        self,
        base_cost_per_translation: float = 0.1,
    ) -> float:
        """Calculate total protocol translation cost across the workflow."""
        return len(self.translation_points()) * base_cost_per_translation


@dataclass
class WorkflowValidationResult:
    """Result of workflow validation."""

    valid: bool
    """Whether the workflow is executable"""

    errors: list[str] = field(default_factory=list)
    """Validation errors"""

    warnings: list[str] = field(default_factory=list)
    """Validation warnings"""

    translation_count: int = 0
    """Number of protocol translations needed"""

    estimated_cost: float = 0.0
    """Estimated total execution cost"""


class WorkflowValidator:
    """Validate workflow executability.

    Checks Theorem 9 conditions:
    1. The workflow forms a valid DAG (no cycles)
    2. Every task has an assigned protocol
    3. Every task's assigned protocol is in its compatible set
    4. For each precedence edge, the protocols are compatible
       (directly or via translation)
    """

    def __init__(
        self,
        compatibility: ProtocolCompatibilityMatrix | None = None,
    ):
        self._compatibility = compatibility

    def validate(
        self,
        workflow: ProtocolWorkflow,
        graph: ProtocolGraph | None = None,
    ) -> WorkflowValidationResult:
        """Validate workflow executability.

        Args:
            workflow: The protocol workflow to validate
            graph: Optional protocol graph for agent-level validation

        Returns:
            WorkflowValidationResult with validation outcome
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Condition 1: DAG check
        try:
            workflow.topological_sort()
        except ValueError as e:
            errors.append(f"DAG violation: {e}")
            return WorkflowValidationResult(
                valid=False,
                errors=errors,
            )

        # Empty workflow is valid
        if not workflow.tasks:
            return WorkflowValidationResult(valid=True)

        # Condition 2: Protocol assignment check
        unassigned = [tid for tid, task in workflow.tasks.items() if task.assigned_protocol is None]
        if unassigned:
            errors.append(f"Tasks without protocol assignment: {unassigned}")

        # Condition 3: Protocol compatibility check
        for tid, task in workflow.tasks.items():
            if (
                task.assigned_protocol is not None
                and task.compatible_protocols
                and task.assigned_protocol not in task.compatible_protocols
            ):
                errors.append(
                    f"Task '{tid}' assigned protocol "
                    f"{task.assigned_protocol.value} not in compatible set "
                    f"{{{', '.join(p.value for p in task.compatible_protocols)}}}"
                )

        # Condition 4: Edge protocol compatibility
        for edge in workflow.edges:
            src_task = workflow.tasks[edge.source]
            tgt_task = workflow.tasks[edge.target]

            if (
                src_task.assigned_protocol is not None
                and tgt_task.assigned_protocol is not None
                and src_task.assigned_protocol != tgt_task.assigned_protocol
            ):
                if self._compatibility is not None:
                    level = self._compatibility.get_compatibility(
                        src_task.assigned_protocol,
                        tgt_task.assigned_protocol,
                    )
                    from .compatibility import CompatibilityLevel

                    if level.level == CompatibilityLevel.NONE:
                        errors.append(
                            f"Incompatible protocols on edge "
                            f"{edge.source}→{edge.target}: "
                            f"{src_task.assigned_protocol.value} → "
                            f"{tgt_task.assigned_protocol.value}"
                        )
                    elif level.level == CompatibilityLevel.PARTIAL:
                        warnings.append(
                            f"Partial compatibility on edge "
                            f"{edge.source}→{edge.target}: "
                            f"{src_task.assigned_protocol.value} → "
                            f"{tgt_task.assigned_protocol.value}"
                        )
                else:
                    warnings.append(
                        f"Protocol translation needed on edge "
                        f"{edge.source}→{edge.target}: "
                        f"{src_task.assigned_protocol.value} → "
                        f"{tgt_task.assigned_protocol.value}"
                    )

        # Agent-level validation
        if graph is not None:
            for tid, task in workflow.tasks.items():
                if task.assigned_agent is not None:
                    if task.assigned_agent not in graph.agents:
                        errors.append(
                            f"Task '{tid}' assigned to unknown agent '{task.assigned_agent}'"
                        )
                    elif (
                        task.assigned_protocol is not None
                        and task.assigned_protocol
                        not in graph.agents[task.assigned_agent].protocols
                    ):
                        errors.append(
                            f"Agent '{task.assigned_agent}' for task '{tid}' "
                            f"does not support protocol "
                            f"{task.assigned_protocol.value}"
                        )

        translation_points = workflow.translation_points()

        return WorkflowValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            translation_count=len(translation_points),
            estimated_cost=workflow.total_translation_cost(),
        )


class OptimalProtocolAssigner:
    """Assign protocols to tasks minimizing total translation cost.

    For tree-structured workflows: DP in O(|T| * |P|^2).
    For general DAGs: greedy heuristic.
    """

    def __init__(
        self,
        available_protocols: set[ProtocolName] | None = None,
        base_translation_cost: float = 0.1,
    ):
        self._protocols = available_protocols or set(ProtocolName)
        self._base_cost = base_translation_cost

    def assign(
        self,
        workflow: ProtocolWorkflow,
    ) -> dict[str, ProtocolName]:
        """Assign protocols to minimize translation cost.

        Uses DP for trees, greedy for general DAGs.

        Args:
            workflow: The workflow to assign protocols for

        Returns:
            Mapping from task_id to assigned protocol
        """
        if not workflow.tasks:
            return {}

        # Only assign to tasks without a fixed assignment
        fixed = {
            tid: task.assigned_protocol
            for tid, task in workflow.tasks.items()
            if task.assigned_protocol is not None
        }

        if len(fixed) == len(workflow.tasks):
            return {tid: p for tid, p in fixed.items() if p is not None}

        if workflow.is_tree():
            return self._assign_tree_dp(workflow, fixed)
        return self._assign_greedy(workflow, fixed)

    def _assign_tree_dp(
        self,
        workflow: ProtocolWorkflow,
        fixed: dict[str, ProtocolName],
    ) -> dict[str, ProtocolName]:
        """DP assignment for tree workflows: O(|T| * |P|^2).

        For each task, compute the minimum cost of assigning each protocol,
        considering children's costs.
        """
        topo_order = workflow.topological_sort()
        protocols = list(self._protocols)

        # cost[tid][p] = min cost of subtree rooted at tid when tid uses protocol p
        cost: dict[str, dict[ProtocolName, float]] = {}
        # best_child_proto[tid][p][child_id] = best protocol for child
        best_child: dict[str, dict[ProtocolName, dict[str, ProtocolName]]] = {}

        # Process in reverse topological order (leaves first)
        for tid in reversed(topo_order):
            task = workflow.tasks[tid]
            candidate_protocols = (
                [fixed[tid]]
                if tid in fixed
                else [
                    p
                    for p in protocols
                    if not task.compatible_protocols or p in task.compatible_protocols
                ]
            )

            cost[tid] = {}
            best_child[tid] = {}

            for p in candidate_protocols:
                total = 0.0
                child_assignments: dict[str, ProtocolName] = {}

                for child_id in workflow.successors(tid):
                    if child_id not in cost:
                        continue

                    # Find best protocol for child given parent uses p
                    min_child_cost = float("inf")
                    best_child_p = protocols[0] if protocols else ProtocolName.MCP

                    for cp in cost[child_id]:
                        edge_cost = 0.0 if cp == p else self._base_cost
                        child_total = cost[child_id][cp] + edge_cost
                        if child_total < min_child_cost:
                            min_child_cost = child_total
                            best_child_p = cp

                    total += min_child_cost
                    child_assignments[child_id] = best_child_p

                cost[tid][p] = total
                best_child[tid][p] = child_assignments

        # Backtrack from roots to build assignment
        assignment: dict[str, ProtocolName] = {}

        for root in workflow.roots():
            if root not in cost:
                continue

            # Pick best protocol for root
            best_p = min(cost[root], key=cost[root].get)  # type: ignore[arg-type]
            self._backtrack_assignment(
                workflow,
                root,
                best_p,
                best_child,
                assignment,
            )

        return assignment

    def _backtrack_assignment(
        self,
        workflow: ProtocolWorkflow,
        tid: str,
        protocol: ProtocolName,
        best_child: dict[str, dict[ProtocolName, dict[str, ProtocolName]]],
        assignment: dict[str, ProtocolName],
    ) -> None:
        """Backtrack from root to assign protocols."""
        assignment[tid] = protocol

        if tid in best_child and protocol in best_child[tid]:
            for child_id, child_p in best_child[tid][protocol].items():
                self._backtrack_assignment(
                    workflow,
                    child_id,
                    child_p,
                    best_child,
                    assignment,
                )

    def _assign_greedy(
        self,
        workflow: ProtocolWorkflow,
        fixed: dict[str, ProtocolName],
    ) -> dict[str, ProtocolName]:
        """Greedy assignment for general DAGs.

        Process tasks in topological order. For each task, choose the protocol
        that minimizes translation cost with already-assigned predecessors.
        """
        topo_order = workflow.topological_sort()
        protocols = list(self._protocols)
        assignment: dict[str, ProtocolName] = dict(fixed)

        for tid in topo_order:
            if tid in assignment:
                continue

            task = workflow.tasks[tid]
            candidates = [
                p
                for p in protocols
                if not task.compatible_protocols or p in task.compatible_protocols
            ]

            if not candidates:
                candidates = protocols

            # Score each candidate by translation cost with predecessors
            best_p = candidates[0]
            best_cost = float("inf")

            for p in candidates:
                cost = 0.0
                for pred_id in workflow.predecessors(tid):
                    if pred_id in assignment and assignment[pred_id] != p:
                        cost += self._base_cost
                # Tie-break: also consider successors already assigned
                for succ_id in workflow.successors(tid):
                    if succ_id in assignment and assignment[succ_id] != p:
                        cost += self._base_cost

                if cost < best_cost:
                    best_cost = cost
                    best_p = p

            assignment[tid] = best_p

        return assignment

    def apply_assignment(
        self,
        workflow: ProtocolWorkflow,
        assignment: dict[str, ProtocolName],
    ) -> None:
        """Apply a protocol assignment to the workflow tasks."""
        for tid, protocol in assignment.items():
            if tid in workflow.tasks:
                workflow.tasks[tid].assigned_protocol = protocol


__all__ = [
    "WorkflowTaskStatus",
    "WorkflowTask",
    "WorkflowEdge",
    "ProtocolWorkflow",
    "WorkflowValidationResult",
    "WorkflowValidator",
    "OptimalProtocolAssigner",
]
