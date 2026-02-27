"""Tests for protocol workflow model.

Covers:
- ProtocolWorkflow construction and DAG operations
- WorkflowValidator executability checks
- OptimalProtocolAssigner (tree DP and greedy DAG)
"""

from __future__ import annotations

import pytest

from agenticraft_foundation.protocols.workflow import (
    OptimalProtocolAssigner,
    ProtocolWorkflow,
    WorkflowValidator,
)
from agenticraft_foundation.types import ProtocolName


@pytest.fixture
def linear_workflow() -> ProtocolWorkflow:
    """Linear workflow: t1 → t2 → t3."""
    wf = ProtocolWorkflow(workflow_id="linear")
    wf.add_task("t1", compatible_protocols={ProtocolName.MCP, ProtocolName.A2A})
    wf.add_task("t2", compatible_protocols={ProtocolName.MCP, ProtocolName.A2A})
    wf.add_task("t3", compatible_protocols={ProtocolName.MCP, ProtocolName.A2A})
    wf.add_precedence("t1", "t2")
    wf.add_precedence("t2", "t3")
    return wf


@pytest.fixture
def diamond_workflow() -> ProtocolWorkflow:
    """Diamond workflow: t1 → {t2, t3} → t4."""
    wf = ProtocolWorkflow(workflow_id="diamond")
    wf.add_task("t1", compatible_protocols={ProtocolName.MCP})
    wf.add_task("t2", compatible_protocols={ProtocolName.MCP, ProtocolName.A2A})
    wf.add_task("t3", compatible_protocols={ProtocolName.A2A})
    wf.add_task("t4", compatible_protocols={ProtocolName.MCP, ProtocolName.A2A})
    wf.add_precedence("t1", "t2")
    wf.add_precedence("t1", "t3")
    wf.add_precedence("t2", "t4")
    wf.add_precedence("t3", "t4")
    return wf


class TestProtocolWorkflow:
    def test_add_task(self):
        wf = ProtocolWorkflow(workflow_id="test")
        task = wf.add_task("t1", ["search"], {ProtocolName.MCP})
        assert task.task_id == "t1"
        assert "t1" in wf.tasks

    def test_add_precedence(self):
        wf = ProtocolWorkflow(workflow_id="test")
        wf.add_task("t1")
        wf.add_task("t2")
        edge = wf.add_precedence("t1", "t2")
        assert edge.source == "t1"
        assert edge.target == "t2"

    def test_add_precedence_unknown_task_raises(self):
        wf = ProtocolWorkflow(workflow_id="test")
        wf.add_task("t1")
        with pytest.raises(ValueError, match="not found"):
            wf.add_precedence("t1", "t_nonexistent")

    def test_topological_sort_linear(self, linear_workflow: ProtocolWorkflow):
        order = linear_workflow.topological_sort()
        assert order == ["t1", "t2", "t3"]

    def test_topological_sort_diamond(self, diamond_workflow: ProtocolWorkflow):
        order = diamond_workflow.topological_sort()
        assert order.index("t1") < order.index("t2")
        assert order.index("t1") < order.index("t3")
        assert order.index("t2") < order.index("t4")
        assert order.index("t3") < order.index("t4")

    def test_cycle_detection(self):
        wf = ProtocolWorkflow(workflow_id="cycle")
        wf.add_task("t1")
        wf.add_task("t2")
        wf.add_task("t3")
        wf.add_precedence("t1", "t2")
        wf.add_precedence("t2", "t3")
        wf.add_precedence("t3", "t1")  # Cycle!
        with pytest.raises(ValueError, match="cycle"):
            wf.topological_sort()

    def test_roots(self, diamond_workflow: ProtocolWorkflow):
        roots = diamond_workflow.roots()
        assert roots == ["t1"]

    def test_leaves(self, diamond_workflow: ProtocolWorkflow):
        leaves = diamond_workflow.leaves()
        assert leaves == ["t4"]

    def test_predecessors(self, diamond_workflow: ProtocolWorkflow):
        preds = diamond_workflow.predecessors("t4")
        assert set(preds) == {"t2", "t3"}

    def test_successors(self, diamond_workflow: ProtocolWorkflow):
        succs = diamond_workflow.successors("t1")
        assert set(succs) == {"t2", "t3"}

    def test_is_tree_linear(self, linear_workflow: ProtocolWorkflow):
        assert linear_workflow.is_tree()

    def test_is_tree_diamond(self, diamond_workflow: ProtocolWorkflow):
        assert not diamond_workflow.is_tree()  # t4 has 2 predecessors

    def test_translation_points(self):
        wf = ProtocolWorkflow(workflow_id="test")
        wf.add_task("t1", assigned_protocol=ProtocolName.MCP)
        wf.add_task("t2", assigned_protocol=ProtocolName.A2A)
        wf.add_precedence("t1", "t2")
        points = wf.translation_points()
        assert len(points) == 1
        assert points[0][2] == ProtocolName.MCP
        assert points[0][3] == ProtocolName.A2A

    def test_no_translation_same_protocol(self):
        wf = ProtocolWorkflow(workflow_id="test")
        wf.add_task("t1", assigned_protocol=ProtocolName.MCP)
        wf.add_task("t2", assigned_protocol=ProtocolName.MCP)
        wf.add_precedence("t1", "t2")
        assert len(wf.translation_points()) == 0

    def test_total_translation_cost(self):
        wf = ProtocolWorkflow(workflow_id="test")
        wf.add_task("t1", assigned_protocol=ProtocolName.MCP)
        wf.add_task("t2", assigned_protocol=ProtocolName.A2A)
        wf.add_task("t3", assigned_protocol=ProtocolName.MCP)
        wf.add_precedence("t1", "t2")
        wf.add_precedence("t2", "t3")
        assert wf.total_translation_cost(base_cost_per_translation=0.5) == 1.0

    def test_empty_workflow(self):
        wf = ProtocolWorkflow(workflow_id="empty")
        assert wf.topological_sort() == []
        assert wf.roots() == []
        assert wf.leaves() == []


class TestWorkflowValidator:
    def test_valid_workflow(self, linear_workflow: ProtocolWorkflow):
        # Assign protocols first
        for task in linear_workflow.tasks.values():
            task.assigned_protocol = ProtocolName.MCP
        validator = WorkflowValidator()
        result = validator.validate(linear_workflow)
        assert result.valid

    def test_unassigned_protocols(self, linear_workflow: ProtocolWorkflow):
        validator = WorkflowValidator()
        result = validator.validate(linear_workflow)
        assert not result.valid
        assert any("without protocol" in e for e in result.errors)

    def test_incompatible_protocol_assignment(self):
        wf = ProtocolWorkflow(workflow_id="test")
        wf.add_task(
            "t1", compatible_protocols={ProtocolName.MCP}, assigned_protocol=ProtocolName.A2A
        )
        validator = WorkflowValidator()
        result = validator.validate(wf)
        assert not result.valid

    def test_empty_workflow_valid(self):
        wf = ProtocolWorkflow(workflow_id="empty")
        validator = WorkflowValidator()
        result = validator.validate(wf)
        assert result.valid

    def test_translation_count_reported(self):
        wf = ProtocolWorkflow(workflow_id="test")
        wf.add_task("t1", assigned_protocol=ProtocolName.MCP)
        wf.add_task("t2", assigned_protocol=ProtocolName.A2A)
        wf.add_precedence("t1", "t2")
        validator = WorkflowValidator()
        result = validator.validate(wf)
        assert result.translation_count == 1


class TestOptimalProtocolAssigner:
    def test_linear_same_protocol(self, linear_workflow: ProtocolWorkflow):
        assigner = OptimalProtocolAssigner()
        assignment = assigner.assign(linear_workflow)
        # Should assign same protocol to minimize translations
        protocols = set(assignment.values())
        assert len(protocols) == 1  # All same protocol

    def test_fixed_assignments_respected(self):
        wf = ProtocolWorkflow(workflow_id="test")
        wf.add_task("t1", assigned_protocol=ProtocolName.MCP)
        wf.add_task("t2")
        wf.add_task("t3", assigned_protocol=ProtocolName.MCP)
        wf.add_precedence("t1", "t2")
        wf.add_precedence("t2", "t3")
        assigner = OptimalProtocolAssigner()
        assignment = assigner.assign(wf)
        assert assignment["t1"] == ProtocolName.MCP
        assert assignment["t3"] == ProtocolName.MCP
        # t2 should also be MCP to minimize cost
        assert assignment["t2"] == ProtocolName.MCP

    def test_diamond_greedy(self, diamond_workflow: ProtocolWorkflow):
        assigner = OptimalProtocolAssigner()
        assignment = assigner.assign(diamond_workflow)
        assert len(assignment) == 4
        # t3 only supports A2A
        assert assignment["t3"] == ProtocolName.A2A

    def test_apply_assignment(self, linear_workflow: ProtocolWorkflow):
        assigner = OptimalProtocolAssigner()
        assignment = assigner.assign(linear_workflow)
        assigner.apply_assignment(linear_workflow, assignment)
        for task in linear_workflow.tasks.values():
            assert task.assigned_protocol is not None

    def test_empty_workflow(self):
        wf = ProtocolWorkflow(workflow_id="empty")
        assigner = OptimalProtocolAssigner()
        assignment = assigner.assign(wf)
        assert assignment == {}

    def test_all_fixed(self):
        wf = ProtocolWorkflow(workflow_id="test")
        wf.add_task("t1", assigned_protocol=ProtocolName.MCP)
        wf.add_task("t2", assigned_protocol=ProtocolName.A2A)
        assigner = OptimalProtocolAssigner()
        assignment = assigner.assign(wf)
        assert assignment["t1"] == ProtocolName.MCP
        assert assignment["t2"] == ProtocolName.A2A
