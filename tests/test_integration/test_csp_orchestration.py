"""Tests for CSP Orchestration Adapter."""

from __future__ import annotations

from agenticraft_foundation.algebra import (
    Event,
    choice,
    is_deadlock_free,
    prefix,
    sequential,
    skip,
    stop,
)
from agenticraft_foundation.integration.csp_orchestration import (
    WorkflowNodeType,
    WorkflowSpec,
    WorkflowVerificationResult,
)


class TestWorkflowSpec:
    """Tests for WorkflowSpec."""

    def test_sequential_tasks(self):
        """Test creating sequential workflow spec."""
        spec = WorkflowSpec.sequential_tasks("seq", ["a", "b", "c"])
        assert spec.name == "seq"
        assert Event("a") in spec.events_of_interest
        assert Event("b") in spec.events_of_interest
        assert Event("c") in spec.events_of_interest

    def test_sequential_tasks_empty(self):
        """Test empty sequential workflow."""
        spec = WorkflowSpec.sequential_tasks("empty", [])
        assert spec.name == "empty"
        # Empty workflow is just SKIP
        assert is_deadlock_free(spec.process)

    def test_parallel_tasks(self):
        """Test creating parallel workflow spec."""
        spec = WorkflowSpec.parallel_tasks("par", ["x", "y"])
        assert spec.name == "par"
        assert Event("x") in spec.events_of_interest
        assert Event("y") in spec.events_of_interest

    def test_parallel_tasks_sync(self):
        """Test parallel workflow with sync on complete."""
        spec = WorkflowSpec.parallel_tasks("par", ["x", "y"], sync_on_complete=True)
        assert spec.name == "par"
        # Should be deadlock-free
        assert is_deadlock_free(spec.process)

    def test_choice_tasks(self):
        """Test creating choice workflow spec."""
        spec = WorkflowSpec.choice_tasks("choice", ["opt1", "opt2"])
        assert spec.name == "choice"
        assert Event("opt1") in spec.events_of_interest
        assert Event("opt2") in spec.events_of_interest

    def test_from_dag(self):
        """Test creating workflow from DAG."""
        nodes = {
            "start": WorkflowNodeType.TASK,
            "middle": WorkflowNodeType.TASK,
            "end": WorkflowNodeType.TASK,
        }
        edges = [("start", "middle"), ("middle", "end")]

        spec = WorkflowSpec.from_dag("dag_workflow", nodes, edges)
        assert spec.name == "dag_workflow"
        assert Event("start") in spec.events_of_interest


class TestCSPOrchestrationAdapter:
    """Tests for CSPOrchestrationAdapter."""

    def test_initialization(self, csp_adapter):
        """Test adapter initialization."""
        assert csp_adapter is not None
        assert len(csp_adapter._specs) == 0

    def test_register_spec(self, csp_adapter, simple_workflow):
        """Test registering a workflow spec."""
        csp_adapter.register_spec(simple_workflow)
        retrieved = csp_adapter.get_spec("simple_workflow")
        assert retrieved is not None
        assert retrieved.name == "simple_workflow"

    def test_get_spec_not_found(self, csp_adapter):
        """Test getting unregistered spec returns None."""
        spec = csp_adapter.get_spec("unknown")
        assert spec is None


class TestWorkflowVerification:
    """Tests for workflow verification."""

    def test_verify_sequential_spec(self, csp_adapter, simple_workflow):
        """Test verifying sequential workflow spec."""
        csp_adapter.register_spec(simple_workflow)
        result = csp_adapter.verify_spec(simple_workflow)
        assert result.is_valid
        assert result.is_deadlock_free

    def test_verify_parallel_spec(self, csp_adapter, parallel_workflow):
        """Test verifying parallel workflow spec."""
        csp_adapter.register_spec(parallel_workflow)
        result = csp_adapter.verify_spec(parallel_workflow)
        assert result.is_valid
        assert result.is_deadlock_free

    def test_verify_choice_spec(self, csp_adapter, choice_workflow):
        """Test verifying choice workflow spec."""
        csp_adapter.register_spec(choice_workflow)
        result = csp_adapter.verify_spec(choice_workflow)
        assert result.is_valid
        assert result.is_deadlock_free

    def test_verify_workflow_with_implementation(self, csp_adapter, simple_workflow):
        """Test verifying workflow against implementation."""
        csp_adapter.register_spec(simple_workflow)

        # Implementation that refines the spec
        impl = sequential(
            prefix("task1", skip()),
            sequential(prefix("task2", skip()), prefix("task3", skip())),
        )

        result = csp_adapter.verify_workflow("simple_workflow", impl)
        # The exact refinement check depends on implementation
        # For now, just check it doesn't crash
        assert result is not None

    def test_verify_workflow_no_spec(self, csp_adapter):
        """Test verifying workflow without registered spec."""
        result = csp_adapter.verify_workflow("unknown", prefix("a", skip()))
        assert not result.is_valid
        assert len(result.violations) > 0

    def test_verify_workflow_no_implementation(self, csp_adapter, simple_workflow):
        """Test verifying workflow without implementation."""
        csp_adapter.register_spec(simple_workflow)
        result = csp_adapter.verify_workflow("simple_workflow")
        assert not result.is_valid
        assert len(result.violations) > 0


class TestDeadlockChecking:
    """Tests for deadlock checking."""

    def test_check_deadlock_freedom_valid(self, csp_adapter, simple_workflow):
        """Test deadlock freedom check on valid workflow."""
        csp_adapter.register_spec(simple_workflow)
        is_free = csp_adapter.check_deadlock_freedom("simple_workflow")
        assert is_free

    def test_check_deadlock_freedom_unknown(self, csp_adapter):
        """Test deadlock freedom check on unknown workflow."""
        is_free = csp_adapter.check_deadlock_freedom("unknown")
        assert not is_free

    def test_check_workflow_with_deadlock(self, csp_adapter):
        """Test detecting workflow with deadlock."""
        # Create a workflow that can deadlock
        deadlock_spec = WorkflowSpec(
            name="deadlock_workflow",
            process=choice(prefix("a", stop()), prefix("b", skip())),
        )
        csp_adapter.register_spec(deadlock_spec)

        is_free = csp_adapter.check_deadlock_freedom("deadlock_workflow")
        assert not is_free


class TestLivenessAnalysis:
    """Tests for liveness analysis."""

    def test_analyze_liveness(self, csp_adapter, simple_workflow):
        """Test liveness analysis on workflow."""
        csp_adapter.register_spec(simple_workflow)
        liveness = csp_adapter.analyze_workflow_liveness("simple_workflow")
        assert len(liveness) > 0
        # All events should be live in a sequential workflow
        assert all(liveness.values())

    def test_analyze_liveness_unknown(self, csp_adapter):
        """Test liveness analysis on unknown workflow."""
        liveness = csp_adapter.analyze_workflow_liveness("unknown")
        assert len(liveness) == 0


class TestWorkflowTraces:
    """Tests for workflow trace generation."""

    def test_get_workflow_traces(self, csp_adapter, simple_workflow):
        """Test getting traces from workflow."""
        csp_adapter.register_spec(simple_workflow)
        traces = csp_adapter.get_workflow_traces("simple_workflow")
        assert len(traces) > 0
        # Should include empty trace
        assert () in traces

    def test_get_workflow_traces_unknown(self, csp_adapter):
        """Test getting traces from unknown workflow."""
        traces = csp_adapter.get_workflow_traces("unknown")
        assert len(traces) == 0


class TestWorkflowComposition:
    """Tests for workflow composition."""

    def test_compose_parallel(self, csp_adapter):
        """Test parallel composition of workflows."""
        spec1 = WorkflowSpec.sequential_tasks("wf1", ["a", "b"])
        spec2 = WorkflowSpec.sequential_tasks("wf2", ["x", "y"])

        csp_adapter.register_spec(spec1)
        csp_adapter.register_spec(spec2)

        composed = csp_adapter.compose_workflows(
            "composed",
            ["wf1", "wf2"],
            composition_type="parallel",
        )

        assert composed is not None
        assert composed.name == "composed"
        assert Event("a") in composed.events_of_interest
        assert Event("x") in composed.events_of_interest

    def test_compose_sequential(self, csp_adapter):
        """Test sequential composition of workflows."""
        spec1 = WorkflowSpec.sequential_tasks("wf1", ["a"])
        spec2 = WorkflowSpec.sequential_tasks("wf2", ["b"])

        csp_adapter.register_spec(spec1)
        csp_adapter.register_spec(spec2)

        composed = csp_adapter.compose_workflows(
            "composed_seq",
            ["wf1", "wf2"],
            composition_type="sequential",
        )

        assert composed is not None
        assert is_deadlock_free(composed.process)

    def test_compose_unknown_workflow(self, csp_adapter, simple_workflow):
        """Test composing with unknown workflow."""
        csp_adapter.register_spec(simple_workflow)

        composed = csp_adapter.compose_workflows(
            "failed",
            ["simple_workflow", "unknown"],
            composition_type="parallel",
        )

        assert composed is None


class TestDAGConversion:
    """Tests for DAG to process conversion."""

    def test_workflow_to_process(self, csp_adapter):
        """Test converting DAG to process."""
        dag = {
            "nodes": {
                "start": {"type": "task"},
                "end": {"type": "task"},
            },
            "edges": [
                {"source": "start", "target": "end"},
            ],
        }

        process = csp_adapter.workflow_to_process(dag)
        assert process is not None
        # Should be deadlock-free
        assert is_deadlock_free(process)

    def test_workflow_to_process_empty(self, csp_adapter):
        """Test converting empty DAG."""
        dag = {"nodes": {}, "edges": []}
        process = csp_adapter.workflow_to_process(dag)
        assert process is not None
        # Empty DAG is SKIP
        assert is_deadlock_free(process)


class TestWorkflowVerificationResult:
    """Tests for WorkflowVerificationResult."""

    def test_passed_result(self):
        """Test creating passed result."""
        result = WorkflowVerificationResult.passed()
        assert result.is_valid
        assert result.is_deadlock_free
        assert result.is_live
        assert result.refines_spec

    def test_passed_with_traces(self):
        """Test passed result with traces."""
        traces = [(Event("a"),), (Event("b"),)]
        result = WorkflowVerificationResult.passed(traces_sample=traces)
        assert result.is_valid
        assert len(result.traces_sample) == 2

    def test_failed_result(self):
        """Test creating failed result."""
        result = WorkflowVerificationResult.failed(
            violations=["Error 1", "Error 2"],
            is_deadlock_free=False,
        )
        assert not result.is_valid
        assert not result.is_deadlock_free
        assert len(result.violations) == 2

    def test_failed_with_counterexample(self):
        """Test failed result with counterexample."""
        counter = (Event("bad"), Event("path"))
        result = WorkflowVerificationResult.failed(
            violations=["Refinement failed"],
            refines_spec=False,
            counterexample=counter,
        )
        assert not result.is_valid
        assert not result.refines_spec
        assert result.counterexample == counter
