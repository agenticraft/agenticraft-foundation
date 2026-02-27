"""Test fixtures for integration adapter tests."""

from __future__ import annotations

import pytest

from agenticraft_foundation.integration.csp_orchestration import (
    CSPOrchestrationAdapter,
    WorkflowSpec,
)
from agenticraft_foundation.integration.mpst_bridge import (
    MPSTBridgeAdapter,
)
from agenticraft_foundation.mpst import (
    msg,
)


@pytest.fixture
def mpst_adapter() -> MPSTBridgeAdapter:
    """Create an MPST bridge adapter."""
    return MPSTBridgeAdapter()


@pytest.fixture
def csp_adapter() -> CSPOrchestrationAdapter:
    """Create a CSP orchestration adapter."""
    return CSPOrchestrationAdapter()


@pytest.fixture
def simple_request_response():
    """A simple request-response session type."""
    return msg(
        "client",
        "server",
        "Request",
        msg("server", "client", "Response"),
    )


@pytest.fixture
def simple_workflow():
    """A simple sequential workflow spec."""
    return WorkflowSpec.sequential_tasks("simple_workflow", ["task1", "task2", "task3"])


@pytest.fixture
def parallel_workflow():
    """A parallel workflow spec."""
    return WorkflowSpec.parallel_tasks(
        "parallel_workflow",
        ["task_a", "task_b", "task_c"],
        sync_on_complete=True,
    )


@pytest.fixture
def choice_workflow():
    """A choice workflow spec."""
    return WorkflowSpec.choice_tasks("choice_workflow", ["option1", "option2"])
