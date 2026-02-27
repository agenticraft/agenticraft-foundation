"""Formal verification integration adapters.

This module provides integration adapters that connect formal verification
components (MPST, Process Algebra) to protocol and workflow systems.

Components:
- MPSTBridgeAdapter: Session type verification for cross-protocol messaging
- CSPOrchestrationAdapter: Process algebra verification for workflows
"""

from __future__ import annotations

from .csp_orchestration import (
    CSPOrchestrationAdapter,
    WorkflowSpec,
    WorkflowVerificationResult,
)
from .mpst_bridge import (
    MPSTBridgeAdapter,
    ProtocolSessionType,
    SessionVerificationResult,
)

__all__ = [
    # MPST Integration
    "MPSTBridgeAdapter",
    "SessionVerificationResult",
    "ProtocolSessionType",
    # CSP Integration
    "CSPOrchestrationAdapter",
    "WorkflowVerificationResult",
    "WorkflowSpec",
]
