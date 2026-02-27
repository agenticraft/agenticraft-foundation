"""Multiparty Session Types (MPST) for formal protocol verification.

This module provides a comprehensive framework for specifying and
verifying multi-agent communication protocols using session types.

Based on: Honda, Yoshida, Carbone (2008) - Multiparty Session Types

Key Components:
- Global Types: Bird's-eye view of session choreography
- Local Types: Participant-specific session behavior
- Projection: Global → Local type derivation
- Type Checking: Static verification of well-formedness
- Session Monitoring: Runtime conformance checking

Example:
    # Define a request-response protocol
    from agenticraft_foundation.mpst import (
        msg, end, request_response, Projector, SessionTypeChecker,
    )

    # Using the convenience pattern
    protocol = request_response("client", "server")

    # Or build manually
    protocol = msg("client", "server", "request",
                   msg("server", "client", "response"))

    # Project to local types
    projector = Projector()
    client_type = projector.project(protocol, "client")
    server_type = projector.project(protocol, "server")

    # Verify well-formedness
    checker = SessionTypeChecker()
    result = checker.verify_global(protocol)
    print(f"Well-formed: {result.is_valid}")
"""

from agenticraft_foundation.mpst.checker import (
    MPSTInvariantRegistry,
    SessionMonitor,
    SessionTypeChecker,
    WellFormednessChecker,
    WellFormednessResult,
)
from agenticraft_foundation.mpst.global_types import (
    ChoiceType,
    EndType,
    MessageType,
    ParallelType,
    RecursionType,
    VariableType,
    choice,
    end,
    msg,
    parallel,
    rec,
    var,
)
from agenticraft_foundation.mpst.local_types import (
    BranchType,
    LocalEndType,
    LocalRecursionType,
    LocalVariableType,
    Projector,
    ReceiveType,
    SelectType,
    SendType,
    project,
    project_all,
)
from agenticraft_foundation.mpst.patterns import (
    ConsensusPattern,
    PipelinePattern,
    RequestResponsePattern,
    ScatterGatherPattern,
    pipeline,
    request_response,
    scatter_gather,
    two_phase_commit,
)
from agenticraft_foundation.mpst.properties import (
    ChoiceConsistency,
    DeadlockFreedom,
    MPSTProperty,
    MPSTPropertyType,
    MPSTSpecification,
    Progress,
    ProjectionDefinedness,
    SessionCompletion,
    TypePreservation,
)
from agenticraft_foundation.mpst.types import (
    MessageLabel,
    MessagePayload,
    ParticipantId,
    ProjectionError,
    SessionContext,
    SessionMessage,
    SessionState,
    SessionType,
    SessionViolation,
    TypeCheckError,
    TypeKind,
)

__all__ = [
    # Core Types
    "ParticipantId",
    "MessageLabel",
    "MessagePayload",
    "SessionType",
    "SessionState",
    "SessionContext",
    "SessionMessage",
    "SessionViolation",
    "TypeKind",
    "ProjectionError",
    "TypeCheckError",
    # Global Types
    "EndType",
    "MessageType",
    "ChoiceType",
    "RecursionType",
    "VariableType",
    "ParallelType",
    # Global Type Constructors
    "msg",
    "choice",
    "rec",
    "var",
    "end",
    "parallel",
    # Local Types
    "LocalEndType",
    "SendType",
    "ReceiveType",
    "SelectType",
    "BranchType",
    "LocalRecursionType",
    "LocalVariableType",
    # Projection
    "Projector",
    "project",
    "project_all",
    # Checking
    "WellFormednessChecker",
    "WellFormednessResult",
    "SessionMonitor",
    "MPSTInvariantRegistry",
    "SessionTypeChecker",
    # Properties
    "MPSTProperty",
    "MPSTPropertyType",
    "ProjectionDefinedness",
    "ChoiceConsistency",
    "DeadlockFreedom",
    "TypePreservation",
    "Progress",
    "SessionCompletion",
    "MPSTSpecification",
    # Patterns
    "RequestResponsePattern",
    "request_response",
    "ScatterGatherPattern",
    "scatter_gather",
    "PipelinePattern",
    "pipeline",
    "ConsensusPattern",
    "two_phase_commit",
]
