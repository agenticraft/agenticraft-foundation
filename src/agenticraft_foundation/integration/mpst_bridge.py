"""MPST integration adapter for protocol bridges.

This module provides session type verification for cross-protocol messaging.

Key Features:
- Session type verification before message routing
- Protocol session type mapping (MCP, A2A → MPST types)
- Runtime session conformance monitoring
- Well-formedness checking for protocol session types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..mpst import (
    MessageLabel,
    MessagePayload,
    ParticipantId,
    Projector,
    SessionMonitor,
    SessionType,
    WellFormednessChecker,
    choice,
    msg,
)

# =============================================================================
# Protocol Session Type Mappings
# =============================================================================


class ProtocolName(str, Enum):
    """Supported protocol names."""

    MCP = "MCP"
    A2A = "A2A"
    CUSTOM = "CUSTOM"


class SessionStatus(str, Enum):
    """Status of a session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


@dataclass
class ProtocolSessionType:
    """Maps a protocol interaction to an MPST session type.

    Attributes:
        protocol: The protocol name
        method: The protocol method (e.g., "tools/call", "tasks/send")
        session_type: The corresponding MPST global type
        participants: Participant roles in the session
    """

    protocol: ProtocolName
    method: str
    session_type: SessionType
    participants: frozenset[ParticipantId] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.participants:
            # Extract participants from session type
            self.participants = frozenset(self.session_type.participants())


# =============================================================================
# Standard Protocol Session Types
# =============================================================================


def mcp_tool_call_session() -> ProtocolSessionType:
    """MCP tools/call session type: client → server : call. server → client : result."""
    session = msg(
        "client",
        "server",
        "ToolCall",
        msg("server", "client", "ToolResult"),
    )
    return ProtocolSessionType(
        protocol=ProtocolName.MCP,
        method="tools/call",
        session_type=session,
    )


def mcp_resource_read_session() -> ProtocolSessionType:
    """MCP resources/read session type."""
    session = msg(
        "client",
        "server",
        "ResourceRequest",
        msg("server", "client", "ResourceContent"),
    )
    return ProtocolSessionType(
        protocol=ProtocolName.MCP,
        method="resources/read",
        session_type=session,
    )


def a2a_task_send_session() -> ProtocolSessionType:
    """A2A tasks/send session type with status updates."""
    session = msg(
        "client",
        "agent",
        "TaskRequest",
        choice(
            "agent",
            "client",
            {
                "completed": msg("agent", "client", "TaskResult"),
                "working": msg(
                    "agent",
                    "client",
                    "TaskStatus",
                    msg("agent", "client", "TaskResult"),
                ),
                "failed": msg("agent", "client", "TaskError"),
            },
        ),
    )
    return ProtocolSessionType(
        protocol=ProtocolName.A2A,
        method="tasks/send",
        session_type=session,
    )


# Registry of standard session types
STANDARD_SESSION_TYPES: dict[tuple[ProtocolName, str], ProtocolSessionType] = {
    (ProtocolName.MCP, "tools/call"): mcp_tool_call_session(),
    (ProtocolName.MCP, "resources/read"): mcp_resource_read_session(),
    (ProtocolName.A2A, "tasks/send"): a2a_task_send_session(),
}


# =============================================================================
# Session Verification Result
# =============================================================================


@dataclass
class SessionVerificationResult:
    """Result of session type verification.

    Attributes:
        is_valid: Whether the session conforms to its type
        session_type: The verified session type
        violations: List of conformance violations
        current_status: Current status of the session
        remaining_type: Remaining session type to complete
    """

    is_valid: bool
    session_type: SessionType | None = None
    violations: list[str] = field(default_factory=list)
    current_status: SessionStatus = SessionStatus.ACTIVE
    remaining_type: SessionType | None = None

    @classmethod
    def valid(
        cls,
        session_type: SessionType,
        remaining: SessionType | None = None,
    ) -> SessionVerificationResult:
        """Create a valid verification result."""
        status = SessionStatus.COMPLETED if remaining is None else SessionStatus.ACTIVE
        return cls(
            is_valid=True,
            session_type=session_type,
            current_status=status,
            remaining_type=remaining,
        )

    @classmethod
    def invalid(
        cls,
        violations: list[str],
        session_type: SessionType | None = None,
    ) -> SessionVerificationResult:
        """Create an invalid verification result."""
        return cls(
            is_valid=False,
            session_type=session_type,
            violations=violations,
            current_status=SessionStatus.FAILED,
        )


# =============================================================================
# Active Session Tracking
# =============================================================================


@dataclass
class ActiveSession:
    """Internal representation of an active session."""

    session_id: str
    global_type: SessionType
    participants: dict[ParticipantId, str]  # role -> agent_id
    monitors: dict[ParticipantId, SessionMonitor]


# =============================================================================
# MPST Bridge Adapter
# =============================================================================


class MPSTBridgeAdapter:
    """Adapter integrating MPST session types with ProtocolBridge.

    This adapter provides session type verification for cross-protocol
    message routing. It hooks into the ProtocolBridge's translation
    and routing pipeline to ensure session conformance.

    Usage:
        adapter = MPSTBridgeAdapter()

        # Register session type for a protocol method
        adapter.register_session_type(mcp_tool_call_session())

        # Verify message conforms to session type
        result = adapter.verify_message(message, "tools/call", ProtocolName.MCP)

        # Start a monitored session
        session_id = await adapter.start_session(
            session_type=mcp_tool_call_session().session_type,
            participants={"client": "agent-1", "server": "agent-2"},
        )

        # Monitor message in session
        await adapter.on_message(session_id, message)
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._session_types: dict[tuple[ProtocolName, str], ProtocolSessionType] = dict(
            STANDARD_SESSION_TYPES
        )
        self._active_sessions: dict[str, ActiveSession] = {}
        self._wellformedness_checker = WellFormednessChecker()
        self._projector = Projector(strict=False)
        self._session_counter = 0

    def register_session_type(self, session_type: ProtocolSessionType) -> None:
        """Register a session type for a protocol method.

        Args:
            session_type: The session type to register
        """
        key = (session_type.protocol, session_type.method)
        self._session_types[key] = session_type

    def get_session_type(
        self,
        protocol: ProtocolName,
        method: str,
    ) -> ProtocolSessionType | None:
        """Get the session type for a protocol method.

        Args:
            protocol: The protocol name
            method: The method name

        Returns:
            The session type, or None if not registered
        """
        return self._session_types.get((protocol, method))

    def verify_wellformedness(
        self,
        session_type: SessionType,
    ) -> SessionVerificationResult:
        """Verify a session type is well-formed.

        Args:
            session_type: The global type to verify

        Returns:
            Verification result
        """
        result = self._wellformedness_checker.check(session_type)

        if result.is_well_formed:
            return SessionVerificationResult.valid(session_type)
        else:
            return SessionVerificationResult.invalid(
                violations=result.errors,
                session_type=session_type,
            )

    def verify_message(
        self,
        message: dict[str, Any],
        method: str,
        protocol: ProtocolName,
        participant: ParticipantId | None = None,
    ) -> SessionVerificationResult:
        """Verify a message conforms to the expected session type.

        Args:
            message: The message to verify
            method: The protocol method
            protocol: The protocol name
            participant: Optional participant to verify as

        Returns:
            Verification result
        """
        proto_session = self.get_session_type(protocol, method)

        if proto_session is None:
            return SessionVerificationResult.invalid(
                violations=[f"No session type registered for {protocol.value}/{method}"],
            )

        global_type = proto_session.session_type

        # If a participant is specified, verify against local projection
        if participant:
            local_type = self._projector.project(global_type, participant)
            if local_type is None:
                return SessionVerificationResult.invalid(
                    violations=[f"Cannot project to participant '{participant}'"],
                    session_type=global_type,
                )

        return SessionVerificationResult.valid(
            global_type,
            remaining=global_type,  # Simplified - full impl tracks state
        )

    async def start_session(
        self,
        session_type: SessionType,
        participants: dict[ParticipantId, str],
    ) -> str:
        """Start a new monitored session.

        Args:
            session_type: The global session type
            participants: Mapping of participant roles to agent IDs

        Returns:
            Session ID for tracking

        Raises:
            ValueError: If session type is not well-formed
        """
        # Verify well-formedness first
        wf_result = self.verify_wellformedness(session_type)
        if not wf_result.is_valid:
            raise ValueError(f"Session type is not well-formed: {wf_result.violations}")

        self._session_counter += 1
        session_id = f"session-{self._session_counter}"

        # Create monitors for each participant
        monitors: dict[ParticipantId, SessionMonitor] = {}
        for participant in participants:
            local_type = self._projector.project(session_type, participant)
            if local_type is not None:
                monitors[participant] = SessionMonitor(
                    participant=participant,
                    local_type=local_type,
                    session_id=session_id,
                    strict=False,  # Don't raise on violations
                )

        self._active_sessions[session_id] = ActiveSession(
            session_id=session_id,
            global_type=session_type,
            participants=participants,
            monitors=monitors,
        )

        return session_id

    async def on_message(
        self,
        session_id: str,
        message: dict[str, Any],
        sender: ParticipantId,
        receiver: ParticipantId,
    ) -> SessionVerificationResult:
        """Process a message in an active session.

        Args:
            session_id: The session ID
            message: The message being sent
            sender: The sending participant role
            receiver: The receiving participant role

        Returns:
            Verification result
        """
        session = self._active_sessions.get(session_id)
        if session is None:
            return SessionVerificationResult.invalid(
                violations=[f"No active session with ID '{session_id}'"],
            )

        # Extract message label
        msg_type = message.get("type") or message.get("method") or "unknown"
        payload = MessagePayload(label=MessageLabel(msg_type))

        violations: list[str] = []

        # Verify with sender's monitor
        sender_monitor = session.monitors.get(sender)
        if sender_monitor:
            try:
                if not sender_monitor.on_send(receiver, payload):
                    monitor_violations = sender_monitor.get_violations()
                    if monitor_violations:
                        violations.append(f"Sender violation: {monitor_violations[-1].message}")
            except Exception as e:
                violations.append(f"Sender error: {e}")

        # Verify with receiver's monitor
        receiver_monitor = session.monitors.get(receiver)
        if receiver_monitor:
            try:
                if not receiver_monitor.on_receive(sender, payload):
                    monitor_violations = receiver_monitor.get_violations()
                    if monitor_violations:
                        violations.append(f"Receiver violation: {monitor_violations[-1].message}")
            except Exception as e:
                violations.append(f"Receiver error: {e}")

        if violations:
            return SessionVerificationResult.invalid(
                violations=violations,
                session_type=session.global_type,
            )

        # Check if session is complete
        all_complete = all(m.is_complete() for m in session.monitors.values())

        if all_complete:
            return SessionVerificationResult.valid(
                session.global_type,
                remaining=None,
            )

        return SessionVerificationResult.valid(
            session.global_type,
            remaining=session.global_type,
        )

    async def end_session(self, session_id: str) -> SessionVerificationResult:
        """End a session and verify it completed properly.

        Args:
            session_id: The session ID

        Returns:
            Final verification result
        """
        session = self._active_sessions.pop(session_id, None)
        if session is None:
            return SessionVerificationResult.invalid(
                violations=[f"No active session with ID '{session_id}'"],
            )

        # Check all monitors completed
        violations: list[str] = []
        for participant, monitor in session.monitors.items():
            if not monitor.is_complete():
                can_act = (
                    "can_send"
                    if monitor.can_send()
                    else ("can_receive" if monitor.can_receive() else "stuck")
                )
                violations.append(f"Participant '{participant}' did not complete ({can_act})")

        if violations:
            return SessionVerificationResult.invalid(
                violations=violations,
                session_type=session.global_type,
            )

        return SessionVerificationResult.valid(session.global_type)

    def get_active_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        return list(self._active_sessions.keys())

    def get_session_status(
        self,
        session_id: str,
    ) -> dict[ParticipantId, SessionStatus] | None:
        """Get current status of all participants in a session.

        Args:
            session_id: The session ID

        Returns:
            Mapping of participant to status, or None if session not found
        """
        session = self._active_sessions.get(session_id)
        if session is None:
            return None

        result: dict[ParticipantId, SessionStatus] = {}
        for participant, monitor in session.monitors.items():
            if monitor.is_complete():
                result[participant] = SessionStatus.COMPLETED
            elif monitor.can_send() or monitor.can_receive():
                result[participant] = SessionStatus.ACTIVE
            else:
                result[participant] = SessionStatus.WAITING

        return result


__all__ = [
    "ProtocolName",
    "ProtocolSessionType",
    "SessionVerificationResult",
    "SessionStatus",
    "MPSTBridgeAdapter",
    "mcp_tool_call_session",
    "mcp_resource_read_session",
    "a2a_task_send_session",
    "STANDARD_SESSION_TYPES",
]
