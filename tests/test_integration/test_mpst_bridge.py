"""Tests for MPST Bridge Adapter."""

from __future__ import annotations

import pytest

from agenticraft_foundation.integration.mpst_bridge import (
    STANDARD_SESSION_TYPES,
    ProtocolName,
    ProtocolSessionType,
    SessionStatus,
    SessionVerificationResult,
    a2a_task_send_session,
    mcp_resource_read_session,
    mcp_tool_call_session,
)
from agenticraft_foundation.mpst import (
    ParticipantId,
    msg,
)


class TestProtocolSessionTypes:
    """Tests for standard protocol session types."""

    def test_mcp_tool_call_session(self):
        """Test MCP tools/call session type."""
        session = mcp_tool_call_session()
        assert session.protocol == ProtocolName.MCP
        assert session.method == "tools/call"
        assert ParticipantId("client") in session.participants
        assert ParticipantId("server") in session.participants

    def test_mcp_resource_read_session(self):
        """Test MCP resources/read session type."""
        session = mcp_resource_read_session()
        assert session.protocol == ProtocolName.MCP
        assert session.method == "resources/read"

    def test_a2a_task_send_session(self):
        """Test A2A tasks/send session type."""
        session = a2a_task_send_session()
        assert session.protocol == ProtocolName.A2A
        assert session.method == "tasks/send"
        assert ParticipantId("client") in session.participants
        assert ParticipantId("agent") in session.participants

    def test_standard_session_types_registry(self):
        """Test standard session types are registered."""
        assert (ProtocolName.MCP, "tools/call") in STANDARD_SESSION_TYPES
        assert (ProtocolName.A2A, "tasks/send") in STANDARD_SESSION_TYPES


class TestMPSTBridgeAdapter:
    """Tests for MPSTBridgeAdapter."""

    def test_initialization(self, mpst_adapter):
        """Test adapter initialization."""
        assert mpst_adapter is not None
        assert len(mpst_adapter._session_types) > 0

    def test_get_session_type(self, mpst_adapter):
        """Test getting registered session type."""
        session = mpst_adapter.get_session_type(ProtocolName.MCP, "tools/call")
        assert session is not None
        assert session.method == "tools/call"

    def test_get_session_type_not_found(self, mpst_adapter):
        """Test getting unregistered session type returns None."""
        session = mpst_adapter.get_session_type(ProtocolName.CUSTOM, "unknown")
        assert session is None

    def test_register_custom_session_type(self, mpst_adapter):
        """Test registering a custom session type."""
        custom_session = ProtocolSessionType(
            protocol=ProtocolName.CUSTOM,
            method="custom/method",
            session_type=msg("a", "b", "Custom"),
        )
        mpst_adapter.register_session_type(custom_session)

        retrieved = mpst_adapter.get_session_type(ProtocolName.CUSTOM, "custom/method")
        assert retrieved is not None
        assert retrieved.method == "custom/method"

    def test_verify_wellformedness_valid(self, mpst_adapter, simple_request_response):
        """Test verifying well-formed session type."""
        result = mpst_adapter.verify_wellformedness(simple_request_response)
        assert result.is_valid
        assert len(result.violations) == 0

    def test_verify_message_valid(self, mpst_adapter):
        """Test verifying a valid message."""
        result = mpst_adapter.verify_message(
            message={"type": "ToolCall"},
            method="tools/call",
            protocol=ProtocolName.MCP,
        )
        assert result.is_valid

    def test_verify_message_unknown_protocol(self, mpst_adapter):
        """Test verifying message for unknown protocol method."""
        result = mpst_adapter.verify_message(
            message={"type": "Unknown"},
            method="unknown/method",
            protocol=ProtocolName.CUSTOM,
        )
        assert not result.is_valid
        assert len(result.violations) > 0


class TestMPSTSessionMonitoring:
    """Tests for session monitoring."""

    @pytest.mark.asyncio
    async def test_start_session(self, mpst_adapter, simple_request_response):
        """Test starting a monitored session."""
        session_id = await mpst_adapter.start_session(
            session_type=simple_request_response,
            participants={
                ParticipantId("client"): "agent-1",
                ParticipantId("server"): "agent-2",
            },
        )

        assert session_id is not None
        assert session_id in mpst_adapter.get_active_sessions()

    @pytest.mark.asyncio
    async def test_session_message_flow(self, mpst_adapter, simple_request_response):
        """Test message flow through a session."""
        session_id = await mpst_adapter.start_session(
            session_type=simple_request_response,
            participants={
                ParticipantId("client"): "agent-1",
                ParticipantId("server"): "agent-2",
            },
        )

        # Send request
        result = await mpst_adapter.on_message(
            session_id=session_id,
            message={"type": "Request"},
            sender=ParticipantId("client"),
            receiver=ParticipantId("server"),
        )
        assert result.is_valid

        # Send response
        result = await mpst_adapter.on_message(
            session_id=session_id,
            message={"type": "Response"},
            sender=ParticipantId("server"),
            receiver=ParticipantId("client"),
        )
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_end_session(self, mpst_adapter, simple_request_response):
        """Test ending a session."""
        session_id = await mpst_adapter.start_session(
            session_type=simple_request_response,
            participants={
                ParticipantId("client"): "agent-1",
                ParticipantId("server"): "agent-2",
            },
        )

        # Complete the session
        await mpst_adapter.on_message(
            session_id=session_id,
            message={"type": "Request"},
            sender=ParticipantId("client"),
            receiver=ParticipantId("server"),
        )
        await mpst_adapter.on_message(
            session_id=session_id,
            message={"type": "Response"},
            sender=ParticipantId("server"),
            receiver=ParticipantId("client"),
        )

        # End session
        result = await mpst_adapter.end_session(session_id)
        assert result.is_valid
        assert session_id not in mpst_adapter.get_active_sessions()

    @pytest.mark.asyncio
    async def test_end_incomplete_session(self, mpst_adapter, simple_request_response):
        """Test ending an incomplete session."""
        session_id = await mpst_adapter.start_session(
            session_type=simple_request_response,
            participants={
                ParticipantId("client"): "agent-1",
                ParticipantId("server"): "agent-2",
            },
        )

        # Don't complete the protocol - just end
        result = await mpst_adapter.end_session(session_id)
        # Session not complete, should report violations
        assert not result.is_valid
        assert len(result.violations) > 0

    @pytest.mark.asyncio
    async def test_get_session_status(self, mpst_adapter, simple_request_response):
        """Test getting session status."""
        session_id = await mpst_adapter.start_session(
            session_type=simple_request_response,
            participants={
                ParticipantId("client"): "agent-1",
                ParticipantId("server"): "agent-2",
            },
        )

        status = mpst_adapter.get_session_status(session_id)
        assert status is not None
        assert ParticipantId("client") in status
        assert ParticipantId("server") in status

    def test_get_session_status_not_found(self, mpst_adapter):
        """Test getting status of non-existent session."""
        status = mpst_adapter.get_session_status("nonexistent")
        assert status is None


class TestSessionVerificationResult:
    """Tests for SessionVerificationResult."""

    def test_valid_result(self):
        """Test creating valid result."""
        session_type = msg("a", "b", "Test")
        result = SessionVerificationResult.valid(session_type)
        assert result.is_valid
        assert result.session_type is not None

    def test_valid_result_completed(self):
        """Test completed valid result."""
        session_type = msg("a", "b", "Test")
        result = SessionVerificationResult.valid(session_type, remaining=None)
        assert result.is_valid
        assert result.current_status == SessionStatus.COMPLETED

    def test_valid_result_in_progress(self):
        """Test in-progress valid result."""
        session_type = msg("a", "b", "Test")
        result = SessionVerificationResult.valid(session_type, remaining=session_type)
        assert result.is_valid
        assert result.current_status == SessionStatus.ACTIVE

    def test_invalid_result(self):
        """Test creating invalid result."""
        result = SessionVerificationResult.invalid(
            violations=["Error 1", "Error 2"],
        )
        assert not result.is_valid
        assert len(result.violations) == 2
        assert result.current_status == SessionStatus.FAILED
