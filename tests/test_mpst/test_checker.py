"""Tests for MPST session type checker."""

from __future__ import annotations

from agenticraft_foundation.mpst import (
    EndType,
    MessageLabel,
    MessagePayload,
    ParticipantId,
    Projector,
    SendType,
    SessionMonitor,
    choice,
    end,
    msg,
    rec,
    var,
)


class TestWellFormednessChecker:
    """Tests for WellFormednessChecker."""

    def test_end_alone_requires_participants(self, well_formedness_checker):
        """Test that EndType alone is flagged (no participants)."""
        result = well_formedness_checker.check(EndType())
        # EndType alone has no participants, which the checker flags
        assert not result.is_well_formed
        assert any("no participants" in e.lower() for e in result.errors)

    def test_simple_message_is_well_formed(self, well_formedness_checker):
        """Test that simple message is well-formed."""
        g = msg("a", "b", "m")
        result = well_formedness_checker.check(g)
        assert result.is_well_formed

    def test_request_response_is_well_formed(self, well_formedness_checker, request_response_type):
        """Test that request-response is well-formed."""
        result = well_formedness_checker.check(request_response_type)
        assert result.is_well_formed

    def test_choice_is_well_formed(self, well_formedness_checker, choice_type):
        """Test that choice type is well-formed."""
        result = well_formedness_checker.check(choice_type)
        assert result.is_well_formed

    def test_recursion_is_well_formed(self, well_formedness_checker, recursive_type):
        """Test that recursive type is well-formed."""
        result = well_formedness_checker.check(recursive_type)
        assert result.is_well_formed

    def test_unguarded_recursion_is_not_well_formed(self, well_formedness_checker):
        """Test that unguarded recursion is not well-formed."""
        # μX.X - unguarded recursion (no participants)
        g = rec("X", var("X"))
        result = well_formedness_checker.check(g)
        assert not result.is_well_formed
        # This has no participants, so that error takes precedence
        assert len(result.errors) > 0

    def test_free_variable_is_not_well_formed(self, well_formedness_checker):
        """Test that free variable is not well-formed."""
        # Just X without binding
        g = msg("a", "b", "m", var("X"))
        result = well_formedness_checker.check(g)
        assert not result.is_well_formed
        # The error message says "Unbound recursion variable"
        assert any("unbound" in e.lower() for e in result.errors)


class TestSessionTypeChecker:
    """Tests for SessionTypeChecker."""

    def test_check_well_formed_valid(self, checker, request_response_type):
        """Test checking a valid global type."""
        result = checker.check_well_formed(request_response_type)
        assert result.is_well_formed

    def test_check_well_formed_with_choice(self, checker, choice_type):
        """Test checking a global type with choice."""
        result = checker.check_well_formed(choice_type)
        assert result.is_well_formed

    def test_project(self, checker, request_response_type):
        """Test projecting a global type."""
        local = checker.project(request_response_type, "client")
        assert isinstance(local, SendType)

    def test_create_session(self, checker, request_response_type):
        """Test creating a session."""
        # create_session returns the monitors directly
        monitors = checker.create_session(
            "test_session",
            request_response_type,
        )
        assert monitors is not None

        # Check monitors are created for both participants
        assert ParticipantId("client") in monitors
        assert ParticipantId("server") in monitors

        # Verify monitors are SessionMonitor instances
        assert isinstance(monitors[ParticipantId("client")], SessionMonitor)
        assert isinstance(monitors[ParticipantId("server")], SessionMonitor)


class TestSessionMonitor:
    """Tests for SessionMonitor."""

    def test_monitor_send_receive(self):
        """Test monitoring send and receive actions."""
        projector = Projector()
        g = msg("client", "server", "request")

        # Create monitor for client
        client_local = projector.project(g, "client")
        monitor = SessionMonitor(
            participant=ParticipantId("client"),
            local_type=client_local,
        )

        assert not monitor.is_complete()
        assert monitor.can_send()
        assert not monitor.can_receive()

        # Perform send action
        payload = MessagePayload(label=MessageLabel("request"))
        result = monitor.on_send(ParticipantId("server"), payload)
        assert result is True

        # After send, client should be done (end)
        assert monitor.is_complete()

    def test_monitor_invalid_action(self):
        """Test that invalid actions are rejected."""
        projector = Projector()
        g = msg("client", "server", "request")

        client_local = projector.project(g, "client")
        monitor = SessionMonitor(
            participant=ParticipantId("client"),
            local_type=client_local,
            strict=False,  # Don't raise exceptions
        )

        # Client should send, not receive
        assert monitor.can_send()
        assert not monitor.can_receive()

        # Try to receive instead of send - should fail
        payload = MessagePayload(label=MessageLabel("request"))
        result = monitor.on_receive(ParticipantId("server"), payload)
        assert result is False

        # Should have recorded a violation
        assert len(monitor.get_violations()) > 0

    def test_monitor_request_response_flow(self):
        """Test monitoring full request-response flow."""
        projector = Projector()
        g = msg(
            "client",
            "server",
            "request",
            msg("server", "client", "response"),
        )

        # Monitor client
        client_local = projector.project(g, "client")
        client_monitor = SessionMonitor(
            participant=ParticipantId("client"),
            local_type=client_local,
        )

        # Monitor server
        server_local = projector.project(g, "server")
        server_monitor = SessionMonitor(
            participant=ParticipantId("server"),
            local_type=server_local,
        )

        request_payload = MessagePayload(label=MessageLabel("request"))
        response_payload = MessagePayload(label=MessageLabel("response"))

        # Client sends request
        assert client_monitor.can_send()
        client_monitor.on_send(ParticipantId("server"), request_payload)

        # Server receives request
        assert server_monitor.can_receive()
        server_monitor.on_receive(ParticipantId("client"), request_payload)

        # Server sends response
        assert server_monitor.can_send()
        server_monitor.on_send(ParticipantId("client"), response_payload)

        # Client receives response
        assert client_monitor.can_receive()
        client_monitor.on_receive(ParticipantId("server"), response_payload)

        # Both should be complete
        assert client_monitor.is_complete()
        assert server_monitor.is_complete()


class TestMonitorChoice:
    """Tests for monitoring choice types."""

    def test_monitor_select(self):
        """Test monitoring select action."""
        projector = Projector()
        g = choice(
            "client",
            "server",
            {
                "buy": msg("server", "client", "confirm"),
                "cancel": end(),
            },
        )

        client_local = projector.project(g, "client")
        monitor = SessionMonitor(
            participant=ParticipantId("client"),
            local_type=client_local,
        )

        # Client can select (it's a select type, which is a send)
        assert monitor.can_send()

        # Select buy
        buy_payload = MessagePayload(label=MessageLabel("buy"))
        monitor.on_send(ParticipantId("server"), buy_payload)

        # Now client should receive confirm
        assert monitor.can_receive()

        confirm_payload = MessagePayload(label=MessageLabel("confirm"))
        monitor.on_receive(ParticipantId("server"), confirm_payload)

        assert monitor.is_complete()

    def test_monitor_branch(self):
        """Test monitoring branch action."""
        projector = Projector()
        g = choice(
            "client",
            "server",
            {
                "buy": msg("server", "client", "confirm"),
                "cancel": end(),
            },
        )

        server_local = projector.project(g, "server")
        monitor = SessionMonitor(
            participant=ParticipantId("server"),
            local_type=server_local,
        )

        # Server expects to receive a choice (branch)
        assert monitor.can_receive()

        # Branch to buy
        buy_payload = MessagePayload(label=MessageLabel("buy"))
        monitor.on_receive(ParticipantId("client"), buy_payload)

        # Server should now send confirm
        assert monitor.can_send()

        confirm_payload = MessagePayload(label=MessageLabel("confirm"))
        monitor.on_send(ParticipantId("client"), confirm_payload)

        assert monitor.is_complete()
