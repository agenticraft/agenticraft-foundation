"""Tests for MPST projection algorithm."""

from __future__ import annotations

import pytest

from agenticraft_foundation.mpst import (
    BranchType,
    EndType,
    LocalEndType,
    LocalRecursionType,
    ParticipantId,
    ReceiveType,
    SelectType,
    SendType,
    choice,
    msg,
)


class TestProjectorBasics:
    """Basic projection tests."""

    def test_project_end(self, projector):
        """Test projecting EndType."""
        g = EndType()
        local = projector.project(g, "alice")
        assert isinstance(local, LocalEndType)

    def test_project_message_as_sender(self, projector, client, server):
        """Test projecting message type for sender."""
        g = msg("client", "server", "request")
        local = projector.project(g, "client")

        assert isinstance(local, SendType)
        assert local.receiver == server
        assert local.payload.label == "request"

    def test_project_message_as_receiver(self, projector, client, server):
        """Test projecting message type for receiver."""
        g = msg("client", "server", "request")
        local = projector.project(g, "server")

        assert isinstance(local, ReceiveType)
        assert local.sender == client
        assert local.payload.label == "request"

    def test_project_message_as_non_participant(self, projector):
        """Test projecting message type for non-participant."""
        g = msg("client", "server", "request")
        local = projector.project(g, "observer")

        # Non-participant projects to the continuation (end in this case)
        assert isinstance(local, LocalEndType)


class TestProjectorRequestResponse:
    """Tests for request-response projection."""

    def test_project_request_response_client(self, projector, request_response_type, server):
        """Test client projection of request-response."""
        local = projector.project(request_response_type, "client")

        # Client: Send(server, request). Recv(server, response). end
        assert isinstance(local, SendType)
        assert local.receiver == server
        assert isinstance(local.continuation, ReceiveType)
        assert local.continuation.sender == server
        assert isinstance(local.continuation.continuation, LocalEndType)

    def test_project_request_response_server(self, projector, request_response_type, client):
        """Test server projection of request-response."""
        local = projector.project(request_response_type, "server")

        # Server: Recv(client, request). Send(client, response). end
        assert isinstance(local, ReceiveType)
        assert local.sender == client
        assert isinstance(local.continuation, SendType)
        assert local.continuation.receiver == client
        assert isinstance(local.continuation.continuation, LocalEndType)


class TestProjectorChoice:
    """Tests for choice projection."""

    def test_project_choice_as_selector(self, projector, choice_type, server):
        """Test projecting choice for the selecting party."""
        local = projector.project(choice_type, "client")

        # Client: Select(server, {buy: ..., cancel: ...})
        assert isinstance(local, SelectType)
        assert local.receiver == server
        assert "buy" in [str(b) for b in local.branches.keys()]
        assert "cancel" in [str(b) for b in local.branches.keys()]

    def test_project_choice_as_brancher(self, projector, choice_type, client):
        """Test projecting choice for the branching party."""
        local = projector.project(choice_type, "server")

        # Server: Branch(client, {buy: ..., cancel: ...})
        assert isinstance(local, BranchType)
        assert local.sender == client
        assert len(local.branches) == 2


class TestProjectorRecursion:
    """Tests for recursion projection."""

    def test_project_recursion_as_participant(self, projector, recursive_type):
        """Test projecting recursive type for a participant."""
        local = projector.project(recursive_type, "client")

        # Should get a local recursion
        assert isinstance(local, LocalRecursionType)
        assert local.variable == "X"

        # Body should be Send
        assert isinstance(local.body, SendType)

    def test_project_recursion_non_participant(self, projector, recursive_type):
        """Test projecting recursive type for non-participant."""
        local = projector.project(recursive_type, "observer")

        # Non-participant gets end
        assert isinstance(local, LocalEndType)


class TestProjectorComplex:
    """Tests for complex projections."""

    def test_three_party_protocol(self, projector):
        """Test projection of three-party protocol.

        A → B : m1.
        B → C : m2.
        C → A : m3.
        end
        """
        g = msg(
            "A",
            "B",
            "m1",
            msg("B", "C", "m2", msg("C", "A", "m3")),
        )

        # Project for A
        local_a = projector.project(g, "A")
        assert isinstance(local_a, SendType)  # Send m1 to B
        # Then receive m3 from C
        assert isinstance(local_a.continuation, ReceiveType)
        assert local_a.continuation.sender == ParticipantId("C")

        # Project for B
        local_b = projector.project(g, "B")
        assert isinstance(local_b, ReceiveType)  # Recv m1 from A
        assert isinstance(local_b.continuation, SendType)  # Send m2 to C

        # Project for C
        local_c = projector.project(g, "C")
        assert isinstance(local_c, ReceiveType)  # Recv m2 from B
        assert isinstance(local_c.continuation, SendType)  # Send m3 to A

    def test_nested_choice(self, projector):
        """Test projection of nested choice."""
        g = choice(
            "a",
            "b",
            {
                "l1": msg("b", "a", "m1"),
                "l2": msg("b", "a", "m2"),
            },
        )

        # Project for a (selector)
        local_a = projector.project(g, "a")
        assert isinstance(local_a, SelectType)

        # Project for b (brancher)
        local_b = projector.project(g, "b")
        assert isinstance(local_b, BranchType)


class TestProjectorErrors:
    """Tests for projection error handling."""

    def test_mergeable_branches(self, projector):
        """Test that branches with same structure merge correctly."""
        # Create a choice where branches have the SAME continuation for c
        g = choice(
            "a",
            "b",
            {
                "l1": msg("b", "c", "m"),  # Same label to c
                "l2": msg("b", "c", "m"),  # Same label to c
            },
        )

        # Both branches send same label to c, so they merge to a single receive
        local_c = projector.project(g, "c")
        assert isinstance(local_c, ReceiveType)

    def test_unmergeable_branches_raise_error(self, projector):
        """Test that branches with incompatible structures raise error."""
        from agenticraft_foundation.mpst.types import ProjectionError

        # Create a choice where branches have different labels to c
        g = choice(
            "a",
            "b",
            {
                "l1": msg("b", "c", "m1"),  # Different label
                "l2": msg("b", "c", "m2"),  # Different label
            },
        )

        # Different labels to c means merge condition is violated
        with pytest.raises(ProjectionError):
            projector.project(g, "c")
