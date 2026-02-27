"""Tests for MPST type system."""

from __future__ import annotations

import pytest

from agenticraft_foundation.mpst import (
    ChoiceType,
    EndType,
    MessageLabel,
    MessagePayload,
    MessageType,
    ParticipantId,
    TypeKind,
    choice,
    end,
    msg,
    parallel,
    rec,
    var,
)


class TestParticipantId:
    """Tests for ParticipantId."""

    def test_creation(self):
        """Test ParticipantId creation."""
        pid = ParticipantId("alice")
        assert pid == "alice"

    def test_from_participant_id(self):
        """Test creating ParticipantId from another ParticipantId."""
        pid1 = ParticipantId("bob")
        pid2 = ParticipantId(pid1)
        assert pid1 == pid2

    def test_equality(self):
        """Test ParticipantId equality."""
        pid1 = ParticipantId("alice")
        pid2 = ParticipantId("alice")
        pid3 = ParticipantId("bob")
        assert pid1 == pid2
        assert pid1 != pid3

    def test_hash(self):
        """Test ParticipantId hashing."""
        pid1 = ParticipantId("alice")
        pid2 = ParticipantId("alice")
        assert hash(pid1) == hash(pid2)
        assert {pid1, pid2} == {pid1}


class TestMessageLabel:
    """Tests for MessageLabel."""

    def test_creation(self):
        """Test MessageLabel creation."""
        label = MessageLabel("request")
        assert label == "request"

    def test_from_message_label(self):
        """Test creating MessageLabel from another MessageLabel."""
        label1 = MessageLabel("response")
        label2 = MessageLabel(label1)
        assert label1 == label2


class TestEndType:
    """Tests for EndType."""

    def test_creation(self):
        """Test EndType creation."""
        e = EndType()
        assert e.kind == TypeKind.END

    def test_is_terminated(self):
        """Test EndType.is_terminated()."""
        e = EndType()
        assert e.is_terminated() is True

    def test_participants(self):
        """Test EndType.participants()."""
        e = EndType()
        assert e.participants() == set()

    def test_convenience_constructor(self):
        """Test end() convenience function."""
        e = end()
        assert isinstance(e, EndType)


class TestMessageType:
    """Tests for MessageType."""

    def test_creation(self, client, server):
        """Test MessageType creation."""
        m = msg("client", "server", "request")
        assert m.kind == TypeKind.MESSAGE
        assert m.sender == client
        assert m.receiver == server
        assert m.payload.label == MessageLabel("request")

    def test_same_sender_receiver_raises(self):
        """Test that same sender and receiver raises error."""
        with pytest.raises(ValueError, match="Sender and receiver must differ"):
            MessageType(
                sender=ParticipantId("alice"),
                receiver=ParticipantId("alice"),
                payload=MessagePayload(label=MessageLabel("msg")),
            )

    def test_is_not_terminated(self):
        """Test MessageType.is_terminated()."""
        m = msg("client", "server", "request")
        assert m.is_terminated() is False

    def test_participants(self, client, server):
        """Test MessageType.participants()."""
        m = msg("client", "server", "request")
        assert m.participants() == {client, server}

    def test_nested_participants(self, client, server):
        """Test participants with nested messages."""
        m = msg(
            "client",
            "server",
            "request",
            msg("server", "client", "response"),
        )
        assert m.participants() == {client, server}

    def test_continuation_default(self):
        """Test MessageType has EndType as default continuation."""
        m = msg("client", "server", "request")
        assert isinstance(m.continuation, EndType)


class TestChoiceType:
    """Tests for ChoiceType."""

    def test_creation(self, client, server):
        """Test ChoiceType creation."""
        c = choice(
            "client",
            "server",
            {
                "buy": msg("server", "client", "confirm"),
                "cancel": msg("server", "client", "cancelled"),
            },
        )
        assert c.kind == TypeKind.CHOICE
        assert c.sender == client
        assert c.receiver == server
        assert len(c.branches) == 2

    def test_empty_branches_raises(self):
        """Test that empty branches raise error."""
        with pytest.raises(ValueError, match="at least one branch"):
            ChoiceType(
                sender=ParticipantId("client"),
                receiver=ParticipantId("server"),
                branches={},
            )

    def test_participants(self, client, server):
        """Test ChoiceType.participants()."""
        c = choice(
            "client",
            "server",
            {
                "buy": msg("server", "client", "confirm"),
                "cancel": end(),
            },
        )
        # Includes sender, receiver, and all participants from branches
        assert client in c.participants()
        assert server in c.participants()


class TestRecursionType:
    """Tests for RecursionType."""

    def test_creation(self):
        """Test RecursionType creation."""
        r = rec("X", msg("client", "server", "ping", var("X")))
        assert r.kind == TypeKind.RECURSION
        assert r.variable == "X"

    def test_unfold_once(self):
        """Test RecursionType.unfold_once()."""
        r = rec("X", msg("client", "server", "ping", var("X")))
        unfolded = r.unfold_once()
        # After unfolding, the variable X is replaced with the recursion
        assert isinstance(unfolded, MessageType)

    def test_participants(self):
        """Test RecursionType.participants()."""
        r = rec("X", msg("client", "server", "ping", var("X")))
        assert ParticipantId("client") in r.participants()
        assert ParticipantId("server") in r.participants()


class TestVariableType:
    """Tests for VariableType."""

    def test_creation(self):
        """Test VariableType creation."""
        v = var("X")
        assert v.kind == TypeKind.VARIABLE
        assert v.name == "X"

    def test_unfold_with_binding(self):
        """Test VariableType.unfold() with binding."""
        v = var("X")
        e = EndType()
        result = v.unfold({"X": e})
        assert result == e

    def test_unfold_without_binding(self):
        """Test VariableType.unfold() without binding."""
        v = var("X")
        result = v.unfold({})
        assert result == v


class TestParallelType:
    """Tests for ParallelType."""

    def test_creation(self):
        """Test ParallelType creation."""
        p = parallel(
            msg("a", "b", "m1"),
            msg("c", "d", "m2"),
        )
        assert p.kind == TypeKind.PARALLEL

    def test_participants(self):
        """Test ParallelType.participants()."""
        p = parallel(
            msg("a", "b", "m1"),
            msg("c", "d", "m2"),
        )
        assert p.participants() == {
            ParticipantId("a"),
            ParticipantId("b"),
            ParticipantId("c"),
            ParticipantId("d"),
        }

    def test_is_terminated_both_end(self):
        """Test ParallelType.is_terminated() when both sides end."""
        p = parallel(end(), end())
        assert p.is_terminated() is True

    def test_is_terminated_one_message(self):
        """Test ParallelType.is_terminated() when one side has message."""
        p = parallel(end(), msg("a", "b", "m"))
        assert p.is_terminated() is False


class TestStructuralEquality:
    """Tests for structural equality of types."""

    def test_end_equality(self):
        """Test EndType equality."""
        e1 = EndType()
        e2 = EndType()
        assert e1 == e2

    def test_message_equality(self):
        """Test MessageType equality."""
        m1 = msg("a", "b", "m")
        m2 = msg("a", "b", "m")
        assert m1 == m2

    def test_message_inequality_different_sender(self):
        """Test MessageType inequality with different sender."""
        m1 = msg("a", "b", "m")
        m2 = msg("c", "b", "m")
        assert m1 != m2

    def test_message_inequality_different_label(self):
        """Test MessageType inequality with different label."""
        m1 = msg("a", "b", "m1")
        m2 = msg("a", "b", "m2")
        assert m1 != m2
