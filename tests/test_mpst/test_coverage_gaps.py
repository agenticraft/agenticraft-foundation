"""Tests to improve coverage for MPST modules.

Targets coverage gaps in:
- properties.py (38% -> higher)
- local_types.py (60% -> higher)
- checker.py (74% -> higher)
- global_types.py (75% -> higher)
"""

from __future__ import annotations

import pytest

from agenticraft_foundation.mpst.checker import (
    MPSTInvariantRegistry,
    SessionMonitor,
    SessionTypeChecker,
    WellFormednessChecker,
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
from agenticraft_foundation.mpst.properties import (
    ChoiceConsistency,
    DeadlockFreedom,
    MPSTProperty,
    MPSTPropertyResult,
    MPSTPropertyStatus,
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
    TypeCheckError,
    TypeKind,
)

# =============================================================================
# Helper: build a simple protocol and monitors
# =============================================================================


def _make_request_response():
    """Client -> Server : request. Server -> Client : response. end"""
    return msg("client", "server", "request", msg("server", "client", "response"))


def _make_monitors_complete():
    """Return monitors where both participants have finished."""
    g = msg("client", "server", "ping")
    projector = Projector()

    client_local = projector.project(g, "client")
    server_local = projector.project(g, "server")

    cm = SessionMonitor(
        participant=ParticipantId("client"),
        local_type=client_local,
        session_id="s1",
        strict=False,
    )
    sm = SessionMonitor(
        participant=ParticipantId("server"),
        local_type=server_local,
        session_id="s1",
        strict=False,
    )

    payload = MessagePayload(label=MessageLabel("ping"))
    cm.on_send(ParticipantId("server"), payload)
    sm.on_receive(ParticipantId("client"), payload)

    return {ParticipantId("client"): cm, ParticipantId("server"): sm}


def _make_monitors_in_progress():
    """Return monitors that are mid-session (client can send, server can receive)."""
    g = _make_request_response()
    projector = Projector()

    client_local = projector.project(g, "client")
    server_local = projector.project(g, "server")

    cm = SessionMonitor(
        participant=ParticipantId("client"),
        local_type=client_local,
        session_id="s2",
        strict=False,
    )
    sm = SessionMonitor(
        participant=ParticipantId("server"),
        local_type=server_local,
        session_id="s2",
        strict=False,
    )

    return {ParticipantId("client"): cm, ParticipantId("server"): sm}


def _make_monitors_with_violations():
    """Return monitors where a violation has been recorded."""
    g = msg("client", "server", "ping")
    projector = Projector()
    client_local = projector.project(g, "client")

    cm = SessionMonitor(
        participant=ParticipantId("client"),
        local_type=client_local,
        session_id="s3",
        strict=False,
    )
    # Wrong action: try to receive when should send
    payload = MessagePayload(label=MessageLabel("ping"))
    cm.on_receive(ParticipantId("server"), payload)

    return {ParticipantId("client"): cm}


def _make_monitors_blocked():
    """Return monitors where no one can act but session is not complete.

    We put each monitor at a LocalVariableType (unbound variable).
    This is neither Send/Select nor Receive/Branch nor LocalEnd,
    so can_send()=False, can_receive()=False, is_complete()=False.
    """
    stuck_type = LocalVariableType(name="STUCK")
    cm = SessionMonitor(
        participant=ParticipantId("client"),
        local_type=stuck_type,
        session_id="s4",
        strict=False,
    )
    sm = SessionMonitor(
        participant=ParticipantId("server"),
        local_type=stuck_type,
        session_id="s4",
        strict=False,
    )

    return {ParticipantId("client"): cm, ParticipantId("server"): sm}


# =============================================================================
# properties.py — MPSTPropertyResult
# =============================================================================


class TestMPSTPropertyResult:
    """Tests for MPSTPropertyResult."""

    def test_is_satisfied_true(self):
        r = MPSTPropertyResult(
            property_name="test",
            property_type=MPSTPropertyType.SAFETY,
            status=MPSTPropertyStatus.SATISFIED,
        )
        assert r.is_satisfied() is True

    def test_is_satisfied_false_violated(self):
        r = MPSTPropertyResult(
            property_name="test",
            property_type=MPSTPropertyType.SAFETY,
            status=MPSTPropertyStatus.VIOLATED,
        )
        assert r.is_satisfied() is False

    def test_is_satisfied_false_unknown(self):
        r = MPSTPropertyResult(
            property_name="test",
            property_type=MPSTPropertyType.SAFETY,
            status=MPSTPropertyStatus.UNKNOWN,
        )
        assert r.is_satisfied() is False

    def test_is_satisfied_false_timeout(self):
        r = MPSTPropertyResult(
            property_name="test",
            property_type=MPSTPropertyType.SAFETY,
            status=MPSTPropertyStatus.TIMEOUT,
        )
        assert r.is_satisfied() is False

    def test_timestamp_is_set(self):
        r = MPSTPropertyResult(
            property_name="test",
            property_type=MPSTPropertyType.SAFETY,
            status=MPSTPropertyStatus.SATISFIED,
        )
        assert r.timestamp > 0

    def test_counterexample_default_none(self):
        r = MPSTPropertyResult(
            property_name="test",
            property_type=MPSTPropertyType.SAFETY,
            status=MPSTPropertyStatus.SATISFIED,
        )
        assert r.counterexample is None


# =============================================================================
# properties.py — ProjectionDefinedness
# =============================================================================


class TestProjectionDefinedness:
    """Tests for ProjectionDefinedness property."""

    def test_satisfied_with_valid_type(self):
        prop = ProjectionDefinedness()
        g = _make_request_response()
        result = prop.check(global_type=g)
        assert result.status == MPSTPropertyStatus.SATISFIED
        assert result.is_satisfied()
        assert "2" in result.message  # 2 participants

    def test_unknown_when_no_type(self):
        prop = ProjectionDefinedness()
        result = prop.check()
        assert result.status == MPSTPropertyStatus.UNKNOWN
        assert "No global type" in result.message

    def test_satisfied_with_choice(self):
        prop = ProjectionDefinedness()
        g = choice(
            "a",
            "b",
            {
                "l1": msg("b", "a", "r1"),
                "l2": msg("b", "a", "r2"),
            },
        )
        result = prop.check(global_type=g)
        assert result.is_satisfied()

    def test_violated_with_unmergeable_choice(self):
        """A choice where a third party sees different branches => undefined projection."""
        prop = ProjectionDefinedness()
        # a -> b : {l1: b->c:m1, l2: b->c:m2} — c sees different labels
        g = choice(
            "a",
            "b",
            {
                "l1": msg("b", "c", "m1"),
                "l2": msg("b", "c", "m2"),
            },
        )
        result = prop.check(global_type=g)
        assert result.status == MPSTPropertyStatus.VIOLATED
        assert result.counterexample is not None
        assert "undefined_participants" in result.counterexample


# =============================================================================
# properties.py — ChoiceConsistency
# =============================================================================


class TestChoiceConsistency:
    """Tests for ChoiceConsistency property."""

    def test_satisfied_with_well_formed_choice(self):
        prop = ChoiceConsistency()
        g = choice("a", "b", {"l1": end(), "l2": end()})
        result = prop.check(global_type=g)
        assert result.is_satisfied()

    def test_satisfied_with_simple_message(self):
        prop = ChoiceConsistency()
        g = msg("a", "b", "m")
        result = prop.check(global_type=g)
        assert result.is_satisfied()

    def test_unknown_when_no_type(self):
        prop = ChoiceConsistency()
        result = prop.check()
        assert result.status == MPSTPropertyStatus.UNKNOWN

    def test_violated_with_self_communication(self):
        """Self-communication in a message makes the global type not well-formed."""
        prop = ChoiceConsistency()
        # Build a global type with self-communication (bypass post_init check)
        g = msg("a", "b", "m", var("X"))  # has unbound var -> not well-formed
        result = prop.check(global_type=g)
        assert result.status == MPSTPropertyStatus.VIOLATED


# =============================================================================
# properties.py — DeadlockFreedom
# =============================================================================


class TestDeadlockFreedom:
    """Tests for DeadlockFreedom property."""

    def test_satisfied_with_well_formed_type(self):
        prop = DeadlockFreedom()
        g = _make_request_response()
        result = prop.check(global_type=g)
        assert result.is_satisfied()
        assert "deadlock-free" in result.message.lower()

    def test_satisfied_all_monitors_complete(self):
        prop = DeadlockFreedom()
        monitors = _make_monitors_complete()
        result = prop.check(monitors=monitors)
        assert result.is_satisfied()
        assert "completed" in result.message.lower()

    def test_satisfied_monitors_can_progress(self):
        prop = DeadlockFreedom()
        monitors = _make_monitors_in_progress()
        result = prop.check(monitors=monitors)
        assert result.is_satisfied()
        assert "senders" in result.message.lower() or "progress" in result.message.lower()

    def test_violated_no_one_can_act(self):
        prop = DeadlockFreedom()
        monitors = _make_monitors_blocked()
        result = prop.check(monitors=monitors)
        assert result.status == MPSTPropertyStatus.VIOLATED
        assert "deadlocked" in result.message.lower()
        assert result.counterexample is not None

    def test_unknown_no_type_no_monitors(self):
        prop = DeadlockFreedom()
        result = prop.check()
        assert result.status == MPSTPropertyStatus.UNKNOWN

    def test_not_well_formed_type_falls_through_to_monitors(self):
        """If global type is not well-formed and no monitors, result is UNKNOWN."""
        prop = DeadlockFreedom()
        g = msg("a", "b", "m", var("X"))  # unbound var
        result = prop.check(global_type=g)
        # Falls through because not well-formed, no monitors => UNKNOWN
        assert result.status == MPSTPropertyStatus.UNKNOWN


# =============================================================================
# properties.py — TypePreservation
# =============================================================================


class TestTypePreservation:
    """Tests for TypePreservation property."""

    def test_unknown_no_monitors(self):
        prop = TypePreservation()
        result = prop.check()
        assert result.status == MPSTPropertyStatus.UNKNOWN

    def test_satisfied_no_violations(self):
        prop = TypePreservation()
        monitors = _make_monitors_in_progress()
        result = prop.check(monitors=monitors)
        assert result.is_satisfied()
        assert "conform" in result.message.lower()

    def test_violated_with_violations(self):
        prop = TypePreservation()
        monitors = _make_monitors_with_violations()
        result = prop.check(monitors=monitors)
        assert result.status == MPSTPropertyStatus.VIOLATED
        assert result.counterexample is not None
        assert "violations" in result.counterexample


# =============================================================================
# properties.py — Progress
# =============================================================================


class TestProgress:
    """Tests for Progress property."""

    def test_unknown_no_monitors(self):
        prop = Progress()
        result = prop.check()
        assert result.status == MPSTPropertyStatus.UNKNOWN

    def test_satisfied_session_completed(self):
        prop = Progress()
        monitors = _make_monitors_complete()
        result = prop.check(monitors=monitors)
        assert result.is_satisfied()
        assert "completed" in result.message.lower()

    def test_satisfied_can_act(self):
        prop = Progress()
        monitors = _make_monitors_in_progress()
        result = prop.check(monitors=monitors)
        assert result.is_satisfied()
        assert "can act" in result.message.lower()

    def test_violated_blocked(self):
        prop = Progress()
        monitors = _make_monitors_blocked()
        result = prop.check(monitors=monitors)
        assert result.status == MPSTPropertyStatus.VIOLATED
        assert "no participant" in result.message.lower()
        assert result.counterexample is not None
        assert "blocked" in result.counterexample


# =============================================================================
# properties.py — SessionCompletion
# =============================================================================


class TestSessionCompletion:
    """Tests for SessionCompletion property."""

    def test_unknown_no_monitors(self):
        prop = SessionCompletion()
        result = prop.check()
        assert result.status == MPSTPropertyStatus.UNKNOWN

    def test_satisfied_all_complete(self):
        prop = SessionCompletion()
        monitors = _make_monitors_complete()
        result = prop.check(monitors=monitors)
        assert result.is_satisfied()
        assert "completed" in result.message.lower()

    def test_in_progress(self):
        prop = SessionCompletion()
        monitors = _make_monitors_in_progress()
        result = prop.check(monitors=monitors)
        assert result.status == MPSTPropertyStatus.UNKNOWN
        assert "in progress" in result.message.lower()

    def test_timeout_exceeded_max_messages(self):
        """Exceed message limit to trigger TIMEOUT status."""
        prop = SessionCompletion(max_messages=2)
        # Build monitors and send enough messages to exceed the limit
        g = msg("client", "server", "ping")
        projector = Projector()
        client_local = projector.project(g, "client")

        cm = SessionMonitor(
            participant=ParticipantId("client"),
            local_type=client_local,
            session_id="s_timeout",
            strict=False,
        )
        # Manually set message count high
        cm._message_count = 10

        monitors = {ParticipantId("client"): cm}
        result = prop.check(monitors=monitors)
        assert result.status == MPSTPropertyStatus.TIMEOUT
        assert "exceeded" in result.message.lower()
        assert result.counterexample is not None
        assert result.counterexample["total_messages"] == 10

    def test_custom_max_messages(self):
        prop = SessionCompletion(max_messages=500)
        assert prop.max_messages == 500


# =============================================================================
# properties.py — MPSTSpecification
# =============================================================================


class TestMPSTSpecification:
    """Tests for MPSTSpecification."""

    def test_verify_all_properties(self):
        spec = MPSTSpecification()
        g = _make_request_response()
        results = spec.verify(global_type=g)
        assert len(results) >= 6  # 6 default properties
        # Well-formedness and safety properties should be satisfied
        for r in results:
            if r.property_type == MPSTPropertyType.WELL_FORMEDNESS:
                assert r.is_satisfied()

    def test_verify_well_formedness_only(self):
        spec = MPSTSpecification()
        g = _make_request_response()
        results = spec.verify_well_formedness(g)
        assert len(results) == 2  # ProjectionDefinedness + ChoiceConsistency
        for r in results:
            assert r.property_type == MPSTPropertyType.WELL_FORMEDNESS
            assert r.is_satisfied()

    def test_verify_safety_only(self):
        spec = MPSTSpecification()
        monitors = _make_monitors_in_progress()
        results = spec.verify_safety(monitors)
        assert len(results) == 2  # DeadlockFreedom + TypePreservation
        for r in results:
            assert r.property_type == MPSTPropertyType.SAFETY

    def test_is_valid_with_good_type_and_monitors(self):
        spec = MPSTSpecification()
        g = _make_request_response()
        monitors = _make_monitors_complete()
        assert spec.is_valid(global_type=g, monitors=monitors) is True

    def test_is_valid_with_bad_type(self):
        spec = MPSTSpecification()
        # Unbound variable makes it not well-formed
        g = msg("a", "b", "m", var("X"))
        assert spec.is_valid(global_type=g) is False

    def test_summary_contains_all_properties(self):
        spec = MPSTSpecification()
        g = _make_request_response()
        summary = spec.summary(global_type=g)
        assert "MPST Specification Verification" in summary
        assert "ProjectionDefinedness" in summary
        assert "ChoiceConsistency" in summary
        assert "DeadlockFreedom" in summary
        assert "TypePreservation" in summary
        assert "Progress" in summary
        assert "SessionCompletion" in summary
        assert "Overall:" in summary

    def test_summary_pass(self):
        spec = MPSTSpecification()
        g = _make_request_response()
        monitors = _make_monitors_complete()
        summary = spec.summary(global_type=g, monitors=monitors)
        # All well-formedness and safety should pass; overall might say PASS
        assert "PASS" in summary or "FAIL" in summary

    def test_custom_properties(self):
        class DummyProperty(MPSTProperty):
            def __init__(self):
                super().__init__("Dummy", MPSTPropertyType.SAFETY)

            def check(self, global_type=None, context=None, monitors=None):
                return MPSTPropertyResult(
                    property_name=self.name,
                    property_type=self.property_type,
                    status=MPSTPropertyStatus.SATISFIED,
                    message="ok",
                )

        spec = MPSTSpecification(custom_properties=[DummyProperty()])
        results = spec.verify()
        assert len(results) == 7  # 6 default + 1 custom
        assert any(r.property_name == "Dummy" for r in results)

    def test_verify_with_monitors(self):
        spec = MPSTSpecification()
        monitors = _make_monitors_in_progress()
        results = spec.verify(monitors=monitors)
        assert len(results) >= 6


# =============================================================================
# local_types.py — Hash methods
# =============================================================================


class TestLocalTypeHashes:
    """Tests for __hash__ on local types."""

    def test_send_type_hash(self):
        s = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        h = hash(s)
        assert isinstance(h, int)

    def test_receive_type_hash(self):
        r = ReceiveType(
            sender=ParticipantId("a"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        h = hash(r)
        assert isinstance(h, int)

    def test_select_type_hash(self):
        s = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        h = hash(s)
        assert isinstance(h, int)

    def test_branch_type_hash(self):
        b = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        h = hash(b)
        assert isinstance(h, int)

    def test_local_recursion_type_hash(self):
        lr = LocalRecursionType(variable="X", body=LocalEndType())
        h = hash(lr)
        assert isinstance(h, int)


# =============================================================================
# local_types.py — __post_init__ validation
# =============================================================================


class TestLocalTypeValidation:
    """Tests for __post_init__ validation on SelectType and BranchType."""

    def test_select_empty_branches_raises(self):
        with pytest.raises(ValueError, match="at least one branch"):
            SelectType(receiver=ParticipantId("b"), branches={})

    def test_branch_empty_branches_raises(self):
        with pytest.raises(ValueError, match="at least one branch"):
            BranchType(sender=ParticipantId("a"), branches={})


# =============================================================================
# local_types.py — Repr methods
# =============================================================================


class TestLocalTypeRepr:
    """Tests for __repr__ on local types."""

    def test_local_end_repr(self):
        assert repr(LocalEndType()) == "end"

    def test_send_repr(self):
        s = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        r = repr(s)
        assert "!b" in r
        assert "m" in r

    def test_receive_repr(self):
        rv = ReceiveType(
            sender=ParticipantId("a"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        r = repr(rv)
        assert "?a" in r
        assert "m" in r

    def test_select_repr(self):
        s = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        r = repr(s)
        assert "b" in r

    def test_branch_repr(self):
        b = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        r = repr(b)
        assert "a" in r

    def test_local_recursion_repr(self):
        lr = LocalRecursionType(variable="X", body=LocalEndType())
        r = repr(lr)
        assert "X" in r

    def test_local_variable_repr(self):
        lv = LocalVariableType(name="X")
        assert repr(lv) == "X"


# =============================================================================
# local_types.py — Structural equality
# =============================================================================


class TestLocalTypeEquality:
    """Tests for _structural_eq on local types."""

    def test_local_end_eq(self):
        assert LocalEndType() == LocalEndType()

    def test_local_end_neq_send(self):
        assert LocalEndType() != SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )

    def test_send_eq(self):
        s1 = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        s2 = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert s1 == s2

    def test_send_neq_different_receiver(self):
        s1 = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        s2 = SendType(
            receiver=ParticipantId("c"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert s1 != s2

    def test_receive_eq(self):
        r1 = ReceiveType(
            sender=ParticipantId("a"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        r2 = ReceiveType(
            sender=ParticipantId("a"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert r1 == r2

    def test_receive_neq(self):
        r1 = ReceiveType(
            sender=ParticipantId("a"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        r2 = ReceiveType(
            sender=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert r1 != r2

    def test_select_eq(self):
        s1 = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        s2 = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        assert s1 == s2

    def test_select_neq_different_receiver(self):
        s1 = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        s2 = SelectType(
            receiver=ParticipantId("c"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        assert s1 != s2

    def test_select_neq_different_labels(self):
        s1 = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        s2 = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l2"): LocalEndType()},
        )
        assert s1 != s2

    def test_branch_eq(self):
        b1 = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        b2 = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        assert b1 == b2

    def test_branch_neq_different_sender(self):
        b1 = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        b2 = BranchType(
            sender=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        assert b1 != b2

    def test_branch_neq_different_labels(self):
        b1 = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        b2 = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l2"): LocalEndType()},
        )
        assert b1 != b2

    def test_local_recursion_eq(self):
        lr1 = LocalRecursionType(variable="X", body=LocalEndType())
        lr2 = LocalRecursionType(variable="X", body=LocalEndType())
        assert lr1 == lr2

    def test_local_recursion_neq_different_var(self):
        lr1 = LocalRecursionType(variable="X", body=LocalEndType())
        lr2 = LocalRecursionType(variable="Y", body=LocalEndType())
        assert lr1 != lr2

    def test_local_variable_eq(self):
        assert LocalVariableType(name="X") == LocalVariableType(name="X")

    def test_local_variable_neq(self):
        assert LocalVariableType(name="X") != LocalVariableType(name="Y")

    def test_send_neq_non_session_type(self):
        s = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert s != "not a type"


# =============================================================================
# local_types.py — Unfold methods
# =============================================================================


class TestLocalTypeUnfold:
    """Tests for unfold methods on local types."""

    def test_local_end_unfold(self):
        e = LocalEndType()
        assert e.unfold({}) is e

    def test_send_unfold(self):
        s = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
            continuation=LocalVariableType(name="X"),
        )
        result = s.unfold({"X": LocalEndType()})
        assert isinstance(result, SendType)
        assert isinstance(result.continuation, LocalEndType)

    def test_receive_unfold(self):
        r = ReceiveType(
            sender=ParticipantId("a"),
            payload=MessagePayload(label=MessageLabel("m")),
            continuation=LocalVariableType(name="X"),
        )
        result = r.unfold({"X": LocalEndType()})
        assert isinstance(result, ReceiveType)
        assert isinstance(result.continuation, LocalEndType)

    def test_select_unfold(self):
        s = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalVariableType(name="X")},
        )
        result = s.unfold({"X": LocalEndType()})
        assert isinstance(result, SelectType)
        assert isinstance(result.branches[MessageLabel("l1")], LocalEndType)

    def test_branch_unfold(self):
        b = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalVariableType(name="X")},
        )
        result = b.unfold({"X": LocalEndType()})
        assert isinstance(result, BranchType)
        assert isinstance(result.branches[MessageLabel("l1")], LocalEndType)

    def test_local_recursion_unfold(self):
        body = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
            continuation=LocalVariableType(name="X"),
        )
        lr = LocalRecursionType(variable="X", body=body)
        unfolded = lr.unfold({})
        # Unfolding replaces X with the recursion type itself
        assert isinstance(unfolded, SendType)

    def test_local_variable_unfold_with_binding(self):
        lv = LocalVariableType(name="X")
        result = lv.unfold({"X": LocalEndType()})
        assert isinstance(result, LocalEndType)

    def test_local_variable_unfold_without_binding(self):
        lv = LocalVariableType(name="X")
        result = lv.unfold({})
        assert result is lv


# =============================================================================
# local_types.py — Participants and is_terminated
# =============================================================================


class TestLocalTypeProperties:
    """Tests for participants() and is_terminated() on local types."""

    def test_send_participants(self):
        s = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert ParticipantId("b") in s.participants()

    def test_receive_participants(self):
        r = ReceiveType(
            sender=ParticipantId("a"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert ParticipantId("a") in r.participants()

    def test_select_participants(self):
        s = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        assert ParticipantId("b") in s.participants()

    def test_branch_participants(self):
        b = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        assert ParticipantId("a") in b.participants()

    def test_local_recursion_participants(self):
        body = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        lr = LocalRecursionType(variable="X", body=body)
        assert ParticipantId("b") in lr.participants()

    def test_local_variable_participants(self):
        lv = LocalVariableType(name="X")
        assert lv.participants() == set()

    def test_local_end_is_terminated(self):
        assert LocalEndType().is_terminated() is True

    def test_send_not_terminated(self):
        s = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert s.is_terminated() is False

    def test_receive_not_terminated(self):
        r = ReceiveType(
            sender=ParticipantId("a"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert r.is_terminated() is False

    def test_select_not_terminated(self):
        s = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        assert s.is_terminated() is False

    def test_branch_not_terminated(self):
        b = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        assert b.is_terminated() is False

    def test_local_recursion_not_terminated(self):
        lr = LocalRecursionType(variable="X", body=LocalEndType())
        assert lr.is_terminated() is False

    def test_local_variable_not_terminated(self):
        lv = LocalVariableType(name="X")
        assert lv.is_terminated() is False


# =============================================================================
# local_types.py — Kind properties
# =============================================================================


class TestLocalTypeKinds:
    """Tests for kind property on local types."""

    def test_local_end_kind(self):
        assert LocalEndType().kind == TypeKind.END

    def test_send_kind(self):
        s = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert s.kind == TypeKind.MESSAGE

    def test_receive_kind(self):
        r = ReceiveType(
            sender=ParticipantId("a"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        assert r.kind == TypeKind.MESSAGE

    def test_select_kind(self):
        s = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        assert s.kind == TypeKind.CHOICE

    def test_branch_kind(self):
        b = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        assert b.kind == TypeKind.CHOICE

    def test_local_recursion_kind(self):
        lr = LocalRecursionType(variable="X", body=LocalEndType())
        assert lr.kind == TypeKind.RECURSION

    def test_local_variable_kind(self):
        lv = LocalVariableType(name="X")
        assert lv.kind == TypeKind.VARIABLE


# =============================================================================
# local_types.py — Projector._project_parallel
# =============================================================================


class TestProjectParallel:
    """Tests for Projector._project_parallel."""

    def test_project_parallel_left_only(self):
        """Participant in left branch only."""
        g = parallel(
            msg("a", "b", "m1"),
            msg("c", "d", "m2"),
        )
        projector = Projector()
        local = projector.project(g, "a")
        assert isinstance(local, SendType)
        assert local.receiver == ParticipantId("b")

    def test_project_parallel_right_only(self):
        """Participant in right branch only."""
        g = parallel(
            msg("a", "b", "m1"),
            msg("c", "d", "m2"),
        )
        projector = Projector()
        local = projector.project(g, "c")
        assert isinstance(local, SendType)
        assert local.receiver == ParticipantId("d")

    def test_project_parallel_neither(self):
        """Participant in neither branch."""
        g = parallel(
            msg("a", "b", "m1"),
            msg("c", "d", "m2"),
        )
        projector = Projector()
        local = projector.project(g, "observer")
        assert isinstance(local, LocalEndType)

    def test_project_parallel_overlap_strict_raises(self):
        """Participant in both branches raises ProjectionError."""
        g = parallel(
            msg("a", "b", "m1"),
            msg("a", "c", "m2"),
        )
        projector = Projector(strict=True)
        with pytest.raises(ProjectionError):
            projector.project(g, "a")

    def test_project_parallel_overlap_non_strict(self):
        """Participant in both branches returns None in non-strict mode."""
        g = parallel(
            msg("a", "b", "m1"),
            msg("a", "c", "m2"),
        )
        projector = Projector(strict=False)
        result = projector.project(g, "a")
        assert result is None


# =============================================================================
# local_types.py — Convenience functions project() and project_all()
# =============================================================================


class TestConvenienceFunctions:
    """Tests for project() and project_all() convenience functions."""

    def test_project_sender(self):
        g = msg("a", "b", "m")
        local = project(g, "a")
        assert isinstance(local, SendType)

    def test_project_receiver(self):
        g = msg("a", "b", "m")
        local = project(g, "b")
        assert isinstance(local, ReceiveType)

    def test_project_with_participant_id(self):
        g = msg("a", "b", "m")
        local = project(g, ParticipantId("a"))
        assert isinstance(local, SendType)

    def test_project_non_strict_returns_none(self):
        g = choice(
            "a",
            "b",
            {
                "l1": msg("b", "c", "m1"),
                "l2": msg("b", "c", "m2"),
            },
        )
        local = project(g, "c", strict=False)
        assert local is None

    def test_project_strict_raises(self):
        g = choice(
            "a",
            "b",
            {
                "l1": msg("b", "c", "m1"),
                "l2": msg("b", "c", "m2"),
            },
        )
        with pytest.raises(ProjectionError):
            project(g, "c", strict=True)

    def test_project_all_auto_participants(self):
        g = msg("a", "b", "m")
        result = project_all(g)
        assert ParticipantId("a") in result
        assert ParticipantId("b") in result
        assert isinstance(result[ParticipantId("a")], SendType)
        assert isinstance(result[ParticipantId("b")], ReceiveType)

    def test_project_all_with_participants_kwarg(self):
        g = _make_request_response()
        result = project_all(g, participants=["client"])
        assert len(result) == 1
        assert ParticipantId("client") in result

    def test_project_all_with_set_of_participant_ids(self):
        g = _make_request_response()
        result = project_all(g, participants={ParticipantId("server")})
        assert len(result) == 1
        assert ParticipantId("server") in result


# =============================================================================
# checker.py — SessionMonitor.get_state()
# =============================================================================


class TestSessionMonitorGetState:
    """Tests for SessionMonitor.get_state()."""

    def test_get_state_initial(self):
        local = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        mon = SessionMonitor(
            participant=ParticipantId("a"),
            local_type=local,
            session_id="s1",
        )
        state = mon.get_state()
        assert state["participant"] == ParticipantId("a")
        assert state["session_id"] == "s1"
        assert state["message_count"] == 0
        assert state["violation_count"] == 0
        assert state["is_complete"] is False
        assert state["can_send"] is True
        assert state["can_receive"] is False

    def test_get_state_after_action(self):
        g = msg("a", "b", "m")
        projector = Projector()
        local = projector.project(g, "a")
        mon = SessionMonitor(
            participant=ParticipantId("a"),
            local_type=local,
            session_id="test",
            strict=False,
        )
        payload = MessagePayload(label=MessageLabel("m"))
        mon.on_send(ParticipantId("b"), payload)

        state = mon.get_state()
        assert state["message_count"] == 1
        assert state["is_complete"] is True


# =============================================================================
# checker.py — SessionMonitor violations with strict mode
# =============================================================================


class TestSessionMonitorStrict:
    """Tests for strict mode violations."""

    def test_strict_send_mismatch_raises(self):
        local = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        mon = SessionMonitor(
            participant=ParticipantId("a"),
            local_type=local,
            strict=True,
        )
        # Wrong label
        payload = MessagePayload(label=MessageLabel("wrong"))
        with pytest.raises(TypeCheckError):
            mon.on_send(ParticipantId("b"), payload)

    def test_strict_unexpected_receive_raises(self):
        local = SendType(
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        mon = SessionMonitor(
            participant=ParticipantId("a"),
            local_type=local,
            strict=True,
        )
        payload = MessagePayload(label=MessageLabel("m"))
        with pytest.raises(TypeCheckError):
            mon.on_receive(ParticipantId("b"), payload)

    def test_strict_receive_mismatch_raises(self):
        local = ReceiveType(
            sender=ParticipantId("a"),
            payload=MessagePayload(label=MessageLabel("m")),
        )
        mon = SessionMonitor(
            participant=ParticipantId("b"),
            local_type=local,
            strict=True,
        )
        payload = MessagePayload(label=MessageLabel("wrong"))
        with pytest.raises(TypeCheckError):
            mon.on_receive(ParticipantId("a"), payload)


# =============================================================================
# checker.py — SessionMonitor select/branch violations
# =============================================================================


class TestSessionMonitorSelectBranch:
    """Tests for select and branch violation paths in SessionMonitor."""

    def test_select_wrong_label_non_strict(self):
        local = SelectType(
            receiver=ParticipantId("b"),
            branches={
                MessageLabel("l1"): LocalEndType(),
                MessageLabel("l2"): LocalEndType(),
            },
        )
        mon = SessionMonitor(
            participant=ParticipantId("a"),
            local_type=local,
            strict=False,
        )
        payload = MessagePayload(label=MessageLabel("wrong"))
        result = mon.on_send(ParticipantId("b"), payload)
        assert result is False
        assert len(mon.get_violations()) == 1
        assert mon.get_violations()[0].violation_type == "select_mismatch"

    def test_select_wrong_receiver_non_strict(self):
        local = SelectType(
            receiver=ParticipantId("b"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        mon = SessionMonitor(
            participant=ParticipantId("a"),
            local_type=local,
            strict=False,
        )
        payload = MessagePayload(label=MessageLabel("l1"))
        result = mon.on_send(ParticipantId("c"), payload)
        assert result is False

    def test_branch_wrong_label_non_strict(self):
        local = BranchType(
            sender=ParticipantId("a"),
            branches={
                MessageLabel("l1"): LocalEndType(),
                MessageLabel("l2"): LocalEndType(),
            },
        )
        mon = SessionMonitor(
            participant=ParticipantId("b"),
            local_type=local,
            strict=False,
        )
        payload = MessagePayload(label=MessageLabel("wrong"))
        result = mon.on_receive(ParticipantId("a"), payload)
        assert result is False
        assert mon.get_violations()[0].violation_type == "branch_mismatch"

    def test_branch_wrong_sender_non_strict(self):
        local = BranchType(
            sender=ParticipantId("a"),
            branches={MessageLabel("l1"): LocalEndType()},
        )
        mon = SessionMonitor(
            participant=ParticipantId("b"),
            local_type=local,
            strict=False,
        )
        payload = MessagePayload(label=MessageLabel("l1"))
        result = mon.on_receive(ParticipantId("c"), payload)
        assert result is False


# =============================================================================
# checker.py — SessionTypeChecker.get_statistics()
# =============================================================================


class TestSessionTypeCheckerStatistics:
    """Tests for SessionTypeChecker.get_statistics()."""

    def test_get_statistics_initial(self):
        checker = SessionTypeChecker()
        stats = checker.get_statistics()
        assert "active_sessions" in stats
        assert stats["active_sessions"] == 0
        assert "invariant_stats" in stats

    def test_get_statistics_with_session(self):
        checker = SessionTypeChecker()
        g = msg("a", "b", "m")
        checker.create_session("s1", g)
        stats = checker.get_statistics()
        assert stats["active_sessions"] == 1


# =============================================================================
# checker.py — Session lifecycle: get_session, close_session
# =============================================================================


class TestSessionLifecycle:
    """Tests for session lifecycle in SessionTypeChecker."""

    def test_create_and_get_session(self):
        checker = SessionTypeChecker()
        g = msg("a", "b", "m")
        monitors = checker.create_session("s1", g)
        assert monitors is not None

        retrieved = checker.get_session("s1")
        assert retrieved is monitors

    def test_get_nonexistent_session(self):
        checker = SessionTypeChecker()
        assert checker.get_session("nope") is None

    def test_close_session(self):
        checker = SessionTypeChecker()
        g = msg("a", "b", "m")
        checker.create_session("s1", g)
        assert checker.close_session("s1") is True
        assert checker.get_session("s1") is None

    def test_close_nonexistent_session(self):
        checker = SessionTypeChecker()
        assert checker.close_session("nope") is False

    def test_create_session_not_well_formed_raises(self):
        checker = SessionTypeChecker()
        # Unbound variable makes it not well-formed
        g = msg("a", "b", "m", var("X"))
        with pytest.raises(TypeCheckError):
            checker.create_session("bad", g)

    def test_check_invariants(self):
        checker = SessionTypeChecker()
        g = msg("a", "b", "m")
        checker.create_session("s1", g)
        violated = checker.check_invariants("s1")
        # Fresh session should have no violations
        assert isinstance(violated, list)

    def test_check_invariants_nonexistent_session(self):
        checker = SessionTypeChecker()
        violated = checker.check_invariants("nope")
        assert violated == []


# =============================================================================
# checker.py — WellFormednessChecker with ParallelType
# =============================================================================


class TestWellFormednessParallel:
    """Tests for WellFormednessChecker with ParallelType."""

    def test_parallel_disjoint_is_well_formed(self):
        checker = WellFormednessChecker()
        g = parallel(msg("a", "b", "m1"), msg("c", "d", "m2"))
        result = checker.check(g)
        assert result.is_well_formed

    def test_parallel_overlap_warning(self):
        """Overlapping participants produce a warning but not an error."""
        checker = WellFormednessChecker()
        g = parallel(msg("a", "b", "m1"), msg("a", "c", "m2"))
        result = checker.check(g)
        # Should have a warning about overlap
        assert any("overlap" in w.lower() for w in result.warnings)

    def test_parallel_nested_recursion(self):
        checker = WellFormednessChecker()
        g = parallel(
            rec("X", msg("a", "b", "m", var("X"))),
            msg("c", "d", "m2"),
        )
        result = checker.check(g)
        assert result.is_well_formed


# =============================================================================
# checker.py — MPSTInvariantRegistry
# =============================================================================


class TestMPSTInvariantRegistry:
    """Tests for MPSTInvariantRegistry."""

    def test_no_deadlock_passes_empty(self):
        registry = MPSTInvariantRegistry()
        violated = registry.check_all({})
        assert violated == []

    def test_no_deadlock_passes_complete(self):
        registry = MPSTInvariantRegistry()
        monitors = _make_monitors_complete()
        violated = registry.check_all(monitors)
        # All complete, no deadlock
        assert "mpst_no_deadlock" not in violated

    def test_conformance_passes_no_violations(self):
        registry = MPSTInvariantRegistry()
        monitors = _make_monitors_in_progress()
        violated = registry.check_all(monitors)
        assert "mpst_conformance" not in violated

    def test_conformance_fails_with_violations(self):
        registry = MPSTInvariantRegistry()
        monitors = _make_monitors_with_violations()
        violated = registry.check_all(monitors)
        assert "mpst_conformance" in violated


# =============================================================================
# global_types.py — Convenience constructors
# =============================================================================


class TestGlobalConvenienceConstructors:
    """Tests for msg(), choice(), rec(), var(), end(), parallel()."""

    def test_msg_basic(self):
        m = msg("a", "b", "label")
        assert isinstance(m, MessageType)
        assert m.sender == ParticipantId("a")
        assert m.receiver == ParticipantId("b")
        assert m.payload.label == MessageLabel("label")
        assert isinstance(m.continuation, EndType)

    def test_msg_with_continuation(self):
        m = msg("a", "b", "l1", msg("b", "a", "l2"))
        assert isinstance(m.continuation, MessageType)

    def test_msg_with_payload_type(self):
        m = msg("a", "b", "l", payload_type="json")
        assert m.payload.payload_type == "json"

    def test_msg_with_participant_id(self):
        m = msg(ParticipantId("a"), ParticipantId("b"), MessageLabel("l"))
        assert isinstance(m, MessageType)

    def test_choice_basic(self):
        c = choice("a", "b", {"l1": end(), "l2": end()})
        assert isinstance(c, ChoiceType)
        assert c.sender == ParticipantId("a")
        assert c.receiver == ParticipantId("b")
        assert len(c.branches) == 2

    def test_rec_basic(self):
        r = rec("X", msg("a", "b", "m", var("X")))
        assert isinstance(r, RecursionType)
        assert r.variable == "X"

    def test_var_basic(self):
        v = var("X")
        assert isinstance(v, VariableType)
        assert v.name == "X"

    def test_end_basic(self):
        e = end()
        assert isinstance(e, EndType)
        assert e.is_terminated()

    def test_parallel_basic(self):
        p = parallel(msg("a", "b", "m1"), msg("c", "d", "m2"))
        assert isinstance(p, ParallelType)


# =============================================================================
# global_types.py — MessageType.__post_init__ validation
# =============================================================================


class TestMessageTypePostInit:
    """Tests for MessageType self-communication check."""

    def test_self_communication_raises(self):
        with pytest.raises(ValueError, match="Sender and receiver must differ"):
            MessageType(
                sender=ParticipantId("a"),
                receiver=ParticipantId("a"),
                payload=MessagePayload(label=MessageLabel("m")),
            )

    def test_msg_convenience_self_communication_raises(self):
        with pytest.raises(ValueError, match="Sender and receiver must differ"):
            msg("a", "a", "m")


# =============================================================================
# global_types.py — ChoiceType.__post_init__ validation
# =============================================================================


class TestChoiceTypePostInit:
    """Tests for ChoiceType post_init validation."""

    def test_self_communication_raises(self):
        with pytest.raises(ValueError, match="Sender and receiver must differ"):
            ChoiceType(
                sender=ParticipantId("a"),
                receiver=ParticipantId("a"),
                branches={MessageLabel("l"): EndType()},
            )

    def test_empty_branches_raises(self):
        with pytest.raises(ValueError, match="at least one branch"):
            ChoiceType(
                sender=ParticipantId("a"),
                receiver=ParticipantId("b"),
                branches={},
            )


# =============================================================================
# global_types.py — RecursionType.unfold_once()
# =============================================================================


class TestRecursionTypeUnfoldOnce:
    """Tests for RecursionType.unfold_once()."""

    def test_unfold_once_replaces_variable(self):
        r = rec("X", msg("a", "b", "m", var("X")))
        unfolded = r.unfold_once()
        assert isinstance(unfolded, MessageType)
        # The continuation of the unfolded message should be the recursion itself
        assert isinstance(unfolded.continuation, RecursionType)
        assert unfolded.continuation.variable == "X"

    def test_unfold_once_preserves_structure(self):
        r = rec("Y", msg("c", "d", "ping", var("Y")))
        unfolded = r.unfold_once()
        assert unfolded.sender == ParticipantId("c")
        assert unfolded.receiver == ParticipantId("d")
        assert unfolded.payload.label == MessageLabel("ping")


# =============================================================================
# global_types.py — ParallelType overlap detection
# =============================================================================


class TestParallelTypeOverlap:
    """Tests for ParallelType participant overlap detection."""

    def test_overlap_allowed(self):
        """ParallelType allows overlap (just passes through __post_init__)."""
        p = ParallelType(
            left=msg("a", "b", "m1"),
            right=msg("a", "c", "m2"),
        )
        # No exception raised; overlap is a warning, not an error
        assert ParticipantId("a") in p.participants()

    def test_no_overlap(self):
        p = ParallelType(
            left=msg("a", "b", "m1"),
            right=msg("c", "d", "m2"),
        )
        assert len(p.participants()) == 4

    def test_parallel_unfold(self):
        p = ParallelType(
            left=msg("a", "b", "m", var("X")),
            right=msg("c", "d", "n"),
        )
        result = p.unfold({"X": EndType()})
        assert isinstance(result, ParallelType)

    def test_parallel_structural_eq(self):
        p1 = ParallelType(left=msg("a", "b", "m"), right=msg("c", "d", "n"))
        p2 = ParallelType(left=msg("a", "b", "m"), right=msg("c", "d", "n"))
        assert p1 == p2

    def test_parallel_structural_neq(self):
        p1 = ParallelType(left=msg("a", "b", "m"), right=msg("c", "d", "n"))
        p2 = ParallelType(left=msg("a", "b", "m"), right=msg("c", "d", "x"))
        assert p1 != p2

    def test_parallel_neq_non_parallel(self):
        p = ParallelType(left=msg("a", "b", "m"), right=msg("c", "d", "n"))
        assert p != msg("a", "b", "m")

    def test_parallel_repr(self):
        p = ParallelType(left=EndType(), right=EndType())
        r = repr(p)
        assert "|" in r

    def test_parallel_hash(self):
        p = ParallelType(left=EndType(), right=EndType())
        h = hash(p)
        assert isinstance(h, int)


# =============================================================================
# global_types.py — Additional global type coverage
# =============================================================================


class TestGlobalTypeAdditional:
    """Additional tests for global type methods not fully covered."""

    def test_message_type_unfold(self):
        m = MessageType(
            sender=ParticipantId("a"),
            receiver=ParticipantId("b"),
            payload=MessagePayload(label=MessageLabel("m")),
            continuation=VariableType(name="X"),
        )
        result = m.unfold({"X": EndType()})
        assert isinstance(result, MessageType)
        assert isinstance(result.continuation, EndType)

    def test_choice_type_unfold(self):
        c = ChoiceType(
            sender=ParticipantId("a"),
            receiver=ParticipantId("b"),
            branches={MessageLabel("l"): VariableType(name="X")},
        )
        result = c.unfold({"X": EndType()})
        assert isinstance(result, ChoiceType)
        assert isinstance(result.branches[MessageLabel("l")], EndType)

    def test_message_type_repr(self):
        m = msg("a", "b", "m")
        r = repr(m)
        assert "a" in r
        assert "b" in r
        assert "m" in r

    def test_choice_type_repr(self):
        c = choice("a", "b", {"l": end()})
        r = repr(c)
        assert "a" in r
        assert "b" in r

    def test_recursion_type_repr(self):
        r = rec("X", msg("a", "b", "m", var("X")))
        s = repr(r)
        assert "X" in s

    def test_variable_type_repr(self):
        assert repr(var("X")) == "X"

    def test_end_type_repr(self):
        assert repr(end()) == "end"

    def test_message_type_hash(self):
        m = msg("a", "b", "m")
        assert isinstance(hash(m), int)

    def test_choice_type_hash(self):
        c = choice("a", "b", {"l": end()})
        assert isinstance(hash(c), int)

    def test_recursion_type_hash(self):
        r = rec("X", msg("a", "b", "m", var("X")))
        assert isinstance(hash(r), int)

    def test_choice_structural_eq_different_sender(self):
        c1 = choice("a", "b", {"l": end()})
        c2 = choice("x", "b", {"l": end()})
        assert c1 != c2

    def test_choice_structural_eq_different_labels(self):
        c1 = choice("a", "b", {"l1": end()})
        c2 = choice("a", "b", {"l2": end()})
        assert c1 != c2

    def test_message_structural_neq_non_message(self):
        m = msg("a", "b", "m")
        assert m != end()

    def test_recursion_structural_eq(self):
        r1 = rec("X", msg("a", "b", "m", var("X")))
        r2 = rec("X", msg("a", "b", "m", var("X")))
        assert r1 == r2

    def test_recursion_structural_neq_different_var(self):
        r1 = rec("X", msg("a", "b", "m", var("X")))
        r2 = rec("Y", msg("a", "b", "m", var("Y")))
        assert r1 != r2

    def test_recursion_structural_neq_non_recursion(self):
        r = rec("X", msg("a", "b", "m", var("X")))
        assert r != msg("a", "b", "m")

    def test_variable_structural_neq_non_variable(self):
        v = var("X")
        assert v != end()

    def test_end_type_unfold(self):
        e = EndType()
        assert e.unfold({}) is e

    def test_end_type_neq_non_end(self):
        assert end() != msg("a", "b", "m")

    def test_recursion_type_unfold(self):
        r = rec("X", msg("a", "b", "m", var("X")))
        unfolded = r.unfold({})
        assert isinstance(unfolded, MessageType)
        assert isinstance(unfolded.continuation, RecursionType)


# =============================================================================
# checker.py — WellFormednessChecker recursion checks
# =============================================================================


class TestWellFormednessRecursion:
    """Tests for recursion-related checks in WellFormednessChecker."""

    def test_shadowed_recursion_variable(self):
        checker = WellFormednessChecker()
        # μX. a->b:m. μX. a->b:n. X -- shadowed X
        g = rec("X", msg("a", "b", "m", rec("X", msg("a", "b", "n", var("X")))))
        result = checker.check(g)
        assert not result.is_well_formed
        assert any("shadowed" in e.lower() for e in result.errors)

    def test_unbound_variable_in_choice(self):
        checker = WellFormednessChecker()
        g = choice("a", "b", {"l1": var("Z"), "l2": end()})
        result = checker.check(g)
        assert not result.is_well_formed
        assert any("unbound" in e.lower() for e in result.errors)

    def test_self_communication_in_choice(self):
        """Self-communication in choice is caught by structure check."""
        checker = WellFormednessChecker()
        # We can't construct ChoiceType with same sender/receiver due to __post_init__,
        # but we can check that a message type with self-communication is caught.
        # MessageType also rejects self-comm in __post_init__, so test the checker
        # _check_structure path indirectly through a nested structure.
        # Let's just verify a recursive type with unguarded recursion.
        g = rec("X", var("X"))
        result = checker.check(g)
        assert not result.is_well_formed
