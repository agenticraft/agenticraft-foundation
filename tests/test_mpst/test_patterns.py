"""Tests for MPST session patterns."""

from __future__ import annotations

import pytest

from agenticraft_foundation.mpst import (
    ChoiceType,
    ConsensusPattern,
    EndType,
    LocalEndType,
    MessageType,
    ParticipantId,
    PipelinePattern,
    Projector,
    ReceiveType,
    RecursionType,
    RequestResponsePattern,
    ScatterGatherPattern,
    SendType,
    pipeline,
    request_response,
    scatter_gather,
    two_phase_commit,
)


class TestRequestResponsePattern:
    """Tests for request-response pattern."""

    def test_basic_pattern(self):
        """Test basic request-response pattern."""
        g = request_response("client", "server")

        # Should be: Client → Server : request. Server → Client : response. end
        assert isinstance(g, MessageType)
        assert g.sender == ParticipantId("client")
        assert g.receiver == ParticipantId("server")
        assert g.payload.label == "request"

        # Continuation
        cont = g.continuation
        assert isinstance(cont, MessageType)
        assert cont.sender == ParticipantId("server")
        assert cont.receiver == ParticipantId("client")
        assert cont.payload.label == "response"

        # Final continuation is end
        assert isinstance(cont.continuation, EndType)

    def test_custom_labels(self):
        """Test request-response with custom labels."""
        g = request_response(
            "alice",
            "bob",
            request_label="query",
            response_label="answer",
        )

        assert g.payload.label == "query"
        assert g.continuation.payload.label == "answer"

    def test_repeatable_pattern(self):
        """Test repeatable request-response pattern."""
        g = request_response("client", "server", repeatable=True)

        # Should be: μX. Client → Server : request. Server → Client : response. X
        assert isinstance(g, RecursionType)
        assert g.variable == "X"

    def test_pattern_class_participants(self):
        """Test RequestResponsePattern.participants()."""
        pattern = RequestResponsePattern("alice", "bob")
        participants = pattern.participants()

        assert ParticipantId("alice") in participants
        assert ParticipantId("bob") in participants

    def test_projection_to_client(self):
        """Test projecting request-response to client."""
        g = request_response("client", "server")
        projector = Projector()
        local = projector.project(g, "client")

        # Client: Send request, Receive response, end
        assert isinstance(local, SendType)
        assert isinstance(local.continuation, ReceiveType)
        assert isinstance(local.continuation.continuation, LocalEndType)


class TestScatterGatherPattern:
    """Tests for scatter-gather pattern."""

    def test_basic_pattern(self):
        """Test basic scatter-gather pattern."""
        g = scatter_gather("coordinator", ["worker1", "worker2"])

        # Should be:
        # Coordinator → Worker1 : task.
        # Coordinator → Worker2 : task.
        # Worker1 → Coordinator : result.
        # Worker2 → Coordinator : result.
        # end
        assert isinstance(g, MessageType)
        assert g.sender == ParticipantId("coordinator")
        assert g.receiver == ParticipantId("worker1")

    def test_custom_workers(self):
        """Test scatter-gather with custom workers."""
        g = scatter_gather("master", ["node1", "node2", "node3"])

        # First message to node1
        assert g.receiver == ParticipantId("node1")

        # Count total messages (should be 6: 3 tasks + 3 results)
        count = 0
        current = g
        while isinstance(current, MessageType):
            count += 1
            current = current.continuation
        assert count == 6

    def test_pattern_class_participants(self):
        """Test ScatterGatherPattern.participants()."""
        pattern = ScatterGatherPattern(
            coordinator="coord",
            workers=["w1", "w2"],
        )
        participants = pattern.participants()

        assert ParticipantId("coord") in participants
        assert ParticipantId("w1") in participants
        assert ParticipantId("w2") in participants

    def test_empty_workers(self):
        """Test scatter-gather with no workers."""
        g = scatter_gather("coordinator", [])

        # Should return EndType
        assert isinstance(g, EndType)


class TestPipelinePattern:
    """Tests for pipeline pattern."""

    def test_basic_pattern(self):
        """Test basic pipeline pattern."""
        g = pipeline(["stage1", "stage2", "stage3"])

        # Should be:
        # Stage1 → Stage2 : data.
        # Stage2 → Stage3 : data.
        # end
        assert isinstance(g, MessageType)
        assert g.sender == ParticipantId("stage1")
        assert g.receiver == ParticipantId("stage2")

        cont = g.continuation
        assert isinstance(cont, MessageType)
        assert cont.sender == ParticipantId("stage2")
        assert cont.receiver == ParticipantId("stage3")

        assert isinstance(cont.continuation, EndType)

    def test_custom_data_label(self):
        """Test pipeline with custom data label."""
        g = pipeline(["a", "b"], data_label="payload")

        assert g.payload.label == "payload"

    def test_repeatable_pipeline(self):
        """Test repeatable pipeline pattern."""
        g = pipeline(["in", "process", "out"], repeatable=True)

        assert isinstance(g, RecursionType)
        assert g.variable == "X"

    def test_pattern_class_minimum_stages(self):
        """Test that pipeline requires at least 2 stages."""
        with pytest.raises(ValueError, match="at least 2 stages"):
            PipelinePattern(stages=["only_one"])

    def test_pattern_class_participants(self):
        """Test PipelinePattern.participants()."""
        pattern = PipelinePattern(stages=["a", "b", "c"])
        participants = pattern.participants()

        assert len(participants) == 3
        assert ParticipantId("a") in participants
        assert ParticipantId("b") in participants
        assert ParticipantId("c") in participants


class TestConsensusPattern:
    """Tests for two-phase commit consensus pattern."""

    def test_simple_voting(self):
        """Test simple voting 2PC."""
        g = two_phase_commit("coordinator", ["p1", "p2"])

        # Should have prepare phase, voting phase, commit phase
        assert isinstance(g, MessageType)

        # First messages should be prepare
        assert g.payload.label == "prepare"
        assert g.sender == ParticipantId("coordinator")

    def test_with_choice_single_participant(self):
        """Test 2PC with choice for single participant."""
        g = two_phase_commit("coord", ["p1"], with_choice=True)

        # Should have a prepare followed by choice
        assert isinstance(g, MessageType)
        assert g.payload.label == "prepare"

        # Continuation should be a choice
        assert isinstance(g.continuation, ChoiceType)

    def test_with_choice_two_participants(self):
        """Test 2PC with choice for two participants."""
        g = two_phase_commit("coord", ["p1", "p2"], with_choice=True)

        # Should create nested choices
        assert isinstance(g, MessageType)

    def test_pattern_class_participants(self):
        """Test ConsensusPattern.participants_set()."""
        pattern = ConsensusPattern(
            coordinator="leader",
            participants=["follower1", "follower2"],
        )
        participants = pattern.participants_set()

        assert ParticipantId("leader") in participants
        assert ParticipantId("follower1") in participants
        assert ParticipantId("follower2") in participants

    def test_minimum_participants(self):
        """Test that consensus requires at least 1 participant."""
        with pytest.raises(ValueError, match="at least 1 participant"):
            ConsensusPattern(coordinator="coord", participants=[])

    def test_falls_back_to_simple_for_many_participants(self):
        """Test that with_choice falls back to simple for >2 participants."""
        g = two_phase_commit(
            "coord",
            ["p1", "p2", "p3"],
            with_choice=True,
        )

        # Should fall back to simple voting (no ChoiceType)
        # Just verify it produces a valid type
        assert isinstance(g, MessageType)


class TestPatternProjection:
    """Tests for projecting patterns to local types."""

    def test_scatter_gather_coordinator_projection(self):
        """Test projecting scatter-gather to coordinator."""
        g = scatter_gather("coord", ["w1", "w2"])
        projector = Projector()
        local = projector.project(g, "coord")

        # Coordinator: Send to w1, Send to w2, Recv from w1, Recv from w2
        assert isinstance(local, SendType)

    def test_scatter_gather_worker_projection(self):
        """Test projecting scatter-gather to worker."""
        g = scatter_gather("coord", ["w1", "w2"])
        projector = Projector()
        local = projector.project(g, "w1")

        # w1: Recv from coord, Send to coord
        assert isinstance(local, ReceiveType)
        assert isinstance(local.continuation, SendType)

    def test_pipeline_middle_stage_projection(self):
        """Test projecting pipeline to middle stage."""
        g = pipeline(["s1", "s2", "s3"])
        projector = Projector()
        local = projector.project(g, "s2")

        # s2: Recv from s1, Send to s3
        assert isinstance(local, ReceiveType)
        assert local.sender == ParticipantId("s1")
        assert isinstance(local.continuation, SendType)
        assert local.continuation.receiver == ParticipantId("s3")
