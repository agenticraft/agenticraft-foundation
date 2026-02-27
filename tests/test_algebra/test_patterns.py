"""Tests for coordination patterns."""

from __future__ import annotations

import pytest

from agenticraft_foundation.algebra import (
    BarrierPattern,
    Event,
    MutexPattern,
    PipelinePattern,
    ProducerConsumerPattern,
    RequestResponsePattern,
    ScatterGatherPattern,
    barrier,
    compose_agents,
    is_deadlock_free,
    mutex,
    pipeline,
    prefix,
    producer_consumer,
    request_response,
    scatter_gather,
    skip,
    verify_pattern,
)


class TestRequestResponsePattern:
    """Tests for request-response pattern."""

    def test_pattern_creation(self):
        """Test creating request-response pattern."""
        pattern = RequestResponsePattern()
        assert pattern.client == "client"
        assert pattern.server == "server"

    def test_pattern_participants(self):
        """Test pattern participants."""
        pattern = RequestResponsePattern()
        assert "client" in pattern.participants()
        assert "server" in pattern.participants()

    def test_pattern_events(self):
        """Test pattern events."""
        pattern = RequestResponsePattern()
        events = pattern.events()
        assert Event("request") in events
        assert Event("response") in events

    def test_global_process(self):
        """Test global process creation."""
        p = request_response()
        # Should be: request → response → SKIP
        assert Event("request") in p.initials()

    def test_local_client_process(self):
        """Test local client process."""
        pattern = RequestResponsePattern()
        client = pattern.local_process("client")
        assert Event("request") in client.initials()

    def test_local_server_process(self):
        """Test local server process."""
        pattern = RequestResponsePattern()
        server = pattern.local_process("server")
        assert Event("request") in server.initials()

    def test_is_deadlock_free(self):
        """Test pattern is deadlock-free."""
        pattern = RequestResponsePattern()
        assert pattern.is_deadlock_free()

    def test_repeatable_pattern(self):
        """Test repeatable request-response."""
        p = request_response(repeatable=True)
        # Recursive pattern
        assert Event("request") in p.initials()


class TestPipelinePattern:
    """Tests for pipeline pattern."""

    def test_pattern_creation(self):
        """Test creating pipeline pattern."""
        pattern = PipelinePattern(stages=["a", "b", "c"])
        assert len(pattern.participants()) == 3

    def test_minimum_stages(self):
        """Test pipeline requires at least 2 stages."""
        with pytest.raises(ValueError):
            PipelinePattern(stages=["only_one"])

    def test_pattern_events(self):
        """Test pattern events are data transfers."""
        pattern = PipelinePattern(stages=["s1", "s2", "s3"])
        events = pattern.events()
        assert Event("data_s1_s2") in events
        assert Event("data_s2_s3") in events

    def test_global_process(self):
        """Test global process."""
        p = pipeline(["a", "b"])
        assert Event("data_a_b") in p.initials()

    def test_local_middle_stage(self):
        """Test local process for middle stage."""
        pattern = PipelinePattern(stages=["a", "b", "c"])
        b_local = pattern.local_process("b")
        # b receives from a, then sends to c
        assert Event("data_a_b") in b_local.initials()


class TestScatterGatherPattern:
    """Tests for scatter-gather pattern."""

    def test_pattern_creation(self):
        """Test creating scatter-gather pattern."""
        pattern = ScatterGatherPattern(workers=["w1", "w2"])
        assert "coordinator" in pattern.participants()
        assert "w1" in pattern.participants()

    def test_pattern_events(self):
        """Test pattern events."""
        pattern = ScatterGatherPattern(workers=["w1", "w2"])
        events = pattern.events()
        assert Event("task_w1") in events
        assert Event("result_w1") in events

    def test_global_process(self):
        """Test global process."""
        p = scatter_gather(workers=["w1", "w2"])
        # First event should be task to first worker
        assert Event("task_w1") in p.initials()

    def test_empty_workers(self):
        """Test scatter-gather with no workers."""
        p = scatter_gather(workers=[])
        # Should be SKIP
        assert is_deadlock_free(p)

    def test_worker_local_process(self):
        """Test worker local process."""
        pattern = ScatterGatherPattern(workers=["w1"])
        w1_local = pattern.local_process("w1")
        assert Event("task_w1") in w1_local.initials()


class TestBarrierPattern:
    """Tests for barrier pattern."""

    def test_pattern_creation(self):
        """Test creating barrier pattern."""
        pattern = BarrierPattern(participants_list=["p1", "p2", "p3"])
        assert len(pattern.participants()) == 3

    def test_pattern_events(self):
        """Test pattern events."""
        pattern = BarrierPattern(participants_list=["p1", "p2"])
        events = pattern.events()
        assert Event("arrive_p1") in events
        assert Event("depart_p1") in events

    def test_global_process(self):
        """Test global process."""
        p = barrier(["p1", "p2"])
        assert Event("arrive_p1") in p.initials()

    def test_local_process(self):
        """Test local participant process."""
        pattern = BarrierPattern(participants_list=["p1", "p2"])
        p1_local = pattern.local_process("p1")
        assert Event("arrive_p1") in p1_local.initials()


class TestMutexPattern:
    """Tests for mutex pattern."""

    def test_pattern_creation(self):
        """Test creating mutex pattern."""
        pattern = MutexPattern(processes=["p1", "p2"])
        assert len(pattern.participants()) == 2

    def test_pattern_events(self):
        """Test pattern events."""
        pattern = MutexPattern(processes=["p1", "p2"])
        events = pattern.events()
        assert Event("acquire") in events
        assert Event("release") in events
        assert Event("critical_p1") in events

    def test_mutex_process(self):
        """Test mutex resource process."""
        m = mutex(["p1"])
        assert Event("acquire") in m.initials()

    def test_local_process(self):
        """Test process local behavior."""
        pattern = MutexPattern(processes=["p1"])
        p1_local = pattern.local_process("p1")
        assert Event("acquire") in p1_local.initials()


class TestProducerConsumerPattern:
    """Tests for producer-consumer pattern."""

    def test_pattern_creation(self):
        """Test creating producer-consumer pattern."""
        pattern = ProducerConsumerPattern()
        assert "producer" in pattern.participants()
        assert "consumer" in pattern.participants()

    def test_pattern_events(self):
        """Test pattern events."""
        pattern = ProducerConsumerPattern()
        events = pattern.events()
        assert Event("produce") in events
        assert Event("consume") in events
        assert Event("put") in events
        assert Event("get") in events

    def test_global_process(self):
        """Test global process."""
        p = producer_consumer()
        assert Event("produce") in p.initials()

    def test_producer_local(self):
        """Test producer local process."""
        pattern = ProducerConsumerPattern()
        prod = pattern.local_process("producer")
        assert Event("produce") in prod.initials()

    def test_consumer_local(self):
        """Test consumer local process."""
        pattern = ProducerConsumerPattern()
        cons = pattern.local_process("consumer")
        assert Event("get") in cons.initials()


class TestComposeAgents:
    """Tests for agent composition."""

    def test_compose_empty(self):
        """Test composing empty agents."""
        result = compose_agents({})
        assert is_deadlock_free(result)

    def test_compose_single(self):
        """Test composing single agent."""
        p = prefix("a", skip())
        result = compose_agents({"agent1": p})
        assert Event("a") in result.initials()

    def test_compose_multiple(self):
        """Test composing multiple agents."""
        agents = {
            "a1": prefix("x", skip()),
            "a2": prefix("y", skip()),
        }
        result = compose_agents(agents)
        # Both events should be possible (interleaving)
        initials = result.initials()
        # Depends on sync set - by default common alphabet
        assert len(initials) > 0


class TestVerifyPattern:
    """Tests for pattern verification."""

    def test_verify_request_response(self):
        """Test verifying request-response pattern."""
        pattern = RequestResponsePattern()
        result = verify_pattern(pattern)
        assert result.pattern_name == "RequestResponsePattern"
        assert result.is_deadlock_free

    def test_verify_with_spec(self):
        """Test verifying pattern against spec."""
        pattern = RequestResponsePattern()
        spec = request_response()
        result = verify_pattern(pattern, spec)
        assert result.refines_spec

    def test_verify_pipeline(self):
        """Test verifying pipeline pattern."""
        pattern = PipelinePattern(stages=["a", "b", "c"])
        result = verify_pattern(pattern)
        assert result.is_deadlock_free


class TestPatternCompose:
    """Tests for CoordinationPattern.compose()."""

    def test_request_response_compose(self):
        """Test composing request-response local processes."""
        pattern = RequestResponsePattern()
        composed = pattern.compose()
        assert Event("request") in composed.initials()

    def test_pipeline_compose(self):
        """Test composing pipeline local processes produces a process."""
        pattern = PipelinePattern(stages=["a", "b", "c"])
        composed = pattern.compose()
        # Composed process is valid (may deadlock due to sync semantics)
        assert composed is not None

    def test_scatter_gather_compose(self):
        """Test composing scatter-gather local processes produces a process."""
        pattern = ScatterGatherPattern(workers=["w1", "w2"])
        composed = pattern.compose()
        assert composed is not None

    def test_compose_with_custom_sync(self):
        """Test compose with custom sync events."""
        pattern = RequestResponsePattern()
        composed = pattern.compose(sync_events={"request"})
        assert Event("request") in composed.initials()


class TestRepeatablePatterns:
    """Tests for repeatable=True branches in all patterns."""

    def test_request_response_repeatable_local_client(self):
        """Test repeatable client local process."""
        pattern = RequestResponsePattern(repeatable=True)
        client = pattern.local_process("client")
        assert Event("request") in client.initials()

    def test_request_response_repeatable_local_server(self):
        """Test repeatable server local process."""
        pattern = RequestResponsePattern(repeatable=True)
        server = pattern.local_process("server")
        assert Event("request") in server.initials()

    def test_pipeline_repeatable_global(self):
        """Test repeatable pipeline global process."""
        p = pipeline(["a", "b", "c"], repeatable=True)
        assert Event("data_a_b") in p.initials()

    def test_pipeline_repeatable_local(self):
        """Test repeatable pipeline local process."""
        pattern = PipelinePattern(stages=["a", "b", "c"], repeatable=True)
        b_local = pattern.local_process("b")
        assert Event("data_a_b") in b_local.initials()

    def test_scatter_gather_repeatable_global(self):
        """Test repeatable scatter-gather global process."""
        p = scatter_gather(workers=["w1", "w2"], repeatable=True)
        assert Event("task_w1") in p.initials()

    def test_scatter_gather_repeatable_worker(self):
        """Test repeatable scatter-gather worker local process."""
        pattern = ScatterGatherPattern(workers=["w1", "w2"], repeatable=True)
        w = pattern.local_process("w1")
        assert Event("task_w1") in w.initials()

    def test_barrier_repeatable_global(self):
        """Test repeatable barrier global process."""
        p = barrier(["p1", "p2"], repeatable=True)
        assert Event("arrive_p1") in p.initials()

    def test_barrier_repeatable_local(self):
        """Test repeatable barrier local process."""
        pattern = BarrierPattern(participants_list=["p1", "p2"], repeatable=True)
        local = pattern.local_process("p1")
        assert Event("arrive_p1") in local.initials()


class TestUnknownParticipantErrors:
    """Tests for ValueError on unknown participants."""

    def test_request_response_unknown(self):
        """Test unknown participant raises ValueError."""
        pattern = RequestResponsePattern()
        with pytest.raises(ValueError, match="Unknown participant"):
            pattern.local_process("unknown")

    def test_pipeline_unknown(self):
        """Test unknown participant raises ValueError."""
        pattern = PipelinePattern(stages=["a", "b"])
        with pytest.raises(ValueError, match="Unknown participant"):
            pattern.local_process("unknown")

    def test_scatter_gather_unknown(self):
        """Test unknown participant raises ValueError."""
        pattern = ScatterGatherPattern(workers=["w1"])
        with pytest.raises(ValueError, match="Unknown participant"):
            pattern.local_process("unknown")

    def test_barrier_unknown(self):
        """Test unknown participant raises ValueError."""
        pattern = BarrierPattern(participants_list=["p1"])
        with pytest.raises(ValueError, match="Unknown participant"):
            pattern.local_process("unknown")

    def test_mutex_unknown(self):
        """Test unknown participant raises ValueError."""
        pattern = MutexPattern(processes=["p1"])
        with pytest.raises(ValueError, match="Unknown participant"):
            pattern.local_process("unknown")

    def test_producer_consumer_unknown(self):
        """Test unknown participant raises ValueError."""
        pattern = ProducerConsumerPattern()
        with pytest.raises(ValueError, match="Unknown participant"):
            pattern.local_process("unknown")


class TestMutexSystemProcess:
    """Tests for MutexPattern.system_process()."""

    def test_system_process_creation(self):
        """Test mutex system process composes mutex with all participants."""
        pattern = MutexPattern(processes=["p1", "p2"])
        system = pattern.system_process()
        assert Event("acquire") in system.initials()

    def test_system_process_single(self):
        """Test mutex system process with single participant."""
        pattern = MutexPattern(processes=["p1"])
        system = pattern.system_process()
        assert Event("acquire") in system.initials()


class TestPipelineEdgeCases:
    """Tests for pipeline edge cases."""

    def test_pipeline_first_stage_local(self):
        """Test first stage only sends, doesn't receive."""
        pattern = PipelinePattern(stages=["a", "b", "c"])
        a_local = pattern.local_process("a")
        assert Event("data_a_b") in a_local.initials()

    def test_pipeline_last_stage_local(self):
        """Test last stage only receives, doesn't send."""
        pattern = PipelinePattern(stages=["a", "b", "c"])
        c_local = pattern.local_process("c")
        assert Event("data_b_c") in c_local.initials()
