"""Round-trip serialization tests for CSP processes, LTS, and protocol graphs."""

from __future__ import annotations

import json

import pytest

from agenticraft_foundation.algebra.csp import (
    Event,
    ExternalChoice,
    Hiding,
    InternalChoice,
    Parallel,
    Prefix,
    Recursion,
    Sequential,
    Skip,
    Stop,
    Variable,
)
from agenticraft_foundation.algebra.operators import (
    Interrupt,
    Pipe,
    Rename,
    Timeout,
)
from agenticraft_foundation.algebra.semantics import (
    DeadlockAnalysis,
    LivenessAnalysis,
    Transition,
    build_lts,
    detect_deadlock,
)
from agenticraft_foundation.protocols.graph import (
    AgentNode,
    NodeType,
    ProtocolEdge,
    ProtocolGraph,
)
from agenticraft_foundation.serialization import (
    agent_node_from_dict,
    agent_node_to_dict,
    deadlock_analysis_from_dict,
    deadlock_analysis_to_dict,
    graph_from_dict,
    graph_from_json,
    graph_to_dict,
    graph_to_json,
    liveness_analysis_from_dict,
    liveness_analysis_to_dict,
    lts_from_dict,
    lts_from_json,
    lts_to_dict,
    lts_to_json,
    process_from_dict,
    process_from_json,
    process_to_dict,
    process_to_json,
    protocol_edge_from_dict,
    protocol_edge_to_dict,
    transition_from_dict,
    transition_to_dict,
)
from agenticraft_foundation.types import ProtocolName

# ── Process round-trip tests ───────────────────────────────────────────


class TestProcessSerialization:
    """Round-trip tests for all Process subclasses."""

    def _roundtrip(self, process, **kwargs):
        """Serialize then deserialize and verify equality."""
        d = process_to_dict(process)
        # Verify JSON-compatible
        json.dumps(d)
        restored = process_from_dict(d, **kwargs)
        assert restored == process
        return d

    def test_stop(self):
        d = self._roundtrip(Stop())
        assert d == {"type": "Stop"}

    def test_skip(self):
        d = self._roundtrip(Skip())
        assert d == {"type": "Skip"}

    def test_variable(self):
        d = self._roundtrip(Variable(name="X"))
        assert d == {"type": "Variable", "name": "X"}

    def test_prefix(self):
        p = Prefix(event=Event("a"), continuation=Stop())
        d = self._roundtrip(p)
        assert d["type"] == "Prefix"
        assert d["event"] == "a"

    def test_prefix_nested(self):
        p = Prefix(
            event=Event("a"),
            continuation=Prefix(event=Event("b"), continuation=Skip()),
        )
        self._roundtrip(p)

    def test_external_choice(self):
        p = ExternalChoice(
            left=Prefix(event=Event("a"), continuation=Stop()),
            right=Prefix(event=Event("b"), continuation=Stop()),
        )
        self._roundtrip(p)

    def test_internal_choice(self):
        p = InternalChoice(
            left=Prefix(event=Event("x"), continuation=Stop()),
            right=Prefix(event=Event("y"), continuation=Skip()),
        )
        self._roundtrip(p)

    def test_parallel_interleaving(self):
        p = Parallel(
            left=Prefix(event=Event("a"), continuation=Stop()),
            right=Prefix(event=Event("b"), continuation=Stop()),
        )
        self._roundtrip(p)

    def test_parallel_synchronized(self):
        sync = frozenset({Event("sync")})
        p = Parallel(
            left=Prefix(event=Event("sync"), continuation=Stop()),
            right=Prefix(event=Event("sync"), continuation=Stop()),
            sync_set=sync,
        )
        d = self._roundtrip(p)
        assert d["sync_set"] == ["sync"]

    def test_sequential(self):
        p = Sequential(
            first=Prefix(event=Event("a"), continuation=Skip()),
            second=Prefix(event=Event("b"), continuation=Stop()),
        )
        self._roundtrip(p)

    def test_hiding(self):
        p = Hiding(
            process=Prefix(event=Event("a"), continuation=Stop()),
            hidden=frozenset({Event("a")}),
        )
        d = self._roundtrip(p)
        assert d["hidden"] == ["a"]

    def test_recursion(self):
        p = Recursion(
            variable="X",
            body=Prefix(event=Event("tick"), continuation=Variable(name="X")),
        )
        self._roundtrip(p)

    def test_interrupt(self):
        p = Interrupt(
            primary=Prefix(event=Event("work"), continuation=Stop()),
            handler=Prefix(event=Event("alert"), continuation=Stop()),
        )
        self._roundtrip(p)

    def test_timeout(self):
        p = Timeout(
            process=Prefix(event=Event("call"), continuation=Stop()),
            duration=30.0,
            fallback=Prefix(event=Event("cached"), continuation=Stop()),
        )
        d = self._roundtrip(p)
        assert d["duration"] == 30.0

    def test_rename(self):
        p = Rename(
            process=Prefix(event=Event("a"), continuation=Stop()),
            mapping=((Event("a"), Event("b")),),
        )
        d = self._roundtrip(p)
        assert d["mapping"] == [["a", "b"]]

    def test_pipe(self):
        p = Pipe(
            producer=Prefix(event=Event("emit"), continuation=Stop()),
            consumer=Prefix(event=Event("emit"), continuation=Stop()),
            channel=frozenset({Event("emit")}),
        )
        d = self._roundtrip(p)
        assert d["channel"] == ["emit"]

    def test_guard_serialization_raises_without_registry(self):
        from agenticraft_foundation.algebra.operators import Guard

        p = Guard(condition=lambda: True, process=Stop())
        d = process_to_dict(p)
        assert d["type"] == "Guard"
        assert d["condition"] == "<callable>"
        with pytest.raises(ValueError, match="condition_registry"):
            process_from_dict(d)

    def test_guard_with_condition_registry(self):
        from agenticraft_foundation.algebra.operators import Guard

        cond = lambda: True  # noqa: E731
        p = Guard(condition=cond, process=Stop())
        d = process_to_dict(p)
        restored = process_from_dict(d, condition_registry={"<callable>": cond})
        assert isinstance(restored, Guard)
        assert restored.process == Stop()

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown process type"):
            process_from_dict({"type": "Nonexistent"})

    def test_complex_nested_tree(self):
        """Test a deeply nested process tree."""
        p = Parallel(
            left=Sequential(
                first=ExternalChoice(
                    left=Prefix(event=Event("a"), continuation=Stop()),
                    right=Prefix(event=Event("b"), continuation=Skip()),
                ),
                second=Prefix(event=Event("c"), continuation=Stop()),
            ),
            right=Hiding(
                process=Prefix(
                    event=Event("d"),
                    continuation=Prefix(event=Event("e"), continuation=Stop()),
                ),
                hidden=frozenset({Event("d")}),
            ),
            sync_set=frozenset({Event("c")}),
        )
        self._roundtrip(p)

    def test_json_roundtrip(self):
        """Verify full JSON string round-trip."""
        p = Prefix(event=Event("hello"), continuation=Skip())
        d = process_to_dict(p)
        json_str = json.dumps(d)
        restored = process_from_dict(json.loads(json_str))
        assert restored == p


# ── LTS round-trip tests ──────────────────────────────────────────────


class TestLTSSerialization:
    """Round-trip tests for LTS types."""

    def test_transition(self):
        t = Transition(source=0, event=Event("a"), target=1)
        d = transition_to_dict(t)
        restored = transition_from_dict(d)
        assert restored == t

    def test_simple_lts(self):
        """Build an LTS from a simple process and round-trip it."""
        process = Prefix(event=Event("a"), continuation=Stop())
        lts = build_lts(process)

        d = lts_to_dict(lts)
        json.dumps(d)  # verify JSON-compatible
        restored = lts_from_dict(d)

        assert restored.initial_state == lts.initial_state
        assert restored.num_states == lts.num_states
        assert restored.num_transitions == lts.num_transitions
        assert restored.alphabet == lts.alphabet

    def test_choice_lts(self):
        """LTS from external choice."""
        process = ExternalChoice(
            left=Prefix(event=Event("a"), continuation=Stop()),
            right=Prefix(event=Event("b"), continuation=Stop()),
        )
        lts = build_lts(process)
        d = lts_to_dict(lts)
        restored = lts_from_dict(d)
        assert restored.num_states == lts.num_states
        assert restored.num_transitions == lts.num_transitions

    def test_lts_state_flags_preserved(self):
        """Verify terminal and deadlock flags survive round-trip."""
        process = Sequential(
            first=Prefix(event=Event("a"), continuation=Skip()),
            second=Stop(),
        )
        lts = build_lts(process)
        d = lts_to_dict(lts)
        restored = lts_from_dict(d)

        for sid, state in lts.states.items():
            restored_state = restored.states[sid]
            assert restored_state.is_terminal == state.is_terminal
            assert restored_state.is_deadlock == state.is_deadlock


# ── Analysis result tests ─────────────────────────────────────────────


class TestAnalysisSerialization:
    """Round-trip tests for analysis results."""

    def test_deadlock_analysis_no_deadlock(self):
        analysis = DeadlockAnalysis(
            has_deadlock=False,
            deadlock_states=[],
            deadlock_traces=[],
        )
        d = deadlock_analysis_to_dict(analysis)
        restored = deadlock_analysis_from_dict(d)
        assert restored.has_deadlock == analysis.has_deadlock
        assert restored.deadlock_states == analysis.deadlock_states

    def test_deadlock_analysis_with_deadlock(self):
        process = ExternalChoice(
            left=Prefix(event=Event("a"), continuation=Stop()),
            right=Prefix(event=Event("b"), continuation=Stop()),
        )
        lts = build_lts(process)
        analysis = detect_deadlock(lts)

        d = deadlock_analysis_to_dict(analysis)
        json.dumps(d)
        restored = deadlock_analysis_from_dict(d)
        assert restored.has_deadlock == analysis.has_deadlock
        assert restored.deadlock_states == analysis.deadlock_states
        assert len(restored.deadlock_traces) == len(analysis.deadlock_traces)

    def test_liveness_analysis(self):
        analysis = LivenessAnalysis(
            is_live=True,
            stuck_states=[],
            live_events={Event("a"): True, Event("b"): False},
        )
        d = liveness_analysis_to_dict(analysis)
        json.dumps(d)
        restored = liveness_analysis_from_dict(d)
        assert restored.is_live == analysis.is_live
        assert restored.stuck_states == analysis.stuck_states
        assert restored.live_events == analysis.live_events


# ── Protocol graph round-trip tests ───────────────────────────────────


class TestProtocolGraphSerialization:
    """Round-trip tests for protocol graph types."""

    def test_agent_node(self):
        node = AgentNode(
            agent_id="agent-1",
            capabilities=["search", "code"],
            protocols={ProtocolName.MCP, ProtocolName.A2A},
            node_type=NodeType.LLM_AGENT,
            metadata={"model": "gpt-4o"},
            avg_latency_ms=100.0,
            reliability=0.95,
        )
        d = agent_node_to_dict(node)
        json.dumps(d)
        restored = agent_node_from_dict(d)
        assert restored.agent_id == node.agent_id
        assert restored.capabilities == node.capabilities
        assert restored.protocols == node.protocols
        assert restored.node_type == node.node_type
        assert restored.metadata == node.metadata
        assert restored.avg_latency_ms == node.avg_latency_ms
        assert restored.reliability == node.reliability

    def test_protocol_edge(self):
        edge = ProtocolEdge(
            source="a",
            target="b",
            protocols={ProtocolName.MCP},
            weights={ProtocolName.MCP: 1.5},
            metadata={"encrypted": True},
            bandwidth_mbps=200.0,
            reliability=0.999,
        )
        d = protocol_edge_to_dict(edge)
        json.dumps(d)
        restored = protocol_edge_from_dict(d)
        assert restored.source == edge.source
        assert restored.target == edge.target
        assert restored.protocols == edge.protocols
        assert restored.weights == edge.weights
        assert restored.bandwidth_mbps == edge.bandwidth_mbps

    def test_protocol_graph(self):
        graph = ProtocolGraph()
        graph.add_agent(
            "agent-1",
            capabilities=["search"],
            protocols={ProtocolName.MCP},
            node_type=NodeType.LLM_AGENT,
        )
        graph.add_agent(
            "agent-2",
            capabilities=["code"],
            protocols={ProtocolName.A2A},
            node_type=NodeType.TOOL_SERVER,
        )
        graph.add_edge(
            "agent-1",
            "agent-2",
            protocols={ProtocolName.MCP},
            weights={ProtocolName.MCP: 2.0},
        )

        d = graph_to_dict(graph)
        json_str = json.dumps(d, indent=2)
        restored = graph_from_dict(json.loads(json_str))

        assert len(restored.agents) == len(graph.agents)
        assert len(restored.edges) == len(graph.edges)
        assert restored.protocols == graph.protocols
        for aid in graph.agents:
            assert restored.agents[aid].agent_id == graph.agents[aid].agent_id
            assert restored.agents[aid].capabilities == graph.agents[aid].capabilities

    def test_empty_graph(self):
        graph = ProtocolGraph(
            protocols={ProtocolName.CUSTOM},
            metadata={"version": "1.0"},
        )
        d = graph_to_dict(graph)
        restored = graph_from_dict(d)
        assert len(restored.agents) == 0
        assert len(restored.edges) == 0
        assert restored.protocols == {ProtocolName.CUSTOM}
        assert restored.metadata == {"version": "1.0"}


# ── JSON convenience wrapper tests ───────────────────────────────────


class TestJSONWrappers:
    """Round-trip tests for to_json/from_json convenience wrappers."""

    def test_process_to_json_returns_string(self):
        p = Prefix(event=Event("a"), continuation=Stop())
        result = process_to_json(p)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["type"] == "Prefix"
        assert parsed["event"] == "a"

    def test_process_json_roundtrip(self):
        p = Parallel(
            left=Prefix(event=Event("a"), continuation=Stop()),
            right=Prefix(event=Event("b"), continuation=Skip()),
            sync_set=frozenset({Event("a")}),
        )
        json_str = process_to_json(p)
        restored = process_from_json(json_str)
        assert restored == p

    def test_process_to_json_with_indent(self):
        p = Prefix(event=Event("x"), continuation=Stop())
        result = process_to_json(p, indent=2)
        assert "\n" in result
        assert "  " in result

    def test_process_json_nested_tree(self):
        p = Sequential(
            first=ExternalChoice(
                left=Prefix(event=Event("a"), continuation=Stop()),
                right=Prefix(event=Event("b"), continuation=Skip()),
            ),
            second=Recursion(
                variable="X",
                body=Prefix(event=Event("tick"), continuation=Variable(name="X")),
            ),
        )
        restored = process_from_json(process_to_json(p))
        assert restored == p

    def test_process_json_guard_with_registry(self):
        from agenticraft_foundation.algebra.operators import Guard

        cond = lambda: True  # noqa: E731
        p = Guard(condition=cond, process=Stop())
        json_str = process_to_json(p)
        restored = process_from_json(json_str, condition_registry={"<callable>": cond})
        assert isinstance(restored, Guard)

    def test_process_json_guard_without_registry_raises(self):
        from agenticraft_foundation.algebra.operators import Guard

        p = Guard(condition=lambda: True, process=Stop())
        json_str = process_to_json(p)
        with pytest.raises(ValueError, match="condition_registry"):
            process_from_json(json_str)

    def test_lts_json_roundtrip(self):
        process = Prefix(event=Event("a"), continuation=Stop())
        lts = build_lts(process)
        json_str = lts_to_json(lts)
        assert isinstance(json_str, str)
        restored = lts_from_json(json_str)
        assert restored.initial_state == lts.initial_state
        assert restored.num_states == lts.num_states
        assert restored.num_transitions == lts.num_transitions
        assert restored.alphabet == lts.alphabet

    def test_lts_json_with_indent(self):
        process = ExternalChoice(
            left=Prefix(event=Event("a"), continuation=Stop()),
            right=Prefix(event=Event("b"), continuation=Stop()),
        )
        lts = build_lts(process)
        result = lts_to_json(lts, indent=2)
        assert "\n" in result

    def test_graph_json_roundtrip(self):
        graph = ProtocolGraph()
        graph.add_agent(
            "agent-1",
            capabilities=["search"],
            protocols={ProtocolName.MCP},
            node_type=NodeType.LLM_AGENT,
        )
        graph.add_agent(
            "agent-2",
            capabilities=["code"],
            protocols={ProtocolName.A2A},
            node_type=NodeType.TOOL_SERVER,
        )
        graph.add_edge(
            "agent-1",
            "agent-2",
            protocols={ProtocolName.MCP},
            weights={ProtocolName.MCP: 2.0},
        )
        json_str = graph_to_json(graph)
        assert isinstance(json_str, str)
        restored = graph_from_json(json_str)
        assert len(restored.agents) == 2
        assert len(restored.edges) == len(graph.edges)
        assert restored.protocols == graph.protocols

    def test_graph_json_with_indent(self):
        graph = ProtocolGraph(protocols={ProtocolName.CUSTOM})
        result = graph_to_json(graph, indent=4)
        assert "\n" in result
        assert "    " in result
