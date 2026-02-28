"""Serialization for CSP processes, LTS, and protocol graphs.

Round-trip guarantee: ``from_dict(to_dict(x)) == x`` for all supported types.

Supported types:
- All 15 CSP Process subclasses (Stop, Skip, Prefix, ExternalChoice,
  InternalChoice, Parallel, Sequential, Hiding, Recursion, Variable,
  Interrupt, Timeout, Guard*, Rename, Pipe)
- LTS, LTSState, Transition
- DeadlockAnalysis, LivenessAnalysis
- ProtocolGraph, AgentNode, ProtocolEdge

*Guard processes contain callables and cannot be fully round-tripped.
``Guard.to_dict()`` stores a placeholder; ``Guard.from_dict()`` raises
``ValueError`` unless a ``condition_registry`` is provided.
"""

from __future__ import annotations

from typing import Any

from agenticraft_foundation.algebra.csp import (
    Event,
    ExternalChoice,
    Hiding,
    InternalChoice,
    Parallel,
    Prefix,
    Process,
    Recursion,
    Sequential,
    Skip,
    Stop,
    Variable,
)
from agenticraft_foundation.algebra.operators import (
    Guard,
    Interrupt,
    Pipe,
    Rename,
    Timeout,
)
from agenticraft_foundation.algebra.semantics import (
    LTS,
    DeadlockAnalysis,
    LivenessAnalysis,
    LTSState,
    Transition,
)
from agenticraft_foundation.protocols.graph import (
    AgentNode,
    NodeType,
    ProtocolEdge,
    ProtocolGraph,
)
from agenticraft_foundation.types import ProtocolName

# ── Process serialization ──────────────────────────────────────────────


def process_to_dict(process: Process) -> dict[str, Any]:
    """Serialize a CSP process tree to a plain dict.

    Args:
        process: Any CSP Process instance.

    Returns:
        A JSON-compatible dict with a ``"type"`` discriminator.

    Raises:
        TypeError: If the process type is unknown.
    """
    if isinstance(process, Stop):
        return {"type": "Stop"}

    if isinstance(process, Skip):
        return {"type": "Skip"}

    if isinstance(process, Variable):
        return {"type": "Variable", "name": process.name}

    if isinstance(process, Prefix):
        return {
            "type": "Prefix",
            "event": str(process.event),
            "continuation": process_to_dict(process.continuation),
        }

    if isinstance(process, ExternalChoice):
        return {
            "type": "ExternalChoice",
            "left": process_to_dict(process.left),
            "right": process_to_dict(process.right),
        }

    if isinstance(process, InternalChoice):
        return {
            "type": "InternalChoice",
            "left": process_to_dict(process.left),
            "right": process_to_dict(process.right),
        }

    if isinstance(process, Parallel):
        return {
            "type": "Parallel",
            "left": process_to_dict(process.left),
            "right": process_to_dict(process.right),
            "sync_set": sorted(str(e) for e in process.sync_set),
        }

    if isinstance(process, Sequential):
        return {
            "type": "Sequential",
            "first": process_to_dict(process.first),
            "second": process_to_dict(process.second),
        }

    if isinstance(process, Hiding):
        return {
            "type": "Hiding",
            "process": process_to_dict(process.process),
            "hidden": sorted(str(e) for e in process.hidden),
        }

    if isinstance(process, Recursion):
        return {
            "type": "Recursion",
            "variable": process.variable,
            "body": process_to_dict(process.body),
        }

    # Extended operators (check before base Process since they inherit it)
    if isinstance(process, Interrupt):
        return {
            "type": "Interrupt",
            "primary": process_to_dict(process.primary),
            "handler": process_to_dict(process.handler),
        }

    if isinstance(process, Timeout):
        return {
            "type": "Timeout",
            "process": process_to_dict(process.process),
            "duration": process.duration,
            "fallback": process_to_dict(process.fallback),
        }

    if isinstance(process, Guard):
        return {
            "type": "Guard",
            "condition": "<callable>",
            "process": process_to_dict(process.process),
        }

    if isinstance(process, Rename):
        return {
            "type": "Rename",
            "process": process_to_dict(process.process),
            "mapping": [[str(k), str(v)] for k, v in process.mapping],
        }

    if isinstance(process, Pipe):
        return {
            "type": "Pipe",
            "producer": process_to_dict(process.producer),
            "consumer": process_to_dict(process.consumer),
            "channel": sorted(str(e) for e in process.channel),
        }

    raise TypeError(f"Unknown process type: {type(process).__name__}")


def process_from_dict(
    data: dict[str, Any],
    condition_registry: dict[str, Any] | None = None,
) -> Process:
    """Reconstruct a CSP process tree from a dict.

    Args:
        data: Dict previously produced by :func:`process_to_dict`.
        condition_registry: Optional mapping of condition names to callables,
            used to reconstruct Guard processes.

    Returns:
        The reconstructed Process.

    Raises:
        ValueError: If the dict contains an unknown or unrestorable type.
    """
    type_tag = data["type"]

    if type_tag == "Stop":
        return Stop()

    if type_tag == "Skip":
        return Skip()

    if type_tag == "Variable":
        return Variable(name=data["name"])

    if type_tag == "Prefix":
        return Prefix(
            event=Event(data["event"]),
            continuation=process_from_dict(data["continuation"], condition_registry),
        )

    if type_tag == "ExternalChoice":
        return ExternalChoice(
            left=process_from_dict(data["left"], condition_registry),
            right=process_from_dict(data["right"], condition_registry),
        )

    if type_tag == "InternalChoice":
        return InternalChoice(
            left=process_from_dict(data["left"], condition_registry),
            right=process_from_dict(data["right"], condition_registry),
        )

    if type_tag == "Parallel":
        return Parallel(
            left=process_from_dict(data["left"], condition_registry),
            right=process_from_dict(data["right"], condition_registry),
            sync_set=frozenset(Event(e) for e in data["sync_set"]),
        )

    if type_tag == "Sequential":
        return Sequential(
            first=process_from_dict(data["first"], condition_registry),
            second=process_from_dict(data["second"], condition_registry),
        )

    if type_tag == "Hiding":
        return Hiding(
            process=process_from_dict(data["process"], condition_registry),
            hidden=frozenset(Event(e) for e in data["hidden"]),
        )

    if type_tag == "Recursion":
        return Recursion(
            variable=data["variable"],
            body=process_from_dict(data["body"], condition_registry),
        )

    if type_tag == "Interrupt":
        return Interrupt(
            primary=process_from_dict(data["primary"], condition_registry),
            handler=process_from_dict(data["handler"], condition_registry),
        )

    if type_tag == "Timeout":
        return Timeout(
            process=process_from_dict(data["process"], condition_registry),
            duration=data["duration"],
            fallback=process_from_dict(data["fallback"], condition_registry),
        )

    if type_tag == "Guard":
        condition_key = data.get("condition", "<callable>")
        if condition_registry and condition_key in condition_registry:
            cond = condition_registry[condition_key]
        else:
            raise ValueError(
                f"Cannot deserialize Guard: condition {condition_key!r} not in "
                "condition_registry. Provide a condition_registry mapping."
            )
        return Guard(
            condition=cond,
            process=process_from_dict(data["process"], condition_registry),
        )

    if type_tag == "Rename":
        return Rename(
            process=process_from_dict(data["process"], condition_registry),
            mapping=tuple((Event(pair[0]), Event(pair[1])) for pair in data["mapping"]),
        )

    if type_tag == "Pipe":
        return Pipe(
            producer=process_from_dict(data["producer"], condition_registry),
            consumer=process_from_dict(data["consumer"], condition_registry),
            channel=frozenset(Event(e) for e in data["channel"]),
        )

    raise ValueError(f"Unknown process type tag: {type_tag!r}")


# ── LTS serialization ─────────────────────────────────────────────────


def transition_to_dict(transition: Transition) -> dict[str, Any]:
    """Serialize a Transition."""
    return {
        "source": transition.source,
        "event": str(transition.event),
        "target": transition.target,
    }


def transition_from_dict(data: dict[str, Any]) -> Transition:
    """Deserialize a Transition."""
    return Transition(
        source=data["source"],
        event=Event(data["event"]),
        target=data["target"],
    )


def lts_to_dict(lts: LTS) -> dict[str, Any]:
    """Serialize an LTS to a plain dict.

    Note: LTSState.process is serialized via :func:`process_to_dict`.
    """
    states = {}
    for state_id, state in lts.states.items():
        states[str(state_id)] = {
            "id": state.id,
            "process": process_to_dict(state.process),
            "is_terminal": state.is_terminal,
            "is_deadlock": state.is_deadlock,
        }
    return {
        "states": states,
        "transitions": [transition_to_dict(t) for t in lts.transitions],
        "initial_state": lts.initial_state,
        "alphabet": sorted(str(e) for e in lts.alphabet),
    }


def lts_from_dict(data: dict[str, Any]) -> LTS:
    """Deserialize an LTS from a dict."""
    lts = LTS(
        initial_state=data["initial_state"],
        alphabet=frozenset(Event(e) for e in data["alphabet"]),
    )
    for state_data in data["states"].values():
        lts.states[state_data["id"]] = LTSState(
            id=state_data["id"],
            process=process_from_dict(state_data["process"]),
            is_terminal=state_data["is_terminal"],
            is_deadlock=state_data["is_deadlock"],
        )
    for t_data in data["transitions"]:
        lts.transitions.append(transition_from_dict(t_data))
    return lts


# ── Analysis result serialization ──────────────────────────────────────


def deadlock_analysis_to_dict(analysis: DeadlockAnalysis) -> dict[str, Any]:
    """Serialize a DeadlockAnalysis."""
    return {
        "has_deadlock": analysis.has_deadlock,
        "deadlock_states": analysis.deadlock_states,
        "deadlock_traces": [[str(e) for e in trace] for trace in analysis.deadlock_traces],
    }


def deadlock_analysis_from_dict(data: dict[str, Any]) -> DeadlockAnalysis:
    """Deserialize a DeadlockAnalysis."""
    return DeadlockAnalysis(
        has_deadlock=data["has_deadlock"],
        deadlock_states=data["deadlock_states"],
        deadlock_traces=[tuple(Event(e) for e in trace) for trace in data["deadlock_traces"]],
    )


def liveness_analysis_to_dict(analysis: LivenessAnalysis) -> dict[str, Any]:
    """Serialize a LivenessAnalysis."""
    return {
        "is_live": analysis.is_live,
        "stuck_states": analysis.stuck_states,
        "live_events": {str(k): v for k, v in analysis.live_events.items()},
    }


def liveness_analysis_from_dict(data: dict[str, Any]) -> LivenessAnalysis:
    """Deserialize a LivenessAnalysis."""
    return LivenessAnalysis(
        is_live=data["is_live"],
        stuck_states=data["stuck_states"],
        live_events={Event(k): v for k, v in data["live_events"].items()},
    )


# ── Protocol graph serialization ──────────────────────────────────────


def agent_node_to_dict(node: AgentNode) -> dict[str, Any]:
    """Serialize an AgentNode."""
    return {
        "agent_id": node.agent_id,
        "capabilities": node.capabilities,
        "protocols": sorted(p.value for p in node.protocols),
        "node_type": node.node_type.value,
        "metadata": node.metadata,
        "avg_latency_ms": node.avg_latency_ms,
        "reliability": node.reliability,
    }


def agent_node_from_dict(data: dict[str, Any]) -> AgentNode:
    """Deserialize an AgentNode."""
    return AgentNode(
        agent_id=data["agent_id"],
        capabilities=data["capabilities"],
        protocols={ProtocolName(p) for p in data["protocols"]},
        node_type=NodeType(data["node_type"]),
        metadata=data.get("metadata", {}),
        avg_latency_ms=data.get("avg_latency_ms", 50.0),
        reliability=data.get("reliability", 0.99),
    )


def protocol_edge_to_dict(edge: ProtocolEdge) -> dict[str, Any]:
    """Serialize a ProtocolEdge."""
    return {
        "source": edge.source,
        "target": edge.target,
        "protocols": sorted(p.value for p in edge.protocols),
        "weights": {p.value: w for p, w in edge.weights.items()},
        "metadata": edge.metadata,
        "bandwidth_mbps": edge.bandwidth_mbps,
        "reliability": edge.reliability,
    }


def protocol_edge_from_dict(data: dict[str, Any]) -> ProtocolEdge:
    """Deserialize a ProtocolEdge."""
    return ProtocolEdge(
        source=data["source"],
        target=data["target"],
        protocols={ProtocolName(p) for p in data["protocols"]},
        weights={ProtocolName(k): v for k, v in data["weights"].items()},
        metadata=data.get("metadata", {}),
        bandwidth_mbps=data.get("bandwidth_mbps", 100.0),
        reliability=data.get("reliability", 0.99),
    )


def graph_to_dict(graph: ProtocolGraph) -> dict[str, Any]:
    """Serialize a ProtocolGraph."""
    edges = {}
    for (src, tgt), edge in graph.edges.items():
        edges[f"{src}->{tgt}"] = protocol_edge_to_dict(edge)
    return {
        "agents": {aid: agent_node_to_dict(node) for aid, node in graph.agents.items()},
        "edges": edges,
        "protocols": sorted(p.value for p in graph.protocols),
        "metadata": graph.metadata,
    }


def graph_from_dict(data: dict[str, Any]) -> ProtocolGraph:
    """Deserialize a ProtocolGraph."""
    graph = ProtocolGraph(
        protocols={ProtocolName(p) for p in data["protocols"]},
        metadata=data.get("metadata", {}),
    )
    for node_data in data["agents"].values():
        node = agent_node_from_dict(node_data)
        graph.agents[node.agent_id] = node
    for edge_key, edge_data in data["edges"].items():
        edge = protocol_edge_from_dict(edge_data)
        src, tgt = edge_key.split("->", 1)
        graph.edges[(src, tgt)] = edge
    return graph


__all__ = [
    # Process
    "process_to_dict",
    "process_from_dict",
    # LTS
    "transition_to_dict",
    "transition_from_dict",
    "lts_to_dict",
    "lts_from_dict",
    # Analysis
    "deadlock_analysis_to_dict",
    "deadlock_analysis_from_dict",
    "liveness_analysis_to_dict",
    "liveness_analysis_from_dict",
    # Protocol graph
    "agent_node_to_dict",
    "agent_node_from_dict",
    "protocol_edge_to_dict",
    "protocol_edge_from_dict",
    "graph_to_dict",
    "graph_from_dict",
]
