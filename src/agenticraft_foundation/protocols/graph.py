"""Protocol graph model: G = (V, E, P, Φ, Γ).

Formal graph-theoretic representation of multi-protocol mesh topology.

The graph represents:
- V: Set of agents (vertices).
- E: Set of edges between agents.
- P: Protocol set {MCP, A2A, CUSTOM, ...}.
- Φ: E → 2ᴾ (protocols supported on each edge).
- Γ: Capability-protocol affinity function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticraft_foundation.types import ProtocolName

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Type of node in the protocol graph.

    Extended taxonomy from graph-theoretic foundations:
    - AGENT: Generic agent (backward-compatible default)
    - LLM_AGENT: LLM-powered agent with inference capabilities
    - TOOL_SERVER: Tool/resource server (MCP servers, API endpoints)
    - COORDINATOR: Orchestrator/coordinator agent managing workflows
    - GATEWAY: Protocol gateway/bridge between protocol domains
    - ROUTER: Routing node for message forwarding
    - TRANSLATOR: Protocol translator for message conversion
    """

    AGENT = "agent"  # Generic agent (backward-compatible)
    LLM_AGENT = "llm_agent"  # LLM-powered agent
    TOOL_SERVER = "tool_server"  # Tool/resource server
    COORDINATOR = "coordinator"  # Orchestrator/coordinator
    GATEWAY = "gateway"  # Protocol gateway/bridge
    ROUTER = "router"  # Routing node
    TRANSLATOR = "translator"  # Protocol translator


@dataclass
class AgentNode:
    """
    Agent node in the protocol graph.

    Represents an agent with its capabilities and supported protocols.
    """

    agent_id: str
    """Unique agent identifier"""

    capabilities: list[str] = field(default_factory=list)
    """Agent capabilities (e.g., ['code_execution', 'web_search'])"""

    protocols: set[ProtocolName] = field(default_factory=set)
    """Protocols this agent supports"""

    node_type: NodeType = NodeType.AGENT
    """Type of node"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional node metadata"""

    # Performance characteristics
    avg_latency_ms: float = 50.0
    """Average response latency in milliseconds"""

    reliability: float = 0.99
    """Node reliability score (0.0 to 1.0)"""

    def supports_protocol(self, protocol: ProtocolName) -> bool:
        """Check if agent supports a specific protocol."""
        return protocol in self.protocols

    def __hash__(self) -> int:
        return hash(self.agent_id)


@dataclass
class ProtocolEdge:
    """
    Edge with protocol capabilities: Φ(u, v).

    Represents a connection between two agents with per-protocol weights.
    """

    source: str
    """Source agent ID"""

    target: str
    """Target agent ID"""

    protocols: set[ProtocolName]
    """Φ(u,v) — protocols supported on this edge"""

    weights: dict[ProtocolName, float] = field(default_factory=dict)
    """wₑ(p) — per-protocol edge weights (cost/latency)"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional edge metadata"""

    # Connection characteristics
    bandwidth_mbps: float = 100.0
    """Estimated bandwidth in Mbps"""

    reliability: float = 0.99
    """Edge reliability score (0.0 to 1.0)"""

    def get_weight(self, protocol: ProtocolName) -> float:
        """
        Get edge weight for a specific protocol.

        Returns infinity if protocol not supported on this edge.
        """
        if protocol not in self.protocols:
            return float("inf")
        return self.weights.get(protocol, 1.0)

    def supports_protocol(self, protocol: ProtocolName) -> bool:
        """Check if edge supports a specific protocol."""
        return protocol in self.protocols

    def diversity_index(self, total_protocols: int) -> float:
        """Protocol Diversity Index: PDI(e) = |Pₑ| / |P|.

        Measures what fraction of the total protocol set is supported
        on this edge. Higher values indicate more versatile connections.

        Args:
            total_protocols: Total number of protocols in the mesh |P|

        Returns:
            PDI in [0.0, 1.0]
        """
        if total_protocols <= 0:
            return 0.0
        return len(self.protocols) / total_protocols

    def __hash__(self) -> int:
        return hash((self.source, self.target))


@dataclass
class ProtocolGraph:
    """
    Multi-protocol mesh graph G = (V, E, P, Φ, Γ).

    V: Set of agents (vertices)
    E: Set of edges between agents
    P: Protocol set {MCP, A2A, CUSTOM, ...}
    Φ: E → 2ᴾ (protocols supported on each edge)
    Γ: Capability-protocol affinity function

    This graph enables:
    - Protocol-aware path finding (Dijkstra with protocol constraints)
    - Optimal protocol selection based on capability affinity
    - Semantic-preserving route discovery
    """

    agents: dict[str, AgentNode] = field(default_factory=dict)
    """V: Set of agent nodes"""

    edges: dict[tuple[str, str], ProtocolEdge] = field(default_factory=dict)
    """E: Set of protocol edges"""

    protocols: set[ProtocolName] = field(
        default_factory=lambda: {
            ProtocolName.MCP,
            ProtocolName.A2A,
            ProtocolName.CUSTOM,
        }
    )
    """P: Supported protocol set"""

    # Graph metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_agent(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        protocols: set[ProtocolName] | None = None,
        node_type: NodeType = NodeType.AGENT,
        metadata: dict[str, Any] | None = None,
        avg_latency_ms: float = 50.0,
        reliability: float = 0.99,
    ) -> AgentNode:
        """
        Add an agent node to the graph.

        Args:
            agent_id: Unique agent identifier
            capabilities: Agent capabilities
            protocols: Supported protocols
            node_type: Type of node
            metadata: Additional metadata
            avg_latency_ms: Average latency
            reliability: Node reliability

        Returns:
            The created or updated AgentNode
        """
        node = AgentNode(
            agent_id=agent_id,
            capabilities=capabilities or [],
            protocols=protocols or set(),
            node_type=node_type,
            metadata=metadata or {},
            avg_latency_ms=avg_latency_ms,
            reliability=reliability,
        )
        self.agents[agent_id] = node
        logger.debug(f"Added agent node: {agent_id}")
        return node

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent node and all connected edges.

        Args:
            agent_id: Agent to remove

        Returns:
            True if removed, False if not found
        """
        if agent_id not in self.agents:
            return False

        # Remove all edges connected to this agent
        edges_to_remove = [key for key in self.edges if key[0] == agent_id or key[1] == agent_id]
        for key in edges_to_remove:
            del self.edges[key]

        del self.agents[agent_id]
        logger.debug(f"Removed agent node: {agent_id}")
        return True

    def add_edge(
        self,
        source: str,
        target: str,
        protocols: set[ProtocolName],
        weights: dict[ProtocolName, float] | None = None,
        metadata: dict[str, Any] | None = None,
        bidirectional: bool = True,
    ) -> ProtocolEdge:
        """
        Add a protocol edge between two agents.

        Args:
            source: Source agent ID
            target: Target agent ID
            protocols: Protocols supported on this edge
            weights: Per-protocol edge weights
            metadata: Additional metadata
            bidirectional: If True, add reverse edge as well

        Returns:
            The created ProtocolEdge

        Raises:
            ValueError: If source or target agent not in graph
        """
        if source not in self.agents:
            raise ValueError(f"Source agent not in graph: {source}")
        if target not in self.agents:
            raise ValueError(f"Target agent not in graph: {target}")

        edge = ProtocolEdge(
            source=source,
            target=target,
            protocols=protocols,
            weights=weights or dict.fromkeys(protocols, 1.0),
            metadata=metadata or {},
        )
        self.edges[(source, target)] = edge

        if bidirectional:
            reverse_edge = ProtocolEdge(
                source=target,
                target=source,
                protocols=protocols,
                weights=weights or dict.fromkeys(protocols, 1.0),
                metadata=metadata or {},
            )
            self.edges[(target, source)] = reverse_edge

        logger.debug(f"Added edge: {source} -> {target} with protocols {protocols}")
        return edge

    def remove_edge(
        self,
        source: str,
        target: str,
        bidirectional: bool = True,
    ) -> bool:
        """
        Remove an edge between two agents.

        Args:
            source: Source agent ID
            target: Target agent ID
            bidirectional: If True, remove reverse edge as well

        Returns:
            True if removed, False if not found
        """
        removed = False
        if (source, target) in self.edges:
            del self.edges[(source, target)]
            removed = True

        if bidirectional and (target, source) in self.edges:
            del self.edges[(target, source)]
            removed = True

        if removed:
            logger.debug(f"Removed edge: {source} <-> {target}")

        return removed

    def get_edge(
        self,
        source: str,
        target: str,
    ) -> ProtocolEdge | None:
        """
        Get edge between two agents.

        Args:
            source: Source agent ID
            target: Target agent ID

        Returns:
            ProtocolEdge or None if not found
        """
        return self.edges.get((source, target))

    def get_neighbors(
        self,
        agent_id: str,
        protocol: ProtocolName | None = None,
    ) -> list[str]:
        """
        Get neighbors of an agent, optionally filtered by protocol.

        Args:
            agent_id: Agent to get neighbors for
            protocol: Optional protocol filter

        Returns:
            List of neighbor agent IDs
        """
        neighbors = []
        for (src, tgt), edge in self.edges.items():
            if src == agent_id:
                if protocol is None or edge.supports_protocol(protocol):
                    neighbors.append(tgt)
        return neighbors

    def get_edges_from(
        self,
        agent_id: str,
        protocol: ProtocolName | None = None,
    ) -> list[ProtocolEdge]:
        """
        Get all outgoing edges from an agent.

        Args:
            agent_id: Source agent ID
            protocol: Optional protocol filter

        Returns:
            List of outgoing ProtocolEdges
        """
        edges = []
        for (src, _tgt), edge in self.edges.items():
            if src == agent_id:
                if protocol is None or edge.supports_protocol(protocol):
                    edges.append(edge)
        return edges

    def is_protocol_valid_path(
        self,
        path: list[str],
        protocol_sequence: list[ProtocolName],
    ) -> bool:
        """
        Check if a path is valid for a given protocol sequence.

        Validates: ∀ (u,v) ∈ π: p_i ∈ Φ(u,v)

        Args:
            path: Sequence of agent IDs
            protocol_sequence: Protocol to use on each edge

        Returns:
            True if path is valid
        """
        if len(path) < 2:
            return True

        if len(protocol_sequence) != len(path) - 1:
            return False

        for i in range(len(path) - 1):
            edge = self.get_edge(path[i], path[i + 1])
            if edge is None:
                return False
            if not edge.supports_protocol(protocol_sequence[i]):
                return False

        return True

    def get_agents_by_capability(
        self,
        capability: str,
        protocol: ProtocolName | None = None,
    ) -> list[AgentNode]:
        """
        Find agents with a specific capability.

        Args:
            capability: Capability to search for
            protocol: Optional protocol filter

        Returns:
            List of matching AgentNodes
        """
        matches = []
        for agent in self.agents.values():
            if capability in agent.capabilities:
                if protocol is None or agent.supports_protocol(protocol):
                    matches.append(agent)
        return matches

    def get_agents_by_protocol(
        self,
        protocol: ProtocolName,
    ) -> list[AgentNode]:
        """
        Find agents that support a specific protocol.

        Args:
            protocol: Protocol to filter by

        Returns:
            List of matching AgentNodes
        """
        return [agent for agent in self.agents.values() if agent.supports_protocol(protocol)]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Dictionary of statistics
        """
        protocol_counts = dict.fromkeys(self.protocols, 0)
        for edge in self.edges.values():
            for p in edge.protocols:
                if p in protocol_counts:
                    protocol_counts[p] += 1

        return {
            "num_agents": len(self.agents),
            "num_edges": len(self.edges),
            "num_protocols": len(self.protocols),
            "edges_per_protocol": protocol_counts,
            "avg_neighbors": (
                sum(len(self.get_neighbors(a)) for a in self.agents) / len(self.agents)
                if self.agents
                else 0
            ),
        }

    def protocol_diversity(self) -> dict[str, float]:
        """Compute Protocol Diversity Index for all edges.

        Returns:
            Dictionary mapping edge key "source->target" to PDI value.
        """
        total_protocols = len(self.protocols)
        result: dict[str, float] = {}
        for (src, tgt), edge in self.edges.items():
            result[f"{src}->{tgt}"] = edge.diversity_index(total_protocols)
        return result

    def is_reachable(
        self,
        source: str,
        target: str,
        allowed_protocols: set[ProtocolName] | None = None,
    ) -> bool:
        """Check reachability from source to target using allowed protocols.

        Implements Theorem 3 from the formal model: protocol-constrained
        reachability in O(|V| + |E|) via BFS on the protocol subgraph.

        Args:
            source: Source agent ID
            target: Target agent ID
            allowed_protocols: If provided, only traverse edges that support
                at least one of these protocols. If None, all edges are used.

        Returns:
            True if target is reachable from source
        """
        if source not in self.agents or target not in self.agents:
            return False
        if source == target:
            return True

        visited: set[str] = {source}
        queue: list[str] = [source]

        while queue:
            current = queue.pop(0)
            for (src, tgt), edge in self.edges.items():
                if src != current:
                    continue
                if tgt in visited:
                    continue
                if allowed_protocols is not None:
                    if not edge.protocols & allowed_protocols:
                        continue
                if tgt == target:
                    return True
                visited.add(tgt)
                queue.append(tgt)

        return False

    def reachable_agents(
        self,
        source: str,
        allowed_protocols: set[ProtocolName] | None = None,
    ) -> set[str]:
        """Find all agents reachable from source using allowed protocols.

        Args:
            source: Source agent ID
            allowed_protocols: If provided, only traverse edges that support
                at least one of these protocols. If None, all edges are used.

        Returns:
            Set of reachable agent IDs (excluding source)
        """
        if source not in self.agents:
            return set()

        visited: set[str] = {source}
        queue: list[str] = [source]

        while queue:
            current = queue.pop(0)
            for (src, tgt), edge in self.edges.items():
                if src != current:
                    continue
                if tgt in visited:
                    continue
                if allowed_protocols is not None:
                    if not edge.protocols & allowed_protocols:
                        continue
                visited.add(tgt)
                queue.append(tgt)

        visited.discard(source)
        return visited

    def clear(self) -> None:
        """Clear all agents and edges."""
        self.agents.clear()
        self.edges.clear()
        logger.debug("Cleared protocol graph")


__all__ = [
    "NodeType",
    "AgentNode",
    "ProtocolEdge",
    "ProtocolGraph",
]
