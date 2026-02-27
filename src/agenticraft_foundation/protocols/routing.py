"""Protocol-aware Dijkstra routing (Algorithm 1).

Finds optimal path considering protocol translation costs.
State space: (agent, incoming_protocol).
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticraft_foundation.types import ProtocolName

from .cost import PathCostCalculator
from .graph import ProtocolEdge

if TYPE_CHECKING:
    from .compatibility import ProtocolCompatibilityMatrix
    from .graph import ProtocolGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingState:
    """
    State for protocol-aware routing.

    State space: (agent_id, incoming_protocol)
    Each state represents being at an agent with a specific incoming protocol.
    """

    agent_id: str
    """Current agent ID"""

    protocol: ProtocolName
    """Incoming protocol at this agent"""

    def __hash__(self) -> int:
        return hash((self.agent_id, self.protocol))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RoutingState):
            return NotImplemented
        return self.agent_id == other.agent_id and self.protocol == other.protocol

    def __lt__(self, other: RoutingState) -> bool:
        """For heap comparison - compare by agent_id then protocol."""
        if self.agent_id != other.agent_id:
            return self.agent_id < other.agent_id
        return self.protocol.value < other.protocol.value


@dataclass
class OptimalRoute:
    """
    Result of protocol-aware routing.

    Contains the optimal path with protocol sequence and cost breakdown.
    """

    path: list[str]
    """Sequence of agent IDs from source to target"""

    protocol_sequence: list[ProtocolName]
    """Protocol to use on each edge (len = len(path) - 1)"""

    total_cost: float
    """Total path cost: Σ w_e(p) + Σ τ(p_i, p_{i+1}, v_i)"""

    translation_points: list[tuple[str, ProtocolName, ProtocolName]]
    """List of (node_id, from_protocol, to_protocol) where translations occur"""

    semantic_loss_estimate: float
    """Estimated semantic loss across all translations (0.0 to 1.0)"""

    # Additional metadata
    edge_costs: list[float] = field(default_factory=list)
    """Individual edge costs"""

    translation_costs: list[float] = field(default_factory=list)
    """Individual translation costs"""

    num_hops: int = 0
    """Number of hops (edges) in the path"""

    @property
    def num_translations(self) -> int:
        """Number of protocol translations (changes)."""
        return len(self.translation_points)

    @property
    def is_single_protocol(self) -> bool:
        """Check if route uses only one protocol (no translations)."""
        return len(self.translation_points) == 0

    def get_summary(self) -> dict[str, Any]:
        """Get route summary."""
        return {
            "path": self.path,
            "protocol_sequence": [p.value for p in self.protocol_sequence],
            "total_cost": self.total_cost,
            "num_hops": self.num_hops,
            "num_translations": self.num_translations,
            "semantic_loss_estimate": self.semantic_loss_estimate,
            "translation_points": [
                {"node": t[0], "from": t[1].value, "to": t[2].value}
                for t in self.translation_points
            ],
        }


@dataclass
class RoutingConfig:
    """Configuration for Protocol-Aware Dijkstra."""

    max_hops: int = 10
    """Maximum number of hops allowed"""

    max_translations: int = 3
    """Maximum number of protocol translations allowed"""

    max_semantic_loss: float = 0.3
    """Maximum allowed semantic loss (0.0 to 1.0)"""

    prefer_single_protocol: bool = True
    """Prefer routes that don't require protocol translation"""

    single_protocol_bonus: float = 0.1
    """Cost reduction for single-protocol routes"""


class ProtocolAwareDijkstra:
    """
    Protocol-Aware Dijkstra Algorithm (Algorithm 1 from formal model).

    State space: (agent, incoming_protocol)
    Transition: Consider all outgoing edges and protocol translations
    Cost: edge_weight + translation_cost

    This algorithm finds the optimal route through a multi-protocol mesh,
    considering both edge weights and protocol translation costs.

    Usage:
        dijkstra = ProtocolAwareDijkstra(graph, compatibility, cost_calculator)

        # Find optimal route
        route = dijkstra.find_optimal_route(
            source="agent1",
            target="agent3",
            source_protocol=ProtocolName.MCP,
        )

        if route:
            print(f"Path: {route.path}")
            print(f"Cost: {route.total_cost}")
            print(f"Protocols: {route.protocol_sequence}")
    """

    def __init__(
        self,
        graph: ProtocolGraph,
        compatibility_matrix: ProtocolCompatibilityMatrix,
        cost_calculator: PathCostCalculator,
        config: RoutingConfig | None = None,
    ):
        """
        Initialize Protocol-Aware Dijkstra.

        Args:
            graph: Protocol graph with agents and edges
            compatibility_matrix: Protocol compatibility information
            cost_calculator: Path cost calculator
            config: Routing configuration
        """
        self._graph = graph
        self._compatibility = compatibility_matrix
        self._cost_calculator = cost_calculator
        self._config = config or RoutingConfig()

        # Statistics
        self._stats = {
            "routes_computed": 0,
            "states_explored": 0,
            "routes_found": 0,
            "routes_not_found": 0,
        }

    def find_optimal_route(
        self,
        source: str,
        target: str,
        source_protocol: ProtocolName,
        target_protocol: ProtocolName | None = None,
        max_hops: int | None = None,
    ) -> OptimalRoute | None:
        """
        Find optimal protocol-aware route.

        Algorithm 1 from formal model:
        1. Initialize dist[source, source_protocol] = 0
        2. For each state (u, p_in) in priority queue:
           a. For each neighbor v with edge (u,v):
              b. For each protocol p_out ∈ Φ(u,v):
                 - cost = dist[u, p_in] + w(u,v, p_out) + τ(p_in, p_out, u)
                 - if cost < dist[v, p_out]: update
        3. Return optimal path to (target, target_protocol)

        Args:
            source: Source agent ID
            target: Target agent ID
            source_protocol: Protocol at source
            target_protocol: Required protocol at target (optional)
            max_hops: Maximum hops (overrides config)

        Returns:
            OptimalRoute if path exists, None otherwise
        """
        self._stats["routes_computed"] += 1

        if source not in self._graph.agents:
            logger.warning(f"Source agent not in graph: {source}")
            return None

        if target not in self._graph.agents:
            logger.warning(f"Target agent not in graph: {target}")
            return None

        # Same source and target
        if source == target:
            return self._create_trivial_route(source, source_protocol)

        max_hops = max_hops or self._config.max_hops

        # Dijkstra data structures
        # dist[state] = minimum cost to reach this state
        dist: dict[RoutingState, float] = {}
        # prev[state] = (previous_state, edge_protocol, edge_cost, translation_cost)
        prev: dict[RoutingState, tuple[RoutingState, ProtocolName, float, float] | None] = {}
        # Priority queue: (cost, state)
        pq: list[tuple[float, RoutingState]] = []

        # Initialize source state
        start_state = RoutingState(agent_id=source, protocol=source_protocol)
        dist[start_state] = 0.0
        prev[start_state] = None
        heapq.heappush(pq, (0.0, start_state))

        # Track visited for statistics
        visited: set[RoutingState] = set()

        while pq:
            current_cost, current_state = heapq.heappop(pq)

            # Skip if we've found a better path
            if current_state in visited:
                continue

            visited.add(current_state)
            self._stats["states_explored"] += 1

            # Check if we've reached the target
            if current_state.agent_id == target:
                if target_protocol is None or current_state.protocol == target_protocol:
                    self._stats["routes_found"] += 1
                    return self._reconstruct_route(current_state, dist, prev, source_protocol)

            # Check hop limit
            path_length = self._count_hops(current_state, prev)
            if path_length >= max_hops:
                continue

            # Explore neighbors
            current_agent = current_state.agent_id
            current_protocol = current_state.protocol

            for edge in self._graph.get_edges_from(current_agent):
                neighbor = edge.target

                # Try each protocol supported on this edge
                for edge_protocol in edge.protocols:
                    # Calculate translation cost (if protocols differ)
                    translation = self._cost_calculator.get_translation_cost(
                        current_protocol, edge_protocol, at_node=current_agent
                    )

                    # Skip if translation not possible
                    if translation.total_cost == float("inf"):
                        continue

                    # Check semantic loss constraint
                    cumulative_loss = self._estimate_cumulative_semantic_loss(
                        current_state, prev, translation.semantic_loss_penalty
                    )
                    if cumulative_loss > self._config.max_semantic_loss:
                        continue

                    # Calculate total cost to reach neighbor
                    edge_cost = edge.get_weight(edge_protocol)
                    new_cost = current_cost + edge_cost + translation.total_cost

                    # Create neighbor state
                    neighbor_state = RoutingState(agent_id=neighbor, protocol=edge_protocol)

                    # Update if better path found
                    if neighbor_state not in dist or new_cost < dist[neighbor_state]:
                        dist[neighbor_state] = new_cost
                        prev[neighbor_state] = (
                            current_state,
                            edge_protocol,
                            edge_cost,
                            translation.total_cost,
                        )
                        heapq.heappush(pq, (new_cost, neighbor_state))

        # Handle case where target requires specific protocol
        if target_protocol is not None:
            # Check if we can reach target with any protocol and translate
            best_route = None
            best_cost = float("inf")

            for protocol in self._graph.protocols:
                final_state = RoutingState(agent_id=target, protocol=protocol)
                if final_state in dist:
                    # Add translation cost to target protocol
                    translation = self._cost_calculator.get_translation_cost(
                        protocol, target_protocol, at_node=target
                    )
                    total = dist[final_state] + translation.total_cost
                    if total < best_cost:
                        best_cost = total
                        best_route = self._reconstruct_route(
                            final_state, dist, prev, source_protocol
                        )

            if best_route:
                self._stats["routes_found"] += 1
                return best_route

        self._stats["routes_not_found"] += 1
        logger.debug(f"No route found from {source} to {target}")
        return None

    def find_all_routes(
        self,
        source: str,
        target: str,
        source_protocol: ProtocolName,
        max_routes: int = 5,
        max_cost_factor: float = 1.5,
    ) -> list[OptimalRoute]:
        """
        Find multiple alternative routes.

        Args:
            source: Source agent ID
            target: Target agent ID
            source_protocol: Protocol at source
            max_routes: Maximum number of routes to return
            max_cost_factor: Max cost as factor of optimal (e.g., 1.5 = 50% worse)

        Returns:
            List of routes sorted by cost
        """
        routes: list[OptimalRoute] = []

        # Find optimal route first
        optimal = self.find_optimal_route(source, target, source_protocol)
        if not optimal:
            return routes

        routes.append(optimal)
        max_cost = optimal.total_cost * max_cost_factor

        # Find routes with each starting protocol
        for protocol in self._graph.protocols:
            if protocol == source_protocol:
                continue

            # Add translation from source_protocol to this protocol
            translation = self._cost_calculator.get_translation_cost(
                source_protocol, protocol, at_node=source
            )

            if translation.total_cost == float("inf"):
                continue

            route = self.find_optimal_route(source, target, protocol)
            if route and route.total_cost + translation.total_cost <= max_cost:
                # Adjust route cost to include initial translation
                route.total_cost += translation.total_cost
                if route.protocol_sequence:
                    route.translation_points.insert(0, (source, source_protocol, protocol))
                routes.append(route)

            if len(routes) >= max_routes:
                break

        # Sort by cost
        routes.sort(key=lambda r: r.total_cost)
        return routes[:max_routes]

    def _create_trivial_route(
        self,
        agent_id: str,
        protocol: ProtocolName,
    ) -> OptimalRoute:
        """Create a trivial route (source == target)."""
        return OptimalRoute(
            path=[agent_id],
            protocol_sequence=[],
            total_cost=0.0,
            translation_points=[],
            semantic_loss_estimate=0.0,
            num_hops=0,
        )

    def _count_hops(
        self,
        state: RoutingState,
        prev: dict[RoutingState, tuple[RoutingState, ProtocolName, float, float] | None],
    ) -> int:
        """Count number of hops to reach a state."""
        count = 0
        current = state
        while prev.get(current) is not None:
            prev_info = prev[current]
            if prev_info:
                current = prev_info[0]
                count += 1
        return count

    def _estimate_cumulative_semantic_loss(
        self,
        state: RoutingState,
        prev: dict[RoutingState, tuple[RoutingState, ProtocolName, float, float] | None],
        additional_loss: float,
    ) -> float:
        """Estimate cumulative semantic loss along path."""
        # Simple additive model (could be multiplicative)
        total_loss = additional_loss
        current = state
        while prev.get(current) is not None:
            prev_info = prev[current]
            if prev_info:
                prev_state, _, _, trans_cost = prev_info
                # Approximate semantic loss from translation cost
                total_loss += min(trans_cost * 0.5, 0.2)  # Cap per translation
                current = prev_state
        return min(total_loss, 1.0)

    def _reconstruct_route(
        self,
        final_state: RoutingState,
        dist: dict[RoutingState, float],
        prev: dict[RoutingState, tuple[RoutingState, ProtocolName, float, float] | None],
        source_protocol: ProtocolName,
    ) -> OptimalRoute:
        """Reconstruct the optimal route from Dijkstra results."""
        path: list[str] = []
        protocol_sequence: list[ProtocolName] = []
        edge_costs: list[float] = []
        translation_costs: list[float] = []
        translation_points: list[tuple[str, ProtocolName, ProtocolName]] = []

        # Trace back from final state
        current = final_state
        while prev.get(current) is not None:
            path.append(current.agent_id)
            prev_info = prev[current]
            if prev_info:
                prev_state, edge_protocol, edge_cost, trans_cost = prev_info
                protocol_sequence.append(edge_protocol)
                edge_costs.append(edge_cost)
                translation_costs.append(trans_cost)

                # Record translation point if protocols differ
                if prev_state.protocol != edge_protocol:
                    translation_points.append(
                        (prev_state.agent_id, prev_state.protocol, edge_protocol)
                    )

                current = prev_state

        # Add source
        path.append(current.agent_id)

        # Reverse everything (we traced backwards)
        path.reverse()
        protocol_sequence.reverse()
        edge_costs.reverse()
        translation_costs.reverse()
        translation_points.reverse()

        # Calculate semantic loss estimate
        semantic_loss = self._calculate_semantic_loss(translation_points)

        return OptimalRoute(
            path=path,
            protocol_sequence=protocol_sequence,
            total_cost=dist[final_state],
            translation_points=translation_points,
            semantic_loss_estimate=semantic_loss,
            edge_costs=edge_costs,
            translation_costs=translation_costs,
            num_hops=len(path) - 1,
        )

    def _calculate_semantic_loss(
        self,
        translation_points: list[tuple[str, ProtocolName, ProtocolName]],
    ) -> float:
        """Calculate estimated semantic loss from translation points."""
        if not translation_points:
            return 0.0

        # Compound semantic loss multiplicatively
        preservation = 1.0
        for _, from_protocol, to_protocol in translation_points:
            compat = self._compatibility.get_compatibility(from_protocol, to_protocol)
            preservation *= 1 - compat.semantic_loss

        return 1 - preservation

    def get_statistics(self) -> dict[str, Any]:
        """Get routing statistics."""
        return {
            **self._stats,
            "config": {
                "max_hops": self._config.max_hops,
                "max_translations": self._config.max_translations,
                "max_semantic_loss": self._config.max_semantic_loss,
            },
        }

    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        self._stats = {
            "routes_computed": 0,
            "states_explored": 0,
            "routes_found": 0,
            "routes_not_found": 0,
        }


class ProtocolConstrainedBFS:
    """
    Protocol-Constrained BFS (Algorithm 2 from formal model).

    Minimum-hop routing ignoring costs — returns shortest path by hop count
    with protocol compatibility checks. Uses BFS instead of priority queue.

    State space: (agent_id, incoming_protocol)

    Usage:
        bfs = ProtocolConstrainedBFS(graph, compatibility_matrix, cost_calculator)
        route = bfs.find_shortest_path("agent1", "agent3", ProtocolName.MCP)
    """

    def __init__(
        self,
        graph: ProtocolGraph,
        compatibility_matrix: ProtocolCompatibilityMatrix,
        cost_calculator: PathCostCalculator,
        config: RoutingConfig | None = None,
    ):
        """Initialize Protocol-Constrained BFS.

        Args:
            graph: Protocol graph with agents and edges
            compatibility_matrix: Protocol compatibility information
            cost_calculator: Path cost calculator (for route cost annotation)
            config: Routing configuration
        """
        self._graph = graph
        self._compatibility = compatibility_matrix
        self._cost_calculator = cost_calculator
        self._config = config or RoutingConfig()

    def find_shortest_path(
        self,
        source: str,
        target: str,
        source_protocol: ProtocolName,
        target_protocol: ProtocolName | None = None,
        max_hops: int | None = None,
    ) -> OptimalRoute | None:
        """Find shortest path by hop count with protocol constraints.

        Args:
            source: Source agent ID
            target: Target agent ID
            source_protocol: Protocol at source
            target_protocol: Required protocol at target (optional)
            max_hops: Maximum hops (overrides config)

        Returns:
            OptimalRoute with minimum hops, or None if unreachable
        """
        if source not in self._graph.agents or target not in self._graph.agents:
            return None

        if source == target:
            return OptimalRoute(
                path=[source],
                protocol_sequence=[],
                total_cost=0.0,
                translation_points=[],
                semantic_loss_estimate=0.0,
                num_hops=0,
            )

        max_hops = max_hops or self._config.max_hops

        # BFS data structures
        # Queue entries: (state, parent_state, edge_protocol)
        from collections import deque

        start_state = RoutingState(agent_id=source, protocol=source_protocol)
        queue: deque[RoutingState] = deque([start_state])
        visited: set[RoutingState] = {start_state}
        # parent: state -> (prev_state, edge_protocol)
        parent: dict[RoutingState, tuple[RoutingState, ProtocolName] | None] = {start_state: None}

        while queue:
            current_state = queue.popleft()

            # Check hop limit
            hops = self._trace_hops(current_state, parent)
            if hops >= max_hops:
                continue

            current_agent = current_state.agent_id
            current_protocol = current_state.protocol

            for edge in self._graph.get_edges_from(current_agent):
                neighbor = edge.target

                for edge_protocol in edge.protocols:
                    # Check translation feasibility
                    translation = self._cost_calculator.get_translation_cost(
                        current_protocol, edge_protocol, at_node=current_agent
                    )
                    if translation.total_cost == float("inf"):
                        continue

                    neighbor_state = RoutingState(agent_id=neighbor, protocol=edge_protocol)

                    if neighbor_state in visited:
                        continue

                    visited.add(neighbor_state)
                    parent[neighbor_state] = (current_state, edge_protocol)

                    # Check if we reached target
                    if neighbor == target:
                        if target_protocol is None or edge_protocol == target_protocol:
                            return self._reconstruct_bfs_route(neighbor_state, parent)

                    queue.append(neighbor_state)

        return None

    def _trace_hops(
        self,
        state: RoutingState,
        parent: dict[RoutingState, tuple[RoutingState, ProtocolName] | None],
    ) -> int:
        """Count hops from start to state."""
        count = 0
        current = state
        while parent.get(current) is not None:
            info = parent[current]
            if info:
                current = info[0]
                count += 1
        return count

    def _reconstruct_bfs_route(
        self,
        final_state: RoutingState,
        parent: dict[RoutingState, tuple[RoutingState, ProtocolName] | None],
    ) -> OptimalRoute:
        """Reconstruct route from BFS parent pointers."""
        path: list[str] = []
        protocol_sequence: list[ProtocolName] = []
        translation_points: list[tuple[str, ProtocolName, ProtocolName]] = []

        current = final_state
        while parent.get(current) is not None:
            path.append(current.agent_id)
            info = parent[current]
            if info:
                prev_state, edge_protocol = info
                protocol_sequence.append(edge_protocol)
                if prev_state.protocol != edge_protocol:
                    translation_points.append(
                        (prev_state.agent_id, prev_state.protocol, edge_protocol)
                    )
                current = prev_state

        path.append(current.agent_id)
        path.reverse()
        protocol_sequence.reverse()
        translation_points.reverse()

        # Calculate actual cost for the found route
        total_cost = 0.0
        for i in range(len(path) - 1):
            edge = self._graph.get_edge(path[i], path[i + 1])
            if edge:
                total_cost += edge.get_weight(protocol_sequence[i])

        # Add translation costs
        for node_id, from_p, to_p in translation_points:
            tc = self._cost_calculator.get_translation_cost(from_p, to_p, node_id)
            total_cost += tc.total_cost

        semantic_loss = self._estimate_semantic_loss(translation_points)

        return OptimalRoute(
            path=path,
            protocol_sequence=protocol_sequence,
            total_cost=total_cost,
            translation_points=translation_points,
            semantic_loss_estimate=semantic_loss,
            num_hops=len(path) - 1,
        )

    def _estimate_semantic_loss(
        self,
        translation_points: list[tuple[str, ProtocolName, ProtocolName]],
    ) -> float:
        """Estimate semantic loss from translations."""
        if not translation_points:
            return 0.0
        preservation = 1.0
        for _, from_p, to_p in translation_points:
            compat = self._compatibility.get_compatibility(from_p, to_p)
            preservation *= 1 - compat.semantic_loss
        return 1 - preservation


class ResilientRouter:
    """
    Resilient Multi-Protocol Router (Algorithm 3 from formal model).

    Three failover strategies:
    1. Same-edge protocol fallback — try alternative protocols on current edge
    2. Alternate path — reroute using different protocol
    3. Gateway insertion — route via gateway nodes

    Creates a residual graph excluding failed protocols/agents, then routes
    on the degraded topology.

    Usage:
        router = ResilientRouter(graph, compatibility_matrix, cost_calculator)
        route = router.find_resilient_route(
            "agent1", "agent3", ProtocolName.MCP,
            failed_protocols={ProtocolName.A2A},
        )
    """

    def __init__(
        self,
        graph: ProtocolGraph,
        compatibility_matrix: ProtocolCompatibilityMatrix,
        cost_calculator: PathCostCalculator,
        config: RoutingConfig | None = None,
    ):
        """Initialize Resilient Router.

        Args:
            graph: Protocol graph
            compatibility_matrix: Protocol compatibility information
            cost_calculator: Path cost calculator
            config: Routing configuration
        """
        self._graph = graph
        self._compatibility = compatibility_matrix
        self._cost_calculator = cost_calculator
        self._config = config or RoutingConfig()

    def find_resilient_route(
        self,
        source: str,
        target: str,
        source_protocol: ProtocolName,
        failed_protocols: set[ProtocolName] | None = None,
        failed_agents: set[str] | None = None,
        max_hops: int | None = None,
    ) -> OptimalRoute | None:
        """Find route avoiding failed protocols and agents.

        Strategy order:
        1. Try routing on the residual graph (excluding failures)
        2. If direct routing fails, try via gateway nodes

        Args:
            source: Source agent ID
            target: Target agent ID
            source_protocol: Protocol at source
            failed_protocols: Protocols to avoid
            failed_agents: Agents to avoid
            max_hops: Maximum hops (overrides config)

        Returns:
            OptimalRoute on degraded topology, or None if unreachable
        """
        failed_protocols = failed_protocols or set()
        failed_agents = failed_agents or set()

        if source in failed_agents or target in failed_agents:
            return None

        if source not in self._graph.agents or target not in self._graph.agents:
            return None

        if source == target:
            return OptimalRoute(
                path=[source],
                protocol_sequence=[],
                total_cost=0.0,
                translation_points=[],
                semantic_loss_estimate=0.0,
                num_hops=0,
            )

        # Build residual graph
        residual = self._build_residual_graph(failed_protocols, failed_agents)

        # Strategy 1: Route on residual graph using Dijkstra
        residual_calc = PathCostCalculator(residual, self._compatibility)
        dijkstra = ProtocolAwareDijkstra(
            residual,
            self._compatibility,
            residual_calc,
            config=self._config,
        )

        # Adjust source protocol if it's failed
        if source_protocol in failed_protocols:
            # Try alternate protocols at source
            source_agent = self._graph.agents.get(source)
            if source_agent:
                for alt_protocol in source_agent.protocols:
                    if alt_protocol not in failed_protocols:
                        route = dijkstra.find_optimal_route(
                            source,
                            target,
                            alt_protocol,
                            max_hops=max_hops,
                        )
                        if route:
                            return route
            return None

        route = dijkstra.find_optimal_route(
            source,
            target,
            source_protocol,
            max_hops=max_hops,
        )
        if route:
            return route

        # Strategy 2: Try via gateway nodes
        from .graph import NodeType

        for agent_id, agent in self._graph.agents.items():
            if agent_id in failed_agents:
                continue
            if agent.node_type in (NodeType.GATEWAY, NodeType.TRANSLATOR):
                # Try source -> gateway -> target
                route_to_gw = dijkstra.find_optimal_route(
                    source,
                    agent_id,
                    source_protocol,
                    max_hops=(max_hops or self._config.max_hops) // 2,
                )
                if route_to_gw is None:
                    continue

                # Find protocol at gateway
                gw_protocol = (
                    route_to_gw.protocol_sequence[-1]
                    if route_to_gw.protocol_sequence
                    else source_protocol
                )

                route_from_gw = dijkstra.find_optimal_route(
                    agent_id,
                    target,
                    gw_protocol,
                    max_hops=(max_hops or self._config.max_hops) // 2,
                )
                if route_from_gw is None:
                    continue

                # Combine routes
                combined = self._combine_routes(route_to_gw, route_from_gw)
                if route is None or combined.total_cost < route.total_cost:
                    route = combined

        return route

    def _build_residual_graph(
        self,
        failed_protocols: set[ProtocolName],
        failed_agents: set[str],
    ) -> ProtocolGraph:
        """Build residual graph excluding failed components."""
        from .graph import ProtocolGraph

        residual = ProtocolGraph(protocols=self._graph.protocols - failed_protocols)

        # Add non-failed agents
        for agent_id, agent in self._graph.agents.items():
            if agent_id not in failed_agents:
                residual.add_agent(
                    agent_id,
                    agent.capabilities,
                    agent.protocols - failed_protocols,
                    agent.node_type,
                    agent.metadata,
                    agent.avg_latency_ms,
                    agent.reliability,
                )

        # Add edges with surviving protocols
        for (src, tgt), edge in self._graph.edges.items():
            if src in failed_agents or tgt in failed_agents:
                continue

            surviving_protocols = edge.protocols - failed_protocols
            if not surviving_protocols:
                continue

            surviving_weights = {p: w for p, w in edge.weights.items() if p in surviving_protocols}

            # Only add if both agents exist in residual
            if src in residual.agents and tgt in residual.agents:
                residual.edges[(src, tgt)] = ProtocolEdge(
                    source=src,
                    target=tgt,
                    protocols=surviving_protocols,
                    weights=surviving_weights or dict.fromkeys(surviving_protocols, 1.0),
                    metadata=edge.metadata,
                    bandwidth_mbps=edge.bandwidth_mbps,
                    reliability=edge.reliability,
                )

        return residual

    def _combine_routes(
        self,
        route1: OptimalRoute,
        route2: OptimalRoute,
    ) -> OptimalRoute:
        """Combine two routes (route1 then route2, sharing junction node)."""
        # route2.path[0] should equal route1.path[-1]
        combined_path = route1.path + route2.path[1:]
        combined_protocols = route1.protocol_sequence + route2.protocol_sequence
        combined_translations = route1.translation_points + route2.translation_points

        # Add translation at junction if protocols differ
        if (
            route1.protocol_sequence
            and route2.protocol_sequence
            and route1.protocol_sequence[-1] != route2.protocol_sequence[0]
        ):
            combined_translations.append(
                (
                    route1.path[-1],
                    route1.protocol_sequence[-1],
                    route2.protocol_sequence[0],
                )
            )

        semantic_loss = 1.0 - (
            (1 - route1.semantic_loss_estimate) * (1 - route2.semantic_loss_estimate)
        )

        return OptimalRoute(
            path=combined_path,
            protocol_sequence=combined_protocols,
            total_cost=route1.total_cost + route2.total_cost,
            translation_points=combined_translations,
            semantic_loss_estimate=semantic_loss,
            edge_costs=route1.edge_costs + route2.edge_costs,
            translation_costs=route1.translation_costs + route2.translation_costs,
            num_hops=len(combined_path) - 1,
        )


__all__ = [
    "RoutingState",
    "OptimalRoute",
    "RoutingConfig",
    "ProtocolAwareDijkstra",
    "ProtocolConstrainedBFS",
    "ResilientRouter",
]
