"""Classical Multi-Agent Systems (MAS) formal mappings.

Maps classical MAS theories to mesh coordination primitives:

1. **BDI** (Belief-Desire-Intention):
   Beliefs → context state, Desires → task objectives, Intentions → active assignments

2. **Joint Intentions** (Cohen & Levesque):
   Mutual belief → consensus state, Persistent goal → task completion

3. **SharedPlans** (Grosz & Kraus):
   Recipe → task decomposition DAG, Subgroup plans → agent clusters

4. **Contract Net Protocol** (Smith 1980):
   Manager broadcasts CFP → bidders respond → manager awards

Each mapping provides bidirectional transformations between classical
MAS concepts and mesh coordination state, with property preservation
verification.

References:
- Rao & Georgeff (1995) - BDI agents
- Cohen & Levesque (1990) - Intention is choice with commitment
- Grosz & Kraus (1996) - Collaborative plans
- Smith (1980) - Contract Net Protocol
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MASTheory(str, Enum):
    """Classical MAS theories."""

    BDI = "bdi"
    """Belief-Desire-Intention architecture"""

    JOINT_INTENTIONS = "joint_intentions"
    """Cohen & Levesque's Joint Intentions"""

    SHARED_PLANS = "shared_plans"
    """Grosz & Kraus's SharedPlans"""

    CONTRACT_NET = "contract_net"
    """Smith's Contract Net Protocol"""


@dataclass
class MeshState:
    """Representation of mesh coordination state.

    Common structure that all MAS mappings convert to/from.
    """

    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Agent states: agent_id → {capabilities, assignments, status, ...}"""

    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Task states: task_id → {objective, status, assigned_agent, ...}"""

    consensus: dict[str, Any] = field(default_factory=dict)
    """Consensus state: {round, decisions, proposed_values, ...}"""

    messages: list[dict[str, Any]] = field(default_factory=list)
    """Message history"""

    metadata: dict[str, Any] = field(default_factory=dict)


# ── BDI Mapping ──────────────────────────────────────────────────────


@dataclass
class BDIState:
    """BDI agent mental state."""

    beliefs: dict[str, Any] = field(default_factory=dict)
    """Agent's beliefs about the world"""

    desires: list[dict[str, Any]] = field(default_factory=list)
    """Agent's desires (goals)"""

    intentions: list[dict[str, Any]] = field(default_factory=list)
    """Agent's committed intentions (active plans)"""

    agent_id: str = ""
    """Owning agent"""


class BDIMapping:
    """Map BDI mental state to/from mesh coordination state.

    - Beliefs → agent context state (world model)
    - Desires → task objectives (unassigned goals)
    - Intentions → active task assignments (committed plans)
    """

    @staticmethod
    def to_mesh(bdi_states: dict[str, BDIState]) -> MeshState:
        """Convert BDI states to mesh state.

        Args:
            bdi_states: BDI states keyed by agent_id

        Returns:
            MeshState representation
        """
        mesh = MeshState()

        for agent_id, bdi in bdi_states.items():
            mesh.agents[agent_id] = {
                "context": bdi.beliefs,
                "status": "active",
                "capabilities": list(bdi.beliefs.get("capabilities", [])),
            }

            # Desires → tasks (unassigned)
            for desire in bdi.desires:
                task_id = desire.get("id", f"{agent_id}_desire_{id(desire)}")
                if task_id not in mesh.tasks:
                    mesh.tasks[task_id] = {
                        "objective": desire.get("goal", ""),
                        "status": "pending",
                        "desired_by": agent_id,
                        "priority": desire.get("priority", 0),
                    }

            # Intentions → active assignments
            for intention in bdi.intentions:
                task_id = intention.get("task_id", f"{agent_id}_int_{id(intention)}")
                mesh.tasks[task_id] = {
                    "objective": intention.get("goal", ""),
                    "status": "in_progress",
                    "assigned_agent": agent_id,
                    "plan": intention.get("plan", []),
                }

        return mesh

    @staticmethod
    def from_mesh(mesh: MeshState) -> dict[str, BDIState]:
        """Convert mesh state to BDI states.

        Args:
            mesh: Mesh coordination state

        Returns:
            BDI states keyed by agent_id
        """
        bdi_states: dict[str, BDIState] = {}

        for agent_id, agent_data in mesh.agents.items():
            bdi = BDIState(agent_id=agent_id)

            # Context → beliefs
            bdi.beliefs = dict(agent_data.get("context", {}))
            bdi.beliefs["capabilities"] = agent_data.get("capabilities", [])

            # Pending tasks desired by this agent → desires
            for task_id, task_data in mesh.tasks.items():
                if task_data.get("desired_by") == agent_id and task_data.get("status") == "pending":
                    bdi.desires.append(
                        {
                            "id": task_id,
                            "goal": task_data.get("objective", ""),
                            "priority": task_data.get("priority", 0),
                        }
                    )

            # Active assignments for this agent → intentions
            for task_id, task_data in mesh.tasks.items():
                if (
                    task_data.get("assigned_agent") == agent_id
                    and task_data.get("status") == "in_progress"
                ):
                    bdi.intentions.append(
                        {
                            "task_id": task_id,
                            "goal": task_data.get("objective", ""),
                            "plan": task_data.get("plan", []),
                        }
                    )

            bdi_states[agent_id] = bdi

        return bdi_states


# ── Joint Intentions Mapping ─────────────────────────────────────────


@dataclass
class JointIntentionState:
    """Joint Intention state (Cohen & Levesque)."""

    team: set[str] = field(default_factory=set)
    """Team members"""

    mutual_beliefs: dict[str, Any] = field(default_factory=dict)
    """Mutually believed facts"""

    persistent_goal: dict[str, Any] = field(default_factory=dict)
    """The persistent goal the team is committed to"""

    joint_commitment: bool = False
    """Whether the team has a joint commitment"""

    goal_status: str = "active"
    """Status: active, achieved, impossible, abandoned"""


class JointIntentionMapping:
    """Map Joint Intentions to/from mesh coordination.

    - Mutual belief → consensus state (agreed-upon facts)
    - Persistent goal → task with completion criteria
    - Joint commitment → all agents assigned to same task
    """

    @staticmethod
    def to_mesh(ji_state: JointIntentionState) -> MeshState:
        """Convert Joint Intention state to mesh state."""
        mesh = MeshState()

        # Team members → agents
        for agent_id in ji_state.team:
            mesh.agents[agent_id] = {
                "status": "active",
                "team_member": True,
            }

        # Mutual beliefs → consensus
        mesh.consensus = {
            "agreed_facts": dict(ji_state.mutual_beliefs),
            "round": 0,
            "participants": list(ji_state.team),
        }

        # Persistent goal → shared task
        if ji_state.persistent_goal:
            goal_id = ji_state.persistent_goal.get("id", "joint_goal")
            status = {
                "active": "in_progress",
                "achieved": "completed",
                "impossible": "failed",
                "abandoned": "cancelled",
            }.get(ji_state.goal_status, "pending")

            mesh.tasks[goal_id] = {
                "objective": ji_state.persistent_goal.get("description", ""),
                "status": status,
                "assigned_agents": list(ji_state.team) if ji_state.joint_commitment else [],
                "joint": True,
                "criteria": ji_state.persistent_goal.get("criteria", {}),
            }

        return mesh

    @staticmethod
    def from_mesh(mesh: MeshState) -> JointIntentionState:
        """Convert mesh state to Joint Intention state."""
        ji = JointIntentionState()

        # Agents → team
        ji.team = set(mesh.agents.keys())

        # Consensus → mutual beliefs
        ji.mutual_beliefs = dict(mesh.consensus.get("agreed_facts", {}))

        # Find joint task → persistent goal
        for task_id, task_data in mesh.tasks.items():
            if task_data.get("joint"):
                ji.persistent_goal = {
                    "id": task_id,
                    "description": task_data.get("objective", ""),
                    "criteria": task_data.get("criteria", {}),
                }
                ji.joint_commitment = len(task_data.get("assigned_agents", [])) > 0

                status_map = {
                    "in_progress": "active",
                    "completed": "achieved",
                    "failed": "impossible",
                    "cancelled": "abandoned",
                    "pending": "active",
                }
                ji.goal_status = status_map.get(task_data.get("status", ""), "active")
                break

        return ji


# ── SharedPlans Mapping ──────────────────────────────────────────────


@dataclass
class SharedPlanState:
    """SharedPlans state (Grosz & Kraus)."""

    recipe: dict[str, list[str]] = field(default_factory=dict)
    """Task decomposition: parent_task → [subtasks]"""

    task_descriptions: dict[str, str] = field(default_factory=dict)
    """Task descriptions: task_id → description"""

    subgroup_assignments: dict[str, set[str]] = field(default_factory=dict)
    """Subgroup plans: task_id → assigned agents"""

    partial_plan: bool = True
    """Whether the plan is still being elaborated"""


class SharedPlanMapping:
    """Map SharedPlans to/from mesh coordination.

    - Recipe → task decomposition DAG
    - Subgroup plans → agent cluster assignments
    """

    @staticmethod
    def to_mesh(sp_state: SharedPlanState) -> MeshState:
        """Convert SharedPlan state to mesh state."""
        mesh = MeshState()

        # Collect all agents from subgroup assignments
        all_agents: set[str] = set()
        for agents in sp_state.subgroup_assignments.values():
            all_agents.update(agents)

        for agent_id in all_agents:
            mesh.agents[agent_id] = {"status": "active"}

        # Recipe → tasks with decomposition
        all_tasks: set[str] = set()
        for parent, subtasks in sp_state.recipe.items():
            all_tasks.add(parent)
            all_tasks.update(subtasks)

        for task_id in all_tasks:
            assigned = sp_state.subgroup_assignments.get(task_id, set())
            subtasks = sp_state.recipe.get(task_id, [])

            mesh.tasks[task_id] = {
                "objective": sp_state.task_descriptions.get(task_id, ""),
                "status": "pending" if sp_state.partial_plan else "ready",
                "assigned_agents": list(assigned),
                "subtasks": subtasks,
                "is_composite": len(subtasks) > 0,
            }

        mesh.metadata["partial_plan"] = sp_state.partial_plan

        return mesh

    @staticmethod
    def from_mesh(mesh: MeshState) -> SharedPlanState:
        """Convert mesh state to SharedPlan state."""
        sp = SharedPlanState()
        sp.partial_plan = mesh.metadata.get("partial_plan", True)

        for task_id, task_data in mesh.tasks.items():
            sp.task_descriptions[task_id] = task_data.get("objective", "")

            subtasks = task_data.get("subtasks", [])
            if subtasks:
                sp.recipe[task_id] = subtasks

            assigned = task_data.get("assigned_agents", [])
            if assigned:
                sp.subgroup_assignments[task_id] = set(assigned)

        return sp


# ── Contract Net Mapping ─────────────────────────────────────────────


class ContractNetPhase(str, Enum):
    """Phases of the Contract Net Protocol."""

    ANNOUNCEMENT = "announcement"
    """Manager announces task (CFP)"""

    BIDDING = "bidding"
    """Bidders submit proposals"""

    EVALUATION = "evaluation"
    """Manager evaluates bids"""

    AWARDING = "awarding"
    """Manager awards contract"""

    EXECUTION = "execution"
    """Contractor executes task"""

    REPORTING = "reporting"
    """Contractor reports results"""


@dataclass
class ContractNetState:
    """Contract Net Protocol state."""

    manager: str = ""
    """Manager agent ID"""

    task: dict[str, Any] = field(default_factory=dict)
    """Task being contracted"""

    phase: ContractNetPhase = ContractNetPhase.ANNOUNCEMENT
    """Current phase"""

    bids: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Bids received: bidder_id → bid details"""

    winner: str | None = None
    """Winning bidder"""

    result: Any | None = None
    """Execution result"""


class ContractNetMapping:
    """Map Contract Net Protocol to/from mesh coordination.

    - Manager broadcasts CFP → task announcement message
    - Bidders respond → agent capability advertisements
    - Manager awards → task assignment
    """

    @staticmethod
    def to_mesh(cn_state: ContractNetState) -> MeshState:
        """Convert Contract Net state to mesh state."""
        mesh = MeshState()

        # Manager agent
        mesh.agents[cn_state.manager] = {
            "status": "active",
            "role": "manager",
        }

        # Bidder agents
        for bidder_id, bid in cn_state.bids.items():
            mesh.agents[bidder_id] = {
                "status": "active",
                "role": "bidder",
                "bid": bid,
            }

        # Task
        task_id = cn_state.task.get("id", "contract_task")
        status_map = {
            ContractNetPhase.ANNOUNCEMENT: "pending",
            ContractNetPhase.BIDDING: "pending",
            ContractNetPhase.EVALUATION: "pending",
            ContractNetPhase.AWARDING: "assigned",
            ContractNetPhase.EXECUTION: "in_progress",
            ContractNetPhase.REPORTING: "completed",
        }

        mesh.tasks[task_id] = {
            "objective": cn_state.task.get("description", ""),
            "status": status_map.get(cn_state.phase, "pending"),
            "manager": cn_state.manager,
            "assigned_agent": cn_state.winner,
            "result": cn_state.result,
        }

        # Messages for each phase
        if cn_state.phase.value >= ContractNetPhase.ANNOUNCEMENT.value:
            mesh.messages.append(
                {
                    "type": "cfp",
                    "from": cn_state.manager,
                    "to": "broadcast",
                    "content": cn_state.task,
                }
            )

        for bidder_id, bid in cn_state.bids.items():
            mesh.messages.append(
                {
                    "type": "bid",
                    "from": bidder_id,
                    "to": cn_state.manager,
                    "content": bid,
                }
            )

        if cn_state.winner:
            mesh.messages.append(
                {
                    "type": "award",
                    "from": cn_state.manager,
                    "to": cn_state.winner,
                    "content": {"task": task_id},
                }
            )

        mesh.metadata["contract_net_phase"] = cn_state.phase.value

        return mesh

    @staticmethod
    def from_mesh(mesh: MeshState) -> ContractNetState:
        """Convert mesh state to Contract Net state."""
        cn = ContractNetState()
        cn.phase = ContractNetPhase(mesh.metadata.get("contract_net_phase", "announcement"))

        # Find manager
        for agent_id, agent_data in mesh.agents.items():
            if agent_data.get("role") == "manager":
                cn.manager = agent_id
            elif agent_data.get("role") == "bidder":
                bid = agent_data.get("bid", {})
                if bid:
                    cn.bids[agent_id] = bid

        # Find task
        for task_id, task_data in mesh.tasks.items():
            cn.task = {
                "id": task_id,
                "description": task_data.get("objective", ""),
            }
            cn.winner = task_data.get("assigned_agent")
            cn.result = task_data.get("result")
            break

        return cn


# ── Verification ─────────────────────────────────────────────────────


def verify_mapping_preservation(
    original_mesh: MeshState,
    mapping_to: Any,
    mapping_from: Any,
    to_func: str = "to_mesh",
    from_func: str = "from_mesh",
) -> dict[str, Any]:
    """Verify that a MAS mapping preserves key properties.

    Checks:
    1. Agent count is preserved
    2. Task count is preserved
    3. Assignment relationships are preserved

    Args:
        original_mesh: Original mesh state
        mapping_to: Mapping class with from_mesh method
        mapping_from: Mapping class with to_mesh method
        to_func: Name of the to_mesh method
        from_func: Name of the from_mesh method

    Returns:
        Verification results
    """
    # Forward: mesh → MAS
    mas_state = getattr(mapping_to, from_func)(original_mesh)

    # Reverse: MAS → mesh
    roundtrip_mesh = getattr(mapping_from, to_func)(mas_state)

    # Check preservation
    agent_preserved = set(original_mesh.agents.keys()) == set(roundtrip_mesh.agents.keys())
    task_preserved = set(original_mesh.tasks.keys()) == set(roundtrip_mesh.tasks.keys())

    return {
        "agent_count_preserved": agent_preserved,
        "original_agents": len(original_mesh.agents),
        "roundtrip_agents": len(roundtrip_mesh.agents),
        "task_count_preserved": task_preserved,
        "original_tasks": len(original_mesh.tasks),
        "roundtrip_tasks": len(roundtrip_mesh.tasks),
        "fully_preserved": agent_preserved and task_preserved,
    }


__all__ = [
    "MASTheory",
    "MeshState",
    # BDI
    "BDIState",
    "BDIMapping",
    # Joint Intentions
    "JointIntentionState",
    "JointIntentionMapping",
    # SharedPlans
    "SharedPlanState",
    "SharedPlanMapping",
    # Contract Net
    "ContractNetPhase",
    "ContractNetState",
    "ContractNetMapping",
    # Verification
    "verify_mapping_preservation",
]
