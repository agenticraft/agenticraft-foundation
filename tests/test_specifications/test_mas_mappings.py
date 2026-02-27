"""Tests for classical MAS formal mappings.

Covers:
- BDI mapping (to/from mesh)
- Joint Intentions mapping
- SharedPlans mapping
- Contract Net Protocol mapping
- Round-trip preservation verification
"""

from __future__ import annotations

from agenticraft_foundation.specifications.mas_mappings import (
    BDIMapping,
    BDIState,
    ContractNetMapping,
    ContractNetPhase,
    ContractNetState,
    JointIntentionMapping,
    JointIntentionState,
    MASTheory,
    MeshState,
    SharedPlanMapping,
    SharedPlanState,
    verify_mapping_preservation,
)


class TestMASTheoryEnum:
    def test_all_theories(self):
        assert MASTheory.BDI.value == "bdi"
        assert MASTheory.JOINT_INTENTIONS.value == "joint_intentions"
        assert MASTheory.SHARED_PLANS.value == "shared_plans"
        assert MASTheory.CONTRACT_NET.value == "contract_net"


# ── BDI Mapping ──────────────────────────────────────────────────────


class TestBDIMapping:
    def test_to_mesh_agents(self):
        bdi = BDIState(
            agent_id="a1",
            beliefs={"capabilities": ["search", "analyze"]},
        )
        mesh = BDIMapping.to_mesh({"a1": bdi})
        assert "a1" in mesh.agents
        assert mesh.agents["a1"]["capabilities"] == ["search", "analyze"]

    def test_to_mesh_desires_become_tasks(self):
        bdi = BDIState(
            agent_id="a1",
            desires=[
                {"id": "d1", "goal": "find information", "priority": 1},
            ],
        )
        mesh = BDIMapping.to_mesh({"a1": bdi})
        assert "d1" in mesh.tasks
        assert mesh.tasks["d1"]["status"] == "pending"
        assert mesh.tasks["d1"]["desired_by"] == "a1"

    def test_to_mesh_intentions_become_assignments(self):
        bdi = BDIState(
            agent_id="a1",
            intentions=[
                {"task_id": "t1", "goal": "process data", "plan": ["step1"]},
            ],
        )
        mesh = BDIMapping.to_mesh({"a1": bdi})
        assert "t1" in mesh.tasks
        assert mesh.tasks["t1"]["status"] == "in_progress"
        assert mesh.tasks["t1"]["assigned_agent"] == "a1"

    def test_from_mesh_beliefs(self):
        mesh = MeshState()
        mesh.agents["a1"] = {
            "context": {"temperature": 72},
            "capabilities": ["search"],
            "status": "active",
        }
        bdi_states = BDIMapping.from_mesh(mesh)
        assert "a1" in bdi_states
        assert bdi_states["a1"].beliefs["temperature"] == 72

    def test_roundtrip_preserves_agents(self):
        bdi = BDIState(
            agent_id="a1",
            beliefs={"capabilities": []},
            desires=[{"id": "d1", "goal": "test"}],
            intentions=[{"task_id": "t1", "goal": "work"}],
        )
        mesh = BDIMapping.to_mesh({"a1": bdi})
        roundtrip = BDIMapping.from_mesh(mesh)
        assert "a1" in roundtrip

    def test_multiple_agents(self):
        states = {
            "a1": BDIState(agent_id="a1", beliefs={"capabilities": []}),
            "a2": BDIState(agent_id="a2", beliefs={"capabilities": []}),
        }
        mesh = BDIMapping.to_mesh(states)
        assert len(mesh.agents) == 2


# ── Joint Intentions Mapping ─────────────────────────────────────────


class TestJointIntentionMapping:
    def test_to_mesh_team(self):
        ji = JointIntentionState(
            team={"a1", "a2", "a3"},
            mutual_beliefs={"task_location": "server_room"},
        )
        mesh = JointIntentionMapping.to_mesh(ji)
        assert len(mesh.agents) == 3
        assert mesh.consensus["agreed_facts"]["task_location"] == "server_room"

    def test_to_mesh_persistent_goal(self):
        ji = JointIntentionState(
            team={"a1", "a2"},
            persistent_goal={"id": "g1", "description": "fix server"},
            joint_commitment=True,
            goal_status="active",
        )
        mesh = JointIntentionMapping.to_mesh(ji)
        assert "g1" in mesh.tasks
        assert mesh.tasks["g1"]["status"] == "in_progress"
        assert mesh.tasks["g1"]["joint"] is True

    def test_from_mesh_extracts_team(self):
        mesh = MeshState()
        mesh.agents = {"a1": {"status": "active"}, "a2": {"status": "active"}}
        mesh.consensus = {"agreed_facts": {"key": "value"}}
        ji = JointIntentionMapping.from_mesh(mesh)
        assert ji.team == {"a1", "a2"}
        assert ji.mutual_beliefs["key"] == "value"

    def test_goal_status_mapping(self):
        for status, expected in [
            ("active", "in_progress"),
            ("achieved", "completed"),
            ("impossible", "failed"),
        ]:
            ji = JointIntentionState(
                team={"a1"},
                persistent_goal={"id": "g1"},
                goal_status=status,
            )
            mesh = JointIntentionMapping.to_mesh(ji)
            assert mesh.tasks["g1"]["status"] == expected


# ── SharedPlans Mapping ──────────────────────────────────────────────


class TestSharedPlanMapping:
    def test_to_mesh_recipe(self):
        sp = SharedPlanState(
            recipe={"main": ["sub1", "sub2"]},
            task_descriptions={"main": "Main task", "sub1": "Sub 1", "sub2": "Sub 2"},
            subgroup_assignments={"sub1": {"a1"}, "sub2": {"a2"}},
        )
        mesh = SharedPlanMapping.to_mesh(sp)
        assert "main" in mesh.tasks
        assert "sub1" in mesh.tasks
        assert mesh.tasks["main"]["subtasks"] == ["sub1", "sub2"]
        assert mesh.tasks["main"]["is_composite"] is True

    def test_to_mesh_agents(self):
        sp = SharedPlanState(
            subgroup_assignments={"t1": {"a1", "a2"}},
            task_descriptions={"t1": "Task 1"},
        )
        mesh = SharedPlanMapping.to_mesh(sp)
        assert "a1" in mesh.agents
        assert "a2" in mesh.agents

    def test_from_mesh_recipe(self):
        mesh = MeshState()
        mesh.tasks["main"] = {
            "objective": "Main",
            "subtasks": ["sub1", "sub2"],
            "assigned_agents": [],
        }
        mesh.tasks["sub1"] = {"objective": "Sub 1", "subtasks": [], "assigned_agents": ["a1"]}
        mesh.tasks["sub2"] = {"objective": "Sub 2", "subtasks": [], "assigned_agents": ["a2"]}
        sp = SharedPlanMapping.from_mesh(mesh)
        assert sp.recipe["main"] == ["sub1", "sub2"]
        assert sp.subgroup_assignments["sub1"] == {"a1"}


# ── Contract Net Mapping ─────────────────────────────────────────────


class TestContractNetMapping:
    def test_to_mesh_manager_and_bidders(self):
        cn = ContractNetState(
            manager="mgr",
            task={"id": "task1", "description": "analyze data"},
            phase=ContractNetPhase.BIDDING,
            bids={"b1": {"cost": 10}, "b2": {"cost": 15}},
        )
        mesh = ContractNetMapping.to_mesh(cn)
        assert mesh.agents["mgr"]["role"] == "manager"
        assert mesh.agents["b1"]["role"] == "bidder"
        assert mesh.agents["b2"]["role"] == "bidder"
        assert len(mesh.agents) == 3

    def test_to_mesh_messages(self):
        cn = ContractNetState(
            manager="mgr",
            task={"id": "task1", "description": "test"},
            phase=ContractNetPhase.BIDDING,
            bids={"b1": {"cost": 10}},
        )
        mesh = ContractNetMapping.to_mesh(cn)
        cfp_msgs = [m for m in mesh.messages if m["type"] == "cfp"]
        bid_msgs = [m for m in mesh.messages if m["type"] == "bid"]
        assert len(cfp_msgs) == 1
        assert len(bid_msgs) == 1

    def test_to_mesh_awarded(self):
        cn = ContractNetState(
            manager="mgr",
            task={"id": "task1"},
            phase=ContractNetPhase.AWARDING,
            winner="b1",
        )
        mesh = ContractNetMapping.to_mesh(cn)
        assert mesh.tasks["task1"]["assigned_agent"] == "b1"
        assert mesh.tasks["task1"]["status"] == "assigned"

    def test_phase_mapping(self):
        for phase, expected_status in [
            (ContractNetPhase.ANNOUNCEMENT, "pending"),
            (ContractNetPhase.EXECUTION, "in_progress"),
            (ContractNetPhase.REPORTING, "completed"),
        ]:
            cn = ContractNetState(
                manager="mgr",
                task={"id": "t1"},
                phase=phase,
            )
            mesh = ContractNetMapping.to_mesh(cn)
            assert mesh.tasks["t1"]["status"] == expected_status

    def test_from_mesh_roundtrip(self):
        cn = ContractNetState(
            manager="mgr",
            task={"id": "task1", "description": "test"},
            phase=ContractNetPhase.BIDDING,
            bids={"b1": {"cost": 10}},
        )
        mesh = ContractNetMapping.to_mesh(cn)
        roundtrip = ContractNetMapping.from_mesh(mesh)
        assert roundtrip.manager == "mgr"
        assert roundtrip.phase == ContractNetPhase.BIDDING


# ── Verification ─────────────────────────────────────────────────────


class TestMappingPreservation:
    def test_bdi_preservation(self):
        mesh = MeshState()
        mesh.agents = {"a1": {"context": {}, "capabilities": [], "status": "active"}}
        mesh.tasks = {"t1": {"objective": "test", "status": "pending", "desired_by": "a1"}}

        result = verify_mapping_preservation(
            mesh,
            BDIMapping,
            BDIMapping,
            to_func="to_mesh",
            from_func="from_mesh",
        )
        assert result["agent_count_preserved"]

    def test_joint_intention_preservation(self):
        mesh = MeshState()
        mesh.agents = {"a1": {"status": "active"}, "a2": {"status": "active"}}
        mesh.consensus = {"agreed_facts": {"key": "val"}}
        mesh.tasks = {
            "g1": {
                "objective": "goal",
                "status": "in_progress",
                "joint": True,
                "assigned_agents": ["a1", "a2"],
                "criteria": {},
            }
        }

        result = verify_mapping_preservation(
            mesh,
            JointIntentionMapping,
            JointIntentionMapping,
            to_func="to_mesh",
            from_func="from_mesh",
        )
        assert result["agent_count_preserved"]
