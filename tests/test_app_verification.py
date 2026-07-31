"""Contract tests for :func:`agenticraft_foundation.verify`.

These tests lock the verification report shape and the severity
decisions for each check. They use bare dicts rather than pydantic
models so the test stays decoupled from any platform package
(matching the production contract of ``verify``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agenticraft_foundation import VerificationCheck, VerificationReport, verify

# =============================================================================
# Manifest fixtures
# =============================================================================


def _manifest(
    *,
    name: str = "test-app",
    agents: list[dict[str, Any]] | None = None,
    connections: list[dict[str, Any]] | None = None,
    workflows: list[dict[str, Any]] | None = None,
    groups: list[dict[str, Any]] | None = None,
    trust: list[dict[str, Any]] | None = None,
    coordination_mode: str = "orchestrated",
) -> dict[str, Any]:
    """Build a minimal manifest dict for tests."""
    return {
        "kind": "app",
        "name": name,
        "manifest_version": 1,
        "agents": agents or [],
        "topology": {
            "connections": connections or [],
            "groups": groups or [],
            "trust": trust or [],
            "coordination_mode": coordination_mode,
        },
        "workflows": workflows or [],
    }


def _agent(aid: str) -> dict[str, Any]:
    return {"id": aid, "module": f"agents.{aid}"}


def _check_named(report: VerificationReport, name: str) -> VerificationCheck:
    """Find the single check with the given name."""
    matches = [c for c in report.checks if c.name == name]
    assert len(matches) >= 1, f"check '{name}' missing from report"
    return matches[0]


# =============================================================================
# Report shape contract
# =============================================================================


class TestReportShape:
    def test_returns_verification_report(self) -> None:
        report = verify(_manifest())
        assert isinstance(report, VerificationReport)
        assert isinstance(report.checks, list)
        assert isinstance(report.passed, bool)

    def test_report_has_timing(self) -> None:
        report = verify(_manifest())
        assert report.duration_ms >= 0.0

    def test_report_preserves_manifest_name(self) -> None:
        report = verify(_manifest(name="alpha-bot"))
        assert report.manifest_name == "alpha-bot"

    def test_report_preserves_manifest_version(self) -> None:
        m = _manifest()
        m["manifest_version"] = 2
        report = verify(m)
        assert report.manifest_version == 2

    def test_unnamed_manifest_falls_back_to_placeholder(self) -> None:
        m = _manifest()
        del m["name"]
        report = verify(m)
        assert report.manifest_name == "<unnamed>"

    def test_summary_is_human_readable(self) -> None:
        text = verify(_manifest()).summary()
        assert "PASSED" in text or "FAILED" in text
        assert "checks run" in text
        assert "strict=" in text

    def test_summary_uses_severity_tag(self) -> None:
        # Confirm each tag bucket appears for the matching severity.
        m = _manifest(agents=[_agent("a"), _agent("b")])  # info disconnect
        text = verify(m).summary()
        assert "[OK ]" in text
        assert "[INF]" in text  # topology.connected info under orchestrated

    def test_errors_property_filters_failed_errors_only(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b")],
            connections=[{"from": "a", "to": "ghost"}],
        )
        report = verify(m)
        assert all(c.severity == "error" for c in report.errors)
        assert all(not c.passed for c in report.errors)

    def test_warnings_property_filters_failed_warnings_only(self) -> None:
        m = _manifest(
            agents=[_agent("hub"), _agent("s1"), _agent("s2")],
            connections=[
                {"from": "hub", "to": "s1"},
                {"from": "hub", "to": "s2"},
            ],  # hub-and-spoke triggers fault_tolerant warning
        )
        report = verify(m)
        assert any(w.name == "topology.fault_tolerant" for w in report.warnings)
        assert all(c.severity == "warning" for c in report.warnings)

    def test_strict_field_defaults_to_false(self) -> None:
        report = verify(_manifest())
        assert report.strict is False

    def test_strict_field_recorded_when_set(self) -> None:
        report = verify(_manifest(), strict=True)
        assert report.strict is True


# =============================================================================
# Per-check timing
# =============================================================================


class TestPerCheckTiming:
    def test_each_check_has_duration_ms(self) -> None:
        report = verify(_manifest(agents=[_agent("a")]))
        for check in report.checks:
            assert check.duration_ms >= 0.0
            assert isinstance(check.duration_ms, float)

    def test_total_duration_at_least_max_individual(self) -> None:
        report = verify(_manifest(agents=[_agent("a")]))
        # Sum of per-check timings should not exceed the wall-clock
        # total report.duration_ms (with a small slack for overhead).
        per_check = max(c.duration_ms for c in report.checks)
        assert report.duration_ms >= per_check


# =============================================================================
# to_dict serialization
# =============================================================================


class TestToDict:
    def test_report_to_dict_is_json_serializable(self) -> None:
        report = verify(_manifest(agents=[_agent("a")]))
        payload = report.to_dict()
        # Round-trip through JSON to confirm no non-serializable values.
        encoded = json.dumps(payload, default=str)
        decoded = json.loads(encoded)
        assert decoded["passed"] == report.passed
        assert decoded["manifest_name"] == report.manifest_name

    def test_check_to_dict_preserves_severity_string(self) -> None:
        report = verify(_manifest(agents=[_agent("a")]))
        for c in report.checks:
            d = c.to_dict()
            assert d["severity"] in {"error", "warning", "info"}
            assert isinstance(d["passed"], bool)
            assert "duration_ms" in d


# =============================================================================
# Coercion contract
# =============================================================================


class TestInputCoercion:
    def test_accepts_dict(self) -> None:
        report = verify(_manifest())
        assert isinstance(report, VerificationReport)

    def test_accepts_model_dump_object(self) -> None:
        class FakeModel:
            def model_dump(self, by_alias: bool = False) -> dict[str, Any]:
                assert by_alias is True, "verify must request by-alias dump"
                return _manifest()

        report = verify(FakeModel())
        assert isinstance(report, VerificationReport)

    def test_rejects_non_dict_non_pydantic(self) -> None:
        with pytest.raises(TypeError, match="dict or an object"):
            verify("not a manifest")  # type: ignore[arg-type]

    def test_rejects_none(self) -> None:
        with pytest.raises(TypeError):
            verify(None)  # type: ignore[arg-type]


# =============================================================================
# Reference integrity (error severity)
# =============================================================================


class TestReferenceIntegrity:
    def test_empty_manifest_passes(self) -> None:
        report = verify(_manifest())
        check = _check_named(report, "references.integrity")
        assert check.passed

    def test_passes_when_all_references_resolve(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b")],
            connections=[{"from": "a", "to": "b"}],
            workflows=[
                {
                    "id": "wf1",
                    "name": "WF 1",
                    "steps": [{"id": "s1", "agent": "a"}],
                }
            ],
        )
        report = verify(m)
        assert report.passed, report.summary()
        assert _check_named(report, "references.integrity").passed

    def test_fails_on_unknown_from_agent(self) -> None:
        m = _manifest(
            agents=[_agent("a")],
            connections=[{"from": "ghost", "to": "a"}],
        )
        report = verify(m)
        assert not report.passed
        check = _check_named(report, "references.integrity")
        assert not check.passed
        assert check.severity == "error"
        assert "ghost" in str(check.details["violations"])

    def test_fails_on_unknown_to_agent(self) -> None:
        m = _manifest(
            agents=[_agent("a")],
            connections=[{"from": "a", "to": "ghost"}],
        )
        assert not verify(m).passed

    def test_fails_on_unknown_workflow_agent(self) -> None:
        m = _manifest(
            agents=[_agent("a")],
            workflows=[
                {
                    "id": "wf",
                    "name": "WF",
                    "steps": [{"id": "s1", "agent": "ghost"}],
                }
            ],
        )
        assert not verify(m).passed

    def test_fails_on_invalid_depends_on(self) -> None:
        m = _manifest(
            agents=[_agent("a")],
            workflows=[
                {
                    "id": "wf",
                    "name": "WF",
                    "steps": [
                        {"id": "s1", "agent": "a", "depends_on": ["does-not-exist"]},
                    ],
                }
            ],
        )
        assert not verify(m).passed

    def test_accepts_from_agent_alias(self) -> None:
        # The pydantic alias is ``from``, but the attribute name is
        # ``from_agent``. Both spellings must work in dict form.
        m = _manifest(
            agents=[_agent("a"), _agent("b")],
            connections=[{"from_agent": "a", "to": "b"}],
        )
        assert _check_named(verify(m), "references.integrity").passed

    def test_to_field_accepts_list(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b"), _agent("c")],
            connections=[{"from": "a", "to": ["b", "c"]}],
        )
        assert _check_named(verify(m), "references.integrity").passed

    def test_fails_on_unknown_broadcast_group_member(self) -> None:
        m = _manifest(
            agents=[_agent("a")],
            groups=[{"name": "g1", "members": ["a", "ghost"]}],
        )
        report = verify(m)
        check = _check_named(report, "references.integrity")
        assert not check.passed
        assert "ghost" in str(check.details["violations"])

    def test_fails_on_unknown_trust_boundary_member(self) -> None:
        m = _manifest(
            agents=[_agent("a")],
            trust=[{"agents": ["a", "ghost"], "level": "full"}],
        )
        report = verify(m)
        check = _check_named(report, "references.integrity")
        assert not check.passed
        assert "ghost" in str(check.details["violations"])

    def test_parallel_step_agent_references_checked(self) -> None:
        m = _manifest(
            agents=[_agent("a")],
            workflows=[
                {
                    "id": "wf",
                    "name": "WF",
                    "steps": [
                        {
                            "id": "step1",
                            "type": "parallel",
                            "agents": [
                                {"agent": "a"},
                                {"agent": "ghost"},
                            ],
                        }
                    ],
                }
            ],
        )
        report = verify(m)
        assert not _check_named(report, "references.integrity").passed


# =============================================================================
# Topology connectivity — severity driven by coordination_mode
# =============================================================================


class TestTopologyConnectivity:
    def test_single_agent_passes_trivially(self) -> None:
        report = verify(_manifest(agents=[_agent("a")]))
        check = _check_named(report, "topology.connected")
        assert check.passed
        assert check.severity == "info"

    def test_empty_manifest_passes_trivially(self) -> None:
        report = verify(_manifest())
        assert _check_named(report, "topology.connected").passed

    def test_orchestrated_mode_disconnected_is_info(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b")],
            coordination_mode="orchestrated",
        )
        report = verify(m)
        check = _check_named(report, "topology.connected")
        assert not check.passed
        assert check.severity == "info"
        assert report.passed  # info doesn't fail

    def test_hybrid_mode_disconnected_is_warning(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b")],
            coordination_mode="hybrid",
        )
        report = verify(m)
        check = _check_named(report, "topology.connected")
        assert not check.passed
        assert check.severity == "warning"
        assert report.passed  # warnings don't fail in non-strict mode

    def test_a2a_mode_disconnected_is_error(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b")],
            coordination_mode="a2a",
        )
        report = verify(m)
        check = _check_named(report, "topology.connected")
        assert not check.passed
        assert check.severity == "error"
        assert not report.passed

    def test_fully_connected_pair_passes_in_a2a_mode(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b")],
            connections=[{"from": "a", "to": "b"}],
            coordination_mode="a2a",
        )
        report = verify(m)
        check = _check_named(report, "topology.connected")
        assert check.passed
        assert check.details["algebraic_connectivity"] > 0
        assert check.details["coordination_mode"] == "a2a"

    def test_disconnected_components_fail_in_a2a_mode(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b"), _agent("c"), _agent("d")],
            connections=[
                {"from": "a", "to": "b"},
                {"from": "c", "to": "d"},
            ],
            coordination_mode="a2a",
        )
        report = verify(m)
        check = _check_named(report, "topology.connected")
        assert not check.passed
        assert check.severity == "error"
        assert check.details["algebraic_connectivity"] < 1e-9

    def test_orphan_agent_flagged_in_details(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b"), _agent("orphan")],
            connections=[{"from": "a", "to": "b"}],
        )
        report = verify(m)
        check = _check_named(report, "topology.connected")
        assert "orphan" in check.details["orphan_agents"]


# =============================================================================
# Topology fault tolerance (warning severity)
# =============================================================================


class TestTopologyFaultTolerant:
    def test_articulation_point_emits_warning(self) -> None:
        # Hub-and-spoke: removing the hub disconnects every spoke.
        m = _manifest(
            agents=[_agent("hub"), _agent("s1"), _agent("s2"), _agent("s3")],
            connections=[
                {"from": "hub", "to": "s1"},
                {"from": "hub", "to": "s2"},
                {"from": "hub", "to": "s3"},
            ],
        )
        report = verify(m)
        check = _check_named(report, "topology.fault_tolerant")
        assert not check.passed
        assert check.severity == "warning"

    def test_ring_has_no_articulation_points(self) -> None:
        m = _manifest(
            agents=[_agent("n1"), _agent("n2"), _agent("n3"), _agent("n4")],
            connections=[
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n3"},
                {"from": "n3", "to": "n4"},
                {"from": "n4", "to": "n1"},
            ],
        )
        report = verify(m)
        check = _check_named(report, "topology.fault_tolerant")
        assert check.passed
        assert check.severity == "info"

    def test_no_edges_is_info_not_warning(self) -> None:
        m = _manifest(agents=[_agent("a"), _agent("b")])
        check = _check_named(verify(m), "topology.fault_tolerant")
        assert check.passed
        assert check.severity == "info"


# =============================================================================
# Topology privilege flow (error severity, new check)
# =============================================================================


class TestTopologyPrivilegeFlow:
    def test_no_trust_boundaries_is_vacuous_pass(self) -> None:
        m = _manifest(agents=[_agent("a"), _agent("b")])
        check = _check_named(verify(m), "topology.privilege_flow")
        assert check.passed
        assert check.severity == "info"

    def test_intra_boundary_connection_does_not_require_attenuation(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b")],
            connections=[{"from": "a", "to": "b"}],
            trust=[{"agents": ["a", "b"], "level": "full"}],
        )
        report = verify(m)
        assert report.passed
        assert _check_named(report, "topology.privilege_flow").passed

    def test_cross_boundary_connection_without_attenuation_fails(self) -> None:
        m = _manifest(
            agents=[_agent("priv"), _agent("unpriv")],
            connections=[{"from": "priv", "to": "unpriv"}],
            trust=[
                {"agents": ["priv"], "level": "full"},
                {"agents": ["unpriv"], "level": "read_only"},
            ],
        )
        report = verify(m)
        check = _check_named(report, "topology.privilege_flow")
        assert not check.passed
        assert check.severity == "error"
        assert not report.passed

    def test_cross_boundary_with_attenuation_passes(self) -> None:
        m = _manifest(
            agents=[_agent("priv"), _agent("unpriv")],
            connections=[
                {
                    "from": "priv",
                    "to": "unpriv",
                    "privilege_attenuation": {"write": "deny"},
                }
            ],
            trust=[
                {"agents": ["priv"], "level": "full"},
                {"agents": ["unpriv"], "level": "read_only"},
            ],
        )
        report = verify(m)
        assert _check_named(report, "topology.privilege_flow").passed
        assert report.passed


# =============================================================================
# Workflow deadlock-freedom (error severity)
# =============================================================================


class TestWorkflowDeadlockFreedom:
    def test_no_workflows_is_info(self) -> None:
        report = verify(_manifest(agents=[_agent("a")]))
        check = _check_named(report, "workflow.deadlock_free")
        assert check.passed
        assert check.severity == "info"

    def test_empty_workflow_passes(self) -> None:
        m = _manifest(
            agents=[_agent("a")],
            workflows=[{"id": "wf", "name": "WF", "steps": []}],
        )
        assert _check_named(verify(m), "workflow.deadlock_free").passed

    def test_sequential_pipeline_passes(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b")],
            connections=[{"from": "a", "to": "b"}],
            workflows=[
                {
                    "id": "pipe",
                    "name": "Pipe",
                    "pattern": "pipeline",
                    "steps": [
                        {"id": "s1", "agent": "a"},
                        {"id": "s2", "agent": "b"},
                    ],
                }
            ],
        )
        report = verify(m)
        check = _check_named(report, "workflow.deadlock_free")
        assert check.passed
        assert check.details["lts_states"] > 0
        assert check.details["lts_transitions"] > 0

    def test_fan_out_then_synthesize_passes(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b"), _agent("c"), _agent("synth")],
            workflows=[
                {
                    "id": "fanout",
                    "name": "FanOut",
                    "pattern": "fan_out_then_synthesize",
                    "steps": [
                        {"id": "s1", "agent": "a"},
                        {"id": "s2", "agent": "b"},
                        {"id": "s3", "agent": "c"},
                        {"id": "synthesis", "agent": "synth"},
                    ],
                }
            ],
        )
        assert _check_named(verify(m), "workflow.deadlock_free").passed

    def test_dag_with_explicit_depends_on_passes(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b"), _agent("c")],
            workflows=[
                {
                    "id": "dag",
                    "name": "DAG",
                    "steps": [
                        {"id": "s1", "agent": "a"},
                        {"id": "s2", "agent": "b", "depends_on": ["s1"]},
                        {"id": "s3", "agent": "c", "depends_on": ["s2"]},
                    ],
                }
            ],
        )
        assert _check_named(verify(m), "workflow.deadlock_free").passed

    def test_parallel_step_passes(self) -> None:
        m = _manifest(
            agents=[_agent("a"), _agent("b"), _agent("c")],
            workflows=[
                {
                    "id": "wf",
                    "name": "WF",
                    "pattern": "pipeline",
                    "steps": [
                        {
                            "id": "fan",
                            "type": "parallel",
                            "agents": [
                                {"agent": "a", "action": "x"},
                                {"agent": "b", "action": "y"},
                            ],
                        },
                        {"id": "report", "agent": "c"},
                    ],
                }
            ],
        )
        report = verify(m)
        check = _check_named(report, "workflow.deadlock_free")
        assert check.passed
        # Parallel sub-events expand the LTS beyond the single-step case.
        assert check.details["lts_states"] >= 2

    def test_malformed_workflow_reports_error_not_crash(self) -> None:
        m = _manifest(
            agents=[_agent("a")],
            workflows=[
                {
                    "id": "bad",
                    "name": "Bad",
                    "steps": [
                        {"agent": "a"},  # missing 'id'
                    ],
                }
            ],
        )
        report = verify(m)
        check = _check_named(report, "workflow.deadlock_free")
        assert not check.passed
        assert check.severity == "error"
        assert "exception_type" in check.details


# =============================================================================
# Strict mode
# =============================================================================


class TestStrictMode:
    def test_strict_promotes_warning_to_failure(self) -> None:
        # hub-and-spoke triggers a fault_tolerant warning.
        m = _manifest(
            agents=[_agent("hub"), _agent("s1"), _agent("s2")],
            connections=[
                {"from": "hub", "to": "s1"},
                {"from": "hub", "to": "s2"},
            ],
        )
        relaxed = verify(m)
        strict = verify(m, strict=True)
        assert relaxed.passed
        assert not strict.passed
        # Errors view is unchanged; the warning set (compared by name)
        # is identical -- only the report verdict flips.
        assert relaxed.errors == []
        assert strict.errors == []
        assert relaxed.warnings  # at least one
        assert {w.name for w in strict.warnings} == {w.name for w in relaxed.warnings}

    def test_strict_does_not_change_warning_severity_label(self) -> None:
        # Warnings remain labeled "warning"; only the report verdict flips.
        m = _manifest(
            agents=[_agent("hub"), _agent("s1"), _agent("s2")],
            connections=[
                {"from": "hub", "to": "s1"},
                {"from": "hub", "to": "s2"},
            ],
        )
        strict = verify(m, strict=True)
        for w in strict.warnings:
            assert w.severity == "warning"

    def test_strict_passes_when_only_info_failures(self) -> None:
        # orchestrated + disconnected = info; strict should still pass.
        m = _manifest(
            agents=[_agent("a"), _agent("b")],
            coordination_mode="orchestrated",
        )
        assert verify(m, strict=True).passed


# =============================================================================
# Overall verdict
# =============================================================================


class TestOverallVerdict:
    def test_pass_with_only_warnings(self) -> None:
        # Hub-and-spoke: fault_tolerant warning, no errors.
        m = _manifest(
            agents=[_agent("hub"), _agent("s1"), _agent("s2")],
            connections=[
                {"from": "hub", "to": "s1"},
                {"from": "hub", "to": "s2"},
            ],
        )
        assert verify(m).passed

    def test_fail_with_reference_violation(self) -> None:
        m = _manifest(
            agents=[_agent("a")],
            connections=[{"from": "ghost", "to": "a"}],
        )
        assert not verify(m).passed

    def test_each_check_is_a_verification_check_instance(self) -> None:
        report = verify(_manifest(agents=[_agent("a")]))
        for c in report.checks:
            assert isinstance(c, VerificationCheck)
            assert c.name and isinstance(c.name, str)
            assert isinstance(c.passed, bool)
            assert c.severity in {"error", "warning", "info"}
            assert isinstance(c.message, str)


# =============================================================================
# Real-world manifest shape: personal-mesh
# =============================================================================


class TestPersonalMeshShape:
    """Locks the verification result for personal-mesh's structural shape."""

    def test_personal_mesh_shape_passes(self) -> None:
        m = _manifest(
            name="personal-mesh",
            agents=[
                _agent("financial"),
                _agent("calendar"),
                _agent("briefing"),
                _agent("grocery"),
                _agent("health"),
            ],
            connections=[
                {
                    "from": "briefing",
                    "to": ["financial", "calendar"],
                    "type": "collect",
                    "purpose": "Gather data for morning briefing",
                },
            ],
            workflows=[
                {
                    "id": "morning-briefing",
                    "name": "Morning Briefing",
                    "pattern": "pipeline",
                    "steps": [
                        {"id": "gather-expenses", "agent": "financial"},
                        {"id": "gather-calendar", "agent": "calendar"},
                        {"id": "synthesize", "agent": "briefing"},
                    ],
                },
            ],
            coordination_mode="orchestrated",
        )
        report = verify(m)
        assert report.passed, report.summary()
        # Orchestrated mode makes the disconnected check INFO; the only
        # warning is fault_tolerant (briefing forms articulation points).
        assert len(report.errors) == 0
        warning_names = {w.name for w in report.warnings}
        assert warning_names == {"topology.fault_tolerant"}

    def test_personal_mesh_in_strict_a2a_mode_fails(self) -> None:
        # Same shape under a2a mode is rejected: orphan agents
        # become errors. Demonstrates the long-term escalation path.
        m = _manifest(
            name="personal-mesh",
            agents=[
                _agent("financial"),
                _agent("calendar"),
                _agent("briefing"),
                _agent("grocery"),
                _agent("health"),
            ],
            connections=[
                {"from": "briefing", "to": ["financial", "calendar"]},
            ],
            coordination_mode="a2a",
        )
        report = verify(m)
        assert not report.passed
        assert any(e.name == "topology.connected" for e in report.errors)
