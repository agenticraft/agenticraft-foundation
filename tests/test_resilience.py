"""Engine tests for the resilience diagnostic.

Covers the canonical topologies (star, chain, ring, voting ensemble), degenerate
sizes, the report shape, target/remediation/cost logic, model swappability, and
the manifest adapter (including a real-schema fixture). CLI tests live in
``test_resilience_cli.py``.

Ground-truth bounds (classical placeholder, over the undirected projection):
    f_crash = max(0, min(κ - 1, (n - 1)//2))
    f_byz   = max(0, min((κ - 1)//2, (n - 1)//3))
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agenticraft_foundation.resilience import (
    CLASSICAL_MODEL_VERSION,
    ClassicalQuorumModel,
    FaultEstimate,
    ResilienceModel,
    ResilienceReport,
    ResilienceTarget,
    analyze_topology,
    graph_from_manifest,
    load_topology,
)
from agenticraft_foundation.topology import (
    ConnectivityAnalyzer,
    NetworkGraph,
    classical_fault_tolerance,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "resilience_star_app.yaml"


def _chain(n: int) -> NetworkGraph:
    """Build a linear pipeline node_0 -- node_1 -- ... -- node_{n-1}."""
    graph = NetworkGraph()
    for i in range(n):
        graph.add_node(f"node_{i}")
    for i in range(n - 1):
        graph.add_edge(f"node_{i}", f"node_{i + 1}")
    return graph


class TestCanonicalTopologies:
    """The acceptance-criteria topologies analyze to the right numbers."""

    def test_star_orchestrator_survives_zero_of_either(self) -> None:
        # Headline demo: one hub => single point of failure for both models.
        report = analyze_topology(NetworkGraph.create_star(6))
        assert (report.f_crash, report.f_byz) == (0, 0)
        assert "single hub" in report.crash_binding
        assert "node_center" in report.articulation_points

    def test_linear_pipeline_breaks_on_any_crash(self) -> None:
        report = analyze_topology(_chain(5))
        assert (report.f_crash, report.f_byz) == (0, 0)
        assert "articulation" in report.crash_binding

    def test_ring_reroutes_so_crash_tolerance_is_positive(self) -> None:
        report = analyze_topology(NetworkGraph.create_ring(5))
        assert report.f_crash == 1  # κ=2 => survives one crash via rerouting
        assert report.f_byz == 0  # no agreement quorum

    def test_voting_ensemble_byzantine_scales_with_replicas(self) -> None:
        assert analyze_topology(NetworkGraph.create_complete(4)).f_byz == 1
        assert analyze_topology(NetworkGraph.create_complete(7)).f_byz == 2
        # f_byz grows with replica count + quorum
        small = analyze_topology(NetworkGraph.create_complete(4)).f_byz
        large = analyze_topology(NetworkGraph.create_complete(7)).f_byz
        assert large > small

    def test_byzantine_never_exceeds_crash(self) -> None:
        # The provable half of the narrative: Byzantine is the harder mode.
        for graph in (
            NetworkGraph.create_star(6),
            _chain(5),
            NetworkGraph.create_ring(7),
            NetworkGraph.create_complete(7),
            NetworkGraph.create_mesh(3, 3),
        ):
            report = analyze_topology(graph)
            assert report.f_byz <= report.f_crash


class TestDegenerateSizes:
    """Single- and two-agent apps are real inputs — handle them cleanly."""

    def test_single_agent(self) -> None:
        graph = NetworkGraph()
        graph.add_node("solo")
        report = analyze_topology(graph)
        assert (report.f_crash, report.f_byz) == (0, 0)
        assert report.n_agents == 1

    def test_two_agents_connected(self) -> None:
        graph = NetworkGraph()
        graph.add_edge("a", "b")
        report = analyze_topology(graph)
        assert (report.f_crash, report.f_byz) == (0, 0)
        assert report.n_agents == 2

    def test_two_agents_disconnected(self) -> None:
        graph = NetworkGraph()
        graph.add_node("a")
        graph.add_node("b")
        report = analyze_topology(graph)
        assert (report.f_crash, report.f_byz) == (0, 0)
        assert "disconnected" in report.crash_binding


class TestReportShape:
    """Report carries provenance and renders the capacity caveat."""

    def test_provenance_fields(self) -> None:
        report = analyze_topology(NetworkGraph.create_star(5))
        assert report.model_version == CLASSICAL_MODEL_VERSION
        assert report.provisional is True

    def test_to_dict_marks_byz_as_capacity(self) -> None:
        report = analyze_topology(NetworkGraph.create_complete(4))
        data = report.to_dict()
        assert data["byz_semantics"] == "capacity"
        assert data["f_byz"] == report.f_byz
        assert data["model_version"] == CLASSICAL_MODEL_VERSION

    def test_redundancy_score_reported(self) -> None:
        full = analyze_topology(NetworkGraph.create_complete(4))
        assert full.redundancy_score == 1.0  # fully meshed
        assert full.to_dict()["redundancy_score"] == 1.0
        sparse = analyze_topology(NetworkGraph.create_star(6))
        assert 0.0 < sparse.redundancy_score < 1.0

    def test_classical_model_leaves_cascade_grade_none(self) -> None:
        # cascade_grade is the empirical model's contribution; classical = None.
        report = analyze_topology(NetworkGraph.create_complete(4))
        assert report.cascade_grade is None
        assert report.to_dict()["cascade_grade"] is None

    def test_to_text_leads_with_numbers_and_caveats(self) -> None:
        text = analyze_topology(NetworkGraph.create_star(6)).to_text()
        assert "f_crash" in text and "f_byz" in text
        assert "not a guarantee" in text  # the Byzantine capacity caveat
        assert "PROVISIONAL" in text  # placeholder is labeled
        assert text == analyze_topology(NetworkGraph.create_star(6)).summary()


class TestTargetAndRemediations:
    """Targets drive meets_target, cost multipliers, and remediations."""

    def test_unmet_byzantine_target_on_star(self) -> None:
        report = analyze_topology(NetworkGraph.create_star(6), target=ResilienceTarget(byzantine=1))
        assert report.meets_target is False
        assert report.byz_cost_multiplier == 4  # 3*1 + 1
        assert any("majority-vote" in r for r in report.remediations)

    def test_met_target_has_no_remediations(self) -> None:
        report = analyze_topology(
            NetworkGraph.create_complete(4),
            target=ResilienceTarget(crash=1, byzantine=1),
        )
        assert report.meets_target is True
        assert report.remediations == []

    def test_no_target_is_trivially_met(self) -> None:
        report = analyze_topology(NetworkGraph.create_ring(5))
        assert report.meets_target is True
        assert report.target is None
        assert report.min_agents_for_target is None

    def test_min_agents_for_target(self) -> None:
        star = NetworkGraph.create_star(6)
        assert (
            analyze_topology(star, target=ResilienceTarget(byzantine=1)).min_agents_for_target
            == 4  # 3*1+1
        )
        assert (
            analyze_topology(star, target=ResilienceTarget(crash=2)).min_agents_for_target
            == 5  # 2*2+1
        )
        both = analyze_topology(
            NetworkGraph.create_complete(4),
            target=ResilienceTarget(crash=2, byzantine=1),
        )
        assert both.min_agents_for_target == 5  # max(2*2+1, 3*1+1)


class TestSwappableModel:
    """The fault rule is isolated behind ResilienceModel and is swappable."""

    def test_custom_model_is_used_verbatim(self) -> None:
        class _FakeModel:
            model_version = "fake-empirical/9.9"
            provisional = False

            def estimate(self, graph: NetworkGraph) -> FaultEstimate:
                return FaultEstimate(
                    f_crash=7,
                    f_byz=5,
                    crash_binding="fake",
                    byz_binding="fake",
                    n_agents=graph.node_count,
                    vertex_connectivity=3,
                    algebraic_connectivity=1.0,
                    redundancy_score=0.5,
                )

        assert isinstance(_FakeModel(), ResilienceModel)
        report = analyze_topology(NetworkGraph.create_complete(4), model=_FakeModel())
        assert (report.f_crash, report.f_byz) == (7, 5)
        assert report.model_version == "fake-empirical/9.9"
        assert report.provisional is False


class TestManifestLoader:
    """The AGNTCY adapter builds a graph from an app-manifest shape."""

    def test_dict_with_from_agent_alias_and_scalar_to(self) -> None:
        data = {
            "agents": [{"id": "a"}, {"id": "b"}],
            "topology": {"connections": [{"from_agent": "a", "to": "b"}]},
        }
        graph = graph_from_manifest(data)
        assert graph.node_count == 2
        assert graph.edge_count == 1

    def test_model_dump_object_with_list_to(self) -> None:
        class _Manifest:
            def model_dump(self, by_alias: bool = False) -> dict[str, object]:
                return {
                    "agents": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
                    "topology": {"connections": [{"from": "a", "to": ["b", "c"]}]},
                }

        graph = graph_from_manifest(_Manifest())
        assert graph.node_count == 3
        assert graph.edge_count == 2

    def test_rejects_unsupported_input(self) -> None:
        with pytest.raises(TypeError):
            graph_from_manifest(42)

    def test_malformed_connection_entry_raises_valueerror(self) -> None:
        # A non-mapping connection must surface as a clean input error
        # (ValueError), not an opaque AttributeError, so CLIs map it to exit 2.
        with pytest.raises(ValueError):
            graph_from_manifest({"topology": {"connections": ["not-a-mapping"]}})

    def test_non_iterable_to_field_raises_valueerror(self) -> None:
        with pytest.raises(ValueError):
            graph_from_manifest({"topology": {"connections": [{"from": "a", "to": 5}]}})

    def test_real_manifest_fixture(self) -> None:
        # Real schema (agents[] + topology.connections w/ list `to`), not a
        # hand-built dict — catches schema divergence before a user does.
        data = yaml.safe_load(_FIXTURE.read_text())
        graph = graph_from_manifest(data)
        assert graph.node_count == 4
        assert graph.edge_count == 3  # orchestrator -> worker_a/b/c

        report = analyze_topology(graph, target=ResilienceTarget(byzantine=1))
        assert (report.f_crash, report.f_byz) == (0, 0)
        assert "orchestrator" in report.crash_binding
        assert report.meets_target is False

    def test_load_topology_from_path(self) -> None:
        graph = load_topology(_FIXTURE)
        assert graph.node_count == 4
        assert graph.edge_count == 3


class TestSharedBoundsConsistency:
    """The topology primitive and the model share one bound formula and agree.

    ``classical_fault_tolerance`` is the single source of truth; both
    ``ConnectivityAnalyzer.analyze_fault_tolerance`` and ``ClassicalQuorumModel``
    consume it, so they can never diverge.
    """

    @pytest.mark.parametrize(
        "graph",
        [
            NetworkGraph.create_star(6),
            NetworkGraph.create_ring(7),
            NetworkGraph.create_complete(5),
            NetworkGraph.create_complete(7),
            _chain(5),
        ],
    )
    def test_primitive_matches_model(self, graph: NetworkGraph) -> None:
        legacy = ConnectivityAnalyzer(graph).analyze_fault_tolerance()
        estimate = ClassicalQuorumModel().estimate(graph)
        assert legacy.crash_tolerance == estimate.f_crash
        assert legacy.byzantine_tolerance == estimate.f_byz

    def test_star_hub_reports_zero(self) -> None:
        # Regression guard for the old off-by-one: a star hub is a single point
        # of failure, so both surfaces must report 0. The legacy formula used
        # min(κ, …) and wrongly returned 1 before the consolidation.
        star = NetworkGraph.create_star(6)
        legacy = ConnectivityAnalyzer(star).analyze_fault_tolerance()
        assert legacy.crash_tolerance == 0
        assert ClassicalQuorumModel().estimate(star).f_crash == 0

    def test_shared_formula_direct(self) -> None:
        assert classical_fault_tolerance(6, 1) == (0, 0)  # star / hub
        assert classical_fault_tolerance(5, 2) == (1, 0)  # ring(5)
        assert classical_fault_tolerance(7, 6) == (3, 2)  # complete(7)
        assert classical_fault_tolerance(1, 0) == (0, 0)  # degenerate

    def test_analyze_fault_tolerance_handles_empty_graph(self) -> None:
        # The sibling primitive must not crash on a degenerate (empty) graph.
        ft = ConnectivityAnalyzer(NetworkGraph()).analyze_fault_tolerance()
        assert (ft.crash_tolerance, ft.byzantine_tolerance) == (0, 0)


class TestReportImmutability:
    """The report is a frozen contract."""

    def test_report_object_is_frozen(self) -> None:
        report = analyze_topology(NetworkGraph.create_ring(5))
        assert isinstance(report, ResilienceReport)
        with pytest.raises((AttributeError, TypeError)):
            report.f_crash = 99  # type: ignore[misc]
