"""Tests for connectivity analysis module."""

from __future__ import annotations

from agenticraft_foundation.topology import (
    ConnectivityAnalyzer,
    NetworkGraph,
    verify_consensus_requirements,
)
from agenticraft_foundation.topology.connectivity import (
    ConnectivityAnalysis,
    FaultToleranceAnalysis,
)


def _make_linear_graph(n: int = 4) -> NetworkGraph:
    """Create a linear graph: 0-1-2-...(n-1)."""
    g = NetworkGraph()
    for i in range(n):
        g.add_node(f"n{i}")
    for i in range(n - 1):
        g.add_edge(f"n{i}", f"n{i + 1}")
    return g


def _make_complete_graph(n: int = 4) -> NetworkGraph:
    """Create a complete graph on n nodes."""
    g = NetworkGraph()
    for i in range(n):
        g.add_node(f"n{i}")
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(f"n{i}", f"n{j}")
    return g


def _make_star_graph(n: int = 5) -> NetworkGraph:
    """Create a star graph with center 'hub' and n-1 leaves."""
    g = NetworkGraph()
    g.add_node("hub")
    for i in range(n - 1):
        g.add_node(f"leaf{i}")
        g.add_edge("hub", f"leaf{i}")
    return g


class TestConnectivityAnalysisSummary:
    """Tests for ConnectivityAnalysis.summary() and can_tolerate_failures()."""

    def test_summary_basic(self):
        """Test summary with no articulation points or bridges."""
        analysis = ConnectivityAnalysis(
            is_connected=True,
            vertex_connectivity=3,
            edge_connectivity=3,
            articulation_points=[],
            bridges=[],
            k_connected=3,
            biconnected_components=[{"a", "b", "c", "d"}],
        )
        s = analysis.summary()
        assert "Connectivity Analysis" in s
        assert "Connected: True" in s
        assert "Vertex Connectivity" in s
        assert "k-Connected: 3" in s
        assert "Biconnected Components: 1" in s
        # No articulation points or bridges sections
        assert "Articulation Points" not in s
        assert "Bridge Edges" not in s

    def test_summary_with_articulation_points(self):
        """Test summary includes articulation points when present."""
        analysis = ConnectivityAnalysis(
            is_connected=True,
            vertex_connectivity=1,
            edge_connectivity=1,
            articulation_points=["b", "c"],
            bridges=[],
            k_connected=1,
            biconnected_components=[{"a", "b"}, {"b", "c"}, {"c", "d"}],
        )
        s = analysis.summary()
        assert "Articulation Points" in s
        assert "- b" in s
        assert "- c" in s

    def test_summary_with_bridges(self):
        """Test summary includes bridges when present."""
        analysis = ConnectivityAnalysis(
            is_connected=True,
            vertex_connectivity=1,
            edge_connectivity=1,
            articulation_points=[],
            bridges=[("a", "b"), ("c", "d")],
            k_connected=1,
            biconnected_components=[{"a", "b", "c", "d"}],
        )
        s = analysis.summary()
        assert "Bridge Edges" in s
        assert "a -- b" in s
        assert "c -- d" in s

    def test_can_tolerate_failures_true(self):
        """Test can_tolerate_failures returns True when connectivity > f."""
        analysis = ConnectivityAnalysis(
            is_connected=True,
            vertex_connectivity=3,
            edge_connectivity=3,
            articulation_points=[],
            bridges=[],
            k_connected=3,
            biconnected_components=[],
        )
        assert analysis.can_tolerate_failures(2)
        assert analysis.can_tolerate_failures(0)

    def test_can_tolerate_failures_false(self):
        """Test can_tolerate_failures returns False when connectivity <= f."""
        analysis = ConnectivityAnalysis(
            is_connected=True,
            vertex_connectivity=1,
            edge_connectivity=1,
            articulation_points=["b"],
            bridges=[],
            k_connected=1,
            biconnected_components=[],
        )
        assert not analysis.can_tolerate_failures(1)
        assert not analysis.can_tolerate_failures(2)


class TestFaultToleranceAnalysisSummary:
    """Tests for FaultToleranceAnalysis.summary()."""

    def test_summary_basic(self):
        """Test summary with critical nodes and no suggested redundancy."""
        analysis = FaultToleranceAnalysis(
            crash_tolerance=2,
            byzantine_tolerance=1,
            critical_nodes=["hub"],
            redundancy_score=0.75,
        )
        s = analysis.summary()
        assert "Fault Tolerance Analysis" in s
        assert "Crash Fault Tolerance: f < 2" in s
        assert "Byzantine Fault Tolerance: f < 1" in s
        assert "75.00%" in s
        assert "Critical Nodes" in s
        assert "- hub" in s
        assert "Suggested Redundancy" not in s

    def test_summary_with_suggested_redundancy(self):
        """Test summary includes suggested edges."""
        analysis = FaultToleranceAnalysis(
            crash_tolerance=1,
            byzantine_tolerance=0,
            critical_nodes=[],
            redundancy_score=0.33,
            suggested_redundancy=[("a", "c"), ("b", "d")],
        )
        s = analysis.summary()
        assert "Suggested Redundancy Edges" in s
        assert "a -- c" in s
        assert "b -- d" in s

    def test_summary_no_critical_nodes(self):
        """Test summary when no critical nodes."""
        analysis = FaultToleranceAnalysis(
            crash_tolerance=3,
            byzantine_tolerance=1,
            critical_nodes=[],
            redundancy_score=1.0,
        )
        s = analysis.summary()
        assert "Critical Nodes" not in s


class TestConnectivityAnalyzerEdgeCases:
    """Tests for ConnectivityAnalyzer edge cases."""

    def test_single_node_graph(self):
        """Test vertex connectivity of single-node graph is 0."""
        g = NetworkGraph()
        g.add_node("a")
        analyzer = ConnectivityAnalyzer(g)
        result = analyzer.analyze()
        assert result.vertex_connectivity == 0

    def test_disconnected_graph(self):
        """Test vertex connectivity of disconnected graph is 0."""
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        # No edges -> disconnected
        analyzer = ConnectivityAnalyzer(g)
        result = analyzer.analyze()
        assert not result.is_connected
        assert result.vertex_connectivity == 0

    def test_linear_graph_has_articulation_points(self):
        """Test linear graph has interior nodes as articulation points."""
        g = _make_linear_graph(4)
        analyzer = ConnectivityAnalyzer(g)
        result = analyzer.analyze()
        # n1 and n2 are articulation points in 0-1-2-3
        assert "n1" in result.articulation_points
        assert "n2" in result.articulation_points

    def test_linear_graph_has_bridges(self):
        """Test linear graph has bridge edges."""
        g = _make_linear_graph(3)
        analyzer = ConnectivityAnalyzer(g)
        result = analyzer.analyze()
        assert len(result.bridges) > 0

    def test_complete_graph_no_articulation_points(self):
        """Test complete graph has no articulation points."""
        g = _make_complete_graph(4)
        analyzer = ConnectivityAnalyzer(g)
        result = analyzer.analyze()
        assert result.articulation_points == []

    def test_biconnected_components_with_articulation(self):
        """Test biconnected components split at articulation points."""
        g = _make_linear_graph(4)
        analyzer = ConnectivityAnalyzer(g)
        result = analyzer.analyze()
        # Linear graph with articulation points should have multiple components
        assert len(result.biconnected_components) > 1

    def test_biconnected_components_complete_graph(self):
        """Test complete graph is a single biconnected component."""
        g = _make_complete_graph(4)
        analyzer = ConnectivityAnalyzer(g)
        result = analyzer.analyze()
        assert len(result.biconnected_components) == 1
        assert result.biconnected_components[0] == {"n0", "n1", "n2", "n3"}


class TestFaultToleranceAnalysis:
    """Tests for analyze_fault_tolerance."""

    def test_star_graph_suggested_redundancy(self):
        """Test star graph suggests edges between low-degree leaf nodes."""
        g = _make_star_graph(4)
        analyzer = ConnectivityAnalyzer(g)
        ft = analyzer.analyze_fault_tolerance()
        # Leaves have degree 1 (<2), so suggestions should appear
        assert len(ft.suggested_redundancy) > 0

    def test_complete_graph_no_suggested_redundancy(self):
        """Test complete graph has no suggested redundancy."""
        g = _make_complete_graph(4)
        analyzer = ConnectivityAnalyzer(g)
        ft = analyzer.analyze_fault_tolerance()
        assert ft.suggested_redundancy == []

    def test_fault_tolerance_scores(self):
        """Test crash/byzantine tolerance values."""
        g = _make_complete_graph(5)
        analyzer = ConnectivityAnalyzer(g)
        ft = analyzer.analyze_fault_tolerance()
        # n=5, vertex_connectivity=4
        # crash: min(4, (5-1)//2) = min(4, 2) = 2
        assert ft.crash_tolerance == 2
        # byzantine: min(4, (5-1)//3) = min(4, 1) = 1
        assert ft.byzantine_tolerance == 1

    def test_redundancy_score(self):
        """Test redundancy score for complete graph is 1.0."""
        g = _make_complete_graph(4)
        analyzer = ConnectivityAnalyzer(g)
        ft = analyzer.analyze_fault_tolerance()
        # 6 edges out of C(4,2)=6 possible
        assert ft.redundancy_score == 1.0


class TestVerifyConsensusRequirements:
    """Tests for verify_consensus_requirements."""

    def test_crash_consensus_insufficient_nodes(self):
        """Test crash consensus fails with too few nodes."""
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        # n=2, f=1: needs 2*1+1=3 nodes
        ok, msg = verify_consensus_requirements(g, "crash", f=1)
        assert not ok
        assert "2f+1" in msg

    def test_byzantine_consensus_insufficient_nodes(self):
        """Test byzantine consensus fails with too few nodes."""
        g = _make_complete_graph(3)
        # n=3, f=1: needs 3*1+1=4 nodes
        ok, msg = verify_consensus_requirements(g, "byzantine", f=1)
        assert not ok
        assert "3f+1" in msg

    def test_crash_consensus_low_connectivity(self):
        """Test crash consensus fails with insufficient connectivity."""
        # 4 nodes in a line: connectivity=1, need >1 for f=1
        g = _make_linear_graph(4)
        ok, msg = verify_consensus_requirements(g, "crash", f=1)
        assert not ok
        assert "vertex connectivity" in msg

    def test_crash_consensus_success(self):
        """Test crash consensus succeeds with enough nodes and connectivity."""
        g = _make_complete_graph(5)
        ok, msg = verify_consensus_requirements(g, "crash", f=1)
        assert ok
        assert "meets requirements" in msg

    def test_byzantine_consensus_success(self):
        """Test byzantine consensus succeeds."""
        g = _make_complete_graph(5)
        ok, msg = verify_consensus_requirements(g, "byzantine", f=1)
        assert ok
        assert "meets requirements" in msg
