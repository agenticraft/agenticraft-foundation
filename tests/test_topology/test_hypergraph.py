"""Tests for hypergraph topology model.

Covers:
- HypergraphNetwork construction
- Hyperedge management
- Connectivity and neighbor queries
- Incidence matrix and Laplacian computation
- Spectral analysis
- Factory functions (from_graph, clique_expansion)
- Group coordination analysis
"""

from __future__ import annotations

import pytest

from agenticraft_foundation.topology.hypergraph import (
    Hyperedge,
    HypergraphNetwork,
)


@pytest.fixture
def simple_hypergraph() -> HypergraphNetwork:
    """Simple hypergraph with 2 overlapping hyperedges."""
    hg = HypergraphNetwork()
    hg.add_node("a")
    hg.add_node("b")
    hg.add_node("c")
    hg.add_node("d")
    hg.add_hyperedge("e1", {"a", "b", "c"})
    hg.add_hyperedge("e2", {"b", "c", "d"})
    return hg


class TestHyperedge:
    def test_degree(self):
        edge = Hyperedge(edge_id="e1", nodes={"a", "b", "c"})
        assert edge.degree == 3

    def test_contains(self):
        edge = Hyperedge(edge_id="e1", nodes={"a", "b"})
        assert edge.contains("a")
        assert not edge.contains("c")

    def test_weight_default(self):
        edge = Hyperedge(edge_id="e1", nodes={"a"})
        assert edge.weight == 1.0


class TestHypergraphNetwork:
    def test_add_node(self):
        hg = HypergraphNetwork()
        hg.add_node("a", {"role": "agent"})
        assert "a" in hg.nodes
        assert hg.nodes["a"]["role"] == "agent"

    def test_add_hyperedge(self):
        hg = HypergraphNetwork()
        hg.add_node("a")
        hg.add_node("b")
        edge = hg.add_hyperedge("e1", {"a", "b"}, weight=2.0)
        assert edge.weight == 2.0
        assert "e1" in hg.hyperedges

    def test_auto_add_nodes(self):
        hg = HypergraphNetwork()
        hg.add_hyperedge("e1", {"x", "y", "z"})
        assert "x" in hg.nodes
        assert "y" in hg.nodes
        assert "z" in hg.nodes

    def test_remove_hyperedge(self, simple_hypergraph: HypergraphNetwork):
        simple_hypergraph.remove_hyperedge("e1")
        assert "e1" not in simple_hypergraph.hyperedges
        assert "e2" in simple_hypergraph.hyperedges

    def test_node_degree(self, simple_hypergraph: HypergraphNetwork):
        # b is in both e1 and e2
        assert simple_hypergraph.node_degree("b") == 2
        # a is only in e1
        assert simple_hypergraph.node_degree("a") == 1

    def test_node_weighted_degree(self, simple_hypergraph: HypergraphNetwork):
        assert simple_hypergraph.node_weighted_degree("b") == 2.0

    def test_incident_edges(self, simple_hypergraph: HypergraphNetwork):
        edges = simple_hypergraph.incident_edges("b")
        assert len(edges) == 2

    def test_neighbors(self, simple_hypergraph: HypergraphNetwork):
        neighbors = simple_hypergraph.neighbors("b")
        assert neighbors == {"a", "c", "d"}

    def test_neighbors_excludes_self(self, simple_hypergraph: HypergraphNetwork):
        neighbors = simple_hypergraph.neighbors("a")
        assert "a" not in neighbors

    def test_is_connected(self, simple_hypergraph: HypergraphNetwork):
        assert simple_hypergraph.is_connected()

    def test_disconnected_graph(self):
        hg = HypergraphNetwork()
        hg.add_node("a")
        hg.add_node("b")
        hg.add_node("c")
        hg.add_hyperedge("e1", {"a", "b"})
        # c is isolated
        assert not hg.is_connected()

    def test_empty_graph_connected(self):
        hg = HypergraphNetwork()
        assert hg.is_connected()


class TestHypergraphMatrices:
    def test_incidence_matrix_shape(self, simple_hypergraph: HypergraphNetwork):
        node_ids, edge_ids, h_mat = simple_hypergraph.incidence_matrix()
        assert len(node_ids) == 4
        assert len(edge_ids) == 2
        assert len(h_mat) == 4
        assert len(h_mat[0]) == 2

    def test_incidence_matrix_values(self):
        hg = HypergraphNetwork()
        hg.add_hyperedge("e1", {"a", "b"})
        node_ids, edge_ids, h_mat = hg.incidence_matrix()
        # Both a and b should be 1 in the single edge column
        for row in h_mat:
            assert row[0] == 1.0

    def test_laplacian_matrix_symmetric(self, simple_hypergraph: HypergraphNetwork):
        node_ids, lap = simple_hypergraph.laplacian_matrix()
        n = len(node_ids)
        for i in range(n):
            for j in range(n):
                assert abs(lap[i][j] - lap[j][i]) < 1e-10

    def test_laplacian_row_sum_zero(self, simple_hypergraph: HypergraphNetwork):
        node_ids, lap = simple_hypergraph.laplacian_matrix()
        n = len(node_ids)
        for i in range(n):
            row_sum = sum(lap[i][j] for j in range(n))
            assert abs(row_sum) < 1e-10


class TestHypergraphAnalysis:
    def test_analyze_basic(self, simple_hypergraph: HypergraphNetwork):
        analysis = simple_hypergraph.analyze()
        assert analysis.num_nodes == 4
        assert analysis.num_hyperedges == 2
        assert analysis.avg_hyperedge_degree == 3.0
        assert analysis.max_hyperedge_degree == 3

    def test_connected_has_positive_lambda2(self, simple_hypergraph: HypergraphNetwork):
        analysis = simple_hypergraph.analyze()
        assert analysis.algebraic_connectivity > 0
        assert analysis.is_connected

    def test_consensus_bound_finite(self, simple_hypergraph: HypergraphNetwork):
        analysis = simple_hypergraph.analyze()
        assert analysis.consensus_bound < float("inf")

    def test_empty_graph_analysis(self):
        hg = HypergraphNetwork()
        analysis = hg.analyze()
        assert analysis.num_nodes == 0
        assert analysis.is_connected

    def test_single_node_analysis(self):
        hg = HypergraphNetwork()
        hg.add_node("a")
        analysis = hg.analyze()
        assert analysis.num_nodes == 1
        assert analysis.is_connected


class TestHypergraphFactories:
    def test_from_graph(self):
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        hg = HypergraphNetwork.from_graph(edges)
        assert len(hg.nodes) == 3
        assert len(hg.hyperedges) == 3
        for edge in hg.hyperedges.values():
            assert edge.degree == 2

    def test_from_graph_with_weights(self):
        edges = [("a", "b")]
        weights = {("a", "b"): 2.5}
        hg = HypergraphNetwork.from_graph(edges, weights)
        edge = list(hg.hyperedges.values())[0]
        assert edge.weight == 2.5

    def test_clique_expansion(self):
        groups = [{"a", "b", "c"}, {"c", "d", "e"}]
        hg = HypergraphNetwork.clique_expansion(groups)
        assert len(hg.nodes) == 5
        assert len(hg.hyperedges) == 2

    def test_clique_expansion_with_weights(self):
        groups = [{"a", "b"}, {"c", "d"}]
        weights = [1.0, 3.0]
        hg = HypergraphNetwork.clique_expansion(groups, weights)
        edges = list(hg.hyperedges.values())
        assert edges[0].weight == 1.0
        assert edges[1].weight == 3.0


class TestGroupCoordination:
    def test_analyze_group_coordination(self, simple_hypergraph: HypergraphNetwork):
        result = simple_hypergraph.analyze_group_coordination()
        assert result["num_nodes"] == 4
        assert result["num_hyperedges"] == 2
        assert result["participation_ratio"] == 1.0
        assert result["avg_group_size"] == 3.0
        assert result["coverage"] > 0
