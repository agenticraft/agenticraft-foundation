"""Tests for complexity annotations system."""

from agenticraft_foundation.complexity import (
    CONSENSUS_BOUNDS,
    ComplexityBound,
    ComplexityComparison,
    DistributedComplexity,
    FaultModel,
    SynchronyModel,
    check_optimality,
    compare_algorithms,
    compare_complexity,
    complexity,
    consensus_complexity,
    get_complexity,
    gossip_complexity,
    parse_big_o,
    validate_fault_tolerance,
)


class TestComplexityBound:
    """Tests for ComplexityBound dataclass."""

    def test_basic_bound(self):
        """Test basic complexity bound creation."""
        bound = ComplexityBound(expression="O(n log n)")
        assert bound.expression == "O(n log n)"
        assert not bound.tight
        assert bound.worst_case

    def test_tight_bound(self):
        """Test tight bound creation."""
        bound = ComplexityBound(expression="Theta(n^2)", tight=True)
        assert bound.tight

    def test_parse_big_o(self):
        """Test parsing Big-O expression."""
        bound = ComplexityBound.parse("O(n^2)")
        assert bound.expression == "O(n^2)"
        assert not bound.tight

        bound = ComplexityBound.parse("Theta(n)")
        assert bound.tight

    def test_str_representation(self):
        """Test string representation."""
        bound = ComplexityBound(expression="O(n)", amortized=True)
        assert "amortized" in str(bound)

        bound = ComplexityBound(expression="O(n)", expected=True)
        assert "expected" in str(bound)


class TestDistributedComplexity:
    """Tests for DistributedComplexity dataclass."""

    def test_basic_complexity(self):
        """Test basic distributed complexity."""
        comp = DistributedComplexity(
            time="O(n^2)",
            messages="O(n^2)",
            rounds="O(1)",
        )
        assert comp.time.expression == "O(n^2)"
        assert comp.messages.expression == "O(n^2)"

    def test_with_fault_model(self):
        """Test complexity with fault model."""
        comp = DistributedComplexity(
            time="O(n)",
            fault_model=FaultModel.BYZANTINE,
            fault_tolerance="f < n/3",
        )
        assert comp.fault_model == FaultModel.BYZANTINE
        assert comp.fault_tolerance == "f < n/3"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        comp = DistributedComplexity(
            time="O(n)",
            messages="O(n)",
            synchrony=SynchronyModel.PARTIAL_SYNCHRONY,
        )
        d = comp.to_dict()
        assert "time" in d
        assert "messages" in d
        assert d["synchrony"] == "partial_synchrony"

    def test_format_docstring(self):
        """Test docstring formatting."""
        comp = DistributedComplexity(
            time="O(n)",
            fault_tolerance="f < n/2",
            assumptions=["partial synchrony"],
        )
        doc = comp.format_docstring()
        assert "Time: O(n)" in doc
        assert "Fault Tolerance: f < n/2" in doc


class TestComplexityDecorator:
    """Tests for @complexity decorator."""

    def test_basic_decorator(self):
        """Test basic complexity decorator."""

        @complexity(time="O(n)", messages="O(n)")
        def my_algorithm():
            pass

        comp = get_complexity(my_algorithm)
        assert comp is not None
        assert comp.time.expression == "O(n)"

    def test_consensus_complexity(self):
        """Test consensus complexity preset."""

        @consensus_complexity("pbft")
        def pbft_protocol():
            pass

        comp = get_complexity(pbft_protocol)
        assert comp is not None
        assert comp.fault_model == FaultModel.BYZANTINE
        assert "n/3" in comp.fault_tolerance

    def test_gossip_complexity(self):
        """Test gossip complexity preset."""

        @gossip_complexity(fanout=3)
        def gossip_protocol():
            pass

        comp = get_complexity(gossip_protocol)
        assert comp is not None
        assert "log n" in comp.time.expression

    def test_docstring_update(self):
        """Test that decorator updates docstring."""

        @complexity(time="O(n)")
        def documented():
            """Original docstring."""
            pass

        assert "Original docstring" in documented.__doc__
        assert "Time: O(n)" in documented.__doc__


class TestParseBigO:
    """Tests for Big-O parsing."""

    def test_parse_constant(self):
        """Test parsing constant complexity."""
        class_name, _ = parse_big_o("O(1)")
        assert class_name == "constant"

    def test_parse_linear(self):
        """Test parsing linear complexity."""
        class_name, _ = parse_big_o("O(n)")
        assert class_name == "linear"

    def test_parse_linearithmic(self):
        """Test parsing linearithmic complexity."""
        class_name, _ = parse_big_o("O(n log n)")
        assert class_name == "linearithmic"

    def test_parse_polynomial(self):
        """Test parsing polynomial complexity."""
        class_name, params = parse_big_o("O(n^2)")
        assert class_name == "polynomial"
        assert params["exponent"] == 2

    def test_parse_exponential(self):
        """Test parsing exponential complexity."""
        class_name, _ = parse_big_o("O(2^n)")
        assert class_name == "exponential"

    def test_parse_unknown(self):
        """Test parsing unknown expression."""
        class_name, params = parse_big_o("O(weird)")
        assert class_name == "unknown"
        assert "raw" in params


class TestCompareComplexity:
    """Tests for complexity comparison."""

    def test_compare_same(self):
        """Test comparing same complexity."""
        result = compare_complexity("O(n)", "O(n)")
        assert result == 0

    def test_compare_different(self):
        """Test comparing different complexities."""
        result = compare_complexity("O(n)", "O(n^2)")
        assert result < 0  # O(n) is less than O(n^2)

        result = compare_complexity("O(n^2)", "O(n)")
        assert result > 0  # O(n^2) is greater than O(n)

    def test_compare_classes(self):
        """Test comparing different complexity classes."""
        result = compare_complexity("O(log n)", "O(n)")
        assert result < 0

        result = compare_complexity("O(n)", "O(2^n)")
        assert result < 0


class TestValidateFaultTolerance:
    """Tests for fault tolerance validation."""

    def test_valid_byzantine(self):
        """Test valid Byzantine configuration."""
        is_valid, _ = validate_fault_tolerance(n=4, f=1, fault_model="byzantine")
        assert is_valid

    def test_invalid_byzantine(self):
        """Test invalid Byzantine configuration."""
        is_valid, msg = validate_fault_tolerance(n=3, f=1, fault_model="byzantine")
        assert not is_valid
        assert "3f+1" in msg

    def test_valid_crash(self):
        """Test valid crash configuration."""
        is_valid, _ = validate_fault_tolerance(n=3, f=1, fault_model="crash")
        assert is_valid

    def test_invalid_crash(self):
        """Test invalid crash configuration."""
        is_valid, msg = validate_fault_tolerance(n=2, f=1, fault_model="crash")
        assert not is_valid
        assert "2f+1" in msg


class TestCheckOptimality:
    """Tests for optimality checking."""

    def test_check_known_problem(self):
        """Test checking optimality for known problem."""
        result = check_optimality("O(n^2)", "byzantine consensus", "messages")
        assert result is not None
        # n^2 matches lower bound for Byzantine consensus messages
        assert result.is_optimal or result.lower_bound != "unknown"

    def test_check_unknown_problem(self):
        """Test checking optimality for unknown problem."""
        result = check_optimality("O(n)", "unknown_problem", "time")
        assert result is not None
        assert result.lower_bound == "unknown"


class TestCompareAlgorithms:
    """Tests for algorithm comparison."""

    def test_compare_consensus_algorithms(self):
        """Test comparing consensus algorithms."""
        algorithms = {
            "PBFT": {"messages": "O(n^2)", "rounds": "O(1)"},
            "HotStuff": {"messages": "O(n)", "rounds": "O(1)"},
            "Raft": {"messages": "O(n)", "rounds": "O(1)"},
        }

        result = compare_algorithms("consensus", algorithms)

        assert isinstance(result, ComplexityComparison)
        assert "messages" in result.optimal_for_metric
        # HotStuff and Raft should be optimal for messages
        assert (
            "HotStuff" in result.optimal_for_metric["messages"]
            or "Raft" in result.optimal_for_metric["messages"]
        )

    def test_pareto_optimal(self):
        """Test finding Pareto optimal algorithms."""
        algorithms = {
            "Algo1": {"time": "O(n)", "space": "O(n^2)"},
            "Algo2": {"time": "O(n^2)", "space": "O(n)"},
            "Algo3": {"time": "O(n^2)", "space": "O(n^2)"},  # Dominated
        }

        result = compare_algorithms("test", algorithms)

        # Algo3 should not be Pareto optimal (dominated by both)
        assert "Algo1" in result.pareto_optimal
        assert "Algo2" in result.pareto_optimal


class TestConsensusBounds:
    """Tests for consensus bounds data."""

    def test_bounds_exist(self):
        """Test that consensus bounds are defined."""
        assert "synchronous_consensus" in CONSENSUS_BOUNDS
        assert "async_consensus_impossibility" in CONSENSUS_BOUNDS
        assert "byzantine_consensus_nodes" in CONSENSUS_BOUNDS

    def test_bound_structure(self):
        """Test bound structure."""
        bound = CONSENSUS_BOUNDS["synchronous_consensus"]
        assert bound.problem
        assert bound.metric
        assert bound.expression
        assert bound.source
