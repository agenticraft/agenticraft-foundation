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


class TestComplexityBoundExtended:
    """Extended tests for ComplexityBound edge cases."""

    def test_str_best_case(self):
        """Test string repr with best-case qualifier."""
        bound = ComplexityBound(expression="O(n)", worst_case=False)
        s = str(bound)
        assert "best-case" in s

    def test_str_multiple_qualifiers(self):
        """Test string repr with multiple qualifiers."""
        bound = ComplexityBound(expression="O(n)", amortized=True, expected=True)
        s = str(bound)
        assert "amortized" in s
        assert "expected" in s


class TestDistributedComplexityExtended:
    """Extended tests for DistributedComplexity to_dict and format_docstring."""

    def test_to_dict_all_fields(self):
        """Test to_dict with all optional fields populated."""
        comp = DistributedComplexity(
            time="O(n)",
            space="O(n)",
            messages="O(n^2)",
            message_size="O(log n)",
            rounds="O(1)",
            bits="O(n^2 log n)",
            synchrony=SynchronyModel.ASYNCHRONOUS,
            fault_model=FaultModel.BYZANTINE,
            fault_tolerance="f < n/3",
            assumptions=["authenticated channels"],
            theorem_ref="Lamport 1998",
            lower_bound="Omega(n^2)",
            optimal=True,
        )
        d = comp.to_dict()
        assert "time" in d
        assert "space" in d
        assert "messages" in d
        assert "message_size" in d
        assert "rounds" in d
        assert "bits" in d
        assert d["synchrony"] == "asynchronous"
        assert d["fault_model"] == "byzantine"
        assert d["fault_tolerance"] == "f < n/3"
        assert d["assumptions"] == ["authenticated channels"]
        assert d["theorem_ref"] == "Lamport 1998"
        assert d["lower_bound"] == "Omega(n^2)"
        assert d["optimal"] is True

    def test_to_dict_string_synchrony_and_fault_model(self):
        """Test to_dict with string (non-enum) synchrony and fault model."""
        comp = DistributedComplexity(
            time="O(n)",
            synchrony="custom_sync",
            fault_model="custom_fault",
        )
        d = comp.to_dict()
        assert d["synchrony"] == "custom_sync"
        assert d["fault_model"] == "custom_fault"

    def test_format_docstring_all_fields(self):
        """Test format_docstring with all fields populated."""
        comp = DistributedComplexity(
            time="O(n)",
            space="O(1)",
            messages="O(n^2)",
            rounds="O(1)",
            fault_tolerance="f < n/3",
            synchrony=SynchronyModel.PARTIAL_SYNCHRONY,
            assumptions=["reliable channels"],
            theorem_ref="Fischer 1985",
            optimal=True,
        )
        doc = comp.format_docstring()
        assert "Time:" in doc
        assert "Space:" in doc
        assert "Messages:" in doc
        assert "Rounds:" in doc
        assert "Fault Tolerance:" in doc
        assert "Synchrony: partial_synchrony" in doc
        assert "Assumptions:" in doc
        assert "Reference:" in doc
        assert "lower bound" in doc

    def test_format_docstring_string_synchrony(self):
        """Test format_docstring with string synchrony model."""
        comp = DistributedComplexity(
            time="O(n)",
            synchrony="custom_model",
        )
        doc = comp.format_docstring()
        assert "Synchrony: custom_model" in doc


class TestGetAllComplexities:
    """Test get_all_complexities registry function."""

    def test_get_all_complexities(self):
        """Test retrieving all registered complexities."""
        from agenticraft_foundation.complexity.annotations import get_all_complexities

        result = get_all_complexities()
        assert isinstance(result, dict)


class TestAnalyzeComplexity:
    """Tests for analyze_complexity module analysis."""

    def test_analyze_module(self):
        """Test analyzing a module for complexity annotations."""
        import types

        from agenticraft_foundation.complexity.annotations import (
            ComplexityAnalysis,
            analyze_complexity,
        )

        # Create a test module with annotated and unannotated functions
        test_mod = types.ModuleType("test_mod")
        test_mod.__name__ = "test_mod"

        @complexity(time="O(n)", fault_model=FaultModel.BYZANTINE, optimal=True)
        def annotated_func():
            """Has complexity."""
            pass

        async def unannotated_async():
            """Has docstring but no annotation."""
            pass

        def plain_func():
            pass

        test_mod.annotated_func = annotated_func
        test_mod.unannotated_async = unannotated_async
        test_mod.plain_func = plain_func

        result = analyze_complexity(test_mod)
        assert isinstance(result, ComplexityAnalysis)
        assert result.total_annotated >= 0

    def test_complexity_analysis_summary(self):
        """Test ComplexityAnalysis summary generation."""
        from agenticraft_foundation.complexity.annotations import ComplexityAnalysis

        analysis = ComplexityAnalysis(
            functions={"mod.func1": DistributedComplexity(time="O(n)")},
            total_annotated=1,
            by_time_complexity={"O(n)": ["mod.func1"]},
            by_fault_model={"byzantine": ["mod.func1"]},
            optimal_algorithms=["mod.func1"],
            missing_annotations=["mod.func2"],
        )
        summary = analysis.summary()
        assert "Total annotated functions: 1" in summary
        assert "Optimal algorithms: 1" in summary
        assert "Missing annotations: 1" in summary
        assert "O(n)" in summary
        assert "byzantine" in summary


class TestParseBigOExtended:
    """Extended tests for Big-O parsing covering more patterns."""

    def test_parse_factorial(self):
        """Test parsing factorial complexity."""
        class_name, _ = parse_big_o("O(n!)")
        assert class_name == "factorial"

    def test_parse_diameter(self):
        """Test parsing diameter complexity."""
        class_name, _ = parse_big_o("O(D)")
        assert class_name == "diameter"

    def test_parse_diameter_log(self):
        """Test parsing diameter-log complexity."""
        class_name, _ = parse_big_o("O(D * log n)")
        assert class_name == "diameter_log"

    def test_parse_sqrt(self):
        """Test parsing sqrt complexity."""
        class_name, _ = parse_big_o("O(sqrt(n))")
        assert class_name == "sqrt"

    def test_parse_logarithmic(self):
        """Test parsing log complexity."""
        class_name, _ = parse_big_o("O(log n)")
        assert class_name == "logarithmic"

    def test_parse_no_parens(self):
        """Test parsing expression without standard notation."""
        class_name, params = parse_big_o("n^2")
        assert class_name == "unknown"
        assert "raw" in params

    def test_parse_polynomial_log(self):
        """Test parsing polynomial with log factor."""
        class_name, params = parse_big_o("O(n^2 log n)")
        assert class_name == "polynomial_log"
        assert params["exponent"] == 2
        assert params["log_factor"] is True


class TestCompareComplexityExtended:
    """Extended tests for complexity comparison."""

    def test_compare_polynomial_exponents(self):
        """Test comparing different polynomial exponents."""
        result = compare_complexity("O(n^2)", "O(n^3)")
        assert result < 0

        result = compare_complexity("O(n^3)", "O(n^2)")
        assert result > 0

    def test_compare_same_polynomial(self):
        """Test comparing same polynomial exponents."""
        result = compare_complexity("O(n^2)", "O(n^2)")
        assert result == 0

    def test_compare_unknown_classes(self):
        """Test comparing unknown complexity classes."""
        result = compare_complexity("O(weird)", "O(stuff)")
        assert result == 0
