"""Complexity annotations for algorithm documentation and analysis.

This module provides decorators and utilities for documenting and analyzing
algorithm complexity, following formal verification principles from distributed
systems research.

Example:
    @complexity(
        time="O(D * log n)",
        space="O(n)",
        messages="Theta(n^2)",
        rounds="O(D)",
        assumptions=["partial synchrony", "f < n/3 Byzantine"],
        theorem_ref="Castro & Liskov 1999"
    )
    async def propose(self, value, context):
        ...
"""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class ComplexityClass(str, Enum):
    """Standard complexity classes."""

    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n^2)"
    CUBIC = "O(n^3)"
    POLYNOMIAL = "O(n^k)"
    EXPONENTIAL = "O(2^n)"
    FACTORIAL = "O(n!)"

    # Distributed system specific
    DIAMETER = "O(D)"
    DIAMETER_LOG = "O(D * log n)"
    SQRT_N = "O(sqrt(n))"
    N_SQUARED = "Theta(n^2)"


class SynchronyModel(str, Enum):
    """Synchrony assumptions for distributed algorithms."""

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    PARTIAL_SYNCHRONY = "partial_synchrony"
    EVENTUAL_SYNCHRONY = "eventual_synchrony"


class FaultModel(str, Enum):
    """Fault tolerance models.

    Classical distributed systems fault models:
    - CRASH_STOP: Process halts, never recovers (f < n/2)
    - CRASH_RECOVERY: Process halts, may recover with stable storage (f < n/2)
    - BYZANTINE: Arbitrary behavior including malicious (f < n/3)
    - OMISSION: Process fails to send or receive messages (f < n/2)
    - AUTHENTICATED_BYZANTINE: Byzantine with digital signatures (f < n/3)

    LLM-specific fault models (map to classical analogs):
    - HALLUCINATION: Generates plausible but incorrect output
      (analog: Byzantine — arbitrary incorrect responses)
    - PROMPT_INJECTION: Adversarial input manipulates behavior
      (analog: Byzantine — attacker-controlled behavior)
    - NON_DETERMINISM: Same input yields different outputs across calls
      (analog: Omission — unreliable message content)
    - CONTEXT_OVERFLOW: Input exceeds context window, truncating information
      (analog: Crash-Recovery — partial state loss, recoverable with chunking)
    """

    # Classical fault models
    CRASH_STOP = "crash_stop"  # f < n/2
    CRASH_RECOVERY = "crash_recovery"  # f < n/2
    BYZANTINE = "byzantine"  # f < n/3
    OMISSION = "omission"  # f < n/2
    AUTHENTICATED_BYZANTINE = "authenticated_byzantine"  # f < n/3

    # LLM-specific fault models
    HALLUCINATION = "hallucination"  # analog: Byzantine
    PROMPT_INJECTION = "prompt_injection"  # analog: Byzantine
    NON_DETERMINISM = "non_determinism"  # analog: Omission
    CONTEXT_OVERFLOW = "context_overflow"  # analog: Crash-Recovery


@dataclass(frozen=True)
class ComplexityBound:
    """Represents a complexity bound with optional proof reference.

    Attributes:
        expression: The complexity expression (e.g., "O(n log n)")
        tight: Whether this is a tight bound (Theta vs O)
        amortized: Whether this is amortized complexity
        expected: Whether this is expected (average) complexity
        worst_case: Whether this is worst-case complexity
        proof_ref: Reference to proof (paper, theorem number)
        notes: Additional notes about the bound
    """

    expression: str
    tight: bool = False
    amortized: bool = False
    expected: bool = False
    worst_case: bool = True
    proof_ref: str | None = None
    notes: str | None = None

    def __str__(self) -> str:
        """Format the complexity bound as a string."""
        qualifiers = []
        if self.amortized:
            qualifiers.append("amortized")
        if self.expected:
            qualifiers.append("expected")
        if not self.worst_case:
            qualifiers.append("best-case")

        qualifier_str = f" ({', '.join(qualifiers)})" if qualifiers else ""
        return f"{self.expression}{qualifier_str}"

    @classmethod
    def parse(cls, expr: str) -> ComplexityBound:
        """Parse a complexity expression string.

        Args:
            expr: Complexity expression like "O(n log n)" or "Theta(n^2)"

        Returns:
            ComplexityBound instance
        """
        tight = expr.startswith("Theta") or expr.startswith("Θ")
        return cls(expression=expr, tight=tight)


@dataclass
class DistributedComplexity:
    """Complexity metrics specific to distributed algorithms.

    Attributes:
        time: Time complexity (sequential operations)
        space: Space complexity per node
        messages: Total message complexity
        message_size: Size of individual messages
        rounds: Round/communication complexity
        bits: Total bit complexity
        synchrony: Required synchrony model
        fault_model: Assumed fault model
        fault_tolerance: Maximum faults tolerated (e.g., "f < n/3")
        assumptions: List of additional assumptions
        theorem_ref: Reference to theorem/paper
        lower_bound: Known lower bound for the problem
        optimal: Whether this achieves the lower bound
    """

    time: ComplexityBound | str | None = None
    space: ComplexityBound | str | None = None
    messages: ComplexityBound | str | None = None
    message_size: ComplexityBound | str | None = None
    rounds: ComplexityBound | str | None = None
    bits: ComplexityBound | str | None = None
    synchrony: SynchronyModel | str | None = None
    fault_model: FaultModel | str | None = None
    fault_tolerance: str | None = None
    assumptions: list[str] = field(default_factory=list)
    theorem_ref: str | None = None
    lower_bound: str | None = None
    optimal: bool = False

    def __post_init__(self) -> None:
        """Convert string bounds to ComplexityBound objects."""
        for attr in ["time", "space", "messages", "message_size", "rounds", "bits"]:
            value = getattr(self, attr)
            if isinstance(value, str):
                object.__setattr__(self, attr, ComplexityBound.parse(value))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {}
        for attr in [
            "time",
            "space",
            "messages",
            "message_size",
            "rounds",
            "bits",
        ]:
            value = getattr(self, attr)
            if value is not None:
                result[attr] = str(value)

        if self.synchrony:
            result["synchrony"] = (
                self.synchrony.value
                if isinstance(self.synchrony, SynchronyModel)
                else self.synchrony
            )
        if self.fault_model:
            result["fault_model"] = (
                self.fault_model.value
                if isinstance(self.fault_model, FaultModel)
                else self.fault_model
            )
        if self.fault_tolerance:
            result["fault_tolerance"] = self.fault_tolerance
        if self.assumptions:
            result["assumptions"] = self.assumptions
        if self.theorem_ref:
            result["theorem_ref"] = self.theorem_ref
        if self.lower_bound:
            result["lower_bound"] = self.lower_bound
        if self.optimal:
            result["optimal"] = True

        return result

    def format_docstring(self) -> str:
        """Format complexity information for docstring inclusion."""
        lines = ["Complexity:"]

        if self.time:
            lines.append(f"    Time: {self.time}")
        if self.space:
            lines.append(f"    Space: {self.space}")
        if self.messages:
            lines.append(f"    Messages: {self.messages}")
        if self.rounds:
            lines.append(f"    Rounds: {self.rounds}")
        if self.fault_tolerance:
            lines.append(f"    Fault Tolerance: {self.fault_tolerance}")
        if self.synchrony:
            sync_str = (
                self.synchrony.value
                if isinstance(self.synchrony, SynchronyModel)
                else self.synchrony
            )
            lines.append(f"    Synchrony: {sync_str}")
        if self.assumptions:
            lines.append(f"    Assumptions: {', '.join(self.assumptions)}")
        if self.theorem_ref:
            lines.append(f"    Reference: {self.theorem_ref}")
        if self.optimal:
            lines.append("    Note: This achieves the known lower bound")

        return "\n".join(lines)


# Registry to store complexity annotations
_complexity_registry: dict[str, DistributedComplexity] = {}


def get_complexity(func: Callable[..., Any]) -> DistributedComplexity | None:
    """Get complexity annotation for a function.

    Args:
        func: The annotated function

    Returns:
        DistributedComplexity if annotated, None otherwise
    """
    key = f"{func.__module__}.{func.__qualname__}"
    return _complexity_registry.get(key)


def get_all_complexities() -> dict[str, DistributedComplexity]:
    """Get all registered complexity annotations.

    Returns:
        Dictionary mapping function keys to complexity info
    """
    return _complexity_registry.copy()


def complexity(
    time: str | ComplexityBound | None = None,
    space: str | ComplexityBound | None = None,
    messages: str | ComplexityBound | None = None,
    message_size: str | ComplexityBound | None = None,
    rounds: str | ComplexityBound | None = None,
    bits: str | ComplexityBound | None = None,
    synchrony: SynchronyModel | str | None = None,
    fault_model: FaultModel | str | None = None,
    fault_tolerance: str | None = None,
    assumptions: list[str] | None = None,
    theorem_ref: str | None = None,
    lower_bound: str | None = None,
    optimal: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to annotate function with complexity information.

    This decorator documents algorithm complexity for formal analysis
    and documentation purposes. It stores the complexity information
    in a registry for later retrieval.

    Args:
        time: Time complexity (e.g., "O(n log n)")
        space: Space complexity per node
        messages: Total message complexity
        message_size: Size of individual messages
        rounds: Round/communication complexity
        bits: Total bit complexity
        synchrony: Required synchrony model
        fault_model: Assumed fault model
        fault_tolerance: Maximum faults tolerated
        assumptions: Additional assumptions
        theorem_ref: Reference to theorem/paper
        lower_bound: Known lower bound
        optimal: Whether this achieves lower bound

    Returns:
        Decorated function with complexity metadata

    Example:
        @complexity(
            time="O(n^2)",
            messages="O(n^2)",
            rounds="O(1)",
            fault_model=FaultModel.BYZANTINE,
            fault_tolerance="f < n/3",
            theorem_ref="Castro & Liskov 1999, PBFT"
        )
        async def pbft_prepare(self, request):
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Create complexity info
        complexity_info = DistributedComplexity(
            time=time,
            space=space,
            messages=messages,
            message_size=message_size,
            rounds=rounds,
            bits=bits,
            synchrony=synchrony,
            fault_model=fault_model,
            fault_tolerance=fault_tolerance,
            assumptions=assumptions or [],
            theorem_ref=theorem_ref,
            lower_bound=lower_bound,
            optimal=optimal,
        )

        # Register complexity
        key = f"{func.__module__}.{func.__qualname__}"
        _complexity_registry[key] = complexity_info

        # Store on function for easy access
        func.__complexity__ = complexity_info  # type: ignore[attr-defined]

        # Update docstring
        original_doc = func.__doc__ or ""
        complexity_doc = complexity_info.format_docstring()

        if original_doc:
            func.__doc__ = f"{original_doc}\n\n{complexity_doc}"
        else:
            func.__doc__ = complexity_doc

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        # Preserve complexity attribute on wrapper
        wrapper.__complexity__ = complexity_info  # type: ignore[attr-defined]

        return wrapper

    return decorator


def consensus_complexity(
    algorithm: str,
    fault_tolerance: str = "f < n/3",
    synchrony: SynchronyModel = SynchronyModel.PARTIAL_SYNCHRONY,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Specialized complexity decorator for consensus algorithms.

    Provides preset complexity bounds for common consensus algorithms.

    Args:
        algorithm: Name of consensus algorithm (raft, pbft, paxos, etc.)
        fault_tolerance: Fault tolerance bound
        synchrony: Synchrony model

    Returns:
        Complexity decorator with preset values
    """
    presets: dict[str, dict[str, Any]] = {
        "raft": {
            "time": "O(n)",
            "messages": "O(n)",
            "rounds": "O(1)",
            "fault_model": FaultModel.CRASH_STOP,
            "fault_tolerance": "f < n/2",
            "theorem_ref": "Ongaro & Ousterhout 2014",
        },
        "pbft": {
            "time": "O(n^2)",
            "messages": "O(n^2)",
            "rounds": "O(1)",
            "fault_model": FaultModel.BYZANTINE,
            "fault_tolerance": "f < n/3",
            "theorem_ref": "Castro & Liskov 1999",
        },
        "paxos": {
            "time": "O(n)",
            "messages": "O(n)",
            "rounds": "O(1)",
            "fault_model": FaultModel.CRASH_STOP,
            "fault_tolerance": "f < n/2",
            "theorem_ref": "Lamport 1998",
        },
        "hotstuff": {
            "time": "O(n)",
            "messages": "O(n)",
            "rounds": "O(1)",
            "fault_model": FaultModel.BYZANTINE,
            "fault_tolerance": "f < n/3",
            "theorem_ref": "Yin et al. 2019",
            "optimal": True,
        },
        "ben-or": {
            "time": "O(2^n)",
            "messages": "O(n^2)",
            "rounds": "O(2^n)",
            "fault_model": FaultModel.BYZANTINE,
            "fault_tolerance": "f < n/5",
            "synchrony": SynchronyModel.ASYNCHRONOUS,
            "theorem_ref": "Ben-Or 1983",
        },
    }

    preset = presets.get(algorithm.lower(), {})

    return complexity(
        time=preset.get("time"),
        messages=preset.get("messages"),
        rounds=preset.get("rounds"),
        fault_model=preset.get("fault_model", FaultModel.BYZANTINE),
        fault_tolerance=preset.get("fault_tolerance", fault_tolerance),
        synchrony=preset.get("synchrony", synchrony),
        theorem_ref=preset.get("theorem_ref"),
        optimal=preset.get("optimal", False),
    )


def gossip_complexity(
    fanout: int = 3,
    rounds: str = "O(log n)",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Specialized complexity decorator for gossip protocols.

    Args:
        fanout: Number of nodes to contact per round
        rounds: Expected rounds to convergence

    Returns:
        Complexity decorator with gossip-specific values
    """
    return complexity(
        time=rounds,
        messages=f"O(n * {fanout} * log n)",
        rounds=rounds,
        synchrony=SynchronyModel.ASYNCHRONOUS,
        fault_model=FaultModel.CRASH_STOP,
        assumptions=[f"fanout = {fanout}", "exponential information spreading"],
        theorem_ref="Epidemic algorithms (Demers et al. 1987)",
    )


@dataclass
class ComplexityAnalysis:
    """Results of complexity analysis on a codebase."""

    functions: dict[str, DistributedComplexity]
    total_annotated: int
    by_time_complexity: dict[str, list[str]]
    by_fault_model: dict[str, list[str]]
    optimal_algorithms: list[str]
    missing_annotations: list[str]

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "Complexity Analysis Summary",
            "=" * 40,
            f"Total annotated functions: {self.total_annotated}",
            f"Optimal algorithms: {len(self.optimal_algorithms)}",
            f"Missing annotations: {len(self.missing_annotations)}",
            "",
            "By Time Complexity:",
        ]

        for complexity_class, funcs in sorted(self.by_time_complexity.items()):
            lines.append(f"  {complexity_class}: {len(funcs)}")

        lines.append("")
        lines.append("By Fault Model:")
        for model, funcs in sorted(self.by_fault_model.items()):
            lines.append(f"  {model}: {len(funcs)}")

        return "\n".join(lines)


def analyze_complexity(module: Any) -> ComplexityAnalysis:
    """Analyze complexity annotations in a module.

    Args:
        module: Python module to analyze

    Returns:
        ComplexityAnalysis with results
    """
    functions: dict[str, DistributedComplexity] = {}
    by_time: dict[str, list[str]] = {}
    by_fault: dict[str, list[str]] = {}
    optimal: list[str] = []
    missing: list[str] = []

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            key = f"{module.__name__}.{name}"
            comp = get_complexity(obj)

            if comp:
                functions[key] = comp

                # Categorize by time complexity
                if comp.time:
                    time_str = str(comp.time)
                    by_time.setdefault(time_str, []).append(key)

                # Categorize by fault model
                if comp.fault_model:
                    fault_str = (
                        comp.fault_model.value
                        if isinstance(comp.fault_model, FaultModel)
                        else comp.fault_model
                    )
                    by_fault.setdefault(fault_str, []).append(key)

                if comp.optimal:
                    optimal.append(key)
            else:
                # Check if it looks like an algorithm (async, has docstring)
                if inspect.iscoroutinefunction(obj) and obj.__doc__:
                    missing.append(key)

    return ComplexityAnalysis(
        functions=functions,
        total_annotated=len(functions),
        by_time_complexity=by_time,
        by_fault_model=by_fault,
        optimal_algorithms=optimal,
        missing_annotations=missing,
    )


def parse_big_o(expression: str) -> tuple[str, dict[str, Any]]:
    """Parse a Big-O expression into components.

    Args:
        expression: Big-O expression like "O(n^2 log n)"

    Returns:
        Tuple of (base_class, parameters)

    Example:
        >>> parse_big_o("O(n^2 log n)")
        ('polynomial_log', {'exponent': 2, 'log_factor': True})
    """
    # Normalize expression
    expr = expression.strip()

    # Extract inner part
    match = re.match(r"[OΘΩoθω]\s*\((.*)\)", expr)
    if not match:
        return ("unknown", {"raw": expr})

    inner = match.group(1).strip()

    # Common patterns
    patterns = [
        (r"^1$", "constant", {}),
        (r"^log\s*n$", "logarithmic", {}),
        (r"^n$", "linear", {}),
        (r"^n\s*log\s*n$", "linearithmic", {}),
        (r"^n\^(\d+)$", "polynomial", lambda m: {"exponent": int(m.group(1))}),
        (
            r"^n\^(\d+)\s*log\s*n$",
            "polynomial_log",
            lambda m: {"exponent": int(m.group(1)), "log_factor": True},
        ),
        (r"^2\^n$", "exponential", {}),
        (r"^n!$", "factorial", {}),
        (r"^D$", "diameter", {}),
        (r"^D\s*\*?\s*log\s*n$", "diameter_log", {}),
        (r"^sqrt\(n\)$", "sqrt", {}),
    ]

    for pattern, class_name, extractor in patterns:
        match = re.match(pattern, inner, re.IGNORECASE)
        if match:
            if callable(extractor):
                params: dict[str, Any] = extractor(match)
            else:
                params = {}
            return (class_name, params)

    return ("unknown", {"raw": inner})


def compare_complexity(a: str, b: str) -> int:
    """Compare two complexity expressions.

    Args:
        a: First complexity expression
        b: Second complexity expression

    Returns:
        -1 if a < b, 0 if equal, 1 if a > b
    """
    ordering = [
        "constant",
        "logarithmic",
        "sqrt",
        "linear",
        "linearithmic",
        "diameter",
        "diameter_log",
        "polynomial",
        "polynomial_log",
        "exponential",
        "factorial",
    ]

    class_a, params_a = parse_big_o(a)
    class_b, params_b = parse_big_o(b)

    try:
        idx_a = ordering.index(class_a)
        idx_b = ordering.index(class_b)
    except ValueError:
        return 0  # Unknown classes treated as equal

    if idx_a != idx_b:
        return -1 if idx_a < idx_b else 1

    # Same class, compare parameters
    if class_a == "polynomial" or class_a == "polynomial_log":
        exp_a = params_a.get("exponent", 1)
        exp_b = params_b.get("exponent", 1)
        if exp_a != exp_b:
            return -1 if exp_a < exp_b else 1

    return 0


__all__ = [
    "ComplexityClass",
    "SynchronyModel",
    "FaultModel",
    "ComplexityBound",
    "DistributedComplexity",
    "complexity",
    "consensus_complexity",
    "gossip_complexity",
    "get_complexity",
    "get_all_complexities",
    "ComplexityAnalysis",
    "analyze_complexity",
    "parse_big_o",
    "compare_complexity",
]
