"""Runtime invariant checking for distributed systems.

This module provides tools for runtime verification of distributed system
invariants, enabling early detection of protocol violations.

Features:
- State transition monitoring
- Invariant assertion and logging
- Violation reporting and debugging
- Thread-safe concurrent access via RLock

Thread Safety:
    All mutable state in InvariantRegistry is protected by RLock,
    enabling safe concurrent invariant checking across threads.

Version: 0.1.0
Date: 2025-12-20
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


class ViolationSeverity(str, Enum):
    """Severity levels for invariant violations."""

    WARNING = "warning"  # Non-critical, log and continue
    ERROR = "error"  # Critical, log and potentially halt
    FATAL = "fatal"  # Unrecoverable, must halt


@dataclass
class Invariant:
    """A runtime invariant to be checked.

    Attributes:
        name: Descriptive name
        condition: Function that returns True if invariant holds
        severity: Severity level if violated
        message: Message to log on violation
        enabled: Whether invariant checking is enabled
    """

    name: str
    condition: Callable[..., bool]
    severity: ViolationSeverity = ViolationSeverity.ERROR
    message: str = ""
    enabled: bool = True

    def check(self, *args: Any, **kwargs: Any) -> bool:
        """Check if invariant holds.

        Args:
            *args: Arguments to pass to condition
            **kwargs: Keyword arguments to pass to condition

        Returns:
            True if invariant holds
        """
        if not self.enabled:
            return True
        return self.condition(*args, **kwargs)


@dataclass
class Violation:
    """Record of an invariant violation.

    Attributes:
        invariant_name: Name of violated invariant
        severity: Severity of violation
        message: Violation message
        context: Context at time of violation
        timestamp: When violation occurred
        stack_trace: Optional stack trace
    """

    invariant_name: str
    severity: ViolationSeverity
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    stack_trace: str | None = None


class InvariantRegistry:
    """Registry for managing system invariants.

    Provides centralized management of invariants with checking
    and violation tracking.

    Thread Safety:
        All mutable state is protected by RLock, enabling safe concurrent
        access from multiple threads. RLock is used (vs Lock) to allow
        re-entrant calls from violation handlers.
    """

    def __init__(self, name: str = "default"):
        """Initialize registry.

        Args:
            name: Registry name for logging
        """
        self.name = name
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._invariants: dict[str, Invariant] = {}
        self._violations: list[Violation] = []
        self._check_count = 0
        self._violation_count = 0
        self._enabled = True
        self._on_violation: list[Callable[[Violation], None]] = []

    def register(
        self,
        name: str,
        condition: Callable[..., bool],
        severity: ViolationSeverity = ViolationSeverity.ERROR,
        message: str = "",
    ) -> Invariant:
        """Register a new invariant.

        Thread-safe: Protected by RLock.

        Args:
            name: Unique name for invariant
            condition: Function returning True if invariant holds
            severity: Severity level if violated
            message: Message to log on violation

        Returns:
            The registered Invariant
        """
        invariant = Invariant(
            name=name,
            condition=condition,
            severity=severity,
            message=message or f"Invariant '{name}' violated",
        )
        with self._lock:
            self._invariants[name] = invariant
        return invariant

    def unregister(self, name: str) -> bool:
        """Unregister an invariant.

        Thread-safe: Protected by RLock.

        Args:
            name: Invariant name to remove

        Returns:
            True if invariant was removed
        """
        with self._lock:
            if name in self._invariants:
                del self._invariants[name]
                return True
            return False

    def check(
        self,
        name: str,
        *args: Any,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Check a specific invariant.

        Thread-safe: Protected by RLock. The invariant condition is evaluated
        outside the lock to avoid blocking concurrent access during potentially
        long-running checks.

        Args:
            name: Invariant name to check
            *args: Arguments to pass to invariant condition
            context: Additional context for violation reporting
            **kwargs: Keyword arguments to pass to condition

        Returns:
            True if invariant holds

        Raises:
            KeyError: If invariant not found
        """
        # Fast path: check if enabled
        with self._lock:
            if not self._enabled:
                return True
            invariant = self._invariants.get(name)
            if not invariant:
                raise KeyError(f"Invariant '{name}' not registered")
            self._check_count += 1
            # Snapshot handlers for notification outside lock
            handlers = list(self._on_violation)

        # Check condition OUTSIDE lock to avoid blocking
        if invariant.check(*args, **kwargs):
            return True

        # Record violation with lock
        violation = Violation(
            invariant_name=name,
            severity=invariant.severity,
            message=invariant.message,
            context=context or {},
        )
        with self._lock:
            self._violation_count += 1
            self._violations.append(violation)

        # Log violation
        log_level = {
            ViolationSeverity.WARNING: logging.WARNING,
            ViolationSeverity.ERROR: logging.ERROR,
            ViolationSeverity.FATAL: logging.CRITICAL,
        }.get(invariant.severity, logging.ERROR)

        logger.log(
            log_level,
            f"Invariant violation: {name} - {invariant.message}",
            extra={"context": context},
        )

        # Notify handlers OUTSIDE lock to avoid deadlock
        for handler in handlers:
            try:
                handler(violation)
            except (TypeError, ValueError) as e:
                logger.error(f"Error in violation handler: {e}")
            except Exception:
                logger.exception("Unexpected error in violation handler")

        return False

    def check_all(
        self,
        *args: Any,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Check all registered invariants.

        Thread-safe: Iterates over a snapshot of invariant names.

        Args:
            *args: Arguments to pass to invariant conditions
            context: Additional context for violation reporting
            **kwargs: Keyword arguments to pass to conditions

        Returns:
            List of violated invariant names
        """
        # Snapshot invariant names under lock
        with self._lock:
            invariant_names = list(self._invariants.keys())

        violated = []
        for name in invariant_names:
            if not self.check(name, *args, context=context, **kwargs):
                violated.append(name)
        return violated

    def on_violation(self, handler: Callable[[Violation], None]) -> None:
        """Register a violation handler.

        Thread-safe: Protected by RLock.

        Args:
            handler: Function to call on violation
        """
        with self._lock:
            self._on_violation.append(handler)

    def get_violations(
        self,
        since: float | None = None,
        severity: ViolationSeverity | None = None,
    ) -> list[Violation]:
        """Get recorded violations.

        Thread-safe: Returns a snapshot copy of violations.

        Args:
            since: Only violations after this timestamp
            severity: Filter by severity level

        Returns:
            List of matching violations
        """
        with self._lock:
            violations = list(self._violations)  # Copy for thread safety

        if since is not None:
            violations = [v for v in violations if v.timestamp >= since]

        if severity is not None:
            violations = [v for v in violations if v.severity == severity]

        return violations

    def clear_violations(self) -> None:
        """Clear all recorded violations.

        Thread-safe: Protected by RLock.
        """
        with self._lock:
            self._violations = []

    def enable(self) -> None:
        """Enable invariant checking.

        Thread-safe: Protected by RLock.
        """
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        """Disable invariant checking.

        Thread-safe: Protected by RLock.
        """
        with self._lock:
            self._enabled = False

    def stats(self) -> dict[str, Any]:
        """Get checking statistics.

        Thread-safe: Returns a snapshot of statistics.

        Returns:
            Dictionary with check and violation counts
        """
        with self._lock:
            return {
                "name": self.name,
                "invariant_count": len(self._invariants),
                "check_count": self._check_count,
                "violation_count": self._violation_count,
                "violation_rate": (
                    self._violation_count / self._check_count if self._check_count > 0 else 0
                ),
                "enabled": self._enabled,
            }

    def record_violation(self, violation: Violation) -> None:
        """Record a violation directly.

        Thread-safe: Protected by RLock.
        Used by assert_invariant for direct violation recording.

        Args:
            violation: The violation to record.
        """
        with self._lock:
            self._violations.append(violation)
            self._violation_count += 1


# Global default registry
_default_registry = InvariantRegistry("global")


def register_invariant(
    name: str,
    condition: Callable[..., bool],
    severity: ViolationSeverity = ViolationSeverity.ERROR,
    message: str = "",
    registry: InvariantRegistry | None = None,
) -> Invariant:
    """Register an invariant in the default or specified registry.

    Args:
        name: Unique name for invariant
        condition: Function returning True if invariant holds
        severity: Severity level if violated
        message: Message to log on violation
        registry: Registry to use (default if None)

    Returns:
        The registered Invariant
    """
    reg = registry or _default_registry
    return reg.register(name, condition, severity, message)


def check_invariant(
    name: str,
    *args: Any,
    context: dict[str, Any] | None = None,
    registry: InvariantRegistry | None = None,
    **kwargs: Any,
) -> bool:
    """Check an invariant in the default or specified registry.

    Args:
        name: Invariant name to check
        *args: Arguments to pass to invariant condition
        context: Additional context for violation reporting
        registry: Registry to use (default if None)
        **kwargs: Keyword arguments to pass to condition

    Returns:
        True if invariant holds
    """
    reg = registry or _default_registry
    return reg.check(name, *args, context=context, **kwargs)


def invariant(
    name: str,
    condition: Callable[..., bool] | None = None,
    severity: ViolationSeverity = ViolationSeverity.ERROR,
    message: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to add invariant checking to a function.

    The invariant is checked before and after the function executes.

    Args:
        name: Invariant name
        condition: Condition to check (receives function args)
        severity: Severity level if violated
        message: Message to log on violation

    Returns:
        Decorated function

    Example:
        @invariant(
            "positive_balance",
            lambda self: self.balance >= 0,
            message="Balance cannot be negative"
        )
        def withdraw(self, amount):
            self.balance -= amount
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Register invariant if condition provided
        if condition:
            register_invariant(name, condition, severity, message)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Check invariant before (use registry directly to avoid
            # ParamSpec incompatibility with check_invariant's signature)
            _default_registry.check(name, *args, **kwargs)

            # Execute function
            result = func(*args, **kwargs)

            # Check invariant after
            _default_registry.check(name, *args, **kwargs)

            return result

        return wrapper

    return decorator


def assert_invariant(
    condition: bool,
    message: str = "Assertion failed",
    severity: ViolationSeverity = ViolationSeverity.ERROR,
) -> None:
    """Assert an invariant condition.

    Thread-safe: Uses thread-safe record_violation method.

    Args:
        condition: Condition that should be true
        message: Message if condition is false
        severity: Severity of violation

    Raises:
        AssertionError: If condition is false and severity is FATAL
    """
    if not condition:
        violation = Violation(
            invariant_name="assertion",
            severity=severity,
            message=message,
        )
        _default_registry.record_violation(violation)

        if severity == ViolationSeverity.FATAL:
            raise AssertionError(message)

        log_level = {
            ViolationSeverity.WARNING: logging.WARNING,
            ViolationSeverity.ERROR: logging.ERROR,
        }.get(severity, logging.ERROR)

        logger.log(log_level, f"Invariant assertion failed: {message}")


class StateTransitionMonitor:
    """Monitors state transitions and checks invariants.

    Useful for verifying state machine correctness.

    Thread Safety:
        All mutable state is protected by RLock for safe concurrent access.
    """

    def __init__(
        self,
        name: str = "state_monitor",
        valid_transitions: dict[str, set[str]] | None = None,
    ):
        """Initialize monitor.

        Args:
            name: Monitor name
            valid_transitions: Map of state to valid next states
        """
        self.name = name
        self._lock = threading.RLock()
        self.valid_transitions = valid_transitions or {}
        self._current_state: str | None = None
        self._transition_history: list[tuple[str, str, float]] = []
        self._invalid_transitions: list[tuple[str, str]] = []

    def set_valid_transitions(self, transitions: dict[str, set[str]]) -> None:
        """Set valid state transitions.

        Thread-safe: Protected by RLock.

        Args:
            transitions: Map of state to valid next states
        """
        with self._lock:
            self.valid_transitions = transitions

    def transition(self, new_state: str) -> bool:
        """Record a state transition.

        Thread-safe: Protected by RLock.

        Args:
            new_state: The new state

        Returns:
            True if transition is valid
        """
        with self._lock:
            is_valid = True

            if self._current_state is not None:
                valid_next = self.valid_transitions.get(self._current_state, set())
                if valid_next and new_state not in valid_next:
                    is_valid = False
                    self._invalid_transitions.append((self._current_state, new_state))
                    logger.warning(
                        f"Invalid state transition: {self._current_state} -> {new_state}"
                    )

            self._transition_history.append(
                (self._current_state or "initial", new_state, time.time())
            )
            self._current_state = new_state

            return is_valid

    @property
    def current_state(self) -> str | None:
        """Get current state.

        Thread-safe: Protected by RLock.
        """
        with self._lock:
            return self._current_state

    def get_history(self) -> list[tuple[str, str, float]]:
        """Get transition history.

        Thread-safe: Returns a copy of the history.

        Returns:
            List of (from_state, to_state, timestamp) tuples
        """
        with self._lock:
            return list(self._transition_history)

    def get_invalid_transitions(self) -> list[tuple[str, str]]:
        """Get invalid transitions that occurred.

        Thread-safe: Returns a copy of invalid transitions.

        Returns:
            List of (from_state, to_state) tuples
        """
        with self._lock:
            return list(self._invalid_transitions)

    def reset(self) -> None:
        """Reset monitor state.

        Thread-safe: Protected by RLock.
        """
        with self._lock:
            self._current_state = None
            self._transition_history = []
            self._invalid_transitions = []


__all__ = [
    "ViolationSeverity",
    "Invariant",
    "Violation",
    "InvariantRegistry",
    "register_invariant",
    "check_invariant",
    "invariant",
    "assert_invariant",
    "StateTransitionMonitor",
]
