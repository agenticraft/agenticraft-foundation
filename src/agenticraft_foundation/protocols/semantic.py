"""Semantic preservation verification.

Verifies Definition 9 (Semantic Preservation): meaning(m) = meaning(T_{p→p'}(m)).
Protocol translations must preserve semantic content, and
round-trip preservation T⁻¹(T(m)) ≈ m.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class SemanticViolationType(str, Enum):
    """Types of semantic violations."""

    FIELD_MISSING = "field_missing"
    """Required field is missing after translation"""

    FIELD_MODIFIED = "field_modified"
    """Field value was unexpectedly modified"""

    TYPE_MISMATCH = "type_mismatch"
    """Field type changed during translation"""

    STRUCTURE_CHANGED = "structure_changed"
    """Overall message structure was altered"""

    ROUND_TRIP_FAILED = "round_trip_failed"
    """T⁻¹(T(m)) ≠ m"""

    SEMANTIC_DRIFT = "semantic_drift"
    """Semantic meaning has drifted beyond threshold"""


@dataclass
class SemanticViolation:
    """A specific semantic violation detected during verification."""

    violation_type: SemanticViolationType
    """Type of violation"""

    field_path: str
    """Path to the affected field (e.g., 'payload.data.value')"""

    original_value: Any | None = None
    """Original value before translation"""

    translated_value: Any | None = None
    """Value after translation"""

    message: str = ""
    """Human-readable description"""

    severity: str = "warning"
    """Severity: 'info', 'warning', 'error', 'critical'"""

    def __repr__(self) -> str:
        return f"SemanticViolation({self.violation_type.value}: {self.message})"


@dataclass
class SemanticVerificationResult:
    """Result of semantic preservation check."""

    preserved: bool
    """Whether semantics were fully preserved"""

    original_hash: str
    """Semantic hash of original message"""

    translated_hash: str
    """Semantic hash of translated message"""

    round_trip_hash: str | None = None
    """Semantic hash after round-trip translation (if performed)"""

    semantic_distance: float = 0.0
    """Distance between original and translated (0.0 = identical, 1.0 = different)"""

    violations: list[SemanticViolation] = field(default_factory=list)
    """List of detected violations"""

    verification_time_ms: float = 0.0
    """Time taken for verification in milliseconds"""

    @property
    def has_violations(self) -> bool:
        """Check if any violations were detected."""
        return len(self.violations) > 0

    @property
    def critical_violations(self) -> list[SemanticViolation]:
        """Get only critical violations."""
        return [v for v in self.violations if v.severity == "critical"]

    @property
    def round_trip_preserved(self) -> bool:
        """Check if round-trip preservation holds."""
        if self.round_trip_hash is None:
            return True  # Not tested
        return self.original_hash == self.round_trip_hash

    def get_summary(self) -> dict[str, Any]:
        """Get verification summary."""
        return {
            "preserved": self.preserved,
            "semantic_distance": self.semantic_distance,
            "round_trip_preserved": self.round_trip_preserved,
            "num_violations": len(self.violations),
            "critical_violations": len(self.critical_violations),
            "verification_time_ms": self.verification_time_ms,
        }


@runtime_checkable
class ProtocolMessage(Protocol):
    """Protocol for messages that can be semantically verified."""

    @property
    def semantic_fields(self) -> dict[str, Any]:
        """Fields that carry semantic meaning."""
        ...

    @property
    def protocol_metadata(self) -> dict[str, Any]:
        """Protocol-specific metadata (not semantic)."""
        ...


@runtime_checkable
class ProtocolTranslator(Protocol):
    """Protocol for translators that can translate messages."""

    def translate(self, message: Any) -> Any:
        """Translate a message to target protocol."""
        ...

    def reverse_translate(self, message: Any) -> Any:
        """Translate back to source protocol."""
        ...


@dataclass
class VerificationConfig:
    """Configuration for semantic verification."""

    semantic_distance_threshold: float = 0.1
    """Maximum allowed semantic distance (0.0 to 1.0)"""

    require_round_trip: bool = False
    """Whether to require round-trip verification"""

    strict_type_checking: bool = True
    """Whether to check field types strictly"""

    ignore_fields: set[str] = field(default_factory=set)
    """Fields to ignore during comparison"""

    critical_fields: set[str] = field(default_factory=set)
    """Fields that must be preserved exactly"""


class SemanticPreservationVerifier:
    """
    Verify semantic preservation in protocol translations.

    Implements Definition 9 (Semantic Preservation): meaning(m) = meaning(T_{p→p'}(m))

    Checks:
    - Field preservation
    - Type preservation
    - Structure preservation
    - Round-trip preservation: T⁻¹(T(m)) ≈ m

    Usage:
        verifier = SemanticPreservationVerifier()

        # Verify a translation
        result = verifier.verify_translation(
            original=original_message,
            translated=translated_message,
        )

        if not result.preserved:
            for violation in result.violations:
                print(f"Violation: {violation.message}")

        # Verify round-trip
        result = verifier.verify_round_trip(
            message=original_message,
            forward_translator=mcp_to_a2a,
            reverse_translator=a2a_to_mcp,
        )
    """

    def __init__(
        self,
        config: VerificationConfig | None = None,
    ):
        """
        Initialize semantic preservation verifier.

        Args:
            config: Verification configuration
        """
        self._config = config or VerificationConfig()

        # Statistics
        self._stats = {
            "verifications_performed": 0,
            "verifications_passed": 0,
            "verifications_failed": 0,
            "round_trips_performed": 0,
            "round_trips_passed": 0,
        }

    def verify_translation(
        self,
        original: dict[str, Any] | ProtocolMessage,
        translated: dict[str, Any] | ProtocolMessage,
    ) -> SemanticVerificationResult:
        """
        Verify semantic preservation in a translation.

        Args:
            original: Original message
            translated: Translated message

        Returns:
            SemanticVerificationResult with preservation status
        """
        import time

        start_time = time.time()
        self._stats["verifications_performed"] += 1

        # Extract semantic fields
        original_fields = self._extract_semantic_fields(original)
        translated_fields = self._extract_semantic_fields(translated)

        # Compute semantic hashes
        original_hash = self._compute_semantic_hash(original_fields)
        translated_hash = self._compute_semantic_hash(translated_fields)

        # Find violations
        violations = self._find_violations(original_fields, translated_fields)

        # Calculate semantic distance
        semantic_distance = self._compute_semantic_distance(original_fields, translated_fields)

        # Determine if preserved
        preserved = (
            semantic_distance <= self._config.semantic_distance_threshold
            and len([v for v in violations if v.severity == "critical"]) == 0
        )

        if preserved:
            self._stats["verifications_passed"] += 1
        else:
            self._stats["verifications_failed"] += 1

        end_time = time.time()

        return SemanticVerificationResult(
            preserved=preserved,
            original_hash=original_hash,
            translated_hash=translated_hash,
            semantic_distance=semantic_distance,
            violations=violations,
            verification_time_ms=(end_time - start_time) * 1000,
        )

    def verify_round_trip(
        self,
        message: dict[str, Any] | ProtocolMessage,
        forward_translator: ProtocolTranslator,
        reverse_translator: ProtocolTranslator,
    ) -> SemanticVerificationResult:
        """
        Verify round-trip preservation: T⁻¹(T(m)) ≈ m.

        Args:
            message: Original message
            forward_translator: Translator from source to target protocol
            reverse_translator: Translator from target back to source

        Returns:
            SemanticVerificationResult with round-trip verification
        """
        self._stats["round_trips_performed"] += 1

        # Translate forward
        translated = forward_translator.translate(message)

        # Translate back
        round_tripped = reverse_translator.reverse_translate(translated)

        # Verify
        result = self.verify_translation(message, round_tripped)

        # Update result with round-trip info
        round_tripped_fields = self._extract_semantic_fields(round_tripped)
        result.round_trip_hash = self._compute_semantic_hash(round_tripped_fields)

        if not result.round_trip_preserved:
            result.violations.append(
                SemanticViolation(
                    violation_type=SemanticViolationType.ROUND_TRIP_FAILED,
                    field_path="",
                    message="Round-trip translation did not preserve semantics",
                    severity="error",
                )
            )

        if result.preserved and result.round_trip_preserved:
            self._stats["round_trips_passed"] += 1

        return result

    def verify_output(
        self,
        output: dict[str, Any],
        expected_schema: dict[str, Any],
    ) -> SemanticVerificationResult:
        """
        Verify output matches expected schema semantically.

        Args:
            output: Actual output
            expected_schema: Expected schema with field types

        Returns:
            SemanticVerificationResult
        """
        violations: list[SemanticViolation] = []

        # Check each expected field
        for field_name, expected_type in expected_schema.items():
            if field_name not in output:
                violations.append(
                    SemanticViolation(
                        violation_type=SemanticViolationType.FIELD_MISSING,
                        field_path=field_name,
                        message=f"Expected field '{field_name}' is missing",
                        severity="error",
                    )
                )
            elif self._config.strict_type_checking:
                actual_type = type(output[field_name]).__name__
                if actual_type != expected_type:
                    violations.append(
                        SemanticViolation(
                            violation_type=SemanticViolationType.TYPE_MISMATCH,
                            field_path=field_name,
                            original_value=expected_type,
                            translated_value=actual_type,
                            message=f"Type mismatch: expected {expected_type}, got {actual_type}",
                            severity="warning",
                        )
                    )

        preserved = len([v for v in violations if v.severity == "error"]) == 0

        return SemanticVerificationResult(
            preserved=preserved,
            original_hash=self._compute_semantic_hash(expected_schema),
            translated_hash=self._compute_semantic_hash(output),
            violations=violations,
        )

    def _extract_semantic_fields(
        self,
        message: dict[str, Any] | ProtocolMessage,
    ) -> dict[str, Any]:
        """Extract semantic fields from a message."""
        if isinstance(message, dict):
            # Filter out protocol metadata
            return {
                k: v
                for k, v in message.items()
                if not k.startswith("_") and k not in self._config.ignore_fields
            }
        elif hasattr(message, "semantic_fields"):
            return message.semantic_fields
        else:
            # Try to extract fields from object
            return {
                k: v
                for k, v in vars(message).items()
                if not k.startswith("_") and k not in self._config.ignore_fields
            }

    def _compute_semantic_hash(self, fields: dict[str, Any]) -> str:
        """Compute semantic hash of fields."""
        # Sort keys for deterministic hashing
        normalized = self._normalize_for_hashing(fields)
        json_str = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _normalize_for_hashing(self, obj: Any) -> Any:
        """Normalize object for deterministic hashing."""
        if isinstance(obj, dict):
            return {k: self._normalize_for_hashing(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [self._normalize_for_hashing(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    def _find_violations(
        self,
        original: dict[str, Any],
        translated: dict[str, Any],
        path_prefix: str = "",
    ) -> list[SemanticViolation]:
        """Find semantic violations between original and translated."""
        violations: list[SemanticViolation] = []

        # Check for missing fields
        for key in original:
            field_path = f"{path_prefix}.{key}" if path_prefix else key

            if key in self._config.ignore_fields:
                continue

            if key not in translated:
                severity = "critical" if key in self._config.critical_fields else "warning"
                violations.append(
                    SemanticViolation(
                        violation_type=SemanticViolationType.FIELD_MISSING,
                        field_path=field_path,
                        original_value=original[key],
                        message=f"Field '{field_path}' missing after translation",
                        severity=severity,
                    )
                )
                continue

            orig_val = original[key]
            trans_val = translated[key]

            # Check type mismatch
            if self._config.strict_type_checking:
                if type(orig_val) is not type(trans_val):
                    violations.append(
                        SemanticViolation(
                            violation_type=SemanticViolationType.TYPE_MISMATCH,
                            field_path=field_path,
                            original_value=type(orig_val).__name__,
                            translated_value=type(trans_val).__name__,
                            message=(
                                f"Type changed from {type(orig_val).__name__}"
                                f" to {type(trans_val).__name__}"
                            ),
                            severity="warning",
                        )
                    )

            # Check value modifications
            if isinstance(orig_val, dict) and isinstance(trans_val, dict):
                # Recursively check nested dicts
                nested = self._find_violations(orig_val, trans_val, field_path)
                violations.extend(nested)
            elif orig_val != trans_val:
                severity = "critical" if key in self._config.critical_fields else "info"
                violations.append(
                    SemanticViolation(
                        violation_type=SemanticViolationType.FIELD_MODIFIED,
                        field_path=field_path,
                        original_value=orig_val,
                        translated_value=trans_val,
                        message=f"Field '{field_path}' modified",
                        severity=severity,
                    )
                )

        # Check for extra fields (structure change)
        extra_fields = set(translated.keys()) - set(original.keys())
        extra_fields -= self._config.ignore_fields
        if extra_fields:
            violations.append(
                SemanticViolation(
                    violation_type=SemanticViolationType.STRUCTURE_CHANGED,
                    field_path=path_prefix or "root",
                    translated_value=list(extra_fields),
                    message=f"Extra fields added: {extra_fields}",
                    severity="info",
                )
            )

        return violations

    def _compute_semantic_distance(
        self,
        original: dict[str, Any],
        translated: dict[str, Any],
    ) -> float:
        """
        Compute semantic distance between two field sets.

        Returns value in [0.0, 1.0] where:
        - 0.0 = semantically identical
        - 1.0 = completely different
        """
        if not original and not translated:
            return 0.0

        if not original or not translated:
            return 1.0

        all_keys = set(original.keys()) | set(translated.keys())
        all_keys -= self._config.ignore_fields

        if not all_keys:
            return 0.0

        matching: float = 0
        total = 0

        for key in all_keys:
            total += 1
            if key in original and key in translated:
                if self._values_match(original[key], translated[key]):
                    matching += 1
                else:
                    # Partial match for similar values
                    similarity = self._compute_value_similarity(original[key], translated[key])
                    matching += similarity

        return 1.0 - (matching / total)

    def _values_match(self, v1: Any, v2: Any) -> bool:
        """Check if two values match semantically."""
        if type(v1) is not type(v2):
            return False
        if isinstance(v1, dict):
            return set(v1.keys()) == set(v2.keys()) and all(
                self._values_match(v1[k], v2[k]) for k in v1
            )
        if isinstance(v1, list):
            if len(v1) != len(v2):
                return False
            return all(self._values_match(a, b) for a, b in zip(v1, v2, strict=False))
        return bool(v1 == v2)

    def _compute_value_similarity(self, v1: Any, v2: Any) -> float:
        """Compute similarity between two values (0.0 to 1.0)."""
        if v1 == v2:
            return 1.0

        if isinstance(v1, str) and isinstance(v2, str):
            # Levenshtein-like similarity
            len1, len2 = len(v1), len(v2)
            if len1 == 0 or len2 == 0:
                return 0.0
            common = sum(1 for c in v1 if c in v2)
            return common / max(len1, len2)

        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # Numeric similarity
            if v1 == 0 and v2 == 0:
                return 1.0
            max_val = max(abs(v1), abs(v2))
            if max_val == 0:
                return 1.0
            return 1.0 - min(abs(v1 - v2) / max_val, 1.0)

        return 0.0

    def get_statistics(self) -> dict[str, Any]:
        """Get verification statistics."""
        return {
            **self._stats,
            "config": {
                "semantic_distance_threshold": self._config.semantic_distance_threshold,
                "strict_type_checking": self._config.strict_type_checking,
            },
        }


__all__ = [
    "SemanticViolationType",
    "SemanticViolation",
    "SemanticVerificationResult",
    "ProtocolMessage",
    "ProtocolTranslator",
    "VerificationConfig",
    "SemanticPreservationVerifier",
]
