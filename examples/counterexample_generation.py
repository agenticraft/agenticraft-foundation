"""Counterexample Generation -- structured failure explanations.

Demonstrates how to get detailed, human-readable explanations when
refinement or equivalence checks fail, pinpointing exactly where
the specification and implementation diverge.
"""

from agenticraft_foundation import (
    Event,
    ExternalChoice,
    InternalChoice,
    Prefix,
    Stop,
    build_lts,
)
from agenticraft_foundation.algebra import (
    failures_refines,
    strong_bisimilar,
    trace_equivalent,
    trace_refines,
)
from agenticraft_foundation.verification import (
    explain_equivalence_failure,
    explain_refinement_failure,
    find_minimal_counterexample,
)

# =============================================================
# Example 1: Trace refinement failure
# =============================================================
print("=== Trace Refinement Failure ===")

req = Event("request")
resp = Event("response")
ack = Event("acknowledge")
err = Event("error")

# Spec: request → response → stop
spec = Prefix(req, Prefix(resp, Stop()))

# Impl: request → response → acknowledge → stop
# (impl does more than spec allows)
impl = Prefix(req, Prefix(resp, Prefix(ack, Stop())))

result = trace_refines(spec, impl)
print(f"Trace refinement holds: {result.is_valid}")

if not result.is_valid:
    explanation = explain_refinement_failure(spec, impl, result)
    print(f"Summary: {explanation.summary}")
    print(f"Divergence at step: {explanation.divergence_point}")
    print(f"Spec allowed: {explanation.spec_allowed}")
    print(f"Impl attempted: {explanation.impl_attempted}")
    print(f"Annotated trace ({len(explanation.annotated_trace)} steps):")
    for step in explanation.annotated_trace:
        print(f"  {step.event} [{step.status.value}]")

# =============================================================
# Example 2: Missing behavior
# =============================================================
print("\n=== Missing Behavior ===")

# Spec: request → (response OR error)
spec2 = Prefix(req, ExternalChoice(
    Prefix(resp, Stop()),
    Prefix(err, Stop()),
))

# Impl: request → response only (missing error handling)
impl2 = Prefix(req, Prefix(resp, Stop()))

result2 = failures_refines(spec2, impl2)
print(f"Failures refinement holds: {result2.is_valid}")

if not result2.is_valid:
    explanation = explain_refinement_failure(spec2, impl2, result2)
    print(f"Summary: {explanation.summary}")

# =============================================================
# Example 3: Extra nondeterminism
# =============================================================
print("\n=== Extra Nondeterminism ===")

# Spec: deterministic choice (external)
spec3 = Prefix(req, ExternalChoice(
    Prefix(resp, Stop()),
    Prefix(err, Stop()),
))

# Impl: nondeterministic choice (internal -- impl decides, not environment)
impl3 = Prefix(req, InternalChoice(
    Prefix(resp, Stop()),
    Prefix(err, Stop()),
))

result3 = failures_refines(spec3, impl3)
print(f"Failures refinement holds: {result3.is_valid}")

if not result3.is_valid:
    explanation = explain_refinement_failure(spec3, impl3, result3)
    print(f"Summary: {explanation.summary}")
    print(f"Divergence at step: {explanation.divergence_point}")

# =============================================================
# Example 4: Bisimulation failure
# =============================================================
print("\n=== Bisimulation Failure ===")

# Two processes that are trace-equivalent but NOT bisimilar
#   P: a → (b → STOP [] c → STOP)
#   Q: (a → b → STOP) [] (a → c → STOP)

a, b, c = Event("a"), Event("b"), Event("c")

p = Prefix(a, ExternalChoice(Prefix(b, Stop()), Prefix(c, Stop())))
q = ExternalChoice(Prefix(a, Prefix(b, Stop())), Prefix(a, Prefix(c, Stop())))

# Same traces...
te_result = trace_equivalent(p, q)
print(f"Trace equivalent: {te_result.is_equivalent}")

# ...but not bisimilar
bisim_result = strong_bisimilar(p, q)
print(f"Strong bisimilar: {bisim_result.is_equivalent}")

if not bisim_result.is_equivalent:
    explanation = explain_equivalence_failure(p, q, bisim_result)
    print(f"Summary: {explanation.summary}")
    print(f"Divergence at step: {explanation.divergence_point}")
    print(f"Annotated trace ({len(explanation.annotated_trace)} steps):")
    for step in explanation.annotated_trace:
        print(f"  {step.event} [{step.status.value}] "
              f"P={step.spec_available} Q={step.impl_available}")

# =============================================================
# Example 5: Minimal counterexample search
# =============================================================
print("\n=== Minimal Counterexample ===")

# Find the shortest trace that distinguishes two processes
recv = Event("recv")
send = Event("send")
drop = Event("drop")
log = Event("log")

# Spec allows recv → send or recv → drop → log
spec5 = Prefix(recv, ExternalChoice(
    Prefix(send, Stop()),
    Prefix(drop, Prefix(log, Stop())),
))

# Impl: recv → send → log (different structure)
impl5 = Prefix(recv, Prefix(send, Prefix(log, Stop())))

result5 = trace_refines(spec5, impl5)
print(f"Refinement holds: {result5.is_valid}")

if not result5.is_valid:
    lts_spec = build_lts(spec5)
    lts_impl = build_lts(impl5)
    minimal = find_minimal_counterexample(lts_spec, lts_impl)
    if minimal:
        print(f"Minimal counterexample: {minimal.summary}")
        print(f"Trace length: {len(minimal.annotated_trace)}")
    else:
        print("No counterexample found (processes may be equivalent)")

print("\nDone.")
