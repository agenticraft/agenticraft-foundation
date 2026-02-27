# Counterexample Generation

**Source:** `examples/counterexample_generation.py`

This example demonstrates structured counterexample generation -- when refinement or equivalence checks fail, the system explains exactly *where* and *why* the specification and implementation diverge.

---

## 1. Trace Refinement: Extra Behavior

The implementation does something the specification doesn't allow.

```python
from agenticraft_foundation import Event, Prefix, Stop
from agenticraft_foundation.algebra import trace_refines
from agenticraft_foundation.verification import explain_refinement_failure

req = Event("request")
resp = Event("response")
ack = Event("acknowledge")

# Spec: request -> response -> stop
spec = Prefix(req, Prefix(resp, Stop()))

# Impl: request -> response -> acknowledge -> stop (extra event!)
impl = Prefix(req, Prefix(resp, Prefix(ack, Stop())))

result = trace_refines(spec, impl)
```

Without counterexample generation, you get `is_valid=False` and a raw trace. With it:

```python
explanation = explain_refinement_failure(spec, impl, result)
print(explanation.summary)
# "Impl violates spec at trace ⟨request, response, acknowledge⟩:
#  After ⟨request, response⟩, spec allows {} but impl performs acknowledge"
```

The explanation pinpoints:

- **Divergence step**: Step 2 (the `acknowledge` event)
- **Spec allowed**: `{}` (nothing -- spec has terminated)
- **Impl attempted**: `acknowledge`

## 2. Failures Refinement: Nondeterminism

When an implementation makes an internal choice that the specification leaves to the environment.

```python
from agenticraft_foundation import ExternalChoice, InternalChoice

# Spec: external choice (environment decides)
spec = Prefix(req, ExternalChoice(
    Prefix(resp, Stop()),
    Prefix(err, Stop()),
))

# Impl: internal choice (impl decides -- may refuse error)
impl = Prefix(req, InternalChoice(
    Prefix(resp, Stop()),
    Prefix(err, Stop()),
))

result = failures_refines(spec, impl)
explanation = explain_refinement_failure(spec, impl, result)
print(explanation.summary)
# "After trace ⟨request⟩, impl refuses {error, request}
#  but spec does not refuse all of these"
```

The `InternalChoice` means the implementation can nondeterministically pick `resp` and refuse `err`, which the specification doesn't allow.

## 3. Bisimulation: Branching Structure

Two processes with the same traces but different branching structure.

```python
from agenticraft_foundation.algebra import trace_equivalent, strong_bisimilar
from agenticraft_foundation.verification import explain_equivalence_failure

a, b, c = Event("a"), Event("b"), Event("c")

# P: a -> (b [] c)  -- choose after a
p = Prefix(a, ExternalChoice(Prefix(b, Stop()), Prefix(c, Stop())))

# Q: (a -> b) [] (a -> c)  -- choose before a
q = ExternalChoice(Prefix(a, Prefix(b, Stop())), Prefix(a, Prefix(c, Stop())))
```

These have identical trace sets `{⟨a,b⟩, ⟨a,c⟩}`, but different branching:

```python
trace_equivalent(p, q).is_equivalent   # True (sometimes False depending on semantics)
strong_bisimilar(p, q).is_equivalent   # False
```

```python
explanation = explain_equivalence_failure(p, q, bisim_result)
print(explanation.summary)
# "After ⟨a⟩, P can do {c}"
```

After event `a`, process P can do `{b, c}` but process Q (having committed to one branch) can only do `{b}`. The explanation shows exactly which events diverge at which step.

## 4. Minimal Counterexample

Find the shortest distinguishing trace between two processes.

```python
from agenticraft_foundation import build_lts
from agenticraft_foundation.verification import find_minimal_counterexample

lts_spec = build_lts(spec)
lts_impl = build_lts(impl)
minimal = find_minimal_counterexample(lts_spec, lts_impl)
print(f"Trace length: {len(minimal.annotated_trace)}")
```

`find_minimal_counterexample` uses BFS over the synchronized product of both LTS to find the shortest trace that distinguishes them.

## Annotated Trace Format

Each step in the annotated trace is an `AnnotatedStep`:

```python
@dataclass(frozen=True)
class AnnotatedStep:
    event: Event              # The event at this step
    status: StepStatus        # OK or VIOLATION
    spec_available: frozenset # Events spec could do here
    impl_available: frozenset # Events impl could do here
    spec_states: frozenset    # Spec LTS states at this step
    impl_states: frozenset    # Impl LTS states at this step
```

Steps with `StepStatus.OK` are where both agree. The first `StepStatus.VIOLATION` marks the divergence point.

## Output

```
=== Trace Refinement Failure ===
Summary: Impl violates spec at trace ⟨request, response, acknowledge⟩:
  After ⟨request, response⟩, spec allows {} but impl performs acknowledge
Divergence at step: 2

=== Extra Nondeterminism ===
Summary: After trace ⟨request⟩, impl refuses {error, request}
  but spec does not refuse all of these

=== Bisimulation Failure ===
Summary: After ⟨a⟩, P can do {c}
Divergence at step: 1
  a [OK]    P={a} Q={a}
  c [VIOLATION] P={b, c} Q={b}

=== Minimal Counterexample ===
Trace length: 3
```
