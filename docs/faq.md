# Frequently Asked Questions

Common questions about AgentiCraft Foundation, CSP process algebra, session types, and formal verification for multi-agent systems.

---

## What is CSP and why use it for agent coordination?

**CSP (Communicating Sequential Processes)** is a formal language for modeling concurrent systems that communicate through message passing. Developed by Tony Hoare, it provides mathematical rigor for specifying, analyzing, and verifying how independent processes interact.

For multi-agent systems, CSP is valuable because it:

- **Guarantees deadlock freedom**: You can prove agents will never reach a stuck state
- **Models coordination precisely**: Capture exact synchronization points and message flows
- **Enables automated verification**: Compile processes to Labeled Transition Systems (LTS) for model checking
- **Supports compositional reasoning**: Verify agents individually, then prove their composition is correct

```python
from agenticraft_foundation import (
    Event, Prefix, Stop, Parallel,
    build_lts, is_deadlock_free,
)

# Model two agents coordinating on shared work
request = Event("request")
process = Event("process")
complete = Event("complete")

agent_a = Prefix(request, Prefix(process, Stop()))
agent_b = Prefix(request, Prefix(complete, Stop()))

# Parallel composition with synchronization on 'request'
system = Parallel(agent_a, agent_b, sync_events={request})

# Verify the system is deadlock-free
assert is_deadlock_free(system)
```

See [Process Algebra (CSP)](concepts/process-algebra.md) for the full operator reference.

---

## How does this relate to the AgentiCraft platform?

AgentiCraft Foundation is the **verification engine** behind [AgentiCraft](https://agenticraft.ai) — an enterprise platform for building production-ready AI agents.

| Foundation | Platform |
|------------|----------|
| CSP process algebra | Agent workflow definitions |
| Multiparty Session Types | Inter-agent protocols |
| Protocol-aware routing | Multi-agent orchestration |
| Topology analysis | Network resilience |
| CTL/DTMC verification | Safety & liveness guarantees |

Foundation provides the **mathematical guarantees**; Platform provides the **developer experience** and runtime infrastructure.

You can use Foundation standalone (it has minimal dependencies) or as part of the full AgentiCraft stack.

---

## What Python version is required?

**Python 3.10 or higher** is required.

Foundation uses modern Python type hints, pattern matching, and other 3.10+ features. The library has minimal runtime dependencies:

- **Core modules** (CSP, MPST, protocols): Pure Python, no external dependencies
- **Topology & verification**: Requires [NumPy](https://numpy.org/) for spectral analysis and probabilistic verification

```bash
# Install with uv (recommended)
uv add agenticraft-foundation

# Or with pip
pip install agenticraft-foundation

# Verify installation
python -c "import agenticraft_foundation; print(agenticraft_foundation.__version__)"
```

See [Installation](getting-started/installation.md) for development setup instructions.

---

## How do I model my own agent protocol?

Use **Multiparty Session Types (MPST)** to define a global protocol, then project it to per-role local types.

```python
from agenticraft_foundation.mpst import (
    GlobalInteraction, GlobalType, Projector, SessionTypeChecker,
)

# Define a three-party protocol: client → processor → responder
protocol = GlobalType(interactions=[
    GlobalInteraction(sender="client", receiver="processor", message_type="Task"),
    GlobalInteraction(sender="processor", receiver="responder", message_type="ProcessedTask"),
    GlobalInteraction(sender="responder", receiver="client", message_type="Result"),
])

# Project to each role's local view
projector = Projector()
client_local = projector.project(protocol, "client")
processor_local = projector.project(protocol, "processor")
responder_local = projector.project(protocol, "responder")

# Verify the protocol is well-formed
checker = SessionTypeChecker()
result = checker.check_well_formedness(protocol)
assert result.is_valid, f"Protocol errors: {result.errors}"
```

For simpler coordination patterns, you can model directly in CSP:

```python
from agenticraft_foundation import Event, Prefix, Skip, Parallel

task = Event("task")
done = Event("done")

worker = Prefix(task, Prefix(done, Skip()))
coordinator = Prefix(task, Prefix(done, Skip()))

# Coordinator and worker synchronize on both events
system = Parallel(coordinator, worker, sync_events={task, done})
```

See [Session Types](concepts/session-types.md) and the [MPST Tutorial](tutorials/mpst-verification.md) for more.

---

## What's the difference between external and internal choice?

Both are choice operators in CSP, but **who decides** differs:

| Aspect | External Choice (`[]`) | Internal Choice (`|\~|`) |
|--------|------------------------|---------------------------|
| **Who decides** | The environment | The process itself |
| **Determinism** | Deterministic | Nondeterministic |
| **Initial events** | Union of both branches | Either branch, unpredictably |
| **Use case** | Wait for external input | Model uncertainty or internal decisions |

### External Choice

The environment chooses based on which initial event occurs first:

```python
from agenticraft_foundation import Event, Prefix, Skip, ExternalChoice

request = Event("request")
cancel = Event("cancel")

# Agent waits for either request or cancel
# The caller decides which path to take
agent = ExternalChoice(
    Prefix(request, Skip()),  # Path A: handle request
    Prefix(cancel, Skip()),   # Path B: handle cancellation
)
```

### Internal Choice

The process nondeterministically selects a branch — useful for modeling uncertainty:

```python
from agenticraft_foundation import Event, Prefix, Skip, InternalChoice

success = Event("success")
failure = Event("failure")

# Agent internally decides success or failure
# The environment cannot influence the choice
flaky_agent = InternalChoice(
    Prefix(success, Skip()),
    Prefix(failure, Skip()),
)
```

**Key insight**: External choice is for **reactive** agents that respond to their environment; internal choice is for modeling **unreliable** or **autonomous** behavior.

---

## How do I check for deadlocks?

Use `is_deadlock_free()` for a boolean check, or `detect_deadlock()` to find the problematic state.

```python
from agenticraft_foundation import (
    Event, Prefix, Stop, Parallel, build_lts,
    is_deadlock_free, detect_deadlock,
)

a = Event("a")
b = Event("b")

# Two agents waiting on each other — classic deadlock
agent1 = Prefix(a, Prefix(b, Stop()))  # Wants a, then b
agent2 = Prefix(b, Prefix(a, Stop()))  # Wants b, then a

# Parallel with synchronization on both events
system = Parallel(agent1, agent2, sync_events={a, b})

# Check for deadlock
lts = build_lts(system)
if not is_deadlock_free(lts):
    deadlock_state = detect_deadlock(lts)
    print(f"Deadlock at state: {deadlock_state}")
    print(f"Trace to deadlock: {lts.trace_to(deadlock_state)}")
```

Common deadlock patterns to watch for:

- **Circular wait**: A waits for B, B waits for A
- **Missing synchronization**: Agents expect different sync events
- **Unmatched send/receive**: Message types don't align across agents

Use [Session Types](concepts/session-types.md) to eliminate entire classes of deadlocks by construction.

---

## Can I use this without the rest of AgentiCraft?

**Yes.** AgentiCraft Foundation is a standalone library with no dependency on the AgentiCraft platform.

- **Zero external dependencies** for CSP, MPST, and protocol modules
- **NumPy only** for topology and probabilistic verification
- **Pure Python implementation** — works anywhere Python 3.10+ runs

```bash
# Just install Foundation
pip install agenticraft-foundation

# Use independently
from agenticraft_foundation import build_lts, is_deadlock_free
```

This makes Foundation suitable for:

- Academic research on formal methods
- Verification of custom agent frameworks
- Integration with other AI/ML toolkits
- Teaching concurrent systems and process algebra

---

## What's the difference between trace refinement and failures refinement?

Both are refinement orderings in CSP — ways to say "implementation Q satisfies specification P" — but they differ in strength.

| Refinement | Checks | Strength |
|------------|--------|----------|
| **Trace refinement** | All behaviors of Q are allowed by P | Weakest |
| **Failures refinement** | Trace refinement + refusal sets match | Stronger |
| **Failures-divergence** | Failures + no infinite internal loops | Strongest |

```python
from agenticraft_foundation import (
    Event, Prefix, Stop, Skip, ExternalChoice,
    trace_refines, failures_refines, fd_refines,
)

a = Event("a")
b = Event("b")

spec = ExternalChoice(Prefix(a, Stop()), Prefix(b, Stop()))
impl = Prefix(a, Stop())  # Only does 'a', never 'b'

# Trace refinement: impl's traces are a subset of spec's traces
assert trace_refines(impl, spec)  # ✓ impl ⊑ spec

# But impl refuses 'b' while spec doesn't — failures refinement fails
# (Uncomment to see the difference)
# assert failures_refines(impl, spec)  # ✗ Would fail
```

**When to use which**:

- **Trace refinement**: When you only care that the implementation doesn't do anything *new*
- **Failures refinement**: When you also care that the implementation doesn't *refuse* actions the spec would accept
- **FD refinement**: When you need the strongest guarantee (no divergence)

---

## How do timeouts and interrupts work?

Both model preemption — one process taking over from another — but differ in the trigger.

### Timeout

Time-based preemption: if the primary process doesn't act within a duration, the fallback takes over.

```python
from agenticraft_foundation import Event, Prefix, Skip, Timeout

process_data = Event("process_data")
return_result = Event("return_result")
return_cached = Event("return_cached")

primary = Prefix(process_data, Prefix(return_result, Skip()))
fallback = Prefix(return_cached, Skip())

# If primary doesn't engage within 30 seconds, fallback runs
bounded = Timeout(process=primary, duration=30.0, fallback=fallback)
```

### Interrupt

Event-based preemption: when a specific event occurs, control transfers to the handler.

```python
from agenticraft_foundation import Event, Prefix, Stop, Interrupt

process_data = Event("process_data")
handle_priority = Event("handle_priority")

task = Prefix(process_data, Stop())
handler = Prefix(handle_priority, Stop())

# Handler can interrupt the task at any point
agent = Interrupt(primary=task, handler=handler)
```

**When to use which**:

- **Timeout**: Bound operations that might hang (network calls, external services)
- **Interrupt**: Handle priority messages or cancellation signals

---

## What verification capabilities are available?

AgentiCraft Foundation provides multiple verification backends:

| Capability | Module | Description |
|------------|--------|-------------|
| **Deadlock detection** | `algebra` | Find states with no outgoing transitions |
| **Trace extraction** | `algebra` | Enumerate all possible event sequences |
| **Bisimulation** | `algebra` | Structural equivalence of processes |
| **Refinement checking** | `algebra` | Trace, failures, and FD refinement |
| **CTL model checking** | `verification` | Temporal logic (AG, AF, EF, AU) |
| **Probabilistic (DTMC)** | `verification` | Reachability, steady-state analysis |
| **Invariant checking** | `verification` | Runtime assertions on states |
| **Counterexamples** | `verification` | Structured explanations of failures |

```python
from agenticraft_foundation.verification import (
    CTLModelChecker, InvariantChecker, DTMCAnalyzer,
)

# Temporal: check "eventually always responding"
checker = CTLModelChecker(lts)
assert checker.check("AG(EF responding)")

# Probabilistic: compute steady-state distribution
analyzer = DTMCAnalyzer(transition_matrix)
steady_state = analyzer.steady_state()

# Invariants: runtime assertions
invariants = InvariantChecker([lambda s: "error" not in s.events])
```

See the [Verification concepts](concepts/verification.md) and [Temporal Tutorial](tutorials/temporal-verification.md) for details.

---

## Where can I learn more?

- **Getting Started**: [Quick Start](getting-started/quickstart.md), [Why Formal Methods](getting-started/why-formal-methods.md)
- **Concepts**: [Process Algebra](concepts/process-algebra.md), [Session Types](concepts/session-types.md), [Verification](concepts/verification.md)
- **Tutorials**: [CSP Coordination](tutorials/csp-coordination.md), [MPST Verification](tutorials/mpst-verification.md), [Temporal Verification](tutorials/temporal-verification.md)
- **API Reference**: [algebra](api/algebra/index.md), [mpst](api/mpst/index.md), [verification](api/verification/index.md)
- **Community**: [GitHub Discussions](https://github.com/agenticraft/agenticraft-foundation/discussions), [Discord](https://discord.gg/hYymTuv9)

---

*Still have questions? [Open an issue](https://github.com/agenticraft/agenticraft-foundation/issues) or join the [Discord](https://discord.gg/hYymTuv9) community.*
