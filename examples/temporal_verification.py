"""CTL Temporal Logic Verification -- model checking over LTS.

Demonstrates CTL formula construction and backward fixpoint model checking
on a multi-agent task assignment system.
"""

from agenticraft_foundation import (
    Event,
    ExternalChoice,
    Prefix,
    Recursion,
    Stop,
    Variable,
    build_lts,
)
from agenticraft_foundation.verification import (
    AF,
    AG,
    EF,
    EU,
    And,
    Atomic,
    Not,
    check_liveness,
    check_safety,
    model_check,
)

# =============================================================
# Model a task assignment agent
# =============================================================
print("=== Task Assignment Agent ===")

# Events
receive = Event("receive_task")
process = Event("process")
succeed = Event("succeed")
fail = Event("fail")
retry = Event("retry")

# Agent: receive task, process, then succeed or fail
# On failure, agent can retry (loops back)
agent = Recursion(
    variable="X",
    body=Prefix(
        receive,
        Prefix(
            process,
            ExternalChoice(
                left=Prefix(succeed, Stop()),
                right=Prefix(fail, Prefix(retry, Variable("X"))),
            ),
        ),
    ),
)

lts = build_lts(agent, max_states=50)
print(f"States: {len(lts.states)}")
print(f"Transitions: {sum(len(list(lts.successors(s))) for s in lts.states)}")

# =============================================================
# Label states with atomic propositions
# =============================================================
print("\n=== State Labeling ===")

labeling: dict[int, set[str]] = {s: set() for s in lts.states}
labeling[lts.initial_state].add("idle")

# Label states by their outgoing events
for s in lts.states:
    outgoing = {e for e, _ in lts.successors(s)}
    if receive in outgoing:
        labeling[s].add("idle")
    if process in outgoing:
        labeling[s].add("received")
    if succeed in outgoing or fail in outgoing:
        labeling[s].add("processing")
    if retry in outgoing:
        labeling[s].add("failed")
    if not outgoing:
        labeling[s].add("done")

for s in sorted(lts.states):
    if labeling[s]:
        print(f"  State {s}: {labeling[s]}")

# =============================================================
# CTL property verification
# =============================================================
print("\n=== Safety: errors never occur ===")
# AG(¬error) -- since we never labeled "error", this trivially holds
result = check_safety(lts, "error", labeling)
print(f"  No error states: {result.satisfied}")

print("\n=== Liveness: task always eventually completes ===")
# AF(done) -- from every state, done is eventually reachable
result = check_liveness(lts, "done", labeling)
print(f"  Always terminates: {result.satisfied}")
if not result.satisfied:
    print(f"  Counterexample trace: {result.counterexample}")

print("\n=== Existential reachability: success is possible ===")
# EF(done) -- there exists a path to done
result = model_check(lts, EF(Atomic("done")), labeling)
print(f"  Can reach done: {result.satisfied}")
print(f"  States where done is reachable: {result.satisfying_states}")

print("\n=== Until: processing holds until done ===")
# Check from processing states: processing Until done
result = model_check(
    lts, EU(Atomic("processing"), Atomic("done")), labeling
)
print(f"  Processing until done (exists path): {result.satisfied}")

print("\n=== Combined: safe progress ===")
# AG(idle → EF(done)) -- from every idle state, done is reachable
result = model_check(
    lts,
    AG(Not(And(Atomic("idle"), Not(EF(Atomic("done")))))),
    labeling,
)
print(f"  Idle always leads to possible completion: {result.satisfied}")

# =============================================================
# Mutual exclusion example
# =============================================================
print("\n=== Mutual Exclusion Model ===")

enter_a = Event("enter_a")
exit_a = Event("exit_a")
enter_b = Event("enter_b")
exit_b = Event("exit_b")

# Two processes that should not be in critical section simultaneously
proc_a = Prefix(enter_a, Prefix(exit_a, Stop()))
proc_b = Prefix(enter_b, Prefix(exit_b, Stop()))

mutex = ExternalChoice(
    left=Prefix(enter_a, Prefix(exit_a, Prefix(enter_b, Prefix(exit_b, Stop())))),
    right=Prefix(enter_b, Prefix(exit_b, Prefix(enter_a, Prefix(exit_a, Stop())))),
)

mutex_lts = build_lts(mutex)
mutex_labels: dict[int, set[str]] = {s: set() for s in mutex_lts.states}

for s in mutex_lts.states:
    mutex_outgoing = {e for e, _ in mutex_lts.successors(s)}
    if exit_a in mutex_outgoing:
        mutex_labels[s].add("critical_a")
    if exit_b in mutex_outgoing:
        mutex_labels[s].add("critical_b")

# AG(¬(critical_a ∧ critical_b)) -- mutual exclusion
result = model_check(
    mutex_lts,
    AG(Not(And(Atomic("critical_a"), Atomic("critical_b")))),
    mutex_labels,
)
print(f"  Mutual exclusion holds: {result.satisfied}")

print("\nDone.")
