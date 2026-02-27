"""Interrupt, Timeout, Guard -- Agent-specific CSP extensions.

Demonstrates the 5 new operators for real agent coordination scenarios.
"""

from agenticraft_foundation import (
    TIMEOUT_EVENT,
    Event,
    Guard,
    Interrupt,
    Pipe,
    Prefix,
    Rename,
    Stop,
    Timeout,
    build_lts,
    is_deadlock_free,
    traces,
)

# =============================================================
# Scenario 1: Interruptible Agent Task
# =============================================================
print("=== Interrupt: Priority Override ===")

process_data = Event("process_data")
return_result = Event("return_result")
handle_priority = Event("handle_priority")
urgent_response = Event("urgent_response")

# Normal task: process data, then return result
task = Prefix(process_data, Prefix(return_result, Stop()))

# Priority handler: handle urgent request
handler = Prefix(handle_priority, Prefix(urgent_response, Stop()))

# Interruptible task: can be preempted at any point
interruptible = Interrupt(primary=task, handler=handler)

print(f"Can do: {interruptible.initials()}")
# process_data (normal) OR handle_priority (interrupt)

# After normal event, still interruptible
after_normal = interruptible.after(process_data)
print(f"After process_data: {after_normal}")
print(f"Still interruptible: {isinstance(after_normal, Interrupt)}")

# After interrupt event, handler takes over
after_interrupt = interruptible.after(handle_priority)
print(f"After interrupt: {after_interrupt}")
print(f"Handler active: {not isinstance(after_interrupt, Interrupt)}")

# =============================================================
# Scenario 2: LLM Call with Timeout
# =============================================================
print("\n=== Timeout: Bounded LLM Call ===")

call_llm = Event("call_gpt4")
parse_response = Event("parse_response")
return_cached = Event("return_cached")

# LLM call process
llm_call = Prefix(call_llm, Prefix(parse_response, Stop()))

# Fallback: return cached response
fallback = Prefix(return_cached, Stop())

# Bounded execution: 30 second timeout
bounded = Timeout(process=llm_call, duration=30.0, fallback=fallback)

print(f"Initials: {bounded.initials()}")
# call_gpt4 (start LLM call) OR tau_timeout (timeout fires)

# If timeout fires, switch to fallback
after_timeout = bounded.after(TIMEOUT_EVENT)
print(f"After timeout: {after_timeout}")
assert return_cached in after_timeout.initials()

# If LLM starts, still has timeout
after_start = bounded.after(call_llm)
print(f"After call_llm: {after_start}")
assert isinstance(after_start, Timeout)

# =============================================================
# Scenario 3: Conditional Agent Activation
# =============================================================
print("\n=== Guard: Budget-Gated Agent ===")

budget = {"remaining": 100.0}

expensive_call = Event("call_gpt4o")
process = Event("process_result")

expensive_agent = Prefix(expensive_call, Prefix(process, Stop()))

# Only activate if budget allows
guarded = Guard(
    condition=lambda: budget["remaining"] > 0,
    process=expensive_agent,
)

print(f"Budget > 0: initials = {guarded.initials()}")
assert expensive_call in guarded.initials()

# Deplete budget
budget["remaining"] = 0
print(f"Budget = 0: initials = {guarded.initials()}")
assert guarded.initials() == frozenset()  # Acts as Stop

# Restore budget
budget["remaining"] = 50.0
print(f"Budget restored: initials = {guarded.initials()}")
assert expensive_call in guarded.initials()

# =============================================================
# Scenario 4: Protocol Bridging via Rename
# =============================================================
print("\n=== Rename: Protocol Bridging ===")

# Agent A uses "task_done" event
task_done = Event("task_done")
agent_a = Prefix(task_done, Stop())

# Agent B expects "work_complete" event
work_complete = Event("work_complete")

# Bridge: rename task_done -> work_complete
bridged = Rename.from_dict(agent_a, {task_done: work_complete})

print(f"Original initials: {agent_a.initials()}")
print(f"Bridged initials: {bridged.initials()}")
assert work_complete in bridged.initials()
assert task_done not in bridged.initials()

# =============================================================
# Scenario 5: RAG Pipeline
# =============================================================
print("\n=== Pipe: RAG Pipeline ===")

query = Event("query")
emit_docs = Event("emit_docs")
summarize = Event("summarize")

# Retriever: receives query, emits documents
retriever = Prefix(query, Prefix(emit_docs, Stop()))

# Processor: receives documents, produces summary
processor = Prefix(emit_docs, Prefix(summarize, Stop()))

# Pipeline: emit_docs is internal channel
rag_pipeline = Pipe(
    producer=retriever,
    consumer=processor,
    channel=frozenset({emit_docs}),
)

print(f"Pipeline alphabet: {rag_pipeline.alphabet()}")
assert query in rag_pipeline.alphabet()
assert summarize in rag_pipeline.alphabet()
assert emit_docs not in rag_pipeline.alphabet()  # Hidden!

# =============================================================
# Analysis
# =============================================================
print("\n=== Analysis ===")

lts = build_lts(interruptible)
t = list(traces(lts, max_length=5))
print(f"Interruptible traces ({len(t)}): {t[:5]}")

print(f"Timeout process deadlock-free: {is_deadlock_free(bounded)}")
print(f"Pipeline deadlock-free: {is_deadlock_free(rag_pipeline)}")

print("\nAll examples passed.")
