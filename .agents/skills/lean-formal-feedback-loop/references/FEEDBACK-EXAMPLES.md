# Feedback Loop Examples (Adversarial Mode)

> Each example follows the same loop: theorem friction -> witness extraction -> route decision -> artifact record.

## Example 1: Notify Lost Wakeup

### Frog
`notify.no_lost_wakeup`: every `notify_one` either wakes exactly one waiter or stores one notification without loss.

### Failure Class
atomic + lock split introduces interleaving window.

### Proof Friction Signal
Lean cannot construct a linearization witness for the transition when `stored_notifications` is read outside lock.

### Witness Extraction
Generate a three-thread interleaving trace and port to lab runtime as deterministic seed.

### Route Decision
`code-first` if witness reproduces missed wake.

### Closure
1. move linearization point into locked region
2. add regression test for discovered schedule
3. theorem discharges with new step semantics

### Artifact Snippet

```json
{
  "frog_id": "notify.no_lost_wakeup",
  "route": "code-first",
  "witness": "seed=42 schedule=A-read,B-inc,C-register,A-lock",
  "test": "src/sync/notify.rs::tests::no_lost_wakeup_interleaving",
  "artifact_hash": "sha256:..."
}
```

## Example 2: Semaphore Cascading Wakeups

### Frog
`semaphore.all_eligible_waiters_wake`: adding N permits eventually wakes all eligible waiters.

### Failure Class
Liveness delegated to incidental polling order.

### Proof Friction Signal
Termination proof fails — queue length does not strictly decrease under current step.

### Witness Extraction
Counterexample: first awakened waiter exits before cascades occur.

### Route Decision
`code-first` if queue stagnation is reproducible.

### Closure
Add explicit wake propagation in `add_permits`, then reprove monotonic decrease of waiting set. Proof shape: define queue length metric, show each wake strictly decreases it, prove no transition increases it without permit deficit, conclude eventual drain under fairness.

## Example 3: Pool Capacity Linearizability

### Frog
`pool.capacity_linearizable`: `entries <= max_size` under all admissible interleavings.

### Failure Class
check-then-act race (TOCTOU).

### Proof Friction Signal
proof requires a linearization point for `acquire`; none exists in current split check/create design.

### Witness Extraction
two-thread trace both passing check before either increment commits.

### Route Decision
`code-first` when overflow reproduced.

### Closure
serialize check+reserve in one critical section (or equivalent atomic protocol), then prove invariant preservation by step induction.

### Common Mistake

Only proving `entries <= max_size` at function exit is insufficient; theorem must quantify over intermediate interleavings as well.

## Example 4: Split-State Drift (Model-First Case)

### Frog
`runtime.split_state_refinement`: split tables refine old flat model.

### Failure Class
abstraction mismatch, not necessarily implementation bug.

### Proof Friction Signal
proof fails only because Lean model assumes atomic joint access to regions+obligations.

### Witness Extraction
Rust tests pass; no counterexample schedule violating intended contract.

### Route Decision
`model-first`.

### Closure
refactor Lean state model with explicit table boundaries + synchronization assumptions, then reprove refinement.

### Artifact Focus

For model-first fixes, artifact should emphasize:
- updated refinement relation
- preserved theorem IDs
- no behavior regression in mapped Rust tests

## Diagnostic Table: Bug vs Model

| Signal | Likely Route |
|---|---|
| concrete failing interleaving reproduced in Rust | code-first |
| proof fails, Rust witness impossible under implementation | model-first |
| conformance fixtures/mappings stale | harness-first |
| theorem stronger than intended API contract | theorem-first |

## Example 5: Theorem-First (Over-Strong Claim)

Claim: "all waiters are woken immediately after permit arrival." But API only guarantees eventual wake under fairness.

Route: `theorem-first` — split into safety (no permit loss, unconditional) + liveness (eventual wake, requires fairness axiom). No Rust or model bug; theorem precision bug fixed.

## Galaxy-Brain Card Template For Stuck Points

Use this card format in issue notes:

```text
Card: linearization_gap
Claim fragment:
Concrete substitution:
Why this blocks:
Best next discriminator:
Expected Bayes factor impact:
```

Example:

```text
Card: linearization_gap
Claim fragment: acquire preserves entries <= max_size
Concrete substitution: entries=max_size-1, two concurrent acquires
Why this blocks: no single commit point for capacity reservation
Best next discriminator: lab test with deterministic two-thread schedule
Expected Bayes factor impact: +4.0 toward code defect
```

## Deep-Math Elicitation Prompt (When Stuck)

Use this only when standard route discrimination stalls:

```text
Now, TRULY think even harder. Surely there is some math invented in the
last 60 years that would be relevant and helpful here? Super hard, esoteric
math that would be ultra accretive and give a ton of alpha for the specific
problems we're trying to solve here, as efficiently as possible?

REALLY RUMINATE ON THIS!!! DIG DEEP!!

STUFF THAT EVEN TERRY TAO WOULD HAVE TO CONCENTRATE SUPER HARD ON!
```

Purpose: find a better theorem decomposition or invariant, not to skip witness/conformance gates.

## Troubleshooting Quick Map

| Symptom | Likely Cause | Route |
|---|---|---|
| Lean proof blocks and Rust witness reproduces failure | Real implementation defect | `code-first` |
| Lean proof blocks and Rust witness is not constructible | Model abstraction drift | `model-first` |
| Proof passes but conformance tests fail | Mapping/tests stale | `harness-first` |
| Safety holds but liveness claim keeps failing | Claim too strong | `theorem-first` |

## Fast Triage Heuristic

If unsure which route to pick:

1. try to build executable witness in Rust
2. if witness fails reliably -> code-first
3. if witness impossible but proof still blocks -> model-first
4. if both appear correct but claim too strong -> theorem-first
